import argparse
import json
import os
import random
import re
from datetime import datetime, timezone

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_loader import clean_source, load_aligned_juliet, load_aligned_realvul


PROMPT = """Rewrite this C function.

Rules:
- Keep exactly the same behavior.
- Do not fix, remove, or introduce any vulnerability.
- Rename variables and the function to different natural names.
- Change formatting and simple loop/branch style only when it is safe.
- Do not add comments.
- Return only C code, no explanation, no markdown.

C function:
{code}
"""


def extract_code(text):
    text = text.strip()
    fenced = re.search(r"```(?:c|cpp|C|C\+\+)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    # Keep the most likely C function if the model adds short prose anyway.
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        prefix = text[:first_brace]
        header_start = max(prefix.rfind("\n"), 0)
        text = text[header_start:last_brace + 1].strip()

    return text


def looks_like_c_function(code):
    if not code or len(code) < 40:
        return False
    if "{" not in code or "}" not in code:
        return False
    if code.count("{") != code.count("}"):
        return False
    if re.search(r"\b(return|if|for|while|switch|memcpy|malloc|free|pthread)\b", code) is None:
        return False
    if re.search(r"\w+\s+\w+\s*\([^;]*\)\s*\{", code, re.DOTALL) is None:
        return False
    return True


def balanced_sample(rows, total, seed):
    per_label = total // 2
    rng = random.Random(seed)
    by_label = {0: [], 1: []}

    for row in rows:
        label = int(row["label"])
        if label in by_label:
            by_label[label].append(row)

    for label, items in by_label.items():
        if len(items) < per_label:
            raise ValueError(
                f"Need {per_label} rows for label {label}, found only {len(items)}."
            )
        rng.shuffle(items)

    sampled = by_label[0][:per_label] + by_label[1][:per_label]
    rng.shuffle(sampled)
    return sampled


def load_rows(dataset_name, total, seed):
    if dataset_name == "juliet":
        dataset = load_aligned_juliet(toy=False)
        rows = []
        for row in dataset["test"]:
            rows.append(
                {
                    "dataset": "juliet",
                    "sample_id": row["file"],
                    "fun_name": row["fun_name"],
                    "source_code": row["source_code"],
                    "label": int(row["label"]),
                }
            )
        return balanced_sample(rows, total, seed)

    if dataset_name == "realvul":
        dataset = load_aligned_realvul(toy=False)
        rows = []
        for row in dataset["test"]:
            rows.append(
                {
                    "dataset": "realvul",
                    "sample_id": row["fun_name"],
                    "fun_name": row["fun_name"],
                    "source_code": row["source_code"],
                    "label": int(row["label"]),
                }
            )
        return balanced_sample(rows, total, seed)

    raise ValueError(f"Unknown dataset: {dataset_name}")


def load_existing_accepted(output_path):
    done = set()
    counts = {0: 0, 1: 0}
    if not os.path.exists(output_path):
        return done, counts

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("rewrite_status") == "accepted":
                done.add(row.get("sample_id"))
                label = int(row.get("label"))
                if label in counts:
                    counts[label] += 1
    return done, counts


def build_messages(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def rewrite_one(model, tokenizer, code, args):
    prompt = PROMPT.format(code=code)
    chat_prompt = build_messages(tokenizer, prompt)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    raw_text = tokenizer.decode(generated, skip_special_tokens=True)
    rewritten = extract_code(raw_text)
    return raw_text, rewritten


def write_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_dataset(dataset_name, model, tokenizer, args):
    output_path = os.path.join(
        args.output_dir,
        f"llm_rewritten_{dataset_name}_test_{args.samples_per_dataset}.jsonl",
    )
    target_per_label = args.samples_per_dataset // 2
    candidate_count = args.samples_per_dataset * args.candidate_multiplier
    rows = load_rows(dataset_name, candidate_count, args.seed)
    existing_ids, accepted_by_label = load_existing_accepted(output_path)

    accepted = 0
    rejected = 0
    skipped = 0

    for row in tqdm(rows, desc=f"Rewriting {dataset_name}"):
        label = int(row["label"])
        if all(count >= target_per_label for count in accepted_by_label.values()):
            break
        if accepted_by_label[label] >= target_per_label:
            skipped += 1
            continue
        if row["sample_id"] in existing_ids:
            skipped += 1
            continue

        raw_text = ""
        rewritten = ""
        status = "rejected"
        reason = ""

        try:
            raw_text, rewritten = rewrite_one(model, tokenizer, row["source_code"], args)
            if not looks_like_c_function(rewritten):
                reason = "not_a_valid_c_function_shape"
            else:
                status = "accepted"
        except Exception as exc:
            reason = f"generation_error: {type(exc).__name__}: {exc}"

        cleaned_rewrite = ""
        if rewritten:
            cleaned_rewrite = clean_source(
                rewritten,
                fun_name=None,
                juliet=(dataset_name == "juliet"),
            )

        out_row = {
            "dataset": dataset_name,
            "split": "test",
            "sample_id": row["sample_id"],
            "original_fun_name": row["fun_name"],
            "label": int(row["label"]),
            "original_source": row["source_code"],
            "rewritten_source": cleaned_rewrite,
            "raw_model_output": raw_text,
            "rewrite_status": status,
            "reject_reason": reason,
            "model_name": args.model_name,
            "prompt_version": "v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_jsonl(output_path, out_row)

        if status == "accepted":
            accepted += 1
            accepted_by_label[label] += 1
        else:
            rejected += 1

    return {
        "dataset": dataset_name,
        "output_path": output_path,
        "target_samples": args.samples_per_dataset,
        "accepted_this_run": accepted,
        "rejected_this_run": rejected,
        "skipped_existing": skipped,
        "accepted_total_by_label": accepted_by_label,
        "complete": all(count >= target_per_label for count in accepted_by_label.values()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate balanced LLM-rewritten Juliet and CompRealVul test sets."
    )
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Hugging Face causal/chat model used for rewriting.",
    )
    parser.add_argument("--samples_per_dataset", type=int, default=200)
    parser.add_argument(
        "--candidate_multiplier",
        type=int,
        default=3,
        help="Sample extra candidates so rejected rewrites do not reduce the final accepted count.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--output_dir", default="./data")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["juliet", "realvul"],
        choices=["juliet", "realvul"],
    )
    parser.add_argument(
        "--torch_dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    args = parser.parse_args()

    if args.samples_per_dataset % 2 != 0:
        raise ValueError("--samples_per_dataset must be even for balanced labels.")

    os.makedirs(args.output_dir, exist_ok=True)

    dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.torch_dtype]

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()

    summaries = []
    for dataset_name in args.datasets:
        summaries.append(generate_dataset(dataset_name, model, tokenizer, args))

    summary_path = os.path.join(args.output_dir, "llm_rewrite_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "samples_per_dataset": args.samples_per_dataset,
                "seed": args.seed,
                "summaries": summaries,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
