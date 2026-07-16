import argparse
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from data_loader import get_dataset


def load_jsonl_as_dataset(path, text_field="rewritten_source", label_field="label"):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append({
                "text": obj[text_field],
                "label": int(obj[label_field]),
                "sample_id": obj.get("sample_id", ""),
            })
    return Dataset.from_list(rows)


def _metrics_from_arrays(y_true, y_pred, y_prob):
    """Compute the same five metrics on plain arrays (used for bootstrap)."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall, "auc": auc}


def bootstrap_metrics(y_true, y_pred, y_prob, n_boot=1000, seed=42, alpha=0.05):
    """Percentile bootstrap 95% CIs for each metric.

    Resamples the test set with replacement `n_boot` times and returns for each
    metric the point estimate on the full set together with lower/upper bounds.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    point = _metrics_from_arrays(y_true, y_pred, y_prob)
    rng = np.random.default_rng(seed)
    stacks = {k: [] for k in point}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = _metrics_from_arrays(y_true[idx], y_pred[idx], y_prob[idx])
        for k, v in m.items():
            stacks[k].append(v)

    lo_p, hi_p = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    ci = {}
    for k, vals in stacks.items():
        vals_np = np.asarray(vals, dtype=float)
        finite = vals_np[np.isfinite(vals_np)]
        if finite.size == 0:
            ci[k] = {"point": point[k], "ci_low": float("nan"), "ci_high": float("nan"),
                     "n_boot": 0, "n_boot_total": n_boot}
        else:
            ci[k] = {
                "point": point[k],
                "ci_low": float(np.percentile(finite, lo_p)),
                "ci_high": float(np.percentile(finite, hi_p)),
                "n_boot": int(finite.size),
                "n_boot_total": n_boot,
            }
    return ci

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Softmax to get probabilities for class 1
    shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    probs_class_1 = probs[:, 1]
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
    acc = accuracy_score(labels, predictions)
    
    try:
        auc = roc_auc_score(labels, probs_class_1)
    except ValueError:
        auc = 0.5
        
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model (supports cross-dataset evaluation)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model checkpoint directory")
    parser.add_argument("--dataset", type=str, choices=["juliet", "realvul"], help="Dataset to evaluate on (ignored if --jsonl_path is set)")
    parser.add_argument("--rep", type=str, choices=["source", "llvm_ir"], help="Code representation (ignored if --jsonl_path is set)")
    parser.add_argument("--jsonl_path", type=str, help="Optional path to a local JSONL file (LLM-rewritten data). Uses 'rewritten_source' and 'label' fields.")
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--toy", action="store_true", help="Use a tiny subset of the dataset")
    parser.add_argument("--output_file", type=str, help="Path to write JSON results file")
    parser.add_argument("--n_boot", type=int, default=1000, help="Bootstrap resamples for 95%% CIs (0 to disable)")
    parser.add_argument("--boot_seed", type=int, default=42, help="Bootstrap RNG seed")
    args = parser.parse_args()

    print(f"=== Starting Evaluation Pipeline ===")
    print(f"Model Path: {args.model_path}")

    # 1. Load test dataset — either from HF (aligned) or from a local JSONL (LLM-rewritten)
    if args.jsonl_path:
        print(f"Evaluating on local JSONL: {args.jsonl_path}")
        test_split = load_jsonl_as_dataset(args.jsonl_path)
        if args.toy:
            test_split = test_split.select(range(min(50, len(test_split))))
    else:
        if not args.dataset or not args.rep:
            parser.error("--dataset and --rep are required when --jsonl_path is not set")
        print(f"Evaluating on Dataset: {args.dataset} | Representation: {args.rep} | Toy Run: {args.toy}")
        dataset = get_dataset(args.dataset, args.rep, toy=args.toy)
        test_split = dataset['test']
    
    # 2. Load Tokenizer from the model path
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # 3. Tokenize Dataset
    print("Tokenizing test dataset...")
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=args.max_length,
            padding=False
        )
        
    tokenized_test = test_split.map(tokenize_function, batched=True)
    
    # 4. Load Model
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    
    # 5. Initialize Trainer with minimal settings for evaluation
    eval_args = TrainingArguments(
        output_dir="./eval_tmp",
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
        disable_tqdm=False
    )
    
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=tokenized_test,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    
    # 6. Run prediction to get per-example logits (needed for bootstrap + comparisons)
    print("Running evaluation on test split...")
    predictions_out = trainer.predict(tokenized_test)
    logits = predictions_out.predictions
    y_true = predictions_out.label_ids
    y_pred = np.argmax(logits, axis=-1)

    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=-1, keepdims=True)
    y_prob = probs[:, 1]

    point = _metrics_from_arrays(y_true, y_pred, y_prob)
    print("\n=== Evaluation Results (point estimates) ===")
    print(f"Accuracy:  {point['accuracy']:.4f}")
    print(f"F1-Score:  {point['f1']:.4f}")
    print(f"Precision: {point['precision']:.4f}")
    print(f"Recall:    {point['recall']:.4f}")
    print(f"ROC-AUC:   {point['auc']:.4f}")

    ci = None
    if args.n_boot > 0:
        print(f"\nBootstrapping 95% CIs (n_boot={args.n_boot})...")
        ci = bootstrap_metrics(y_true, y_pred, y_prob, n_boot=args.n_boot, seed=args.boot_seed)
        for k in ["accuracy", "f1", "precision", "recall", "auc"]:
            lo, hi = ci[k]["ci_low"], ci[k]["ci_high"]
            print(f"  {k:<9} {point[k]:.4f}  [{lo:.4f}, {hi:.4f}]")

    # 7. Save results and per-example predictions
    if args.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        out = {
            "model_path": args.model_path,
            "n_samples": int(len(y_true)),
            "point_estimates": point,
            "bootstrap_ci_95": ci,
            "n_boot": int(args.n_boot),
            "boot_seed": int(args.boot_seed),
        }
        if args.jsonl_path:
            out["jsonl_path"] = args.jsonl_path
        else:
            out["dataset"] = args.dataset
            out["rep"] = args.rep
        with open(args.output_file, "w") as f:
            json.dump(out, f, indent=4)
        print(f"Saved results to {args.output_file}")

        # Per-example predictions for post-hoc comparisons (McNemar, paired bootstrap).
        pred_path = os.path.splitext(args.output_file)[0] + "_predictions.jsonl"
        sample_ids = tokenized_test["sample_id"] if "sample_id" in tokenized_test.column_names else [""] * len(y_true)
        with open(pred_path, "w") as f:
            for i, (yt, yp, pr, sid) in enumerate(zip(y_true, y_pred, y_prob, sample_ids)):
                f.write(json.dumps({
                    "idx": i,
                    "sample_id": sid,
                    "label": int(yt),
                    "pred": int(yp),
                    "prob_class1": float(pr),
                }) + "\n")
        print(f"Saved per-example predictions to {pred_path}")

if __name__ == "__main__":
    main()
