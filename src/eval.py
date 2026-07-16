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
            rows.append({"text": obj[text_field], "label": int(obj[label_field])})
    return Dataset.from_list(rows)

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
    
    # 6. Run Evaluation
    print("Running evaluation on test split...")
    results = trainer.evaluate()
    
    print("\n=== Evaluation Results ===")
    print(f"Accuracy:  {results['eval_accuracy']:.4f}")
    print(f"F1-Score:  {results['eval_f1']:.4f}")
    print(f"Precision: {results['eval_precision']:.4f}")
    print(f"Recall:    {results['eval_recall']:.4f}")
    print(f"ROC-AUC:   {results['eval_auc']:.4f}")
    
    # 7. Save results
    if args.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Saved results to {args.output_file}")

if __name__ == "__main__":
    main()
