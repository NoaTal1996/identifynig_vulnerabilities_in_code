import argparse
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from data_loader import get_dataset
from eval import _metrics_from_arrays, bootstrap_metrics

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Softmax to get probabilities for class 1 (needed for ROC-AUC)
    # subtracting max for numerical stability
    shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    probs_class_1 = probs[:, 1]
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
    acc = accuracy_score(labels, predictions)
    
    try:
        auc = roc_auc_score(labels, probs_class_1)
    except ValueError:
        # Fallback if the evaluation slice contains only one class (often happens in toy runs)
        auc = 0.5
        
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }

def main():
    parser = argparse.ArgumentParser(description="Fine-tune CodeBERT for Vulnerability Detection")
    parser.add_argument("--dataset", type=str, required=True, choices=["juliet", "realvul"], help="Dataset to load")
    parser.add_argument("--rep", type=str, required=True, choices=["source", "llvm_ir"], help="Code representation")
    parser.add_argument("--model_name", type=str, default="microsoft/codebert-base", help="Model pre-trained weights path or HF repo")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length for tokenization")
    parser.add_argument("--toy", action="store_true", help="Use a tiny subset of the dataset for testing")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save checkpoints and metrics")
    parser.add_argument("--n_boot", type=int, default=1000, help="Bootstrap resamples for 95%% CIs on the test-set metrics (0 to disable)")
    parser.add_argument("--boot_seed", type=int, default=42, help="Bootstrap RNG seed")
    args = parser.parse_args()
    
    print(f"=== Starting Training Pipeline ===")
    print(f"Dataset: {args.dataset} | Representation: {args.rep} | Toy Run: {args.toy}")
    print(f"Model: {args.model_name} | Epochs: {args.epochs} | LR: {args.lr}")
    
    # 1. Load aligned dataset
    dataset = get_dataset(args.dataset, args.rep, toy=args.toy)
    
    # 2. Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # 3. Preprocess / Tokenize Dataset
    print("Tokenizing datasets...")
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=args.max_length,
            padding=False  # Padding is handled dynamically by data collator
        )
        
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # 4. Load Model
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    
    # 5. Define Training Arguments
    run_name = f"{args.dataset}_{args.rep}"
    if args.toy:
        run_name += "_toy"
    run_output_dir = os.path.join(args.output_dir, run_name)
    
    training_args = TrainingArguments(
        output_dir=run_output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10 if args.toy else 100,
        report_to="none",  # disable logging to wandb/tensorboard for simplicity
        disable_tqdm=False
    )
    
    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    
    # 7. Train model
    print("Training model...")
    trainer.train()

    # Save the best model (load_best_model_at_end=True) and tokenizer to the
    # run root, so eval.py can load --model_path <run_output_dir> directly
    # instead of hunting for a checkpoint-* folder.
    trainer.save_model(run_output_dir)
    tokenizer.save_pretrained(run_output_dir)
    print(f"Saved best model and tokenizer to {run_output_dir}")


    # 8. Evaluate on test set (predict → per-example predictions → point + bootstrap CI)
    print("Evaluating model on test set...")
    predictions_out = trainer.predict(tokenized_datasets['test'])
    logits = predictions_out.predictions
    y_true = predictions_out.label_ids
    y_pred = np.argmax(logits, axis=-1)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=-1, keepdims=True)
    y_prob = probs[:, 1]

    point = _metrics_from_arrays(y_true, y_pred, y_prob)
    print(f"\n=== Test Results for {run_name} (point estimates) ===")
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

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, f"{run_name}_metrics.json")
    with open(results_path, "w") as f:
        json.dump({
            "run_name": run_name,
            "dataset": args.dataset,
            "rep": args.rep,
            "n_samples": int(len(y_true)),
            "point_estimates": point,
            "bootstrap_ci_95": ci,
            "n_boot": int(args.n_boot),
            "boot_seed": int(args.boot_seed),
        }, f, indent=4)
    print(f"Saved final metrics to {results_path}")

    pred_path = os.path.join(args.output_dir, f"{run_name}_predictions.jsonl")
    with open(pred_path, "w") as f:
        for i, (yt, yp, pr) in enumerate(zip(y_true, y_pred, y_prob)):
            f.write(json.dumps({
                "idx": i,
                "sample_id": "",
                "label": int(yt),
                "pred": int(yp),
                "prob_class1": float(pr),
            }) + "\n")
    print(f"Saved per-example predictions to {pred_path}")

if __name__ == "__main__":
    main()
