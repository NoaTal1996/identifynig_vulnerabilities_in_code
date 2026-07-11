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
        evaluation_strategy="epoch",
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
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    
    # 7. Train model
    print("Training model...")
    trainer.train()
    
    # 8. Evaluate on test set
    print("Evaluating model on test set...")
    test_results = trainer.evaluate(tokenized_datasets['test'])
    print(f"\n=== Test Results for {run_name} ===")
    print(f"Accuracy:  {test_results['eval_accuracy']:.4f}")
    print(f"F1-Score:  {test_results['eval_f1']:.4f}")
    print(f"Precision: {test_results['eval_precision']:.4f}")
    print(f"Recall:    {test_results['eval_recall']:.4f}")
    print(f"ROC-AUC:   {test_results['eval_auc']:.4f}")
    
    # Save test results to a file
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, f"{run_name}_metrics.json")
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=4)
    print(f"Saved final metrics to {results_path}")

if __name__ == "__main__":
    main()
