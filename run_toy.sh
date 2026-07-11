#!/bin/bash
set -e

echo "============================================="
echo "Running End-to-End Toy Pipeline Verification"
echo "============================================="

# Create results directory if it doesn't exist
mkdir -p results

# Step 1: Run Juliet Source Code Toy Training (1 epoch, tiny batch size, short max_length)
echo -e "\n--> Running Juliet Source Code Toy Training..."
python3 src/train.py \
  --dataset juliet \
  --rep source \
  --epochs 1 \
  --batch_size 2 \
  --max_length 128 \
  --toy \
  --output_dir ./results

# Step 2: Run Cross-Dataset Evaluation (using the trained source checkpoint to evaluate LLVM IR)
# This verifies that the evaluation pipeline can load checkpoints and run cross-representation transfer.
echo -e "\n--> Running Juliet LLVM IR Evaluation (using the trained checkpoint)..."
python3 src/eval.py \
  --model_path ./results/juliet_source_toy \
  --dataset juliet \
  --rep llvm_ir \
  --batch_size 2 \
  --max_length 128 \
  --toy \
  --output_file ./results/juliet_llvm_cross_metrics.json

echo -e "\n============================================="
echo "Verification Completed Successfully!"
echo "============================================="
