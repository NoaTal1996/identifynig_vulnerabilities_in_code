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

# Step 2: Run Same-Form Cross-Dataset Evaluation (Juliet source model -> RealVul source test)
# Mirrors the real experiment design (exam E3): transfer across datasets, never across forms.
echo -e "\n--> Running RealVul Source Evaluation (zero-shot transfer, same form)..."
python3 src/eval.py \
  --model_path ./results/juliet_source_toy \
  --dataset realvul \
  --rep source \
  --batch_size 2 \
  --max_length 128 \
  --toy \
  --output_file ./results/juliet_to_realvul_source_metrics.json

echo -e "\n============================================="
echo "Verification Completed Successfully!"
echo "============================================="
