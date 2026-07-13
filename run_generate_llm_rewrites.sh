#!/bin/bash
# Generate the AI-style exam data:
# - 200 balanced Juliet test rewrites
# - 200 balanced CompRealVul test rewrites
#
# Recommended on Kaggle GPU:
#   bash run_generate_llm_rewrites.sh
#
set -e

mkdir -p data

python3 src/generate_llm_rewrites.py \
  --model_name Qwen/Qwen2.5-Coder-7B-Instruct \
  --samples_per_dataset 200 \
  --candidate_multiplier 3 \
  --seed 42 \
  --temperature 0 \
  --max_new_tokens 768 \
  --output_dir ./data \
  --datasets juliet realvul
