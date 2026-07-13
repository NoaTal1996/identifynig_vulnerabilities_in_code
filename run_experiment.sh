#!/bin/bash
# End-to-end experiment: Tier 1 (M1-M4, exams E1/E2/E3).
# E1 = home exam for M1/M2 (test on Juliet)      -> produced automatically by train.py
# E2 = home exam for M3/M4 (test on CompRealVul) -> produced automatically by train.py
# E3 = away exam for M1/M2 (zero-shot on CompRealVul, same form)
#
# Usage:
#   ./run_experiment.sh              # full run
#   ./run_experiment.sh toy          # tiny sanity-check run
#
set -e

MODE=${1:-full}
if [ "$MODE" = "toy" ]; then
  TOY_FLAG="--toy"
  EPOCHS=1
  BATCH=2
  MAXLEN=128
  SUFFIX="_toy"
  echo "==> Running in TOY mode"
else
  TOY_FLAG=""
  EPOCHS=3
  BATCH=8
  MAXLEN=512
  SUFFIX=""
  echo "==> Running in FULL mode"
fi

OUT=./results
mkdir -p "$OUT"

train_one () {
  local dataset=$1
  local rep=$2
  local tag=$3   # human name (M1/M2/M3/M4)
  echo -e "\n=============================================="
  echo "Training $tag  (dataset=$dataset  rep=$rep)"
  echo "=============================================="
  python3 src/train.py \
    --dataset "$dataset" \
    --rep "$rep" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH" \
    --max_length "$MAXLEN" \
    $TOY_FLAG \
    --output_dir "$OUT"
}

eval_transfer () {
  local model_dir=$1
  local dataset=$2
  local rep=$3
  local outfile=$4
  echo -e "\n----------------------------------------------"
  echo "Zero-shot: $model_dir  -->  $dataset/$rep"
  echo "----------------------------------------------"
  python3 src/eval.py \
    --model_path "$model_dir" \
    --dataset "$dataset" \
    --rep "$rep" \
    --batch_size "$BATCH" \
    --max_length "$MAXLEN" \
    $TOY_FLAG \
    --output_file "$outfile"
}

# ---- Train the 4 models (E1 & E2 are computed at the end of each training run) ----
train_one juliet  source   M1
train_one juliet  llvm_ir  M2
train_one realvul source   M3
train_one realvul llvm_ir  M4

# ---- E3: zero-shot transfer, same representation, Juliet -> CompRealVul ----
eval_transfer "$OUT/juliet_source$SUFFIX"  realvul source \
              "$OUT/E3_M1_juliet_source_to_realvul${SUFFIX}.json"

eval_transfer "$OUT/juliet_llvm_ir$SUFFIX" realvul llvm_ir \
              "$OUT/E3_M2_juliet_llvm_to_realvul${SUFFIX}.json"

echo -e "\n=============================================="
echo "Done. Results in $OUT/"
echo "  E1 (M1/M2 on Juliet test)      -> juliet_source${SUFFIX}_metrics.json, juliet_llvm_ir${SUFFIX}_metrics.json"
echo "  E2 (M3/M4 on CompRealVul test) -> realvul_source${SUFFIX}_metrics.json, realvul_llvm_ir${SUFFIX}_metrics.json"
echo "  E3 (M1/M2 zero-shot -> CompRealVul) -> E3_*.json"
echo "=============================================="
