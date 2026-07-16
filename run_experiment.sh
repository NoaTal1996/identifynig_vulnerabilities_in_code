#!/bin/bash
# End-to-end experiment (E1-E5).
# E1 = home exam for M1/M2 (test on Juliet)      -> produced automatically by train.py
# E2 = home exam for M3/M4 (test on CompRealVul) -> produced automatically by train.py
# E3 = away exam for M1/M2 (zero-shot on CompRealVul, same form)
# E4 = LLM-Juliet       (source), tested on both source models M1 and M3
# E5 = LLM-CompRealVul  (source), tested on both source models M1 and M3
#
# Usage:
#   ./run_experiment.sh              # full run   (~1-2 h on a Kaggle T4 GPU)
#   ./run_experiment.sh toy          # tiny sanity-check run (~3 min, CPU/MPS ok)
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

eval_on_jsonl () {
  local model_dir=$1
  local jsonl_path=$2
  local outfile=$3
  echo -e "\n----------------------------------------------"
  echo "E4 (LLM restyle): $model_dir  -->  $jsonl_path"
  echo "----------------------------------------------"
  python3 src/eval.py \
    --model_path "$model_dir" \
    --jsonl_path "$jsonl_path" \
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

# ---- E4/E5: LLM-restyled source, tested on the source-trained models (M1 and M3). ----
# ---- IR models (M2, M4) are not tested here — LLM restyling only produces source,   ----
# ---- and there is no IR of the restyled code (see handoff). Files pre-generated     ----
# ---- with Qwen; see data_llm/.                                                       ----
LLM_JULIET=./data_llm/llm_rewritten_juliet_source_200_repaired_balanced.jsonl
LLM_REALVUL=./data_llm/llm_rewritten_realvul_source_200_repaired_balanced.jsonl

# E4: LLM-Juliet, tested on both source models (M1 and M3).
if [ -f "$LLM_JULIET" ]; then
  eval_on_jsonl "$OUT/juliet_source$SUFFIX"  "$LLM_JULIET" \
                "$OUT/E4_M1_on_llm_juliet_source${SUFFIX}.json"
  eval_on_jsonl "$OUT/realvul_source$SUFFIX" "$LLM_JULIET" \
                "$OUT/E4_M3_on_llm_juliet_source${SUFFIX}.json"
else
  echo "Skipping E4: $LLM_JULIET not found."
fi

# E5: LLM-CompRealVul, tested on both source models (M1 and M3).
if [ -f "$LLM_REALVUL" ]; then
  eval_on_jsonl "$OUT/juliet_source$SUFFIX"  "$LLM_REALVUL" \
                "$OUT/E5_M1_on_llm_realvul_source${SUFFIX}.json"
  eval_on_jsonl "$OUT/realvul_source$SUFFIX" "$LLM_REALVUL" \
                "$OUT/E5_M3_on_llm_realvul_source${SUFFIX}.json"
else
  echo "Skipping E5: $LLM_REALVUL not found."
fi

echo -e "\n=============================================="
echo "Done. Results in $OUT/"
echo "  E1 (M1/M2 on Juliet test)            -> juliet_{source,llvm_ir}${SUFFIX}_metrics.json"
echo "  E2 (M3/M4 on CompRealVul test)       -> realvul_{source,llvm_ir}${SUFFIX}_metrics.json"
echo "  E3 (M1/M2 zero-shot -> CompRealVul)  -> E3_*.json"
echo "  E4 (M1/M3 on LLM-Juliet)             -> E4_*.json"
echo "  E5 (M1/M3 on LLM-CompRealVul)        -> E5_*.json"
echo "=============================================="
