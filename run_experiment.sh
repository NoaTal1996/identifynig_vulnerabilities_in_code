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

# Paired same-test-set statistical comparison (McNemar + paired bootstrap).
compare () {
  local a=$1
  local b=$2
  local label_a=$3
  local label_b=$4
  local outfile=$5
  local extra=$6   # e.g. "--canonicalize_juliet_ids"
  if [ ! -f "$a" ] || [ ! -f "$b" ]; then
    echo "Skipping compare ($label_a vs $label_b): missing $a or $b"
    return
  fi
  echo -e "\n----------------------------------------------"
  echo "Compare: $label_a  vs  $label_b"
  echo "----------------------------------------------"
  python3 src/compare_experiments.py \
    --a "$a" --b "$b" \
    --label_a "$label_a" --label_b "$label_b" \
    --output_file "$outfile" \
    $extra || echo "Compare failed (non-fatal): $label_a vs $label_b"
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

# ---- Statistical comparisons: McNemar + paired bootstrap of the metric diff. ----
# Uses per-example *_predictions.jsonl files written by train.py and eval.py.
JS_PRED=$OUT/juliet_source${SUFFIX}_predictions.jsonl
JI_PRED=$OUT/juliet_llvm_ir${SUFFIX}_predictions.jsonl
RS_PRED=$OUT/realvul_source${SUFFIX}_predictions.jsonl
RI_PRED=$OUT/realvul_llvm_ir${SUFFIX}_predictions.jsonl
E3_M1_PRED=$OUT/E3_M1_juliet_source_to_realvul${SUFFIX}_predictions.jsonl
E3_M2_PRED=$OUT/E3_M2_juliet_llvm_to_realvul${SUFFIX}_predictions.jsonl
E4_M1_PRED=$OUT/E4_M1_on_llm_juliet_source${SUFFIX}_predictions.jsonl
E4_M3_PRED=$OUT/E4_M3_on_llm_juliet_source${SUFFIX}_predictions.jsonl
E5_M1_PRED=$OUT/E5_M1_on_llm_realvul_source${SUFFIX}_predictions.jsonl
E5_M3_PRED=$OUT/E5_M3_on_llm_realvul_source${SUFFIX}_predictions.jsonl

# Q1 — Does representation matter on the same dataset?
compare "$JS_PRED" "$JI_PRED"  "M1 (Juliet source)"     "M2 (Juliet IR)"      "$OUT/cmp_M1_vs_M2_on_E1${SUFFIX}.json"
compare "$RS_PRED" "$RI_PRED"  "M3 (RealVul source)"    "M4 (RealVul IR)"     "$OUT/cmp_M3_vs_M4_on_E2${SUFFIX}.json"

# Q2 — Does training data matter under LLM restyling?
compare "$E4_M1_PRED" "$E4_M3_PRED"  "M1 on LLM-Juliet"  "M3 on LLM-Juliet"    "$OUT/cmp_M1_vs_M3_on_E4${SUFFIX}.json"
compare "$E5_M1_PRED" "$E5_M3_PRED"  "M1 on LLM-RealVul" "M3 on LLM-RealVul"   "$OUT/cmp_M1_vs_M3_on_E5${SUFFIX}.json"

# Q3 — Does LLM restyling hurt each model, matched by original instance?
compare "$JS_PRED" "$E4_M1_PRED"  "M1 on Juliet"        "M1 on LLM-Juliet"    "$OUT/cmp_M1_E1_vs_E4${SUFFIX}.json"  "--canonicalize_juliet_ids"
compare "$RS_PRED" "$E5_M3_PRED"  "M3 on RealVul"       "M3 on LLM-RealVul"   "$OUT/cmp_M3_E2_vs_E5${SUFFIX}.json"

# Note: E1 vs E3 (same model, different test sets — Juliet vs CompRealVul) is
# not runnable as a paired test. Report each side's bootstrap CI from its own
# *_metrics.json instead.

echo -e "\n=============================================="
echo "Done. Results in $OUT/"
echo "  E1 (M1/M2 on Juliet test)            -> juliet_{source,llvm_ir}${SUFFIX}_metrics.json"
echo "  E2 (M3/M4 on CompRealVul test)       -> realvul_{source,llvm_ir}${SUFFIX}_metrics.json"
echo "  E3 (M1/M2 zero-shot -> CompRealVul)  -> E3_*.json"
echo "  E4 (M1/M3 on LLM-Juliet)             -> E4_*.json"
echo "  E5 (M1/M3 on LLM-CompRealVul)        -> E5_*.json"
echo "  Statistical comparisons              -> cmp_*.json"
echo "=============================================="
