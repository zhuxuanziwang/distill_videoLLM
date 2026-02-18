#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/teacher_cache}"
RUNS_DIR="${RUNS_DIR:-$ROOT_DIR/runs/msrvtt_suite}"

TRAIN_FILE="${TRAIN_FILE:-train.videollava.redo.jsonl}"
VAL_FILE="${VAL_FILE:-val.videollava.redo.jsonl}"
TEST_FILE="${TEST_FILE:-test.videollava.redo.jsonl}"

EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LR="${LR:-5e-5}"
LAMBDA_SOFT="${LAMBDA_SOFT:-0.4}"
LAMBDA_HARD="${LAMBDA_HARD:-0.5}"

D_MODEL="${D_MODEL:-512}"
NUM_FRAMES="${NUM_FRAMES:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
TOKENIZER="${TOKENIZER:-clip}"
CLIP_MODEL_ID="${CLIP_MODEL_ID:-openai/clip-vit-base-patch16}"

RESULTS_CSV="$RUNS_DIR/results.csv"
mkdir -p "$RUNS_DIR"

if [ ! -f "$RESULTS_CSV" ]; then
  echo "timestamp,group,tag,train_file,val_file,test_file,prompt_variant,distill_mode,epochs,batch_size,lr,lambda_hard,lambda_soft,d_model,num_frames,image_size,tokenizer,clip_model_id,top1_acc,checkpoint" > "$RESULTS_CSV"
fi

train_and_eval() {
  local group="$1"
  local tag="$2"
  local train_file="$3"
  local val_file="$4"
  local test_file="$5"
  local prompt_variant="$6"
  local distill_mode="$7"

  local out_dir="$RUNS_DIR/${group}_${tag}"
  mkdir -p "$out_dir"

  if [ ! -f "$DATA_DIR/$train_file" ] || [ ! -f "$DATA_DIR/$val_file" ] || [ ! -f "$DATA_DIR/$test_file" ]; then
    echo "[skip] missing files for $group/$tag"
    return
  fi

  "$PYTHON_BIN" "$ROOT_DIR/train.py" \
    --data_dir "$DATA_DIR" \
    --train_file "$train_file" \
    --val_file "$val_file" \
    --dataset_name msrvtt_qa \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --lr "$LR" \
    --lambda_soft "$LAMBDA_SOFT" \
    --lambda_hard "$LAMBDA_HARD" \
    --d_model "$D_MODEL" \
    --num_frames "$NUM_FRAMES" \
    --image_size "$IMAGE_SIZE" \
    --tokenizer "$TOKENIZER" \
    --clip_model_id "$CLIP_MODEL_ID" \
    --prompt_variant "$prompt_variant" \
    --distill_mode "$distill_mode" \
    --out_dir "$out_dir"

  local eval_out
  eval_out="$("$PYTHON_BIN" "$ROOT_DIR/eval.py" \
    --data_dir "$DATA_DIR" \
    --split_file "$test_file" \
    --checkpoint "$out_dir/best.pt" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS")"
  echo "$eval_out"
  local top1
  top1="$(echo "$eval_out" | awk -F'top1_acc=' 'NF>1{print $2}' | tail -n 1)"

  local ts
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "$ts,$group,$tag,$train_file,$val_file,$test_file,$prompt_variant,$distill_mode,$EPOCHS,$BATCH_SIZE,$LR,$LAMBDA_HARD,$LAMBDA_SOFT,$D_MODEL,$NUM_FRAMES,$IMAGE_SIZE,$TOKENIZER,$CLIP_MODEL_ID,$top1,$out_dir/best.pt" >> "$RESULTS_CSV"
}

echo "== Prompt Ablation =="
for prompt in none random frame event; do
  train_and_eval "prompt" "$prompt" "$TRAIN_FILE" "$VAL_FILE" "$TEST_FILE" "$prompt" "hybrid"
done

echo "== Distillation Ablation =="
for mode in none hard soft hybrid; do
  train_and_eval "distill" "$mode" "$TRAIN_FILE" "$VAL_FILE" "$TEST_FILE" "event" "$mode"
done

echo "== LLM Generator Ablation =="
# Expected files in DATA_DIR:
# train.prompt_vicuna.jsonl / val.prompt_vicuna.jsonl / test.prompt_vicuna.jsonl ...
# train.prompt_mistral.jsonl / ...
# train.prompt_llama3.jsonl / ...
# train.prompt_gpt4.jsonl / ...
for llm in gpt4 vicuna mistral llama3; do
  train_and_eval "llm" "$llm" \
    "train.prompt_${llm}.jsonl" \
    "val.prompt_${llm}.jsonl" \
    "test.prompt_${llm}.jsonl" \
    "event" "hybrid"
done

echo "done. results: $RESULTS_CSV"
