#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/teacher_cache}"
RUNS_DIR="${RUNS_DIR:-$ROOT_DIR/runs/paper_mini}"
D_MODEL="${D_MODEL:-256}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
IMAGE_SIZE="${IMAGE_SIZE:-64}"
NUM_FRAMES="${NUM_FRAMES:-6}"

mkdir -p "$RUNS_DIR"

train_once() {
  local tag="$1"
  local train_file="$2"
  local val_file="$3"
  local prompt_variant="$4"
  local distill_mode="$5"
  if [ ! -f "$DATA_DIR/$train_file" ]; then
    echo "Missing file: $DATA_DIR/$train_file"
    exit 1
  fi
  if [ ! -f "$DATA_DIR/$val_file" ]; then
    echo "Missing file: $DATA_DIR/$val_file"
    exit 1
  fi
  "$PYTHON_BIN" "$ROOT_DIR/train.py" \
    --data_dir "$DATA_DIR" \
    --train_file "$train_file" \
    --val_file "$val_file" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --d_model "$D_MODEL" \
    --image_size "$IMAGE_SIZE" \
    --num_frames "$NUM_FRAMES" \
    --prompt_variant "$prompt_variant" \
    --distill_mode "$distill_mode" \
    --out_dir "$RUNS_DIR/$tag"
}

for prompt in none random frame event; do
  train_once "prompt_${prompt}" "train.videollava.jsonl" "val.videollava.jsonl" "$prompt" "hybrid"
done

for mode in none hard soft hybrid; do
  train_once "distill_${mode}" "train.videollava.jsonl" "val.videollava.jsonl" "event" "$mode"
done

for teacher in videollava flamingo chatunivl; do
  train_once "teacher_${teacher}" "train.${teacher}.jsonl" "val.${teacher}.jsonl" "event" "hybrid"
done

echo "Mini paper suite completed. Outputs: $RUNS_DIR"
