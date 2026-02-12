#!/usr/bin/env bash
set -euo pipefail

python scripts/make_toy_data.py \
  --out_dir data/toy \
  --train_size 128 \
  --val_size 32 \
  --test_size 32 \
  --num_frames 6 \
  --d_model 256

python train.py \
  --data_dir data/toy \
  --epochs 2 \
  --batch_size 8 \
  --num_workers 0 \
  --num_frames 6 \
  --image_size 64 \
  --d_model 256 \
  --out_dir runs/smoke

python eval.py \
  --data_dir data/toy \
  --checkpoint runs/smoke/best.pt \
  --num_workers 0 \
  --split_file test.jsonl
