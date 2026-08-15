#!/bin/bash
set -euo pipefail

DATASETS_DIR="${1:-../EEGdenoiseNet/data}"
OUTPUT_DIR="${2:-./output/eegdenoisenet}"

python train_eeg.py \
  --dataset eegdenoisenet \
  --datasets_dir "$DATASETS_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --model JiT-S/16 \
  --num_eeg_channels 1 \
  --target_length 512 \
  --eeg_patch_size 64 \
  --batch_size 128 \
  --epochs 200 \
  --blr 5e-5 \
  --loss_type l2 \
  --num_sampling_steps 50 \
  --sampling_method heun
