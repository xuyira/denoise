#!/bin/bash
set -euo pipefail

DATASETS_DIR="${1:-../EEGdenoiseNet/data}"
CKPT="${2:-./output/eegdenoisenet}"
OUTPUT_DIR="${3:-./output/eegdenoisenet_eval}"

python inference.py \
  --dataset eegdenoisenet \
  --datasets_dir "$DATASETS_DIR" \
  --resume "$CKPT" \
  --output_dir "$OUTPUT_DIR" \
  --model JiT-S/16 \
  --num_eeg_channels 1 \
  --target_length 512 \
  --eeg_patch_size 64 \
  --gen_bsz 64 \
  --num_sampling_steps 50 \
  --sampling_method heun
