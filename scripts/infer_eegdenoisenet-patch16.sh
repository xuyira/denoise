#!/bin/bash
set -euo pipefail

DATASETS_DIR="${1:-../EEGdenoiseNet/data}"
CKPT="${2:-./output/eegdenoisenet}"
OUTPUT_ROOT="${3:-./output/eegdenoisenet_eval}"
NOISE_TYPE="${4:-eog}"
EMA_MODE="${5:-ema1}"
OUTPUT_DIR="${OUTPUT_ROOT}_${NOISE_TYPE}"

python inference.py \
  --dataset eegdenoisenet \
  --datasets_dir "$DATASETS_DIR" \
  --resume "$CKPT" \
  --output_dir "$OUTPUT_DIR" \
  --noise_types "$NOISE_TYPE" \
  --model JiT-S/16 \
  --num_eeg_channels 1 \
  --target_length 512 \
  --eeg_patch_size 16 \
  --gen_bsz 64 \
  --ema_mode "$EMA_MODE" \
  --num_sampling_steps 50 \
  --sampling_method heun 
