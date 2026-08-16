#!/bin/bash
set -euo pipefail

DATASETS_DIR="${1:-../EEGdenoiseNet/data}"
CKPT="${2:-./output/eegdenoisenet_clean_dual_patch16_emg}"
OUTPUT_ROOT="${3:-./output/eegdenoisenet_clean_dual_patch16_eval}"
NOISE_TYPE="${4:-emg}"
EMA_MODE="${5:-raw}"
DENOISE_MODE="${6:-direct}"
OUTPUT_DIR="${OUTPUT_ROOT}_${NOISE_TYPE}_${EMA_MODE}_${DENOISE_MODE}"

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
  --dual_branch \
  --prediction_target clean \
  --denoise_mode "$DENOISE_MODE" \
  --gen_bsz 64 \
  --ema_mode "$EMA_MODE" \
  --num_sampling_steps 50 \
  --sampling_method heun
