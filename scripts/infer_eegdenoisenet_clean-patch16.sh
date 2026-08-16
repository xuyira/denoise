#!/bin/bash
set -euo pipefail

DATASETS_DIR="${1:-../EEGdenoiseNet/data}"
CKPT="${2:-./output/eegdenoisenet_clean_patch16_none_emg}"
OUTPUT_ROOT="${3:-./output/eegdenoisenet_clean_patch16_eval}"
NOISE_TYPE="${4:-emg}"
EMA_MODE="${5:-raw}"
DENOISE_MODE="${6:-direct}"
VARIANT="${7:-none}"
OUTPUT_DIR="${OUTPUT_ROOT}_${VARIANT}_${NOISE_TYPE}_${EMA_MODE}_${DENOISE_MODE}"

EXTRA_ARGS=()
if [[ "$VARIANT" == "conv" ]]; then
  EXTRA_ARGS=(
    --conv_refiner
    --conv_refiner_channels 64
    --conv_refiner_kernel 3
  )
elif [[ "$VARIANT" == "cond" ]]; then
  EXTRA_ARGS=(
    --conv_refiner
    --conv_refiner_channels 64
    --conv_refiner_kernel 3
    --condition_mode refiner
  )
elif [[ "$VARIANT" != "none" ]]; then
  echo "VARIANT must be 'none', 'conv', or 'cond', got '$VARIANT'" >&2
  exit 1
fi

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
  --prediction_target clean \
  --denoise_mode "$DENOISE_MODE" \
  --gen_bsz 64 \
  --ema_mode "$EMA_MODE" \
  --num_sampling_steps 50 \
  --sampling_method heun \
  "${EXTRA_ARGS[@]}"
