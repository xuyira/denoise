#!/bin/bash
set -euo pipefail

DATASETS_DIR="${1:-../EEGdenoiseNet/data}"
OUTPUT_ROOT="${2:-./output/eegdenoisenet_clean_mix_patch16}"
NOISE_TYPE="${3:-emg}"
OUTPUT_DIR="${OUTPUT_ROOT}_${NOISE_TYPE}"

python train_eeg.py \
  --dataset eegdenoisenet \
  --datasets_dir "$DATASETS_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --noise_types "$NOISE_TYPE" \
  --model JiT-S/16 \
  --num_eeg_channels 1 \
  --target_length 512 \
  --eeg_patch_size 16 \
  --batch_size 128 \
  --epochs 200 \
  --blr 5e-5 \
  --prediction_target clean \
  --loss_type mix \
  --loss_weight_recon 0 \
  --loss_weight_stft 0.02 \
  --loss_weight_corr 0.05 \
  --loss_weight_stat 0 \
  --loss_weight_tv 0 \
  --num_sampling_steps 50 \
  --sampling_method heun
