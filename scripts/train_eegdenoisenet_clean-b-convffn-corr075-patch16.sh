#!/bin/bash
set -euo pipefail

DATASETS_DIR="${1:-../EEGdenoiseNet/data}"
OUTPUT_ROOT="${2:-./output/eegdenoisenet_clean_b_convffn_corr075_patch16}"
NOISE_TYPE="${3:-emg}"
OUTPUT_DIR="${OUTPUT_ROOT}_${NOISE_TYPE}"

python train_eeg.py \
  --dataset eegdenoisenet \
  --datasets_dir "$DATASETS_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --noise_types "$NOISE_TYPE" \
  --model JiT-B/16 \
  --num_eeg_channels 1 \
  --target_length 512 \
  --eeg_patch_size 16 \
  --use_convffn \
  --convffn_kernel_size 3 \
  --batch_size 128 \
  --epochs 200 \
  --early_stop_patience 20 \
  --early_stop_min_delta 1e-3 \
  --blr 5e-5 \
  --prediction_target clean \
  --loss_type mix \
  --loss_weight_recon 0 \
  --loss_weight_stft 0 \
  --loss_weight_corr 0.075 \
  --loss_weight_stat 0 \
  --loss_weight_tv 0 \
  --num_sampling_steps 50 \
  --sampling_method heun
