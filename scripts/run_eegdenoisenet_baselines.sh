#!/bin/bash
set -euo pipefail

DATASETS_DIR="${1:-../EEGdenoiseNet/data}"
OUTPUT_DIR="${2:-./output/eegdenoise_baselines}"

MODELS=(fcnn simple_cnn complex_cnn rnn_lstm)
NOISE_TYPES=(eog emg)

for noise_type in "${NOISE_TYPES[@]}"; do
  for model in "${MODELS[@]}"; do
    python benchmark_eegdenoise.py \
      --datasets_dir "$DATASETS_DIR" \
      --output_dir "$OUTPUT_DIR" \
      --model "$model" \
      --noise_types "$noise_type"
  done
done
