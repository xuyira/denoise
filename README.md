# JET Flow Matching for EEG Denoising

This repository adapts the JET EEG flow-matching code to supervised EEG denoising on EEGdenoiseNet.

The training task is:

```text
noisy EEG -> clean EEG
```

Unlike the original class-conditional JET setup, the model does not use class labels. The noisy EEG is the flow source and the clean EEG is the target.

## Data

EEGdenoiseNet provides separate clean EEG and artifact epochs:

```text
EEG_all_epochs.npy  clean EEG, shape [4514, 512]
EOG_all_epochs.npy  ocular artifacts, shape [3400, 512]
EMG_all_epochs.npy  muscular artifacts, shape [5598, 512]
```

The dataset loader synthesizes noisy EEG online:

```text
noisy = clean + scaled_artifact
```

The artifact scale is chosen by SNR. For `eog`, training/validation/test use `[-7, 2]` dB. For `emg`, training/validation/test use `[-7, 4]` dB.

Expected layout:

```text
EEGdenoiseNet/data/
├── EEG_all_epochs.npy
├── EOG_all_epochs.npy
└── EMG_all_epochs.npy
```

## Installation

```bash
conda create -n jet python=3.10 -y
conda activate jet
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

For Apple Silicon, install the PyTorch build appropriate for your local environment.

## Training

```bash
bash scripts/train_eegdenoisenet.sh ../EEGdenoiseNet/data ./output/eegdenoisenet eog
```

Or run directly:

```bash
python train_eeg.py \
  --dataset eegdenoisenet \
  --datasets_dir ../EEGdenoiseNet/data \
  --output_dir ./output/eegdenoisenet_eog \
  --noise_types eog \
  --model JiT-S/16 \
  --num_eeg_channels 1 \
  --target_length 512 \
  --eeg_patch_size 64 \
  --batch_size 128 \
  --epochs 200 \
  --loss_type mix \
  --loss_weight_recon 0.5
```

Checkpoints are written under `--output_dir`:

```text
checkpoint-last.pth
checkpoint-best.pth
```

## Evaluation

```bash
bash scripts/infer_eegdenoisenet.sh ../EEGdenoiseNet/data ./output/eegdenoisenet_eog ./output/eegdenoisenet_eval eog
```

Evaluation writes:

```text
eval_batch.npz
metrics.json
```

The reported metrics are:

```text
RRMSE_temporal
RRMSE_spectral
CC
```

`metrics.json` reports the overall metrics and grouped results by:

```text
per_snr
per_noise_type
per_noise_type_snr
```
