from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


_SPLIT_RATIOS = {"train": 0.72, "val": 0.18, "test": 0.10}
_SNR_RANGES = {
    "eog": (-5.0, 5.0),
    "emg": (-5.0, 5.0),
}


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std < 1e-8:
        return x - mean
    return (x - mean) / std


def _permute_rows(x: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(x)[rng.permutation(len(x))]


def _balance_clean_to_noise(clean: np.ndarray, noise_len: int) -> np.ndarray:
    if len(clean) == noise_len:
        return np.asarray(clean)
    if len(clean) > noise_len:
        return np.asarray(clean[:noise_len])
    reuse_num = noise_len - len(clean)
    return np.vstack([clean[:reuse_num], clean])


def _split_indices(n: int, split: str, seed: int) -> np.ndarray:
    if split not in _SPLIT_RATIOS:
        raise ValueError(f"Unsupported split '{split}'.")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    train_end = int(round(n * _SPLIT_RATIOS["train"]))
    val_end = train_end + int(round(n * _SPLIT_RATIOS["val"]))
    if split == "train":
        return indices[:train_end]
    if split == "val":
        return indices[train_end:val_end]
    return indices[val_end:]


def benchmark_snr_range(noise_type: str):
    noise_type = noise_type.lower()
    if noise_type not in _SNR_RANGES:
        raise ValueError(f"Unsupported noise type '{noise_type}'.")
    return _SNR_RANGES[noise_type]


def benchmark_eval_snr_levels(noise_type: str):
    snr_min, snr_max = benchmark_snr_range(noise_type)
    return tuple(float(v) for v in range(int(snr_min), int(snr_max) + 1))


class EEGDenoiseNetDataset(Dataset):
    """Build paired noisy/clean EEG examples from EEGdenoiseNet epochs."""

    def __init__(
        self,
        data_dir,
        split: str = "train",
        noise_types: Sequence[str] = ("eog", "emg"),
        train_snr_range=(-5.0, 5.0),
        eval_snr_levels: Iterable[float] = tuple(range(-5, 6)),
        combin_num: int = 11,
        seed: int = 0,
        normalize: bool = True,
        return_metadata: bool = False,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.noise_types = tuple(n.lower() for n in noise_types)
        self.train_snr_range = (float(train_snr_range[0]), float(train_snr_range[1]))
        self.eval_snr_levels = tuple(float(v) for v in eval_snr_levels)
        self.combin_num = int(combin_num)
        self.seed = int(seed)
        self.normalize = normalize
        self.return_metadata = return_metadata
        self._rng = np.random.default_rng(self.seed + {"train": 0, "val": 1, "test": 2}[split])

        self.clean = np.load(self.data_dir / "EEG_all_epochs.npy", mmap_mode="r")
        if self.clean.ndim != 2:
            raise ValueError("EEG_all_epochs.npy must have shape [num_epochs, length].")

        self.artifacts = {}
        self.matched_clean = {}
        for noise_type in self.noise_types:
            if noise_type not in {"eog", "emg"}:
                raise ValueError("noise_types entries must be 'eog' or 'emg'.")
            filename = f"{noise_type.upper()}_all_epochs.npy"
            artifact = np.load(self.data_dir / filename, mmap_mode="r")
            if artifact.ndim != 2:
                raise ValueError(f"{filename} must have shape [num_epochs, length].")
            if artifact.shape[1] != self.clean.shape[1]:
                raise ValueError(f"{filename} length {artifact.shape[1]} does not match EEG length {self.clean.shape[1]}.")
            self.artifacts[noise_type] = _permute_rows(artifact, self.seed + 100 + len(self.artifacts))

            clean_perm = _permute_rows(self.clean, self.seed)
            self.matched_clean[noise_type] = _balance_clean_to_noise(clean_perm, len(self.artifacts[noise_type]))

        self.clean_indices = {
            noise_type: _split_indices(len(clean_matched), split, self.seed + 10 + i)
            for i, (noise_type, clean_matched) in enumerate(self.matched_clean.items())
        }
        self.artifact_indices = {
            noise_type: _split_indices(len(artifact), split, self.seed + 100 + i)
            for i, (noise_type, artifact) in enumerate(self.artifacts.items())
        }

        if split == "train":
            train_base = sum(len(self.clean_indices[noise_type]) for noise_type in self.noise_types)
            self.length = max(1, train_base * max(1, self.combin_num))
        else:
            self.eval_plan = [
                (noise_type, clean_idx, artifact_idx, snr_db)
                for noise_type in self.noise_types
                for snr_db in self.eval_snr_levels
                for clean_idx, artifact_idx in zip(self.clean_indices[noise_type], self.artifact_indices[noise_type])
            ]
            self.length = len(self.eval_plan)

    def __len__(self):
        return self.length

    @staticmethod
    def _mix(clean: np.ndarray, artifact: np.ndarray, snr_db: float) -> np.ndarray:
        snr_linear = 10.0 ** (float(snr_db) / 20.0)
        artifact_rms = max(_rms(artifact), 1e-8)
        scale = _rms(clean) / (artifact_rms * snr_linear)
        return clean + artifact * scale

    def _sample_train(self, idx: int):
        noise_type = self.noise_types[idx % len(self.noise_types)]
        clean_pool = self.clean_indices[noise_type]
        artifact_pool = self.artifact_indices[noise_type]
        clean_idx = clean_pool[self._rng.integers(0, len(clean_pool))]
        artifact_idx = artifact_pool[self._rng.integers(0, len(artifact_pool))]
        snr_db = self._rng.uniform(*self.train_snr_range)
        return noise_type, clean_idx, artifact_idx, snr_db

    def _sample_eval(self, idx: int):
        noise_type, clean_idx, artifact_idx, snr_db = self.eval_plan[idx]
        return noise_type, clean_idx, artifact_idx, snr_db

    def __getitem__(self, idx):
        if self.split == "train":
            noise_type, clean_idx, artifact_idx, snr_db = self._sample_train(idx)
        else:
            noise_type, clean_idx, artifact_idx, snr_db = self._sample_eval(idx)

        clean = np.asarray(self.matched_clean[noise_type][clean_idx], dtype=np.float32)
        artifact = np.asarray(self.artifacts[noise_type][artifact_idx], dtype=np.float32)
        if self.normalize:
            clean = _standardize(clean)
            artifact = _standardize(artifact)
        noisy = self._mix(clean, artifact, snr_db).astype(np.float32)

        noisy = torch.from_numpy(noisy.astype(np.float32)).unsqueeze(0)
        clean = torch.from_numpy(clean.astype(np.float32)).unsqueeze(0)
        if self.return_metadata:
            metadata = {
                "noise_type": noise_type,
                "snr_db": float(snr_db),
                "clean_idx": int(clean_idx),
                "artifact_idx": int(artifact_idx),
            }
            return noisy, clean, metadata
        return noisy, clean
