from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


_SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))


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


class EEGDenoiseNetDataset(Dataset):
    """Build paired noisy/clean EEG examples from EEGdenoiseNet epochs."""

    def __init__(
        self,
        data_dir,
        split: str = "train",
        noise_types: Sequence[str] = ("eog", "emg"),
        train_snr_range=(-5.0, 5.0),
        eval_snr_levels: Iterable[float] = tuple(range(-5, 6)),
        combin_num: int = 10,
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
        for noise_type in self.noise_types:
            if noise_type not in {"eog", "emg"}:
                raise ValueError("noise_types entries must be 'eog' or 'emg'.")
            filename = f"{noise_type.upper()}_all_epochs.npy"
            artifact = np.load(self.data_dir / filename, mmap_mode="r")
            if artifact.ndim != 2:
                raise ValueError(f"{filename} must have shape [num_epochs, length].")
            if artifact.shape[1] != self.clean.shape[1]:
                raise ValueError(f"{filename} length {artifact.shape[1]} does not match EEG length {self.clean.shape[1]}.")
            self.artifacts[noise_type] = artifact

        # 8:1:1 split over each source before mixing.
        self.clean_indices = _split_indices(len(self.clean), split, self.seed)
        self.artifact_indices = {
            noise_type: _split_indices(len(artifact), split, self.seed + 100 + i)
            for i, (noise_type, artifact) in enumerate(self.artifacts.items())
        }

        if split == "train":
            self.length = len(self.clean_indices) * max(1, self.combin_num)
        else:
            self.eval_plan = [
                (clean_idx, noise_type, snr_db)
                for noise_type in self.noise_types
                for snr_db in self.eval_snr_levels
                for clean_idx in self.clean_indices
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

    def _normalize_pair(self, noisy: np.ndarray, clean: np.ndarray):
        if not self.normalize:
            return noisy, clean
        std = float(np.std(noisy))
        if std < 1e-8:
            std = 1.0
        return noisy / std, clean / std

    def _sample_train(self, idx: int):
        clean_idx = self.clean_indices[idx % len(self.clean_indices)]
        noise_type = self.noise_types[idx % len(self.noise_types)]
        artifact_pool = self.artifact_indices[noise_type]
        artifact_idx = artifact_pool[self._rng.integers(0, len(artifact_pool))]
        snr_db = self._rng.uniform(*self.train_snr_range)
        return clean_idx, noise_type, artifact_idx, snr_db

    def _sample_eval(self, idx: int):
        clean_idx, noise_type, snr_db = self.eval_plan[idx]
        artifact_pool = self.artifact_indices[noise_type]
        artifact_idx = artifact_pool[idx % len(artifact_pool)]
        return clean_idx, noise_type, artifact_idx, snr_db

    def __getitem__(self, idx):
        if self.split == "train":
            clean_idx, noise_type, artifact_idx, snr_db = self._sample_train(idx)
        else:
            clean_idx, noise_type, artifact_idx, snr_db = self._sample_eval(idx)

        clean = np.asarray(self.clean[clean_idx], dtype=np.float32)
        artifact = np.asarray(self.artifacts[noise_type][artifact_idx], dtype=np.float32)
        noisy = self._mix(clean, artifact, snr_db).astype(np.float32)
        noisy, clean = self._normalize_pair(noisy, clean)

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
