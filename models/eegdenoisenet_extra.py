from typing import Iterable, Tuple

import numpy as np
import torch
from torch import nn
from scipy.interpolate import CubicSpline
from scipy.signal import butter, find_peaks, filtfilt, iirnotch, sosfiltfilt


class TransformerDenoiser(nn.Module):
    def __init__(self, target_length: int, d_model: int = 128, nhead: int = 8, num_layers: int = 4):
        super().__init__()
        self.input_proj = nn.Conv1d(1, d_model, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, target_length, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, 1)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x).transpose(1, 2)
        x = x + self.pos_embed[:, : x.shape[1], :]
        x = self.encoder(x)
        x = self.output_proj(x).transpose(1, 2)
        return x


def build_transformer_denoiser(target_length: int, d_model: int = 128, nhead: int = 8, num_layers: int = 4):
    return TransformerDenoiser(target_length=target_length, d_model=d_model, nhead=nhead, num_layers=num_layers)


class GanGenerator(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=7, padding=3),
            nn.InstanceNorm1d(hidden, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.InstanceNorm1d(hidden, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.InstanceNorm1d(hidden, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden, 1, kernel_size=7, padding=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GanDiscriminator(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden, hidden * 2, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm1d(hidden * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden * 2, hidden * 4, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm1d(hidden * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def gan_losses(real_logits: torch.Tensor, fake_logits: torch.Tensor):
    bce = nn.BCEWithLogitsLoss()
    real_targets = torch.ones_like(real_logits)
    fake_targets = torch.zeros_like(fake_logits)
    d_loss = bce(real_logits, real_targets) + bce(fake_logits, fake_targets)
    g_adv_loss = bce(fake_logits, real_targets)
    return g_adv_loss, d_loss


def build_gan_models(hidden: int = 64):
    return GanGenerator(hidden=hidden), GanDiscriminator(hidden=hidden)


def _to_numpy_signal(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _bandpass_sos(sampling_rate: float, lowcut: float, highcut: float, order: int):
    nyquist = sampling_rate * 0.5
    low = max(lowcut / nyquist, 1e-4)
    high = min(highcut / nyquist, 0.999)
    if not low < high:
        raise ValueError("lowcut must be smaller than highcut after Nyquist normalization.")
    return butter(order, [low, high], btype="bandpass", output="sos")


def filter_denoise_batch(
    noisy: torch.Tensor,
    sampling_rate: float = 256.0,
    lowcut: float = 0.5,
    highcut: float = 45.0,
    order: int = 4,
    notch_freq: float = 60.0,
    notch_q: float = 30.0,
):
    device = noisy.device
    dtype = noisy.dtype
    batch = noisy.detach().cpu().numpy()
    sos = _bandpass_sos(sampling_rate, lowcut, highcut, order)
    nyquist = sampling_rate * 0.5
    apply_notch = 0.0 < notch_freq < nyquist
    if apply_notch:
        b_notch, a_notch = iirnotch(notch_freq / nyquist, notch_q)
    denoised = []
    for sample in batch:
        signal = np.asarray(sample, dtype=np.float64).reshape(-1)
        filtered = sosfiltfilt(sos, signal)
        if apply_notch:
            filtered = filtfilt(b_notch, a_notch, filtered)
        denoised.append(filtered.astype(np.float32))
    denoised = torch.from_numpy(np.asarray(denoised, dtype=np.float32)).unsqueeze(1)
    return denoised.to(device=device, dtype=dtype)


def _extrema_indices(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "max":
        idx, _ = find_peaks(x)
    elif kind == "min":
        idx, _ = find_peaks(-x)
    else:
        raise ValueError("kind must be 'max' or 'min'.")
    if idx.size == 0:
        return idx
    idx = np.unique(np.concatenate(([0], idx, [len(x) - 1])))
    return idx


def _envelope(x: np.ndarray, extrema_idx: np.ndarray) -> np.ndarray:
    if extrema_idx.size < 2:
        return np.full_like(x, float(np.mean(x)))
    y = x[extrema_idx]
    if extrema_idx.size < 4:
        return np.interp(np.arange(len(x)), extrema_idx, y)
    spline = CubicSpline(extrema_idx, y, bc_type="natural", extrapolate=True)
    return spline(np.arange(len(x)))


def _zero_crossings(x: np.ndarray) -> int:
    signs = np.signbit(x)
    return int(np.count_nonzero(signs[:-1] != signs[1:]))


def _emd_once(x: np.ndarray, max_sift: int = 10, sift_tol: float = 0.05) -> np.ndarray:
    h = x.copy()
    for _ in range(max_sift):
        max_idx = _extrema_indices(h, "max")
        min_idx = _extrema_indices(h, "min")
        if max_idx.size < 2 or min_idx.size < 2:
            break
        upper = _envelope(h, max_idx)
        lower = _envelope(h, min_idx)
        mean_env = 0.5 * (upper + lower)
        prev = h
        h = h - mean_env
        extrema = max_idx.size + min_idx.size
        zero_cross = _zero_crossings(h)
        if abs(extrema - zero_cross * 2) <= 2 and np.max(np.abs(mean_env)) <= sift_tol * max(np.max(np.abs(prev)), 1e-8):
            break
    return h


def _emd_decompose(x: np.ndarray, max_imfs: int = 8, max_sift: int = 10, sift_tol: float = 0.05) -> Tuple[list[np.ndarray], np.ndarray]:
    residual = x.copy()
    imfs: list[np.ndarray] = []
    for _ in range(max_imfs):
        max_idx = _extrema_indices(residual, "max")
        min_idx = _extrema_indices(residual, "min")
        if max_idx.size < 2 or min_idx.size < 2:
            break
        imf = _emd_once(residual, max_sift=max_sift, sift_tol=sift_tol)
        if np.allclose(imf, 0.0):
            break
        imfs.append(imf)
        residual = residual - imf
        if _extrema_indices(residual, "max").size < 2 or _extrema_indices(residual, "min").size < 2:
            break
    return imfs, residual


def emd_denoise_batch(noisy: torch.Tensor, noise_type: str | Iterable[str], max_imfs: int = 8):
    device = noisy.device
    dtype = noisy.dtype
    batch = noisy.detach().cpu().numpy()
    if isinstance(noise_type, str):
        noise_types = [noise_type] * len(batch)
    else:
        noise_types = list(noise_type)
        if len(noise_types) != len(batch):
            raise ValueError("noise_type sequence must match batch size.")
    denoised = []
    for sample, kind in zip(batch, noise_types):
        signal = np.asarray(sample, dtype=np.float64).reshape(-1)
        keep_from = 1 if str(kind).lower() == "eog" else 2
        imfs, residual = _emd_decompose(signal, max_imfs=max_imfs)
        if not imfs:
            reconstructed = signal
        else:
            kept_imfs = imfs[keep_from:] if len(imfs) > keep_from else []
            reconstructed = residual.copy()
            for imf in kept_imfs:
                reconstructed = reconstructed + imf
        denoised.append(reconstructed.astype(np.float32))
    denoised = torch.from_numpy(np.asarray(denoised, dtype=np.float32)).unsqueeze(1)
    return denoised.to(device=device, dtype=dtype)
