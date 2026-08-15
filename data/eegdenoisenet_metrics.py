from collections import defaultdict

import numpy as np
import torch


def to_signal(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], x.shape[1], -1).float()


def rrmse_temporal(pred: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    num = torch.sqrt(torch.mean((pred - clean) ** 2, dim=(-2, -1)))
    den = torch.sqrt(torch.mean(clean ** 2, dim=(-2, -1))).clamp_min(1e-8)
    return num / den


def rrmse_spectral(
    pred: torch.Tensor,
    clean: torch.Tensor,
    sampling_rate: float,
    max_freq: float,
) -> torch.Tensor:
    pred_psd = torch.fft.rfft(pred, dim=-1).abs().pow(2)
    clean_psd = torch.fft.rfft(clean, dim=-1).abs().pow(2)
    freqs = torch.fft.rfftfreq(pred.shape[-1], d=1.0 / sampling_rate)
    freq_mask = freqs <= max_freq
    pred_psd = pred_psd[..., freq_mask]
    clean_psd = clean_psd[..., freq_mask]
    num = torch.sqrt(torch.mean((pred_psd - clean_psd) ** 2, dim=(-2, -1)))
    den = torch.sqrt(torch.mean(clean_psd ** 2, dim=(-2, -1))).clamp_min(1e-8)
    return num / den


def cc(pred: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    pred_f = pred.reshape(pred.shape[0], -1)
    clean_f = clean.reshape(clean.shape[0], -1)
    pred_c = pred_f - pred_f.mean(dim=1, keepdim=True)
    clean_c = clean_f - clean_f.mean(dim=1, keepdim=True)
    num = (pred_c * clean_c).sum(dim=1)
    den = torch.sqrt((pred_c.pow(2).sum(dim=1) + 1e-8) * (clean_c.pow(2).sum(dim=1) + 1e-8))
    return (num / den).clamp(-1.0, 1.0)


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else None,
        "std": float(arr.std()) if arr.size else None,
    }


def new_metric_store():
    return {"RRMSE_temporal": [], "RRMSE_spectral": [], "CC": []}


def append_group_metrics(store, key, rrmse_temporal_values, rrmse_spectral_values, cc_values):
    store[key]["RRMSE_temporal"].extend(rrmse_temporal_values)
    store[key]["RRMSE_spectral"].extend(rrmse_spectral_values)
    store[key]["CC"].extend(cc_values)


def summarize_grouped(grouped):
    return {
        str(key): {metric: summarize(values) for metric, values in metrics.items()}
        for key, metrics in sorted(grouped.items(), key=lambda item: str(item[0]))
    }
