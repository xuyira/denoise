"""Denoise EEGdenoiseNet samples and report denoising metrics."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.data import build_base_dataset
from denoiser import Denoiser


def get_args_parser():
    parser = argparse.ArgumentParser("EEG denoising inference", add_help=False)

    parser.add_argument("--dataset", default="eegdenoisenet", choices=["eegdenoisenet"])
    parser.add_argument("--datasets_dir", type=str, required=True)
    parser.add_argument("--resume", type=str, required=True,
                        help="Path to a .pth checkpoint or a directory containing checkpoint-last.pth.")
    parser.add_argument("--output_dir", default="./output/eval")
    parser.add_argument("--eval_split", default="test", choices=["train", "val", "test"])

    parser.add_argument("--model", default="JiT-B/16")
    parser.add_argument("--num_eeg_channels", type=int, default=1)
    parser.add_argument("--eeg_patch_size", type=int, default=64)
    parser.add_argument("--target_length", type=int, default=512)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dropout", type=float, default=0.0)

    parser.add_argument("--noise_types", nargs="+", default=["eog", "emg"], choices=["eog", "emg"])
    parser.add_argument("--train_snr_min", type=float, default=-5.0)
    parser.add_argument("--train_snr_max", type=float, default=5.0)
    parser.add_argument("--eval_snr_levels", nargs="+", type=float,
                        default=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    parser.add_argument("--combin_num", type=int, default=10)

    parser.add_argument("--ema_decay1", type=float, default=0.9999)
    parser.add_argument("--ema_decay2", type=float, default=0.9996)

    parser.add_argument("--sampling_method", default="heun", choices=["euler", "heun"])
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--gen_bsz", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=0,
                        help="0 evaluates the whole split.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")

    return parser


def _resolve_ckpt_path(resume: str) -> Path:
    p = Path(resume)
    if p.is_file():
        return p
    candidate = p / "checkpoint-last.pth"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Checkpoint not found at {resume}")


def _to_signal(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], x.shape[1], -1).float()


def _rrmse_time(pred: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    num = torch.sqrt(torch.mean((pred - clean) ** 2, dim=(-2, -1)))
    den = torch.sqrt(torch.mean(clean ** 2, dim=(-2, -1))).clamp_min(1e-8)
    return num / den


def _rrmse_freq(pred: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    pred_mag = torch.fft.rfft(pred, dim=-1).abs()
    clean_mag = torch.fft.rfft(clean, dim=-1).abs()
    num = torch.sqrt(torch.mean((pred_mag - clean_mag) ** 2, dim=(-2, -1)))
    den = torch.sqrt(torch.mean(clean_mag ** 2, dim=(-2, -1))).clamp_min(1e-8)
    return num / den


def _corr(pred: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    pred_f = pred.reshape(pred.shape[0], -1)
    clean_f = clean.reshape(clean.shape[0], -1)
    pred_c = pred_f - pred_f.mean(dim=1, keepdim=True)
    clean_c = clean_f - clean_f.mean(dim=1, keepdim=True)
    num = (pred_c * clean_c).sum(dim=1)
    den = torch.sqrt((pred_c.pow(2).sum(dim=1) + 1e-8) * (clean_c.pow(2).sum(dim=1) + 1e-8))
    return (num / den).clamp(-1.0, 1.0)


def _summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else None,
        "std": float(arr.std()) if arr.size else None,
    }


def _append_group_metrics(store, key, rrmse_time, rrmse_freq, corr):
    store[key]["rrmse_time"].extend(rrmse_time)
    store[key]["rrmse_freq"].extend(rrmse_freq)
    store[key]["corr"].extend(corr)


def _summarize_grouped(grouped):
    return {
        str(key): {metric: _summarize(values) for metric, values in metrics.items()}
        for key, metrics in sorted(grouped.items(), key=lambda item: str(item[0]))
    }


def main(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.target_length % args.eeg_patch_size != 0:
        raise ValueError("target_length must be divisible by eeg_patch_size")
    args.eeg_patch_num = args.target_length // args.eeg_patch_size
    args.class_num = 0

    dataset_kwargs = {
        "noise_types": args.noise_types,
        "train_snr_range": (args.train_snr_min, args.train_snr_max),
        "eval_snr_levels": args.eval_snr_levels,
        "combin_num": args.combin_num,
        "seed": args.seed,
        "return_metadata": True,
    }
    dataset = build_base_dataset(args.dataset, args.eval_split, Path(args.datasets_dir), **dataset_kwargs)
    if args.num_samples and args.num_samples > 0:
        indices = list(range(min(args.num_samples, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=args.gen_bsz, shuffle=False, num_workers=0, pin_memory=True)

    model = Denoiser(args).to(device)
    model.eval()
    ckpt_path = _resolve_ckpt_path(args.resume)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.ema_params1 = [ckpt["model_ema1"][n].to(device) for n, _ in model.named_parameters()]
    model.ema_params2 = [ckpt["model_ema2"][n].to(device) for n, _ in model.named_parameters()]
    print(f"Loaded weights from {ckpt_path}")

    noisy_chunks, clean_chunks, denoised_chunks = [], [], []
    metrics = {"rrmse_time": [], "rrmse_freq": [], "corr": []}
    per_snr = defaultdict(lambda: {"rrmse_time": [], "rrmse_freq": [], "corr": []})
    per_noise_type = defaultdict(lambda: {"rrmse_time": [], "rrmse_freq": [], "corr": []})
    per_noise_type_snr = defaultdict(lambda: {"rrmse_time": [], "rrmse_freq": [], "corr": []})

    for step, (noisy, clean, metadata) in enumerate(loader):
        noisy = noisy.to(device, non_blocking=True).float()
        clean = clean.to(device, non_blocking=True).float()
        denoised = model.denoise(noisy).detach()

        noisy_s = _to_signal(noisy).cpu()
        clean_s = _to_signal(clean).cpu()
        denoised_s = _to_signal(denoised).cpu()

        batch_rrmse_time = _rrmse_time(denoised_s, clean_s).tolist()
        batch_rrmse_freq = _rrmse_freq(denoised_s, clean_s).tolist()
        batch_corr = _corr(denoised_s, clean_s).tolist()

        metrics["rrmse_time"].extend(batch_rrmse_time)
        metrics["rrmse_freq"].extend(batch_rrmse_freq)
        metrics["corr"].extend(batch_corr)

        noise_types = metadata["noise_type"]
        snr_values = metadata["snr_db"].tolist()
        for i, (noise_type, snr_db) in enumerate(zip(noise_types, snr_values)):
            snr_key = f"{float(snr_db):g}"
            combo_key = f"{noise_type}_{snr_key}dB"
            _append_group_metrics(per_snr, snr_key, [batch_rrmse_time[i]], [batch_rrmse_freq[i]], [batch_corr[i]])
            _append_group_metrics(per_noise_type, noise_type, [batch_rrmse_time[i]], [batch_rrmse_freq[i]], [batch_corr[i]])
            _append_group_metrics(per_noise_type_snr, combo_key, [batch_rrmse_time[i]], [batch_rrmse_freq[i]], [batch_corr[i]])

        noisy_chunks.append(noisy_s)
        clean_chunks.append(clean_s)
        denoised_chunks.append(denoised_s)
        print(f"  denoised {(step + 1) * noisy.shape[0]}/{len(dataset)}")

    noisy_all = torch.cat(noisy_chunks, dim=0)
    clean_all = torch.cat(clean_chunks, dim=0)
    denoised_all = torch.cat(denoised_chunks, dim=0)

    np.savez_compressed(
        output_dir / "eval_batch.npz",
        noisy=noisy_all.numpy(),
        clean=clean_all.numpy(),
        denoised=denoised_all.numpy(),
    )

    summary = {
        "num_samples": int(denoised_all.shape[0]),
        "dataset": args.dataset,
        "split": args.eval_split,
        "noise_types": args.noise_types,
        "eval_snr_levels": args.eval_snr_levels,
        "metrics": {name: _summarize(values) for name, values in metrics.items()},
        "per_snr": _summarize_grouped(per_snr),
        "per_noise_type": _summarize_grouped(per_noise_type),
        "per_noise_type_snr": _summarize_grouped(per_noise_type_snr),
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary["metrics"], indent=2))
    print(f"Wrote {output_dir / 'eval_batch.npz'} and {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
