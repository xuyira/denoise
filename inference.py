"""Denoise EEGdenoiseNet samples and report denoising metrics."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.data import build_base_dataset
from data.eegdenoisenet_dataset import benchmark_eval_snr_levels, benchmark_snr_range
from data.eegdenoisenet_metrics import (
    append_group_metrics,
    cc,
    new_metric_store,
    rrmse_spectral,
    rrmse_temporal,
    summarize,
    summarize_grouped,
    to_signal,
)
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
    parser.add_argument("--conv_refiner", action="store_true",
                        help="Add an EEGDfus-style Conv1D residual refiner after clean prediction.")
    parser.add_argument("--conv_refiner_channels", type=int, default=64)
    parser.add_argument("--conv_refiner_kernel", type=int, default=3)
    parser.add_argument("--condition_mode", default="none", choices=["none", "refiner"],
                        help="Use noisy EEG as a fixed condition for the Conv1D refiner.")

    parser.add_argument("--noise_types", nargs="+", default=["eog", "emg"], choices=["eog", "emg"])
    parser.add_argument("--train_snr_min", type=float, default=None)
    parser.add_argument("--train_snr_max", type=float, default=None)
    parser.add_argument("--eval_snr_levels", nargs="+", type=float, default=None)
    parser.add_argument("--combin_num", type=int, default=11)

    parser.add_argument("--ema_decay1", type=float, default=0.9999)
    parser.add_argument("--ema_decay2", type=float, default=0.9996)
    parser.add_argument("--loss_weight_velocity", type=float, default=0.0,
                        help="Recorded for clean-target mix experiments; unused during inference.")
    parser.add_argument("--prediction_target", default="velocity", choices=["velocity", "clean"],
                        help="Network target used during training: direct velocity or final clean EEG.")
    parser.add_argument("--clean_output", default="direct", choices=["direct", "residual"],
                        help="For clean target, predict clean EEG directly or a residual correction from z_t.")
    parser.add_argument("--denoise_mode", default="ode", choices=["ode", "direct"],
                        help="Use iterative ODE denoising or one-step direct clean prediction.")
    parser.add_argument("--t_eps", type=float, default=1e-5,
                        help="Minimum denominator when converting clean prediction to velocity.")

    parser.add_argument("--sampling_method", default="heun", choices=["euler", "heun"])
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--gen_bsz", type=int, default=64)
    parser.add_argument("--ema_mode", default="ema1", choices=["raw", "ema1", "ema2"],
                        help="Weights used during denoising: raw model, slow EMA1, or faster EMA2.")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="0 evaluates the whole split.")
    parser.add_argument("--sampling_rate", type=float, default=256.0,
                        help="Sampling rate used for EEGdenoiseNet spectral metrics.")
    parser.add_argument("--spectral_max_freq", type=float, default=120.0,
                        help="Upper frequency bound for EEGdenoiseNet RRMSE_spectral.")
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

    if len(args.noise_types) != 1 and (args.train_snr_min is None or args.train_snr_max is None or args.eval_snr_levels is None):
        raise ValueError("For EEGdenoiseNet benchmark runs, use one noise type at a time or pass explicit SNR settings.")

    noise_type = args.noise_types[0] if len(args.noise_types) == 1 else None
    if args.train_snr_min is None or args.train_snr_max is None:
        args.train_snr_min, args.train_snr_max = benchmark_snr_range(noise_type)
    if args.eval_snr_levels is None:
        args.eval_snr_levels = list(benchmark_eval_snr_levels(noise_type))

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
    print(
        "Inference config: "
        f"noise_types={args.noise_types}, "
        f"prediction_target={args.prediction_target}, "
        f"clean_output={args.clean_output}, "
        f"denoise_mode={args.denoise_mode}, "
        f"ema_mode={args.ema_mode}, "
        f"conv_refiner={args.conv_refiner}, "
        f"condition_mode={args.condition_mode}, "
        f"sampling_method={args.sampling_method}, "
        f"num_sampling_steps={args.num_sampling_steps}"
    )

    noisy_chunks, clean_chunks, denoised_chunks = [], [], []
    metrics = new_metric_store()
    per_snr = defaultdict(new_metric_store)
    per_noise_type = defaultdict(new_metric_store)
    per_noise_type_snr = defaultdict(new_metric_store)

    for step, (noisy, clean, metadata) in enumerate(loader):
        noisy = noisy.to(device, non_blocking=True).float()
        clean = clean.to(device, non_blocking=True).float()
        denoised = model.denoise(noisy, use_ema=args.ema_mode).detach()

        noisy_s = to_signal(noisy).cpu()
        clean_s = to_signal(clean).cpu()
        denoised_s = to_signal(denoised).cpu()

        batch_rrmse_temporal = rrmse_temporal(denoised_s, clean_s).tolist()
        batch_rrmse_spectral = rrmse_spectral(
            denoised_s,
            clean_s,
            sampling_rate=args.sampling_rate,
            max_freq=args.spectral_max_freq,
        ).tolist()
        batch_cc = cc(denoised_s, clean_s).tolist()

        metrics["RRMSE_temporal"].extend(batch_rrmse_temporal)
        metrics["RRMSE_spectral"].extend(batch_rrmse_spectral)
        metrics["CC"].extend(batch_cc)

        noise_types = metadata["noise_type"]
        snr_values = metadata["snr_db"].tolist()
        for i, (noise_type, snr_db) in enumerate(zip(noise_types, snr_values)):
            snr_key = f"{float(snr_db):g}"
            combo_key = f"{noise_type}_{snr_key}dB"
            append_group_metrics(
                per_snr,
                snr_key,
                [batch_rrmse_temporal[i]],
                [batch_rrmse_spectral[i]],
                [batch_cc[i]],
            )
            append_group_metrics(
                per_noise_type,
                noise_type,
                [batch_rrmse_temporal[i]],
                [batch_rrmse_spectral[i]],
                [batch_cc[i]],
            )
            append_group_metrics(
                per_noise_type_snr,
                combo_key,
                [batch_rrmse_temporal[i]],
                [batch_rrmse_spectral[i]],
                [batch_cc[i]],
            )

        noisy_chunks.append(noisy_s)
        clean_chunks.append(clean_s)
        denoised_chunks.append(denoised_s)
        print(f"  denoised {min((step + 1) * args.gen_bsz, len(dataset))}/{len(dataset)}")

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
        "sampling_rate": args.sampling_rate,
        "spectral_max_freq": args.spectral_max_freq,
        "ema_mode": args.ema_mode,
        "prediction_target": args.prediction_target,
        "clean_output": args.clean_output,
        "denoise_mode": args.denoise_mode,
        "conv_refiner": args.conv_refiner,
        "conv_refiner_channels": args.conv_refiner_channels,
        "conv_refiner_kernel": args.conv_refiner_kernel,
        "condition_mode": args.condition_mode,
        "loss_weight_velocity": args.loss_weight_velocity,
        "metrics": {name: summarize(values) for name, values in metrics.items()},
        "per_snr": summarize_grouped(per_snr),
        "per_noise_type": summarize_grouped(per_noise_type),
        "per_noise_type_snr": summarize_grouped(per_noise_type_snr),
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary["metrics"], indent=2))
    print(f"Wrote {output_dir / 'eval_batch.npz'} and {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
