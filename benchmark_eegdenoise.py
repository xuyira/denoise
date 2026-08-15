"""Train and evaluate EEGdenoiseNet benchmark baselines in PyTorch."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data.data import build_dataloaders
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
from models.eegdenoise_baselines import build_baseline_model
from models.eegdenoisenet_extra import (
    build_gan_models,
    gan_losses,
)


TRAINABLE_METHODS = {"fcnn", "simple_cnn", "complex_cnn", "rnn_lstm", "gan"}


def get_args_parser():
    parser = argparse.ArgumentParser("EEGdenoiseNet baseline benchmark", add_help=True)
    parser.add_argument("--datasets_dir", type=str, required=True)
    parser.add_argument("--output_dir", default="./output/eegdenoise_baselines")
    parser.add_argument(
        "--model",
        choices=sorted(TRAINABLE_METHODS),
        required=True,
    )
    parser.add_argument("--noise_types", nargs="+", default=["eog"], choices=["eog", "emg"])
    parser.add_argument("--train_snr_min", type=float, default=None)
    parser.add_argument("--train_snr_max", type=float, default=None)
    parser.add_argument("--eval_snr_levels", nargs="+", type=float, default=None)
    parser.add_argument("--combin_num", type=int, default=10)
    parser.add_argument("--target_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sampling_rate", type=float, default=256.0)
    parser.add_argument("--spectral_max_freq", type=float, default=120.0)
    parser.add_argument("--gan_lr", type=float, default=1e-3)
    parser.add_argument("--gan_recon_weight", type=float, default=1.0)
    parser.add_argument("--gan_adv_weight", type=float, default=5e-4)
    parser.add_argument("--gan_l1_weight", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_mem", action="store_true", default=True)
    return parser


def _make_loaders(args):
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
    loaders, _ = build_dataloaders(
        dataset_name="eegdenoisenet",
        datasets_dir=Path(args.datasets_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=None,
        use_weighted_sampler=False,
        dataset_kwargs=dataset_kwargs,
    )
    return loaders


def _evaluate_mse(model, loader, device):
    model.eval()
    total = 0.0
    count = 0
    criterion = nn.MSELoss(reduction="sum")
    with torch.no_grad():
        for noisy, clean, _ in loader:
            noisy = noisy.to(device).float()
            clean = clean.to(device).float()
            pred = model(noisy)
            loss = criterion(pred, clean)
            total += float(loss.item())
            count += int(clean.numel())
    return total / max(1, count)


def _evaluate_no_train(args, model_name, model, loader, device):
    metrics = new_metric_store()
    per_snr = {}
    per_noise_type = {}
    per_noise_type_snr = {}
    noisy_chunks, clean_chunks, denoised_chunks = [], [], []
    with torch.no_grad():
        for noisy, clean, metadata in loader:
            noisy = noisy.to(device).float()
            clean = clean.to(device).float()
            pred = model(noisy)

            noisy_s = to_signal(noisy).cpu()
            clean_s = to_signal(clean).cpu()
            pred_s = to_signal(pred).cpu()

            batch_rrmse_temporal = rrmse_temporal(pred_s, clean_s).tolist()
            batch_rrmse_spectral = rrmse_spectral(
                pred_s, clean_s, sampling_rate=args.sampling_rate, max_freq=args.spectral_max_freq
            ).tolist()
            batch_cc = cc(pred_s, clean_s).tolist()

            metrics["RRMSE_temporal"].extend(batch_rrmse_temporal)
            metrics["RRMSE_spectral"].extend(batch_rrmse_spectral)
            metrics["CC"].extend(batch_cc)

            for i, (noise_type, snr_db) in enumerate(zip(metadata["noise_type"], metadata["snr_db"].tolist())):
                snr_key = f"{float(snr_db):g}"
                combo_key = f"{noise_type}_{snr_key}dB"
                per_snr.setdefault(snr_key, new_metric_store())
                per_noise_type.setdefault(noise_type, new_metric_store())
                per_noise_type_snr.setdefault(combo_key, new_metric_store())
                append_group_metrics(per_snr, snr_key, [batch_rrmse_temporal[i]], [batch_rrmse_spectral[i]], [batch_cc[i]])
                append_group_metrics(
                    per_noise_type, noise_type, [batch_rrmse_temporal[i]], [batch_rrmse_spectral[i]], [batch_cc[i]]
                )
                append_group_metrics(
                    per_noise_type_snr, combo_key, [batch_rrmse_temporal[i]], [batch_rrmse_spectral[i]], [batch_cc[i]]
                )

            noisy_chunks.append(noisy_s)
            clean_chunks.append(clean_s)
            denoised_chunks.append(pred_s)

    return metrics, per_snr, per_noise_type, per_noise_type_snr, noisy_chunks, clean_chunks, denoised_chunks


def _train_regression_model(args, model, train_loader, val_loader, device, run_dir):
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_path = run_dir / "checkpoint-best.pth"
    last_path = run_dir / "checkpoint-last.pth"

    for epoch in range(args.epochs):
        model.train()
        train_sum = 0.0
        train_count = 0
        for noisy, clean, _ in train_loader:
            noisy = noisy.to(device).float()
            clean = clean.to(device).float()
            optimizer.zero_grad(set_to_none=True)
            pred = model(noisy)
            loss = criterion(pred, clean)
            loss.backward()
            optimizer.step()
            train_sum += float(loss.item()) * clean.shape[0]
            train_count += int(clean.shape[0])

        val_loss = _evaluate_mse(model, val_loader, device)
        train_loss = train_sum / max(1, train_count)
        torch.save(_checkpoint_state(model, optimizer, epoch, best_loss), last_path)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(_checkpoint_state(model, optimizer, epoch, best_loss), best_path)
        print(f"Epoch {epoch + 1}/{args.epochs} train_mse={train_loss:.6f} val_mse={val_loss:.6f} best={best_loss:.6f}")

    ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    return model, best_path, last_path


def _train_gan_model(args, generator, discriminator, train_loader, val_loader, device, run_dir):
    g_opt = torch.optim.Adam(generator.parameters(), lr=args.gan_lr, betas=(0.9, 0.999), eps=1e-8)
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=args.gan_lr, betas=(0.9, 0.999), eps=1e-8)
    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    best_loss = float("inf")
    best_path = run_dir / "checkpoint-best.pth"
    last_path = run_dir / "checkpoint-last.pth"

    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        train_sum = 0.0
        train_count = 0
        for noisy, clean, _ in train_loader:
            noisy = noisy.to(device).float()
            clean = clean.to(device).float()

            with torch.no_grad():
                fake_for_d = generator(noisy)
            real_logits = discriminator(clean)
            fake_logits = discriminator(fake_for_d.detach())
            _, d_loss = gan_losses(real_logits, fake_logits)
            d_opt.zero_grad(set_to_none=True)
            d_loss.backward()
            d_opt.step()

            fake = generator(noisy)
            fake_logits = discriminator(fake)
            real_logits = discriminator(clean)
            g_adv_loss, _ = gan_losses(real_logits.detach(), fake_logits)
            recon_loss = mse(fake, clean)
            if args.gan_l1_weight > 0:
                recon_loss = recon_loss + args.gan_l1_weight * l1(fake, clean)
            g_loss = args.gan_recon_weight * recon_loss + args.gan_adv_weight * g_adv_loss
            g_opt.zero_grad(set_to_none=True)
            g_loss.backward()
            g_opt.step()

            train_sum += float(g_loss.item()) * clean.shape[0]
            train_count += int(clean.shape[0])

        val_loss = _evaluate_noisy_to_clean_mse(generator, val_loader, device, args)
        train_loss = train_sum / max(1, train_count)
        torch.save(
            {
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer_g": g_opt.state_dict(),
                "optimizer_d": d_opt.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss,
            },
            last_path,
        )
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "optimizer_g": g_opt.state_dict(),
                    "optimizer_d": d_opt.state_dict(),
                    "epoch": epoch,
                    "best_loss": best_loss,
                },
                best_path,
            )
        print(f"Epoch {epoch + 1}/{args.epochs} train_gan={train_loss:.6f} val_mse={val_loss:.6f} best={best_loss:.6f}")

    ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
    generator.load_state_dict(ckpt["generator"])
    generator.to(device)
    return generator, best_path, last_path


def _evaluate_noisy_to_clean_mse(model, loader, device, args):
    model.eval()
    total = 0.0
    count = 0
    criterion = nn.MSELoss(reduction="sum")
    with torch.no_grad():
        for noisy, clean, _ in loader:
            noisy = noisy.to(device).float()
            clean = clean.to(device).float()
            pred = model(noisy)
            total += float(criterion(pred, clean).item())
            count += int(clean.numel())
    return total / max(1, count)


def _checkpoint_state(model, optimizer, epoch, best_loss):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
    }


def main(args):
    args.model = args.model.lower()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / "_".join(args.noise_types) / args.model
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.target_length % 64 != 0:
        raise ValueError("target_length must be divisible by 64 for benchmark comparison.")

    loaders = _make_loaders(args)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    best_path = None
    last_path = None
    if args.model == "gan":
        generator, discriminator = build_gan_models()
        generator = generator.to(device)
        discriminator = discriminator.to(device)
        model, best_path, last_path = _train_gan_model(args, generator, discriminator, train_loader, val_loader, device, run_dir)
        model.eval()
    else:
        model = build_baseline_model(args.model, args.target_length).to(device)
        model, best_path, last_path = _train_regression_model(args, model, train_loader, val_loader, device, run_dir)
        model.eval()

    metrics, per_snr, per_noise_type, per_noise_type_snr, noisy_chunks, clean_chunks, denoised_chunks = _evaluate_no_train(
        args, args.model, model, test_loader, device
    )

    noisy_all = torch.cat(noisy_chunks, dim=0)
    clean_all = torch.cat(clean_chunks, dim=0)
    denoised_all = torch.cat(denoised_chunks, dim=0)

    np.savez_compressed(
        run_dir / "eval_batch.npz",
        noisy=noisy_all.numpy(),
        clean=clean_all.numpy(),
        denoised=denoised_all.numpy(),
    )

    summary = {
        "model": args.model,
        "noise_types": args.noise_types,
        "train_snr_range": [args.train_snr_min, args.train_snr_max],
        "eval_snr_levels": args.eval_snr_levels,
        "metrics": {name: summarize(values) for name, values in metrics.items()},
        "per_snr": summarize_grouped(per_snr),
        "per_noise_type": summarize_grouped(per_noise_type),
        "per_noise_type_snr": summarize_grouped(per_noise_type_snr),
    }
    if best_path is not None and last_path is not None:
        summary["checkpoint_best"] = str(best_path)
        summary["checkpoint_last"] = str(last_path)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary["metrics"], indent=2))
    if best_path is not None and last_path is not None:
        print(f"Wrote {best_path}, {last_path}, {run_dir / 'metrics.json'}")
    else:
        print(f"Wrote {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main(get_args_parser().parse_args())
