"""Train JET-style flow matching for EEG denoising."""

import argparse
import datetime
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from denoiser import Denoiser
from engine_eeg import evaluate, train_one_epoch
import util.misc as misc

from data.data import build_dataloaders
from data.eegdenoisenet_dataset import benchmark_eval_snr_levels, benchmark_snr_range


_TARGET_LENGTH_DEFAULTS = {"tuab": 2000, "tuev": 1000, "tusz": 1000, "eegdenoisenet": 512}


def get_args_parser():
    parser = argparse.ArgumentParser("JET training", add_help=False)

    # architecture
    parser.add_argument("--model", default="JiT-B/16", type=str, metavar="MODEL")
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dropout", type=float, default=0.0)
    parser.add_argument("--eeg_patch_size", type=int, default=64)
    parser.add_argument("--use_convffn", action="store_true",
                        help="Replace the block MLP with a depthwise Conv1d FFN over patch tokens.")
    parser.add_argument("--convffn_kernel_size", type=int, default=3)

    # dataset
    parser.add_argument("--dataset", default="eegdenoisenet", choices=["eegdenoisenet"])
    parser.add_argument("--datasets_dir", type=str, required=True,
                        help="Root folder of the EEG dataset.")
    parser.add_argument("--num_eeg_channels", type=int, default=1)
    parser.add_argument("--target_length", type=int, default=None,
                        help="Length of each EEG segment; defaults to the dataset's recommended value.")
    parser.add_argument("--noise_types", nargs="+", default=["eog", "emg"], choices=["eog", "emg"],
                        help="EEGdenoiseNet artifact types used to synthesize noisy EEG.")
    parser.add_argument("--train_snr_min", type=float, default=None)
    parser.add_argument("--train_snr_max", type=float, default=None)
    parser.add_argument("--eval_snr_levels", nargs="+", type=float, default=None)
    parser.add_argument("--combin_num", type=int, default=11,
                        help="Synthetic combinations per clean EEG epoch for EEGdenoiseNet training.")

    # training
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--blr", type=float, default=5e-5, help="Base learning rate (per 256 samples).")
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--lr_schedule", type=str, default="constant", choices=["constant", "cosine"])
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--ema_decay1", type=float, default=0.9999)
    parser.add_argument("--ema_decay2", type=float, default=0.9996)

    # loss
    parser.add_argument("--loss_type", default="l2", choices=["l1", "l2", "mix"])
    parser.add_argument("--loss_weight_recon", type=float, default=0.5)
    parser.add_argument("--loss_weight_stft", type=float, default=0.0)
    parser.add_argument("--loss_weight_stat", type=float, default=0.0)
    parser.add_argument("--loss_weight_tv", type=float, default=0.0)
    parser.add_argument("--loss_weight_corr", type=float, default=0.0)
    parser.add_argument("--loss_weight_velocity", type=float, default=0.0,
                        help="Auxiliary velocity MSE weight for clean-target mix loss.")
    parser.add_argument("--prediction_target", default="velocity", choices=["velocity", "clean"],
                        help="Network target: direct flow velocity or final clean EEG.")
    parser.add_argument("--clean_output", default="direct", choices=["direct", "residual"],
                        help="For clean target, predict clean EEG directly or a residual correction from z_t.")
    parser.add_argument("--denoise_mode", default="ode", choices=["ode", "direct"],
                        help="Denoising mode used by validation/inference for clean-target models.")
    parser.add_argument("--t_eps", type=float, default=1e-5,
                        help="Minimum denominator when converting clean prediction to velocity.")

    # sampling
    parser.add_argument("--sampling_method", default="heun", choices=["euler", "heun"])
    parser.add_argument("--num_sampling_steps", default=50, type=int)

    # data loading
    parser.add_argument("--weighted_sampler", action="store_true", dest="weighted_sampler", default=False)
    parser.add_argument("--no_weighted_sampler", action="store_false", dest="weighted_sampler")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin_mem", action="store_true", default=True)

    # checkpointing
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--resume", default="", type=str,
                        help="Path to a checkpoint file or a directory containing checkpoint-last.pth.")
    parser.add_argument("--save_last_freq", type=int, default=5)
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--device", default="cuda")

    # early stopping
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="Epochs without improvement before stopping (0 disables).")
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-3)

    return parser


def main(args):
    args.distributed = False
    args.rank = 0
    args.gpu = 0
    args.world_size = 1

    print("Arguments:\n{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    os.makedirs(args.output_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.output_dir)

    if args.target_length is None:
        args.target_length = _TARGET_LENGTH_DEFAULTS[args.dataset.lower()]

    if args.target_length % args.eeg_patch_size != 0:
        raise ValueError("target_length must be divisible by eeg_patch_size")
    args.eeg_patch_num = args.target_length // args.eeg_patch_size

    if len(args.noise_types) != 1 and (args.train_snr_min is None or args.train_snr_max is None or args.eval_snr_levels is None):
        raise ValueError("For EEGdenoiseNet benchmark runs, use one noise type at a time or pass explicit SNR settings.")

    noise_type = args.noise_types[0] if len(args.noise_types) == 1 else None
    if args.train_snr_min is None or args.train_snr_max is None:
        args.train_snr_min, args.train_snr_max = benchmark_snr_range(noise_type)
    if args.eval_snr_levels is None:
        args.eval_snr_levels = list(benchmark_eval_snr_levels(noise_type))

    dataset_kwargs = {}
    if args.dataset.lower() == "eegdenoisenet":
        dataset_kwargs = {
            "noise_types": args.noise_types,
            "train_snr_range": (args.train_snr_min, args.train_snr_max),
            "eval_snr_levels": args.eval_snr_levels,
            "combin_num": args.combin_num,
            "seed": args.seed,
        }

    data_loader_dict, num_classes = build_dataloaders(
        dataset_name=args.dataset,
        datasets_dir=args.datasets_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=None,
        use_weighted_sampler=args.weighted_sampler,
        dataset_kwargs=dataset_kwargs,
    )
    args.class_num = num_classes
    data_loader_train = data_loader_dict["train"]
    data_loader_val = data_loader_dict["val"]

    model = Denoiser(args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    eff_batch_size = args.batch_size
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    param_groups = misc.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    if args.resume:
        resume_path = Path(args.resume)
        ckpt_path = resume_path if resume_path.is_file() else resume_path / "checkpoint-last.pth"
        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model"])
            model.ema_params1 = [ckpt["model_ema1"][n].to(device) for n, _ in model.named_parameters()]
            model.ema_params2 = [ckpt["model_ema2"][n].to(device) for n, _ in model.named_parameters()]
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "epoch" in ckpt:
                args.start_epoch = ckpt["epoch"] + 1
            print(f"Resumed from {ckpt_path}")
        else:
            print(f"Warning: resume path {args.resume} not found; training from scratch.")
            model.ema_params1 = [p.detach().clone() for p in model.parameters()]
            model.ema_params2 = [p.detach().clone() for p in model.parameters()]
    else:
        model.ema_params1 = [p.detach().clone() for p in model.parameters()]
        model.ema_params2 = [p.detach().clone() for p in model.parameters()]

    early_stop_enabled = args.early_stop_patience is not None and args.early_stop_patience > 0
    best_loss = float("inf")
    epochs_since_improvement = 0

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_loss = train_one_epoch(
            model, model, data_loader_train, optimizer, device, epoch,
            log_writer=log_writer, args=args,
        )
        val_loss = evaluate(model, data_loader_val, device, epoch=epoch, log_writer=log_writer, args=args)
        monitor_loss = val_loss if val_loss is not None else epoch_loss

        if monitor_loss is not None:
            if monitor_loss + args.early_stop_min_delta < best_loss:
                best_loss = monitor_loss
                epochs_since_improvement = 0
                misc.save_model(args=args, model_without_ddp=model, optimizer=optimizer,
                                epoch=epoch, epoch_name="best")
            else:
                epochs_since_improvement += 1

        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(args=args, model_without_ddp=model, optimizer=optimizer,
                            epoch=epoch, epoch_name="last")

        if log_writer is not None:
            log_writer.flush()

        if early_stop_enabled and epochs_since_improvement >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch} (best loss {best_loss:.6f})")
            break

    print("Training time:", str(datetime.timedelta(seconds=int(time.time() - start_time))))


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
