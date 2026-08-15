import math
import sys

import torch
import contextlib

import util.misc as misc
import util.lr_sched as lr_sched


def _autocast_context(device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (noisy, clean) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        noisy = noisy.to(device, non_blocking=True).to(torch.float32)
        clean = clean.to(device, non_blocking=True).to(torch.float32)

        with _autocast_context(device):
            loss = model(noisy, clean)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    avg_loss = metric_logger.meters['loss'].global_avg if 'loss' in metric_logger.meters else None
    return avg_loss


@torch.no_grad()
def evaluate(model, data_loader, device, epoch=0, log_writer=None, args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Val: [{}]".format(epoch)
    print_freq = 20

    for noisy, clean in metric_logger.log_every(data_loader, print_freq, header):
        noisy = noisy.to(device, non_blocking=True).to(torch.float32)
        clean = clean.to(device, non_blocking=True).to(torch.float32)

        with _autocast_context(device):
            loss = model(noisy, clean)

        metric_logger.update(loss=loss.item())

    metric_logger.synchronize_between_processes()
    print("Validation stats:", metric_logger)
    avg_loss = metric_logger.meters["loss"].global_avg if "loss" in metric_logger.meters else None
    if log_writer is not None and avg_loss is not None:
        log_writer.add_scalar("val_loss", misc.all_reduce_mean(avg_loss), epoch)
    return avg_loss
