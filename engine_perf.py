import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from util import lr_sched, misc


class TrainingObjective(nn.Module):

    def __init__(self, denoiser, irepa=None):
        super().__init__()
        self.denoiser, self.irepa = denoiser, irepa
        self.metrics = {}

    def set_epoch(self, epoch):
        return self.irepa.set_epoch(epoch) if self.irepa is not None else False

    def forward(self, images, labels):
        if self.irepa is not None and self.irepa.active:
            flow, features = self.denoiser(images, labels, return_persistent=True)
            alignment = self.irepa(features, images)
            loss = flow + self.irepa.weight * alignment
            self.metrics = {"flow": flow.detach(), "irepa": alignment.detach()}
        else:
            flow = self.denoiser(images, labels)
            loss = flow + self.irepa.inactive_loss() if self.irepa is not None else flow
            self.metrics = {"flow": flow.detach()}
        return loss


def train_one_epoch(model, objective, loader, optimizer, ema, device, epoch, args, log_writer=None):
    model.train()
    active = objective.set_epoch(epoch)
    if misc.rank() == 0:
        print(
            f"Epoch {epoch + 1}/{args.epochs}: iREPA {'on' if active else 'off'}",
            flush=True,
        )
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    for step, (images, labels) in enumerate(metric_logger.log_every(loader, 20, f"Epoch: [{epoch}]")):
        lr = lr_sched.adjust_learning_rate(optimizer, epoch + step / len(loader), args)
        images = images.to(device, non_blocking=True).float().div_(255).mul_(2).sub_(1)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            loss = model(images, labels)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        ema.update(objective.denoiser.net)
        metric_logger.update(loss=loss_value, lr=lr, **objective.metrics)
        reduced = torch.stack([loss.detach(), *objective.metrics.values()]).float()
        if dist.is_initialized():
            dist.all_reduce(reduced)
            reduced /= misc.world_size()
        if log_writer is not None and step % args.log_freq == 0:
            epoch_1000x = int((step / len(loader) + epoch) * 1000)
            log_writer.add_scalar("train_loss", reduced[0].item(), epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)
            for key, value in zip(objective.metrics, reduced[1:]):
                log_writer.add_scalar(f"train_{key}", value.item(), epoch_1000x)


def calculate_metrics(samples, reference, device):
    import torch_fidelity

    return torch_fidelity.calculate_metrics(
        input1=str(samples),
        input2=None,
        fid_statistics_file=str(reference),
        cuda=device.type == "cuda",
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        verbose=False,
    )


def resolve_fid_stats(image_size):
    if image_size not in (256, 512):
        raise ValueError(f"No ImageNet FID statistics for resolution {image_size}")
    path = Path(__file__).resolve().parent / "fid_stats" / f"jit_in{image_size}_stats.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing ImageNet FID reference statistics: {path}")
    return path


@torch.no_grad()
def evaluate(denoiser, args, device, log_writer=None, epoch=0):
    denoiser.eval()
    reference = resolve_fid_stats(denoiser.net.input_size)
    if args.num_images % denoiser.net.num_classes != 0:
        raise ValueError("Number of images per class must be the same")
    samples = Path(args.output_dir) / (
        f"{denoiser.method}-steps{denoiser.steps}-cfg{args.cfg}"
        f"-interval{args.cfg_interval_min}-{args.cfg_interval_max}"
        f"-pg{args.pg}-pginterval{args.pg_interval_min}-{args.pg_interval_max}"
        f"-image{args.num_images}-res{denoiser.net.input_size}"
    )
    if misc.rank() == 0:
        samples.mkdir(parents=True, exist_ok=True)
    misc.barrier()
    batch_size, world_size = args.gen_bsz, misc.world_size()
    num_steps = args.num_images // (batch_size * world_size) + 1
    class_labels = np.arange(denoiser.net.num_classes).repeat(
        args.num_images // denoiser.net.num_classes
    )
    class_labels = np.hstack([class_labels, np.zeros(batch_size * world_size)])
    for step in range(num_steps):
        start = world_size * batch_size * step + misc.rank() * batch_size
        labels = torch.as_tensor(class_labels[start:start + batch_size], device=device, dtype=torch.long)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            images = denoiser.generate(labels)
        misc.barrier()
        images = ((images + 1) / 2).detach().cpu()
        for offset, pixels in enumerate(images):
            index = start + offset
            if index >= args.num_images:
                break
            pixels = np.round(np.clip(pixels.numpy().transpose(1, 2, 0) * 255, 0, 255))
            cv2.imwrite(str(samples / f"{index:05d}.png"), pixels.astype(np.uint8)[:, :, ::-1])
        if misc.rank() == 0:
            print(f"Generation step {step}/{num_steps}", flush=True)
    misc.barrier()
    if misc.rank() == 0:
        metrics = calculate_metrics(samples, reference, device)
        fid = float(metrics["frechet_inception_distance"])
        inception_score = float(metrics["inception_score_mean"])
        if log_writer is not None:
            postfix = f"_cfg{args.cfg}_pg{args.pg}_res{denoiser.net.input_size}"
            log_writer.add_scalar(f"fid{postfix}", fid, epoch)
            log_writer.add_scalar(f"is{postfix}", inception_score, epoch)
        result = {
            "fid": fid,
            "inception_score": inception_score,
            "save_folder": str(samples),
            "method": denoiser.method,
            "steps": denoiser.steps,
            "cfg": args.cfg,
            "cfg_interval": [args.cfg_interval_min, args.cfg_interval_max],
            "pg": args.pg,
            "pg_interval": [args.pg_interval_min, args.pg_interval_max],
            "num_images": args.num_images,
            "img_size": denoiser.net.input_size,
            "epoch": int(epoch),
        }
        Path(f"{samples}.metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"FID: {fid:.4f}, Inception Score: {inception_score:.4f}")
        print(f"Saved evaluation metrics to {samples}.metrics.json")
    misc.barrier()
