import argparse
import datetime
import time
from collections.abc import Mapping
from functools import partial
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from denoiser import Denoiser
from engine_perf import TrainingObjective, evaluate, train_one_epoch
from model_perf import MODEL_CONFIGS, build_model
from util import misc


def get_args_parser():
    parser = argparse.ArgumentParser(description="PerF")

    # architecture
    parser.add_argument("--model", choices=tuple(MODEL_CONFIGS), default=None)
    parser.add_argument("--img_size", type=int, default=None, help="Image size; inferred from the model when omitted")
    parser.add_argument("--attn_dropout", type=float, default=None)
    parser.add_argument("--proj_dropout", type=float, default=None)
    parser.add_argument("--p_to_a_dropout", type=float, default=None)

    # training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128, help="Per-GPU training batch size")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--blr", type=float, default=5e-5, help="lr = blr * global_batch / 256")
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--lr_schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--ema_decay1", type=float, default=0.9999)
    parser.add_argument("--ema_decay2", type=float, default=0.9996)
    parser.add_argument("--P_mean", type=float, default=-0.8)
    parser.add_argument("--P_std", type=float, default=0.8)
    parser.add_argument("--noise_scale", type=float, default=None)
    parser.add_argument("--t_eps", type=float, default=0.05)
    parser.add_argument("--label_drop_prob", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # irepa
    parser.add_argument("--irepa_earlystop", type=int, default=100, help="Number of iREPA epochs, 0 disables it",)
    parser.add_argument("--irepa_weight", type=float, default=0.1)
    parser.add_argument("--irepa_kernel_size", type=int, default=3)
    parser.add_argument("--irepa_spnorm_alpha", type=float, default=0.8)
    parser.add_argument("--irepa_spnorm_eps", type=float, default=1e-6)
    parser.add_argument("--dino_repo", default="facebookresearch/dinov2")
    parser.add_argument("--dino_source", choices=("github", "local"), default="github")
    parser.add_argument("--dino_checkpoint", default="")

    # sampling
    parser.add_argument("--sampling_method", choices=("heun", "euler"), default="heun")
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--cfg_interval_min", type=float, default=0.0)
    parser.add_argument("--cfg_interval_max", type=float, default=1.0)
    parser.add_argument("--pg", type=float, default=1.0)
    parser.add_argument("--pg_interval_min", type=float, default=0.0)
    parser.add_argument("--pg_interval_max", type=float, default=1.0)
    parser.add_argument("--num_images", type=int, default=50000)
    parser.add_argument("--gen_bsz", type=int, default=256)

    # evaluation
    parser.add_argument("--evaluate_gen", action="store_true")

    # dataset
    parser.add_argument("--data_path", default="./data/imagenet", help="ImageFolder root")
    parser.add_argument("--class_num", type=int, default=1000)

    # checkpointing
    parser.add_argument("--output_dir", default="./output_dir")
    parser.add_argument("--resume", default="", help="Checkpoint file or directory; directories use checkpoint-last.pth")
    parser.add_argument("--save_last_freq", type=int, default=5)
    parser.add_argument("--log_freq", type=int, default=100)

    # runtime
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")

    # distributed training
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")

    return parser


def resolve_resume_path(resume, evaluate=False):
    path = Path(resume).expanduser()
    if path.is_dir():
        path = path / "checkpoint-last.pth"
    if not path.is_file() and evaluate:
        raise FileNotFoundError(path)
    return path


def evaluation_weights(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a dictionary")
    checkpoint_format = checkpoint.get("format")
    if checkpoint_format != "perf-v1":
        raise ValueError(f"Unsupported PerF checkpoint format: {checkpoint_format!r}")
    if "model_args" not in checkpoint:
        raise ValueError("Evaluation checkpoint is missing model_args")
    if "model_ema1" not in checkpoint:
        raise ValueError("Evaluation checkpoint is missing model_ema1")
    state, source = checkpoint["model_ema1"], "model_ema1"
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint {source} must be a state_dict")
    return state, checkpoint["model_args"], source


def resolve_model_args(args, saved=None):
    defaults = dict(
        model="PerF-B/16",
        num_classes=1000,
        p_to_a_dropout=0.1,
        attn_drop=0.0,
        proj_drop=0.0,
    )
    names = dict(
        model="model",
        num_classes="class_num",
        p_to_a_dropout="p_to_a_dropout",
        attn_drop="attn_dropout",
        proj_drop="proj_dropout",
    )
    if saved is not None and set(saved) != set(defaults):
        raise ValueError(
            "Unrecognized model metadata; legacy conversion is not implemented"
        )
    result = dict(defaults if saved is None else saved)
    if saved is None and args.model and args.model.startswith("PerF-H/"):
        result["proj_drop"] = 0.2
    for key, attr in names.items():
        value = getattr(args, attr)
        if value is not None:
            if saved is not None and value != saved[key]:
                raise ValueError(f"--{attr} conflicts with checkpoint metadata")
            result[key] = value
        setattr(args, attr, result[key])
    image_size = MODEL_CONFIGS[args.model]["input_size"]
    if args.img_size is not None and args.img_size != image_size:
        raise ValueError(f"--img_size must be {image_size} for {args.model}")
    args.img_size = image_size
    if args.noise_scale is None:
        args.noise_scale = 2.0 if args.img_size == 512 else 1.0
    return result


def create_denoiser(net, args):
    return Denoiser(
        net,
        label_drop_prob=args.label_drop_prob,
        P_mean=args.P_mean,
        P_std=args.P_std,
        t_eps=args.t_eps,
        noise_scale=args.noise_scale,
        cfg_scale=args.cfg,
        pg_scale=args.pg,
        cfg_interval=(args.cfg_interval_min, args.cfg_interval_max),
        pg_interval=(args.pg_interval_min, args.pg_interval_max),
        sampling_method=args.sampling_method,
        num_sampling_steps=args.num_sampling_steps,
    )


def main(args):
    for name in (
        "batch_size",
        "epochs",
        "save_last_freq",
        "log_freq",
        "gen_bsz",
        "num_images",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if (
        args.irepa_earlystop < 0
        or args.irepa_weight < 0
        or args.num_workers < 0
        or args.warmup_epochs < 0
        or args.start_epoch < 0
    ):
        raise ValueError("Invalid training configuration")
    device = misc.setup_distributed(args)
    cudnn.benchmark = True
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False
    misc.seed_everything(args.seed + misc.rank())
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter

    log_writer = SummaryWriter(log_dir=args.output_dir) if misc.rank() == 0 else None
    if args.evaluate_gen:
        if not args.resume:
            raise ValueError("Evaluation requires --resume pointing to a PerF checkpoint or its directory")
        args.resume = str(resolve_resume_path(args.resume, evaluate=True))
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        state, saved_model_args, source = evaluation_weights(checkpoint)
        model_args = resolve_model_args(args, saved_model_args)
        net = build_model(
            model_args["model"], **{k: v for k, v in model_args.items() if k != "model"}
        )
        net.load_state_dict(state, strict=True)
        if misc.rank() == 0:
            print(f"Loaded evaluation weights from {source}: {args.resume}")
        del checkpoint
        denoiser = create_denoiser(net, args).to(device)
        with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
            torch.manual_seed(args.seed + misc.rank())
            evaluate(denoiser, args, device, log_writer=log_writer)
        if log_writer is not None:
            log_writer.close()
        return
    if not args.data_path:
        raise ValueError("Training requires --data_path")
    resume_checkpoint = None
    start_epoch = args.start_epoch
    if args.resume:
        args.resume = str(resolve_resume_path(args.resume))
        if Path(args.resume).is_file():
            resume_checkpoint = misc.load_checkpoint(args.resume)
            start_epoch = misc.checkpoint_start_epoch(resume_checkpoint)
            model_args = resolve_model_args(args, resume_checkpoint["model_args"])
        else:
            print("No checkpoint found; training from scratch")
            model_args = resolve_model_args(args)
    else:
        model_args = resolve_model_args(args)
    from torchvision import datasets, transforms

    from util.crop import center_crop_arr

    transform = transforms.Compose(
        [
            transforms.Lambda(partial(center_crop_arr, image_size=args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(
        str(Path(args.data_path) / "train"), transform=transform
    )
    if len(dataset.classes) != args.class_num:
        raise ValueError(
            f"Dataset has {len(dataset.classes)} classes, expected {args.class_num}"
        )
    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=misc.world_size(),
        rank=misc.rank(),
        shuffle=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError("Dataset is too small for the configured per-rank batch size")
    net = build_model(
        model_args["model"], **{k: v for k, v in model_args.items() if k != "model"}
    )
    auxiliary = None
    if start_epoch < args.irepa_earlystop and args.irepa_weight > 0:
        from util.irepa import PersistentIREPA

        auxiliary = PersistentIREPA(
            net.persistent_width,
            weight=args.irepa_weight,
            earlystop=args.irepa_earlystop,
            kernel_size=args.irepa_kernel_size,
            spnorm_alpha=args.irepa_spnorm_alpha,
            spnorm_eps=args.irepa_spnorm_eps,
            teacher_repo=args.dino_repo,
            teacher_source=args.dino_source,
            teacher_checkpoint=args.dino_checkpoint,
        )
    objective = TrainingObjective(create_denoiser(net, args), auxiliary).to(device)
    if args.lr is None:
        args.lr = args.blr * args.batch_size * misc.world_size() / 256
    optimizer = torch.optim.AdamW(
        misc.add_weight_decay(objective, args.weight_decay),
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    model = objective
    if dist.is_initialized():
        model = DistributedDataParallel(
            objective, device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,  # Fixed RoPE tables need no per-step broadcast.
        )
    # DDP broadcasts rank-zero parameters before EMA is initialized.
    ema = misc.EMA(net, (args.ema_decay1, args.ema_decay2))
    if resume_checkpoint is not None:
        start_epoch = misc.resume_training(
            resume_checkpoint, objective, optimizer, ema, args
        )
    if start_epoch >= args.epochs:
        raise ValueError(
            f"Checkpoint has completed {start_epoch} epochs; --epochs must be larger"
        )
    objective.set_epoch(start_epoch)
    if misc.rank() == 0:
        print(
            f"{args.model}: {sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6:.3f}M parameters"
        )
        print(
            f"Global batch={args.batch_size * misc.world_size()}, learning rate={args.lr}"
        )
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        train_one_epoch(
            model, objective, loader, optimizer, ema, device, epoch, args, log_writer=log_writer
        )
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_training(
                Path(args.output_dir) / "checkpoint-last.pth",
                objective,
                optimizer,
                ema,
                epoch,
                args,
                model_args,
                device,
            )
            misc.barrier()
        if epoch > 0 and epoch % 50 == 0:
            misc.save_training(
                Path(args.output_dir) / f"checkpoint-{epoch}.pth",
                objective, optimizer, ema, epoch, args, model_args, device,
            )
        if log_writer is not None:
            log_writer.flush()
    if log_writer is not None:
        log_writer.close()
    print("Training time:", str(datetime.timedelta(seconds=int(time.time() - start_time))))


if __name__ == "__main__":
    try:
        main(get_args_parser().parse_args())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
