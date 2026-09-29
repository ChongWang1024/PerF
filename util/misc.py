import datetime
import os
import random
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def barrier():
    if dist.is_initialized():
        dist.barrier()


def setup_distributed(args):
    if isinstance(args, str):
        from argparse import Namespace
        args = Namespace(device=args, world_size=1, local_rank=-1, dist_on_itp=False, dist_url="env://")
    args.rank, args.gpu = 0, max(args.local_rank, 0)
    args.distributed = False
    if args.dist_on_itp:
        args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        args.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        args.gpu = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        args.dist_url = "tcp://%s:%s" % (os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])
        args.distributed = True
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
        args.distributed = True
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ.get("SLURM_NTASKS", args.world_size))
        args.gpu = args.rank % max(torch.cuda.device_count(), 1)
        args.distributed = True
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu for local checks")
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda", args.gpu)
    else:
        device = torch.device(args.device)
    if args.distributed:
        args.dist_backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        barrier()
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def add_weight_decay(model, weight_decay=0, skip_list=()):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim == 1 or name.endswith(".bias") or name in skip_list or "diffloss" in name else decay).append(
            parameter
        )
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


class EMA:

    def __init__(self, net, decays):
        self.decays = tuple(decays)
        if not self.decays or any(not 0 < d < 1 for d in self.decays):
            raise ValueError("EMA decays must be in (0, 1)")
        self.states = [
            {k: v.detach().clone() for k, v in net.state_dict().items()}
            for _ in self.decays
        ]

    @torch.no_grad()
    def update(self, net):
        parameters = dict(net.named_parameters())
        for decay, state in zip(self.decays, self.states):
            for name, value in net.state_dict().items():
                if name in parameters:
                    state[name].mul_(decay).add_(value.detach(), alpha=1 - decay)
                else:
                    state[name].copy_(value)

    def load(self, states, net):
        expected = net.state_dict()
        if len(states) != len(self.decays):
            raise ValueError("EMA count changed on resume")
        for state in states:
            if state.keys() != expected.keys() or any(
                state[k].shape != v.shape for k, v in expected.items()
            ):
                raise ValueError("Invalid backbone EMA state")
        self.states = [
            {
                k: v.to(device=expected[k].device, dtype=expected[k].dtype)
                for k, v in state.items()
            }
            for state in states
        ]


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def rng_state(device):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)


def optimizer_state_for_parameters(optimizer, parameters):
    keep = {id(parameter) for parameter in parameters}
    state = optimizer.state_dict()
    if len(optimizer.param_groups) != len(state["param_groups"]):
        raise ValueError("Optimizer parameter groups changed while saving")
    kept_ids = set()
    for live_group, saved_group in zip(optimizer.param_groups, state["param_groups"]):
        if len(live_group["params"]) != len(saved_group["params"]):
            raise ValueError("Optimizer parameter layout changed while saving")
        saved_group["params"] = [
            saved_id
            for parameter, saved_id in zip(live_group["params"], saved_group["params"])
            if id(parameter) in keep
        ]
        kept_ids.update(saved_group["params"])
    if len(kept_ids) != len(keep):
        raise ValueError("Optimizer does not contain every requested model parameter")
    state["state"] = {
        parameter_id: value
        for parameter_id, value in state["state"].items()
        if parameter_id in kept_ids
    }
    return state


def save_training(path, objective, optimizer, ema, epoch, args, model_args, device):
    if rank() == 0:
        keep_irepa = (
            objective.irepa is not None
            and objective.irepa.weight > 0
            and epoch + 1 < objective.irepa.earlystop
        )
        optimizer_state = (
            optimizer.state_dict()
            if keep_irepa
            else optimizer_state_for_parameters(
                optimizer,
                (p for p in objective.denoiser.net.parameters() if p.requires_grad),
            )
        )
        atomic_save(
            {
                "format": "perf-v1",
                "model": cpu_tree(objective.denoiser.net.state_dict()),
                "model_args": model_args,
                "model_ema1": cpu_tree(ema.states[0]),
                "model_ema2": cpu_tree(ema.states[1]),
                "irepa": cpu_tree(objective.irepa.state_dict())
                if keep_irepa
                else None,
                "optimizer": cpu_tree(optimizer_state),
                "epoch": epoch,
                "args": vars(args),
            },
            path,
        )


def load_checkpoint(path):
    # Resume checkpoints can contain optimizer state; load only trusted files.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != "perf-v1":
        raise ValueError("--resume requires a perf-v1 checkpoint")
    required = {"model", "model_args", "model_ema1", "model_ema2", "epoch"}
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f"Checkpoint is missing required entries: {missing}")
    return checkpoint


def checkpoint_start_epoch(checkpoint):
    epoch = checkpoint.get("epoch")
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f"Invalid checkpoint epoch: {epoch!r}")
    return epoch + 1


def resume_training(checkpoint, objective, optimizer, ema, args):
    start_epoch = checkpoint_start_epoch(checkpoint)
    objective.denoiser.net.load_state_dict(checkpoint["model"], strict=True)
    needs_irepa = start_epoch < args.irepa_earlystop and args.irepa_weight > 0
    if needs_irepa:
        if objective.irepa is None or checkpoint.get("irepa") is None:
            raise ValueError(
                f"Checkpoint requires iREPA to resume at epoch {start_epoch}"
            )
        objective.irepa.load_state_dict(checkpoint["irepa"], strict=True)
    if checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    elif rank() == 0:
        print("Checkpoint has no optimizer state; initialized a new optimizer")
    args.start_epoch = start_epoch
    ema.load([checkpoint["model_ema1"], checkpoint["model_ema2"]], objective.denoiser.net)
    return args.start_epoch


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not dist.is_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda' if dist.get_backend() == 'nccl' else 'cpu')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if rank() == 0 and (i % print_freq == 0 or i == len(iterable) - 1):
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if rank() == 0:
            print('{} Total time: {} ({:.4f} s / it)'.format(
                header, total_time_str, total_time / len(iterable)))
