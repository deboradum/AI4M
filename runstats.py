#!/usr/bin/env python3

# MIT License
#
# Cost tracking of a training run: time per epoch, throughput, peak memory,
# parameter count, device, git commit. Written to <dest>/stats.json after
# every epoch so that a crashed or stopped run still has its numbers.
# Works on CPU, CUDA and MPS; memory counters that do not exist on a
# device are reported as null rather than failing.

import json
import time
import subprocess
from pathlib import Path

import torch
from torch import nn


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return device.type


def peak_memory_mb(device: torch.device) -> float | None:
    # Peak memory allocated by this process on the accelerator. Independent of
    # what other users run on a shared GPU, unlike nvidia-smi.
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    if device.type == "mps" and hasattr(torch.mps, "driver_allocated_memory"):
        return torch.mps.driver_allocated_memory() / 2**20
    return None


def peak_rss_mb() -> float | None:
    # Peak resident memory of the main process (not the DataLoader workers)
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is in kilobytes on Linux, bytes on macOS
        import sys
        return rss / 2**20 if sys.platform == "darwin" else rss / 2**10
    except ImportError:  # Windows
        return None


def synchronize(device: torch.device) -> None:
    # GPU kernels are asynchronous: without this, timings measure the launch, not the work
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class RunStats:
    def __init__(self, net: nn.Module, device: torch.device, dest: Path,
                 n_train: int, n_val: int, batch_size: int) -> None:
        self.device = device
        self.dest = dest
        self.t_start = time.perf_counter()
        self.t_phase = self.t_start

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        self.stats: dict = {
            "git_commit": git_commit(),
            "device": device.type,
            "device_name": device_name(device),
            "n_params": count_params(net),
            "n_train_slices": n_train,
            "n_val_slices": n_val,
            "batch_size": batch_size,
            "epochs": [],
        }

    def phase_start(self) -> None:
        synchronize(self.device)
        self.t_phase = time.perf_counter()

    def phase_end(self, name: str, epoch: int) -> None:
        synchronize(self.device)
        elapsed = time.perf_counter() - self.t_phase
        if len(self.stats["epochs"]) <= epoch:
            self.stats["epochs"].append({"epoch": epoch})
        self.stats["epochs"][epoch][f"{name}_s"] = round(elapsed, 2)

    def epoch_end(self, epoch: int, val_dice: float, best_epoch: int, best_val_dice: float) -> None:
        epochs = self.stats["epochs"]
        epochs[epoch]["val_dice"] = round(val_dice, 4)

        train_s = [ep["train_s"] for ep in epochs if "train_s" in ep]
        total_s = time.perf_counter() - self.t_start
        self.stats.update({
            "epochs_done": epoch + 1,
            "best_epoch": best_epoch,
            "best_val_dice": round(best_val_dice, 4),
            "total_s": round(total_s, 1),
            "total_min": round(total_s / 60, 2),
            "train_s_per_epoch": round(sum(train_s) / len(train_s), 2),
            "train_slices_per_s": round(self.stats["n_train_slices"] / (sum(train_s) / len(train_s)), 1),
            "peak_gpu_mem_mb": None if peak_memory_mb(self.device) is None else round(peak_memory_mb(self.device), 1),
            "peak_cpu_rss_mb": None if peak_rss_mb() is None else round(peak_rss_mb(), 1),
        })

        with open(self.dest / "stats.json", 'w') as f:
            json.dump(self.stats, f, indent=2)
