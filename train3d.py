#!/usr/bin/env python3

# MIT License
#
# Training of the 3D U-Net on SegTHOR. Volumes are the stacked 2D slices of
# the configured dataset (see segthor/volumes.py), so E_F07 shares E_F05/E_F06's
# preprocessing and split, and keeps their recipe where it carries over to 3D:
# CE + soft Dice, AdamW (wd 1e-4) with cosine decay to 0, in-plane
# rotation/scale + intensity augmentation, checkpoint at the best validation
# Dice. What changes: random 3D patches instead of slices, an "epoch" is a
# fixed number of iterations, and validation is the 3D Dice of sliding-window
# predictions over whole volumes, every `val_every` epochs.
#
# python train3d.py --config configs/full/E_F07_unet3d_adamw_cosine.yaml --dest results/full/E_F07 --gpu

import json
import time
import random
import argparse
from pathlib import Path
from pprint import pprint
from dataclasses import dataclass, asdict
from typing import Optional, List

import yaml
import torch
import numpy as np
import torch.nn.functional as F

from segthor.dataset import AugParams
from segthor.models.UNet3D import UNet3D
from segthor.runstats import count_params, git_commit
from segthor.volumes import load_split, PatchSampler, augment_batch, sliding_window_probs, dice_3d
from segthor.wandb_logger import init_run as init_wandb_run, log_epoch as log_wandb_epoch, finish_run as finish_wandb_run


@dataclass
class TrainConfig3D:
    dataset: str
    K: int
    B: int
    kernels: int
    factor: int
    lr: float
    betas: List[float]
    epochs: int
    iters_per_epoch: int
    patch_size: List[int]
    seed: int
    net_name: str = "UNet3D"
    loss_fn: str = "ce_dice"
    max_channels: int = 320
    optimizer: str = "AdamW"
    weight_decay: float = 0.0
    lr_scheduler: Optional[str] = "cosine"
    lr_min: float = 0.0
    warmup_epochs: int = 0
    fg_ratio: float = 0.33
    val_every: int = 5
    val_overlap: float = 0.5
    amp: bool = True
    temperature: float = 1.0
    augment: bool = True
    aug_rotation: float = 15.0
    aug_scale_min: float = 0.9
    aug_scale_max: float = 1.1
    aug_intensity: float = 0.1
    wandb_enabled: bool = False
    wandb_project: str = "AI4M"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None

    def __post_init__(self) -> None:
        assert self.net_name == "UNet3D", self.net_name
        assert self.loss_fn == "ce_dice", self.loss_fn
        assert len(self.patch_size) == 3
        assert all(p % 2 ** self.factor == 0 for p in self.patch_size), (self.patch_size, self.factor)
        assert self.lr_scheduler in [None, "cosine"], self.lr_scheduler


def ce_dice_loss(probs: torch.Tensor, gt_1h: torch.Tensor) -> torch.Tensor:
    """Same formula as segthor.losses.CEDiceLoss over all classes, without its
    simplex/sset asserts: torch.unique on a 3D batch costs ~2.5 s per call,
    and both properties hold by construction (softmax, one_hot)."""
    mask = gt_1h.float()
    ce = -(mask * (probs + 1e-10).log()).sum() / (mask.sum() + 1e-10)
    dims = tuple(range(2, probs.ndim))
    eps = 1e-5
    dice = (2 * (probs * mask).sum(dims) + eps) / (probs.sum(dims) + mask.sum(dims) + eps)
    return ce + 1 - dice.mean()


def load_config(path: Path) -> TrainConfig3D:
    with open(path) as f:
        return TrainConfig3D(**yaml.safe_load(f))


def build_net(config: TrainConfig3D) -> UNet3D:
    return UNet3D(1, config.K, kernels=config.kernels, factor=config.factor,
                  max_channels=config.max_channels)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic 3D convolutions are much slower; the seed fixes the
    # initialisation and the patch/augmentation draws, not the cuDNN kernels.
    torch.backends.cudnn.benchmark = True
    print(f"Set seed {seed}")


def lr_lambda(config: TrainConfig3D):
    def f(epoch: int) -> float:
        if epoch < config.warmup_epochs:
            return (epoch + 1) / config.warmup_epochs
        if config.lr_scheduler is None:
            return 1.0
        t = (epoch - config.warmup_epochs) / max(config.epochs - config.warmup_epochs, 1)
        lo = config.lr_min / config.lr
        return lo + (1 - lo) * 0.5 * (1 + np.cos(np.pi * t))
    return f


@torch.no_grad()
def validate(net, val_volumes, config: TrainConfig3D, device) -> np.ndarray:
    net.eval()
    dices = []
    for id_, (img, gt) in val_volumes.items():
        probs = sliding_window_probs(net, img, tuple(config.patch_size), config.K, device,
                                     overlap=config.val_overlap, amp=config.amp,
                                     temperature=config.temperature)
        pred = probs.argmax(0)
        dices.append(dice_3d(pred, torch.from_numpy(gt).to(device), config.K).numpy())
    net.train()
    return np.stack(dices)  # (n_patients, K)


def run(args, config: TrainConfig3D) -> None:
    device = torch.device("cuda") if args.gpu and torch.cuda.is_available() else torch.device("cpu")
    print(f">> Picked {device} to run experiments")
    dest: Path = args.dest

    root = Path("data") / config.dataset
    train_volumes = load_split(root, "train")
    val_volumes = load_split(root, "val")
    if args.debug:
        train_volumes = dict(list(train_volumes.items())[:2])
        val_volumes = dict(list(val_volumes.items())[:1])

    sampler = PatchSampler(train_volumes, tuple(config.patch_size), device, fg_ratio=config.fg_ratio)
    aug_params = AugParams(rotation_deg=config.aug_rotation, scale_min=config.aug_scale_min,
                           scale_max=config.aug_scale_max, intensity_shift=config.aug_intensity,
                           elastic_alpha=0.0)

    net = build_net(config)
    net.init_weights()
    net.to(device)
    n_params = count_params(net)
    print(f">> {n_params / 1e6:.2f} M parameters")

    optimizer = getattr(torch.optim, config.optimizer)(net.parameters(), lr=config.lr,
                                                       betas=tuple(config.betas),
                                                       weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(config))

    log: list[dict] = []
    best_dice, best_epoch = 0.0, -1
    start_epoch: int = 0
    wandb_id: str | None = None

    # Resume: this runs for hours outside a scheduler, so a crash, a reboot or a
    # killed process should not cost the whole run. checkpoint.pt is rewritten
    # (atomically) after every epoch and is the only thing needed to continue.
    ckpt_path = dest / "checkpoint.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["net"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_dice, best_epoch = ckpt["best_dice"], ckpt["best_epoch"]
        log = ckpt["log"]
        wandb_id = ckpt.get("wandb_id")
        torch.set_rng_state(ckpt["cpu_rng"].cpu())  # map_location moved it to the device
        if device.type == "cuda" and ckpt.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ckpt["cuda_rng"].cpu(), device)
        print(f">> Resumed from {ckpt_path} at epoch {start_epoch} "
              f"(best {best_dice:.4f} at epoch {best_epoch})")
    elif args.resume:
        print(f">> --resume given but {ckpt_path} does not exist: starting from scratch")

    wandb_run = init_wandb_run(config, dest, run_id=wandb_id, resume=wandb_id is not None)
    if wandb_run is not None:
        wandb_id = wandb_run.id
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for e in range(start_epoch, config.epochs):
        net.train()
        t0 = time.perf_counter()
        losses = []
        for _ in range(config.iters_per_epoch):
            img, gt = sampler.sample(config.B)
            if config.augment:
                img, gt = augment_batch(img, gt, aug_params)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.amp and device.type == "cuda"):
                logits = net(img)
            # Loss in fp32: the softmax/log of the CE are not bf16-safe
            probs = F.softmax(logits.float() / config.temperature, dim=1)
            gt_1h = F.one_hot(gt, config.K).permute(0, 4, 1, 2, 3)
            loss = ce_dice_loss(probs, gt_1h)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        t_train = time.perf_counter() - t0

        entry: dict = {"epoch": e, "lr": optimizer.param_groups[0]["lr"],
                       "train_loss": float(np.mean(losses)), "train_s": t_train}

        if (e + 1) % config.val_every == 0 or e == config.epochs - 1:
            t0 = time.perf_counter()
            dices = validate(net, val_volumes, config, device)
            per_class = np.nanmean(dices, axis=0)
            current = float(np.nanmean(per_class[1:]))
            entry.update({"val_s": time.perf_counter() - t0, "val_dice_fg": current,
                          "val_dice_per_class": per_class.tolist()})
            msg = (f"epoch {e:4d} | loss {entry['train_loss']:.4f} | val 3D Dice {current:.4f} | "
                   + " ".join(f"{k}:{d:.3f}" for k, d in enumerate(per_class) if k > 0))
            if current > best_dice:
                msg = ">>> Improved " + msg + f" (was {best_dice:.4f})"
                best_dice, best_epoch = current, e
                torch.save(net.state_dict(), dest / "bestweights.pt")
                with open(dest / "best_epoch.txt", "w") as f:
                    f.write(msg + "\n" + json.dumps({"per_patient": dict(zip(sorted(val_volumes), dices.tolist()))}) + "\n")
            print(msg, flush=True)
        else:
            print(f"epoch {e:4d} | loss {entry['train_loss']:.4f} | {t_train:.0f}s", flush=True)

        entry.update({"best_dice_fg": best_dice, "best_epoch": best_epoch})
        log.append(entry)
        with open(dest / "log.json", "w") as f:
            json.dump(log, f, indent=1)

        wandb_metrics = {"epoch": e + 1, "learning_rate": entry["lr"], "train/loss": entry["train_loss"],
                         "best/val_dice_foreground": best_dice, "best/epoch": best_epoch + 1}
        if "val_dice_fg" in entry:
            wandb_metrics["val/dice_foreground"] = entry["val_dice_fg"]
            for k in range(1, config.K):
                wandb_metrics[f"val/dice_class_{k}"] = entry["val_dice_per_class"][k]
        log_wandb_epoch(wandb_run, wandb_metrics)

        scheduler.step()

        # Written last and atomically: a crash mid-save leaves the previous
        # checkpoint intact rather than a truncated one.
        torch.save({"epoch": e, "net": net.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "best_dice": best_dice,
                    "best_epoch": best_epoch, "log": log, "wandb_id": wandb_id,
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None},
                   dest / "checkpoint.pt.tmp")
        (dest / "checkpoint.pt.tmp").replace(ckpt_path)

    stats = {"params": n_params, "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
             "commit": git_commit(), "best_epoch": best_epoch, "best_val_dice_fg": best_dice,
             "train_s_total": sum(x["train_s"] for x in log),
             "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None}
    with open(dest / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    # Final-epoch weights (same file as main.py writes): chosen without looking at
    # the validation scores, so a cross-validation fold's val metrics stay unbiased.
    torch.save(net.state_dict(), dest / "lastweights.pt")
    finish_wandb_run(wandb_run)
    print(f">>> Done. Best 3D val Dice {best_dice:.4f} at epoch {best_epoch}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dest', type=Path, required=True)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true', help="2 train / 1 val volume")
    parser.add_argument('--resume', action='store_true',
                        help="Continue from <dest>/checkpoint.pt if it exists (same W&B run)")
    parser.add_argument('--override', nargs='*', default=[], metavar="KEY=VALUE",
                        help="Override config entries (YAML values), e.g. epochs=2 iters_per_epoch=10")
    args = parser.parse_args()

    with open(args.config) as f:
        yaml_config = yaml.safe_load(f)
    for kv in args.override:
        key, value = kv.split("=", 1)
        yaml_config[key] = yaml.safe_load(value)
    config = TrainConfig3D(**yaml_config)
    set_seed(config.seed)
    pprint(asdict(config))

    args.dest.mkdir(parents=True, exist_ok=True)
    with open(args.dest / 'config_dump.yaml', 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)

    run(args, config)


if __name__ == '__main__':
    main()
