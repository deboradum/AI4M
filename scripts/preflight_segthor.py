#!/usr/bin/env python3
"""GPU preflight for a SegTHOR experiment configuration.

It checks the dataset layout and executes one real batch through the same
model, loss, backward pass, and optimizer used by main.py. This catches CUDA,
Triton, Python-header, data-loader, input-channel, and target-shape failures
before a multi-hour training job is launched.
"""

import argparse
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from configType import TrainConfig, NETWORKS
from dataset import SliceDataset
from losses import CrossEntropy
from main import img_transform, gt_transform


def load_config(path: Path) -> TrainConfig:
    with open(path) as f:
        return TrainConfig(**yaml.safe_load(f))


def main(config_path: Path) -> None:
    config = load_config(config_path)
    assert torch.cuda.is_available(), "CUDA is unavailable in this Slurm allocation"
    device = torch.device("cuda")
    root = Path("data") / config.dataset

    for subset in ["train", "val"]:
        for kind in ["img", "gt"]:
            path = root / subset / kind
            assert path.is_dir() and any(path.glob("*.png")), f"Missing PNG data at {path}"

    dataset = SliceDataset("train", root,
                           img_transform=img_transform,
                           gt_transform=partial(gt_transform, config.K),
                           in_slices=config.in_slices)
    loader = DataLoader(dataset, batch_size=min(config.B, 2), num_workers=0, shuffle=False)
    batch = next(iter(loader))
    images = batch["images"].to(device)
    gts = batch["gts"].to(device)

    net = NETWORKS[config.net_name](config.in_slices, config.K,
                                    kernels=config.kernels, factor=config.factor).to(device)
    net.init_weights()
    optimizer = getattr(torch.optim, config.optimizer)(net.parameters(), lr=config.lr,
                                                         betas=tuple(config.betas))
    loss_fn = CrossEntropy(idk=list(range(config.K)))

    optimizer.zero_grad()
    logits = net(images)
    assert logits.shape == gts.shape, (logits.shape, gts.shape)
    probabilities = F.softmax(logits / config.temperature, dim=1)
    loss = loss_fn(probabilities, gts)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

    assert torch.isfinite(loss), loss
    print(f"Preflight passed: input={tuple(images.shape)}, target={tuple(gts.shape)}, "
          f"loss={loss.item():.6f}, GPU={torch.cuda.get_device_name(0)}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate one real GPU training step before a full run")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    get_args_ = get_args()
    main(get_args_.config)
