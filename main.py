#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec, Caroline Magg

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import yaml
import random
import argparse
import warnings
from typing import Any
from pathlib import Path
from pprint import pprint
from operator import itemgetter
from shutil import copytree, rmtree

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader

from functools import partial

from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   save_images)

from losses import (CrossEntropy)

from configType import TrainConfig, NETWORKS
from runstats import RunStats
from metrics import iou_coef, precision_coef, recall_coef

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.mps.is_available():
        torch.mps.manual_seed(seed)

    print(f"Set seed {seed}")

def img_transform(img):
        img = img.convert('L')
        img = np.array(img)[np.newaxis, ...]
        img = img / 255  # max <= 1
        img = torch.tensor(img, dtype=torch.float32)
        return img

def gt_transform(K, img):
        img = np.array(img)[...]
        # The idea is that the classes are mapped to {0, 255} for binary cases
        # {0, 85, 170, 255} for 4 classes
        # {0, 51, 102, 153, 204, 255} for 6 classes
        # Very sketchy but that works here and that simplifies visualization
        img = img / (255 / (K - 1)) if K != 5 else img / 63  # max <= 1
        img = torch.tensor(img, dtype=torch.int64)[None, ...]  # Add one dimension to simulate batch
        img = class2one_hot(img, K=K)
        return img[0]

def setup(args, config: TrainConfig) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
    # Networks and scheduler
    gpu: bool = args.gpu and torch.cuda.is_available()
    mps: bool = args.mps and torch.mps.is_available()
    device = torch.device("cuda") if gpu else torch.device("mps") if mps else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    K: int = config.K
    net_class = NETWORKS[config.net_name]
    net = net_class(1, K, kernels=config.kernels, factor=config.factor)
    net.init_weights()
    net.to(device)

    optim_class = getattr(torch.optim, config.optimizer)
    optimizer = optim_class(net.parameters(), lr=config.lr, betas=tuple(config.betas))

    # Dataset part
    B: int = config.B
    root_dir = Path("data") / config.dataset

    train_set = SliceDataset('train',
                             root_dir,
                             img_transform=img_transform,
                             gt_transform= partial(gt_transform, K),
                             debug=args.debug)
    train_loader = DataLoader(train_set,
                              batch_size=B,
                              num_workers=config.num_workers,
                              shuffle=True)

    val_set = SliceDataset('val',
                           root_dir,
                           img_transform=img_transform,
                           gt_transform=partial(gt_transform, K),
                           debug=args.debug)
    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=config.num_workers,
                            shuffle=False)

    args.dest.mkdir(parents=True, exist_ok=True)

    return (net, optimizer, device, train_loader, val_loader, K)


def runTraining(args, config: TrainConfig):
    print(f">>> Setting up to train on {config.dataset} with {config.mode}")
    net, optimizer, device, train_loader, val_loader, K = setup(args, config)

    if config.mode == "full":
        loss_fn = CrossEntropy(idk=list(range(K)))
    elif config.mode in ["partial"] and config.dataset == 'SEGTHOR':
        loss_fn = CrossEntropy(idk=[0, 1, 3, 4])
    else:
        raise ValueError(config.mode, config.dataset)

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((config.epochs, len(train_loader)))
    log_loss_val: Tensor = torch.zeros((config.epochs, len(val_loader)))

    # Dice: Measures overall volume overlap (2 * Intersection / (Area 1 + Area 2))
    log_dice_tra: Tensor = torch.zeros((config.epochs, len(train_loader.dataset), K))
    log_dice_val: Tensor = torch.zeros((config.epochs, len(val_loader.dataset), K))

    # IoU: Measures strict overlap (Intersection / Union)
    log_iou_tra: Tensor = torch.zeros((config.epochs, len(train_loader.dataset), K))
    log_iou_val: Tensor = torch.zeros((config.epochs, len(val_loader.dataset), K))

    # Precision: Measures over-segmentation (Out of all predicted organ pixels, how many were actually the organ?)
    log_prec_tra: Tensor = torch.zeros((config.epochs, len(train_loader.dataset), K))
    log_prec_val: Tensor = torch.zeros((config.epochs, len(val_loader.dataset), K))

    # Recall: Measures under-segmentation (Out of all actual organ pixels, how many did the model find?)
    log_rec_tra: Tensor = torch.zeros((config.epochs, len(train_loader.dataset), K))
    log_rec_val: Tensor = torch.zeros((config.epochs, len(val_loader.dataset), K))

    # Whether each class is present in the GT of each slice. The per-slice metrics
    # above score 1.0 when a class is absent from both GT and prediction, which
    # inflates them a lot (an organ is absent from most slices). Used to also
    # report the metrics restricted to slices that actually contain the organ.
    log_present_tra: Tensor = torch.zeros((config.epochs, len(train_loader.dataset), K), dtype=torch.bool)
    log_present_val: Tensor = torch.zeros((config.epochs, len(val_loader.dataset), K), dtype=torch.bool)

    best_dice: float = 0
    best_epoch: int = -1
    epochs_without_improvement: int = 0

    stats = RunStats(net, device, args.dest,
                     n_train=len(train_loader.dataset), n_val=len(val_loader.dataset), batch_size=config.B)

    for e in range(config.epochs):
        for m in ['train', 'val']:
            match m:
                case 'train':
                    net.train()
                    opt = optimizer
                    cm = Dcm
                    desc = f">> Training   ({e: 4d})"
                    loader = train_loader
                    log_loss = log_loss_tra
                    log_dice = log_dice_tra
                    log_iou = log_iou_tra
                    log_prec = log_prec_tra
                    log_rec = log_rec_tra
                    log_present = log_present_tra
                case 'val':
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val
                    log_iou = log_iou_val
                    log_prec = log_prec_val
                    log_rec = log_rec_val
                    log_present = log_present_val

            stats.phase_start()
            with cm():  # Either dummy context manager, or the torch.no_grad for validation
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data['images'].to(device)
                    gt = data['gts'].to(device)

                    if opt:  # So only for training
                        opt.zero_grad()

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    B, _, W, H = img.shape

                    pred_logits = net(img)
                    pred_probs = F.softmax(pred_logits / config.temperature, dim=1)

                    # Metrics computation, not used for training
                    pred_seg = probs2one_hot(pred_probs)
                    log_dice[e, j:j + B, :] = dice_coef(pred_seg, gt)  # One DSC value per sample and per class
                    log_iou[e, j:j + B, :] = iou_coef(pred_seg, gt)
                    log_prec[e, j:j + B, :] = precision_coef(pred_seg, gt)
                    log_rec[e, j:j + B, :] = recall_coef(pred_seg, gt)
                    log_present[e, j:j + B, :] = gt.sum(dim=(2, 3)) > 0

                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = loss.item()  # One loss value per batch (averaged in the loss)

                    if opt:  # Only for training
                        loss.backward()
                        opt.step()

                    if m == 'val':
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning)
                            predicted_class: Tensor = probs2class(pred_probs)
                            mult: int = 63 if K == 5 else (255 / (K - 1))
                            save_images(predicted_class * mult,
                                        data['stems'],
                                        args.dest / f"iter{e:03d}" / m)

                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {"Loss": f"{log_loss[e, :i + 1].mean():5.2e}",
                                                    "Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                                                    "IoU": f"{log_iou[e, :j, 1:].mean():05.3f}",
                                                    "Prec": f"{log_prec[e, :j, 1:].mean():05.3f}",
                                                    "Rec": f"{log_rec[e, :j, 1:].mean():05.3f}"}

                    # Printing all this just overflows in the terminal
                    # if K > 2:
                    #     for k in range(1, K):
                    #         postfix_dict[f"D-{k}"] = f"{log_dice[e, :j, k].mean():05.3f}"
                    #         postfix_dict[f"I-{k}"] = f"{log_iou[e, :j, k].mean():05.3f}"
                    #         postfix_dict[f"P-{k}"] = f"{log_prec[e, :j, k].mean():05.3f}"
                    #         postfix_dict[f"R-{k}"] = f"{log_rec[e, :j, k].mean():05.3f}"
                    tq_iter.set_postfix(postfix_dict)
            stats.phase_end(m, e)

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)

        np.save(args.dest / "iou_tra.npy", log_iou_tra)
        np.save(args.dest / "iou_val.npy", log_iou_val)
        np.save(args.dest / "prec_tra.npy", log_prec_tra)
        np.save(args.dest / "prec_val.npy", log_prec_val)
        np.save(args.dest / "rec_tra.npy", log_rec_tra)
        np.save(args.dest / "rec_val.npy", log_rec_val)
        np.save(args.dest / "present_tra.npy", log_present_tra)
        np.save(args.dest / "present_val.npy", log_present_val)

        current_dice: float = log_dice_val[e, :, 1:].mean().item()
        current_iou: float = log_iou_val[e, :, 1:].mean().item()
        current_prec: float = log_prec_val[e, :, 1:].mean().item()
        current_rec: float = log_rec_val[e, :, 1:].mean().item()

        if current_dice > best_dice:
            epochs_without_improvement = 0

            message = f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC | IoU: {current_iou:05.3f} | Prec: {current_prec:05.3f} | Rec: {current_rec:05.3f}\n"
            if K > 2:
                message += "Per-Organ Breakdown (all slices | slices containing the organ):\n"
                for k in range(1, K):
                    present = log_present_val[e, :, k]
                    dice_present = log_dice_val[e, present, k].mean().item() if present.any() else float('nan')
                    message += (f"  - Organ {k}: Dice = {log_dice_val[e, :, k].mean().item():05.3f} | {dice_present:.3f} "
                                f"({int(present.sum())}/{len(present)} slices), "
                                f"IoU = {log_iou_val[e, :, k].mean().item():05.3f}, Prec = {log_prec_val[e, :, k].mean().item():05.3f}, Rec = {log_rec_val[e, :, k].mean().item():05.3f}\n")

            print(message.strip())
            best_dice = current_dice
            best_epoch = e
            with open(args.dest / "best_epoch.txt", 'w') as f:
                f.write(message)

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")
        else:
            epochs_without_improvement += 1

        stats.epoch_end(e, current_dice, best_epoch, best_dice)

        # patience=-1 disables it
        if config.patience != -1 and epochs_without_improvement >= config.patience:
            print(f">>> Early stopping triggered after {e} epochs (no improvement for {config.patience} epochs).")
            break

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=Path, required=True,
                        help="Path to the YAML configuration file.")
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights).")

    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--mps', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logics around epochs and logging easily.")

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        yaml_config = yaml.safe_load(file)
    config = TrainConfig(**yaml_config)

    set_seed(config.seed)

    print("Parsed arguments:")
    pprint(vars(args))

    print("\nLoaded Configuration:")
    pprint(yaml_config)

    args.dest.mkdir(parents=True, exist_ok=True)
    with open(args.dest / 'config_dump.yaml', 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)

    runTraining(args, config)


if __name__ == '__main__':
    main()

# python -O main.py --config configs/TOY2_default_config.yaml --dest results/toy2/ce_with_cfg --mps
