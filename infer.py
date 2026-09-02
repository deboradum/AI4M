#!/usr/bin/env python3

# MIT License
#
# Inference: run a trained network over a folder of 2D slices, save the
# predicted .png (same encoding as main.py), and optionally stitch them back
# into 3D NIfTI volumes with stitch.py. This is the path used for the test-set
# submission and for measuring inference time per patient.

import time
import argparse
from pathlib import Path
from pprint import pprint

import yaml
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from main import img_transform
from configType import TrainConfig, NETWORKS
from stitch import group_by_patient, stitch_patient
from utils import probs2class, save_images, tqdm_


class ImageFolder(Dataset):
    # Images only, no ground truth: the test set has none.
    def __init__(self, folder: Path, img_transform):
        self.files: list[Path] = sorted(folder.glob("*.png"))
        assert len(self.files) > 0, f"No .png found in {folder}"
        self.img_transform = img_transform
        print(f">> Created inference dataset with {len(self)} images from {folder}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        path: Path = self.files[index]
        return {"images": self.img_transform(Image.open(path)),
                "stems": path.stem}


def load_config(path: Path) -> TrainConfig:
    with open(path, 'r') as f:
        return TrainConfig(**yaml.safe_load(f))


def build_net(config: TrainConfig, weights: Path, device: torch.device) -> torch.nn.Module:
    net = NETWORKS[config.net_name](1, config.K, kernels=config.kernels, factor=config.factor)
    state_dict = torch.load(weights, map_location=device, weights_only=True)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()
    return net


@torch.no_grad()
def predict(net: torch.nn.Module, loader: DataLoader, device: torch.device,
            K: int, temperature: float, dest: Path) -> int:
    mult: float = 63 if K == 5 else (255 / (K - 1))  # Same encoding as main.py

    n: int = 0
    for data in tqdm_(loader, desc=">> Inference"):
        img: Tensor = data['images'].to(device)
        assert 0 <= img.min() and img.max() <= 1

        pred_logits: Tensor = net(img)
        pred_probs: Tensor = F.softmax(pred_logits / temperature, dim=1)
        predicted_class: Tensor = probs2class(pred_probs)

        save_images(predicted_class * mult, data['stems'], dest)
        n += len(data['stems'])

    return n


def main(args: argparse.Namespace) -> None:
    config: TrainConfig = load_config(args.config)
    K: int = config.K

    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run inference")

    net = build_net(config, args.weights, device)

    dataset = ImageFolder(args.img_folder, img_transform)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size or config.B,
                        num_workers=config.num_workers,
                        shuffle=False)

    png_dest: Path = args.dest / "png"
    png_dest.mkdir(parents=True, exist_ok=True)

    if gpu:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_slices: int = predict(net, loader, device, K, config.temperature, png_dest)
    if gpu:
        torch.cuda.synchronize()
    t_forward = time.perf_counter() - t0
    print(f">> Predicted {n_slices} slices in {t_forward:.1f} s ({1000 * t_forward / n_slices:.1f} ms/slice)")

    if args.scan_pattern is None:
        print(">> No --scan_pattern given, skipping the stitching")
        return

    groups: dict[str, list[Path]] = group_by_patient(sorted(png_dest.glob("*.png")), args.grp_regex)
    nii_dest: Path = args.dest / "nii"
    nii_dest.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    for id_ in tqdm_(sorted(groups), desc=">> Stitching"):
        stitch_patient(id_, groups[id_], nii_dest, K, args.scan_pattern)
    t_stitch = time.perf_counter() - t0

    n_patients: int = len(groups)
    print(f">> Stitched {n_patients} patients to {nii_dest} in {t_stitch:.1f} s")
    print(f">> Inference time per patient (forward + stitch): "
          f"{(t_forward + t_stitch) / n_patients:.2f} s "
          f"(forward {t_forward / n_patients:.2f} s, stitch {t_stitch / n_patients:.2f} s)")

    with open(args.dest / "timing.txt", 'w') as f:
        f.write(f"slices {n_slices}\npatients {n_patients}\n"
                f"forward_s {t_forward:.3f}\nstitch_s {t_stitch:.3f}\n"
                f"per_patient_s {(t_forward + t_stitch) / n_patients:.3f}\n"
                f"device {device}\n")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run a trained network on 2D slices and rebuild the 3D volumes')
    parser.add_argument('--config', type=Path, required=True,
                        help="YAML config of the run (e.g. results/X/config_dump.yaml), to rebuild the network")
    parser.add_argument('--weights', type=Path, required=True,
                        help="State dict saved by main.py, e.g. results/X/bestweights.pt")
    parser.add_argument('--img_folder', type=Path, required=True,
                        help="Folder of .png slices to predict, e.g. data/SEGTHOR/val/img or data/SEGTHOR/test/img")
    parser.add_argument('--dest', type=Path, required=True,
                        help="Output folder: <dest>/png for the slices, <dest>/nii for the volumes")
    parser.add_argument('--scan_pattern', type=str, default=None,
                        help="CT pattern with {id_} for stitching, e.g. 'data/segthor_part1/train/{id_}/{id_}.nii.gz'. "
                             "Omit to only save the .png predictions.")
    parser.add_argument('--grp_regex', type=str, default=r"(Patient_\d+)_\d+")
    parser.add_argument('--batch_size', type=int, default=None, help="Defaults to the config's B")
    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()
    pprint(vars(args))

    return args


if __name__ == "__main__":
    main(get_args())
