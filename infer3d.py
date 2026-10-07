#!/usr/bin/env python3

# MIT License
#
# Inference with the 3D U-Net: sliding-window prediction over every volume of a
# slice folder, written back as per-slice .png (same names and encoding as
# infer.py), then stitched to NIfTI with stitch.py so metrics3d.py scores the
# 3D model exactly like the 2D/2.5D ones.

import time
import argparse
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from PIL import Image

from stitch import group_by_patient, stitch_patient
from segthor.utils import tqdm_
from segthor.volumes import group_slices, load_volume, sliding_window_probs
from train3d import load_config, build_net


def main(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    K = config.K
    device = torch.device("cuda") if args.gpu and torch.cuda.is_available() else torch.device("cpu")
    print(f">> Picked {device} to run inference")

    net = build_net(config)
    net.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    net.to(device).eval()

    groups = group_slices(args.img_folder)
    png_dest: Path = args.dest / "png"
    png_dest.mkdir(parents=True, exist_ok=True)
    mult = 63 if K == 5 else 255 / (K - 1)  # Same encoding as main.py

    t_forward = 0.0
    n_slices = 0
    for id_ in tqdm_(sorted(groups), desc=">> Inference"):
        img = load_volume(groups[id_])
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        probs = sliding_window_probs(net, img, tuple(config.patch_size), K, device,
                                     overlap=args.overlap, amp=config.amp, temperature=config.temperature)
        pred = probs.argmax(0).to(torch.uint8).cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_forward += time.perf_counter() - t0
        for z, path in enumerate(groups[id_]):
            Image.fromarray((pred[:, :, z] * mult).astype(np.uint8)).save(png_dest / f"{path.stem}.png")
        n_slices += pred.shape[2]
    print(f">> Predicted {len(groups)} volumes ({n_slices} slices) in {t_forward:.1f} s")

    if args.scan_pattern is None:
        print(">> No --scan_pattern given, skipping the stitching")
        return

    png_groups = group_by_patient(sorted(png_dest.glob("*.png")), args.grp_regex)
    nii_dest: Path = args.dest / "nii"
    nii_dest.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for id_ in tqdm_(sorted(png_groups), desc=">> Stitching"):
        stitch_patient(id_, png_groups[id_], nii_dest, K, args.scan_pattern)
    t_stitch = time.perf_counter() - t0

    n = len(png_groups)
    print(f">> Inference time per patient (forward + stitch): {(t_forward + t_stitch) / n:.2f} s")
    with open(args.dest / "timing.txt", 'w') as f:
        f.write(f"slices {n_slices}\npatients {n}\nforward_s {t_forward:.3f}\nstitch_s {t_stitch:.3f}\n"
                f"per_patient_s {(t_forward + t_stitch) / n:.3f}\ndevice {device}\n")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='3D U-Net sliding-window inference and stitching')
    parser.add_argument('--config', type=Path, required=True, help="results/X/config_dump.yaml")
    parser.add_argument('--weights', type=Path, required=True, help="results/X/bestweights.pt")
    parser.add_argument('--img_folder', type=Path, required=True, help="e.g. data/SEGTHOR_FULL_huwide/val/img")
    parser.add_argument('--dest', type=Path, required=True)
    parser.add_argument('--scan_pattern', type=str, default=None,
                        help="e.g. 'data/segthor_train_full/train/{id_}/{id_}.nii.gz'")
    parser.add_argument('--grp_regex', type=str, default=r"(Patient_\d+)_\d+")
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    pprint(vars(args))
    return args


if __name__ == "__main__":
    main(get_args())
