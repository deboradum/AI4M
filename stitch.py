#!/usr/bin/env python3

# MIT License
#
# Group re-implementation of the stitching step (assignment 03), replacing the
# original script by Hoel Kervadec. Reconstructs 3D NIfTI volumes from the 2D
# prediction slices saved as .png by main.py / infer.py.
#
# Differences from the original:
# * absent classes are allowed (part-1 data has no aorta label);
# * geometry (shape, affine, header) is taken from the CT scan, which carries
#   the true voxel spacing, and not from the GT file whose affine is identity;
# * the label encoding of the .png is derived from the number of classes, the
#   same way main.py encodes it, instead of a hardcoded division by 63;
# * optional multiprocessing, one patient per process.

import re
import argparse
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from typing import Match, Pattern

import numpy as np
import nibabel as nib
from skimage.io import imread
from skimage.transform import resize

from utils import map_, tqdm_


def label_scale(K: int) -> float:
    # Must mirror the encoding used in main.py when saving predictions:
    #   mult = 63 if K == 5 else 255 / (K - 1)
    return 63 if K == 5 else 255 / (K - 1)


def get_z(image: Path) -> int:
    return int(image.stem.split('_')[-1])


def decode_slice(png: Path, K: int, scale: float, target_shape: tuple[int, int]) -> np.ndarray:
    raw: np.ndarray = imread(png)
    assert raw.ndim == 2, f"{png}: expected a grayscale image, got shape {raw.shape}"
    assert raw.dtype == np.uint8, f"{png}: expected uint8, got {raw.dtype}"

    labels: np.ndarray = np.rint(raw / scale).astype(np.uint8)
    assert set(np.unique(labels)) <= set(range(K)), (png, np.unique(raw))

    if labels.shape != target_shape:
        # Nearest neighbour only: interpolating labels would create new classes
        labels = resize(labels, target_shape,
                        order=0,
                        mode="constant",
                        preserve_range=True,
                        anti_aliasing=False).astype(np.uint8)

    assert labels.shape == target_shape
    return labels


def stitch_patient(id_: str, images: list[Path], dest_folder: Path,
                   K: int, source_pattern: str) -> Path:
    ct_nib = nib.load(source_pattern.format(id_=id_))
    X, Y, Z = ct_nib.shape
    assert len(ct_nib.shape) == 3, ct_nib.shape

    zs: list[int] = map_(get_z, images)
    assert len(images) == Z, f"{id_}: {len(images)} slices found but the scan has {Z}"
    assert sorted(zs) == list(range(Z)), f"{id_}: slice indices are not 0..{Z - 1}"

    scale: float = label_scale(K)
    volume: np.ndarray = np.zeros((X, Y, Z), dtype=np.uint8)
    for png, z in zip(images, zs):
        volume[:, :, z] = decode_slice(png, K, scale, (X, Y))

    assert volume.shape == ct_nib.shape, (volume.shape, ct_nib.shape)
    assert set(np.unique(volume)) <= set(range(K)), np.unique(volume)
    # Deliberately no assertion that every class is present:
    # a class can legitimately be absent from a patient, or from the labels

    out_nib = nib.nifti1.Nifti1Image(volume, affine=ct_nib.affine, header=ct_nib.header)
    out_nib.set_data_dtype(np.uint8)
    out_path: Path = (dest_folder / id_).with_suffix(".nii.gz")
    nib.save(out_nib, str(out_path))

    return out_path


def group_by_patient(images: list[Path], grp_regex: str) -> dict[str, list[Path]]:
    pattern: Pattern = re.compile(grp_regex)

    groups: dict[str, list[Path]] = {}
    for image in images:
        match: Match | None = pattern.match(image.stem)
        assert match is not None, f"{image.stem} does not match {grp_regex}"
        groups.setdefault(match.group(1), []).append(image)

    for id_, files in groups.items():
        files.sort(key=get_z)

    assert sum(len(v) for v in groups.values()) == len(images)
    return groups


def main(args: argparse.Namespace) -> None:
    images: list[Path] = sorted(Path(args.data_folder).glob("*.png"))
    assert len(images) > 0, f"No .png found in {args.data_folder}"

    groups: dict[str, list[Path]] = group_by_patient(images, args.grp_regex)
    ids: list[str] = sorted(groups.keys())
    print(f"Found {len(ids)} patients out of {len(images)} images ; regex: {args.grp_regex}")
    print(ids)

    args.dest_folder.mkdir(parents=True, exist_ok=True)

    pfun = partial(stitch_patient,
                   dest_folder=args.dest_folder,
                   K=args.num_classes,
                   source_pattern=args.source_scan_pattern)
    jobs: list[tuple[str, list[Path]]] = [(id_, groups[id_]) for id_ in ids]

    outputs: list[Path]
    match args.process:
        case 1:
            outputs = [pfun(id_, files) for id_, files in tqdm_(jobs)]
        case -1:
            outputs = Pool().starmap(pfun, jobs)
        case _ as p:
            outputs = Pool(p).starmap(pfun, jobs)

    print(f"Saved {len(outputs)} volumes to {args.dest_folder}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Stitch 2D .png predictions back into 3D NIfTI volumes')
    parser.add_argument('--data_folder', type=Path, required=True,
                        help="Folder containing the predicted .png slices, e.g. results/X/best_epoch/val")
    parser.add_argument('--dest_folder', type=Path, required=True,
                        help="Where to write the Patient_XX.nii.gz volumes")
    parser.add_argument('--source_scan_pattern', type=str, required=True,
                        help="Pattern of the original CT scan, with {id_} as placeholder, "
                             "e.g. 'data/segthor_part1/train/{id_}/{id_}.nii.gz'. "
                             "Use the CT and not the GT: only the CT carries the correct spacing/affine.")
    parser.add_argument('--grp_regex', type=str, default=r"(Patient_\d+)_\d+",
                        help="Regex on the .png stem whose first group is the patient id")
    parser.add_argument('--num_classes', '-K', type=int, default=5,
                        help="Number of classes K (including background) used to encode the .png")
    parser.add_argument('--process', '-p', type=int, default=1,
                        help="Number of processes (1: sequential, -1: all cores)")

    args = parser.parse_args()
    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
