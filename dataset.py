#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

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

from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Union

import math

import torch
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as TF

from utils import class2one_hot


@dataclass
class AugParams:
    # Rotation/scale ranges are sampled uniformly per slice. No horizontal
    # flip: the thorax is not left-right symmetric.
    rotation_deg: float = 15.0
    scale_min: float = 0.9
    scale_max: float = 1.1
    intensity_shift: float = 0.1   # additive, in [0, 1] image units
    elastic_alpha: float = 10.0    # displacement magnitude in pixels (light)
    elastic_sigma: float = 4.0     # smoothing of the displacement field


def _sample_elastic_displacement(shape: tuple[int, int], alpha: float, sigma: float) -> Tensor:
    # One shared displacement field for image and GT of the same slice.
    h, w = shape
    field = torch.rand(1, 2, h, w) * 2 - 1
    kernel = 2 * math.ceil(3 * sigma) + 1
    field = TF.gaussian_blur(field, kernel_size=kernel, sigma=sigma)
    field = field * alpha
    return field.permute(0, 2, 3, 1)  # (1, H, W, 2) as expected by elastic_transform


def augment_sample(img: Tensor, gt: Tensor, params: AugParams, K: int) -> tuple[Tensor, Tensor]:
    # img: (C, H, W) float in [0, 1]; gt: (K, H, W) one-hot.
    # Geometric transforms are identical for every input channel and the GT.
    angle = (torch.rand(()).item() * 2 - 1) * params.rotation_deg
    scale = params.scale_min + torch.rand(()).item() * (params.scale_max - params.scale_min)

    gt_idx = gt.argmax(dim=0, keepdim=True).to(torch.float32)  # (1, H, W) class indices

    img = TF.affine(img, angle=angle, translate=[0, 0], scale=scale, shear=[0, 0],
                    interpolation=InterpolationMode.BILINEAR)
    gt_idx = TF.affine(gt_idx, angle=angle, translate=[0, 0], scale=scale, shear=[0, 0],
                       interpolation=InterpolationMode.NEAREST)

    if params.elastic_alpha > 0:
        displacement = _sample_elastic_displacement(tuple(img.shape[-2:]),
                                                    params.elastic_alpha, params.elastic_sigma)
        img = TF.elastic_transform(img, displacement, interpolation=InterpolationMode.BILINEAR)
        gt_idx = TF.elastic_transform(gt_idx, displacement, interpolation=InterpolationMode.NEAREST)

    if params.intensity_shift > 0:
        shift = (torch.rand(()).item() * 2 - 1) * params.intensity_shift
        img = (img + shift).clamp(0, 1)

    gt_aug = class2one_hot(gt_idx.round().to(torch.int64), K=K)[0]
    return img, gt_aug


def slice_id(path: Path) -> tuple[str, int]:
    """Return the patient id and z-index encoded in ``Patient_XX_####.png``."""
    try:
        patient_id, z = path.stem.rsplit('_', maxsplit=1)
        return patient_id, int(z)
    except ValueError as exc:
        raise ValueError(f"Expected a '<patient>_<z>.png' slice name, got {path.name}") from exc


def make_slice_windows(images: list[Path], in_slices: int) -> dict[Path, tuple[Path, ...]]:
    """Map every centre image to its one- or three-slice input window.

    Windows are built independently per patient, never from the globally sorted
    image list. At volume boundaries the centre slice is replicated.
    """
    assert in_slices in [1, 3], f"Only 1 and 3 input slices are supported, got {in_slices}"

    by_patient: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for image in images:
        patient_id, z = slice_id(image)
        by_patient[patient_id].append((z, image))

    windows: dict[Path, tuple[Path, ...]] = {}
    for patient_id, numbered_images in by_patient.items():
        numbered_images.sort(key=lambda item: item[0])
        zs = [z for z, _ in numbered_images]
        assert len(zs) == len(set(zs)), f"Duplicate z-index for {patient_id}: {zs}"
        paths = [path for _, path in numbered_images]

        for i, path in enumerate(paths):
            if in_slices == 1:
                windows[path] = (path,)
            else:
                windows[path] = (paths[max(i - 1, 0)], path, paths[min(i + 1, len(paths) - 1)])

    assert len(windows) == len(images)
    return windows


def make_dataset(root, subset) -> list[tuple[Path, Path | None]]:
    assert subset in ['train', 'val', 'test']

    root = Path(root)
    print(f"> {root=}")

    img_path = root / subset / 'img'
    full_path = root / subset / 'gt'

    images: list[Path] = sorted(img_path.glob("*.png"))
    full_labels: list[Path | None]
    if subset != 'test':
        labels_by_stem = {label.stem: label for label in full_path.glob("*.png")}
        assert len(labels_by_stem) == len(images), (len(labels_by_stem), len(images))
        assert set(labels_by_stem) == {image.stem for image in images}
        full_labels = [labels_by_stem[image.stem] for image in images]
    else:
        full_labels = [None] * len(images)

    return list(zip(images, full_labels))


class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, augment=False, equalize=False, debug=False,
                 in_slices: int = 1, aug_params: AugParams | None = None):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment
        self.aug_params: AugParams = aug_params if aug_params is not None else AugParams()
        self.equalize: bool = equalize
        self.in_slices: int = in_slices

        self.test_mode: bool = subset == 'test'

        self.files = make_dataset(root_dir, subset)
        self.windows = make_slice_windows([image for image, _ in self.files], in_slices)
        if debug:
            self.files = self.files[:10]

        print(f">> Created {subset} dataset with {len(self)} images ({in_slices} input slice(s))...")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]

        # Concatenation puts z-1, z, z+1 into channels 0, 1, 2. For future
        # geometric augmentation, sample one transform and apply it to every
        # image here and to the centre-slice GT below.
        img_parts = [self.img_transform(Image.open(path)) for path in self.windows[img_path]]
        assert all(part.shape == img_parts[0].shape for part in img_parts)
        img: Tensor = torch.cat(img_parts, dim=0)
        assert img.shape[0] == self.in_slices

        data_dict = {"images": img,
                     "stems": img_path.stem}

        if not self.test_mode:
            gt: Tensor = self.gt_transform(Image.open(gt_path))

            _, W, H = img.shape
            K, _, _ = gt.shape
            assert gt.shape == (K, W, H)

            if self.augmentation:
                img, gt = augment_sample(img, gt, self.aug_params, K)
                data_dict["images"] = img

            data_dict["gts"] = gt

        return data_dict
