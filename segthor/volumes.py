#!/usr/bin/env python3

# MIT License
#
# 3D side of the pipeline. The volumes are rebuilt by stacking the 2D slices
# written by slice_segthor.py (e.g. data/SEGTHOR_FULL_huwide), so the 3D U-Net
# sees exactly the same preprocessing as the 2D/2.5D runs (HU window, 256x256
# in-plane resize, stratified 32/8 split) and its predictions can be written
# back as slices and go through the same stitch.py / metrics3d.py evaluation.
#
# Layout: a volume is (H, W, Z) like the slices stacked along the last axis;
# network inputs are (B, C, H, W, Z).

from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool

import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from segthor.dataset import slice_id, AugParams


def _read_png(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert('L'))


def group_slices(folder: Path) -> dict[str, list[Path]]:
    by_patient: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in sorted(folder.glob("*.png")):
        patient_id, z = slice_id(path)
        by_patient[patient_id].append((z, path))

    groups: dict[str, list[Path]] = {}
    for patient_id, numbered in by_patient.items():
        numbered.sort(key=lambda item: item[0])
        zs = [z for z, _ in numbered]
        assert zs == list(range(len(zs))), f"{patient_id}: slice indices are not 0..{len(zs) - 1}"
        groups[patient_id] = [path for _, path in numbered]
    return groups


def load_volume(paths: list[Path]) -> np.ndarray:
    return np.stack([_read_png(p) for p in paths], axis=-1)  # (H, W, Z) uint8


def load_split(root: Path, subset: str, with_gt: bool = True,
               processes: int = 16) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    """{patient: (image uint8 (H, W, Z) in 0..255, labels uint8 (H, W, Z) in 0..K-1)}."""
    img_groups = group_slices(root / subset / "img")
    ids = sorted(img_groups)
    with Pool(processes) as pool:
        imgs = pool.map(load_volume, [img_groups[i] for i in ids])
        gts: list[np.ndarray | None] = [None] * len(ids)
        if with_gt:
            gt_groups = group_slices(root / subset / "gt")
            assert set(gt_groups) == set(ids)
            gts = pool.map(load_volume, [gt_groups[i] for i in ids])

    volumes = {}
    for id_, img, gt in zip(ids, imgs, gts):
        if gt is not None:
            assert gt.shape == img.shape, (id_, gt.shape, img.shape)
            assert set(np.unique(gt)) <= {0, 63, 126, 189, 252}, (id_, np.unique(gt))
            gt = (gt // 63).astype(np.uint8)  # Same encoding as slice_segthor.py
        volumes[id_] = (img, gt)
    print(f">> Loaded {len(volumes)} {subset} volumes from {root} "
          f"(z from {min(v[0].shape[2] for v in volumes.values())} "
          f"to {max(v[0].shape[2] for v in volumes.values())})")
    return volumes


class PatchSampler:
    """Random (H, W, Z) patches from volumes kept on the device.

    A fraction ``fg_ratio`` of the patches is forced to contain a randomly
    chosen foreground voxel, otherwise the position is uniform.
    """
    def __init__(self, volumes: dict[str, tuple[np.ndarray, np.ndarray]], patch_size: tuple[int, int, int],
                 device: torch.device, fg_ratio: float = 0.33):
        self.ids = sorted(volumes)
        self.patch_size = patch_size
        self.fg_ratio = fg_ratio
        self.imgs = [torch.from_numpy(volumes[i][0]).to(device) for i in self.ids]
        self.gts = [torch.from_numpy(volumes[i][1]).to(device) for i in self.ids]
        # Foreground coordinates per patient, subsampled to keep it small
        self.fg = []
        for gt in self.gts:
            coords = torch.nonzero(gt > 0)
            if len(coords) > 200_000:
                coords = coords[torch.randperm(len(coords), device=coords.device)[:200_000]]
            self.fg.append(coords)
        for img in self.imgs:
            assert all(s >= p for s, p in zip(img.shape, patch_size)), (img.shape, patch_size)

    def _start(self, shape, center: Tensor | None) -> list[int]:
        starts = []
        for d, (s, p) in enumerate(zip(shape, self.patch_size)):
            if center is None:
                starts.append(int(torch.randint(0, s - p + 1, ()).item()))
            else:
                c = int(center[d].item()) - p // 2 + int(torch.randint(-p // 4, p // 4 + 1, ()).item())
                starts.append(min(max(c, 0), s - p))
        return starts

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor]:
        imgs, gts = [], []
        for _ in range(batch_size):
            i = int(torch.randint(0, len(self.ids), ()).item())
            center = None
            if torch.rand(()).item() < self.fg_ratio and len(self.fg[i]) > 0:
                center = self.fg[i][torch.randint(0, len(self.fg[i]), ())]
            (h, w, z) = self._start(self.imgs[i].shape, center)
            ph, pw, pz = self.patch_size
            imgs.append(self.imgs[i][h:h + ph, w:w + pw, z:z + pz])
            gts.append(self.gts[i][h:h + ph, w:w + pw, z:z + pz])
        img = torch.stack(imgs)[:, None].float() / 255  # (B, 1, H, W, Z)
        gt = torch.stack(gts).long()                    # (B, H, W, Z)
        return img, gt


def augment_batch(img: Tensor, gt: Tensor, params: AugParams) -> tuple[Tensor, Tensor]:
    """In-plane rotation/scale (one transform per volume, shared by every slice
    and the labels) and an additive intensity shift, as in the 2D augmentation.
    No flips: the thorax is not left-right symmetric. Runs on the device."""
    B, C, H, W, Z = img.shape
    angle = (torch.rand(B, device=img.device) * 2 - 1) * math.radians(params.rotation_deg)
    scale = params.scale_min + torch.rand(B, device=img.device) * (params.scale_max - params.scale_min)
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.zeros(B, 2, 3, device=img.device)
    theta[:, 0, 0], theta[:, 0, 1] = cos, -sin
    theta[:, 1, 0], theta[:, 1, 1] = sin, cos
    # Every z slice of volume b uses theta[b]: fold z into the batch for 2D grid_sample
    theta = theta.repeat_interleave(Z, dim=0)
    grid = F.affine_grid(theta, (B * Z, C, H, W), align_corners=False)

    img2d = img.permute(0, 4, 1, 2, 3).reshape(B * Z, C, H, W)
    gt2d = gt.permute(0, 3, 1, 2).reshape(B * Z, 1, H, W).float()
    img2d = F.grid_sample(img2d, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    gt2d = F.grid_sample(gt2d, grid, mode='nearest', padding_mode='zeros', align_corners=False)
    img = img2d.reshape(B, Z, C, H, W).permute(0, 2, 3, 4, 1)
    gt = gt2d.reshape(B, Z, H, W).permute(0, 2, 3, 1).round().long()

    if params.intensity_shift > 0:
        shift = (torch.rand(B, 1, 1, 1, 1, device=img.device) * 2 - 1) * params.intensity_shift
        img = (img + shift).clamp(0, 1)
    return img.contiguous(), gt.contiguous()


def _starts(size: int, patch: int, step: int) -> list[int]:
    if size <= patch:
        return [0]
    starts = list(range(0, size - patch + 1, step))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def _gaussian_weight(patch_size: tuple[int, int, int], device: torch.device) -> Tensor:
    # Centre voxels of a window are more reliable than its borders (nnU-Net)
    axes = [torch.exp(-0.5 * ((torch.arange(p, device=device) - (p - 1) / 2) / (p / 8)) ** 2)
            for p in patch_size]
    w = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return (w / w.max()).clamp_min(1e-3)


@torch.no_grad()
def sliding_window_probs(net: torch.nn.Module, img: np.ndarray, patch_size: tuple[int, int, int],
                         K: int, device: torch.device, overlap: float = 0.5,
                         amp: bool = True, temperature: float = 1.0) -> Tensor:
    """Softmax probabilities (K, H, W, Z) for a whole uint8 volume (H, W, Z)."""
    vol = torch.from_numpy(img).to(device).float()[None, None] / 255
    shape = vol.shape[-3:]
    # Pad up to the patch size where the volume is smaller
    pads = [max(p - s, 0) for s, p in zip(shape, patch_size)]
    if any(pads):
        vol = F.pad(vol, (0, pads[2], 0, pads[1], 0, pads[0]))
    H, W, Z = vol.shape[-3:]

    weight = _gaussian_weight(patch_size, device)
    acc = torch.zeros((K, H, W, Z), device=device)
    norm = torch.zeros((H, W, Z), device=device)
    steps = [max(int(p * (1 - overlap)), 1) for p in patch_size]
    ph, pw, pz = patch_size
    for h in _starts(H, ph, steps[0]):
        for w in _starts(W, pw, steps[1]):
            for z in _starts(Z, pz, steps[2]):
                patch = vol[..., h:h + ph, w:w + pw, z:z + pz]
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                    logits = net(patch)
                probs = F.softmax(logits.float() / temperature, dim=1)[0]
                acc[:, h:h + ph, w:w + pw, z:z + pz] += probs * weight
                norm[h:h + ph, w:w + pw, z:z + pz] += weight
    probs = acc / norm
    return probs[:, :shape[0], :shape[1], :shape[2]]


def dice_3d(pred: Tensor, gt: Tensor, K: int) -> Tensor:
    """Per-class 3D Dice (K,), NaN where the class is absent from both."""
    res = torch.full((K,), float('nan'))
    for k in range(K):
        p, g = pred == k, gt == k
        denom = p.sum() + g.sum()
        if denom > 0:
            res[k] = (2 * (p & g).sum() / denom).item()
    return res
