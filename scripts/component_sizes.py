#!/usr/bin/env python3

"""Inspect connected-component sizes in 3D segmentation predictions. Used to
determine values for the minimum size component postprocessing technique.

For every NIfTI prediction, this script finds all 3D connected components
for the selected foreground classes and prints their sizes in voxels and mm^3.
The input should be the RAW stitched predictions, before post-processing.
"""

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import generate_binary_structure, label
import scipy.ndimage as ndi

from segthor.utils import tqdm_


def connectivity_structure(connectivity: int) -> np.ndarray:
    """Return a 3D binary structuring element for 6/18/26-connectivity."""
    if connectivity not in {6, 18, 26}:
        raise ValueError(f"Unsupported connectivity: {connectivity}")

    rank = {6: 1, 18: 2, 26: 3}[connectivity]
    return generate_binary_structure(rank=3, connectivity=rank)


def component_sizes(mask, structure):
    labeled, num_components = ndi.label(mask, structure=structure)

    if num_components == 0:
        return []

    counts = np.bincount(labeled.ravel())

    # counts[0] is background, so skip it
    return sorted(
        (int(n) for n in counts[1:] if n > 0),
        reverse=True
    )


def format_sizes(sizes: list[int], max_components: int) -> str:
    """Format component sizes for terminal output."""
    shown = sizes[:max_components]
    text = ", ".join(f"{n:,}" for n in shown)
    if len(sizes) > max_components:
        text += f", ... (+{len(sizes) - max_components} more)"
    return text or "none"


def main(args: argparse.Namespace) -> None:
    files = sorted(args.input_folder.glob("*.nii.gz"))
    assert files, f"No .nii.gz found in {args.input_folder}"

    classes = args.classes or list(range(1, args.num_classes))
    assert all(1 <= k < args.num_classes for k in classes), classes

    structure = connectivity_structure(args.connectivity)

    print(f">> Found {len(files)} NIfTI predictions in {args.input_folder}")
    print(f">> Classes: {classes}")
    print(f">> Connectivity: {args.connectivity}")
    print(f">> Component sizes shown per class: top {args.max_components}")

    # Accumulate all component sizes across patients for each class.
    all_sizes: dict[int, list[int]] = {k: [] for k in classes}
    csv_rows: list[list[object]] = []

    for path in tqdm_(files, desc=">> Inspecting"):
        nii = nib.load(str(path))
        arr = np.asarray(nii.dataobj)
        assert arr.ndim == 3, (path, arr.shape)
        assert np.issubdtype(arr.dtype, np.integer), (path, arr.dtype)

        unique = np.unique(arr)
        assert set(unique) <= set(range(args.num_classes)), (path, unique)

        spacing = tuple(float(z) for z in nii.header.get_zooms()[:3])
        voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
        patient = path.name.removesuffix(".nii.gz")

        print(f"\n{patient}  spacing={spacing[0]:.3f} x {spacing[1]:.3f} x {spacing[2]:.3f} mm")

        for k in classes:
            sizes = component_sizes(arr == k, structure)
            all_sizes[k].extend(sizes)

            volumes = [n * voxel_volume_mm3 for n in sizes]
            print(
                f"  class {k}: {len(sizes)} component(s) | "
                f"voxels: {format_sizes(sizes, args.max_components)}"
            )
            if args.show_mm3:
                formatted_volumes = ", ".join(f"{v:,.1f}" for v in volumes[:args.max_components])
                if len(volumes) > args.max_components:
                    formatted_volumes += f", ... (+{len(volumes) - args.max_components} more)"
                print(f"           mm^3:   {formatted_volumes or 'none'}")

            for index, (size, volume) in enumerate(zip(sizes, volumes), start=1):
                csv_rows.append([patient, k, index, size, f"{volume:.3f}"])

    print("\n" + "=" * 80)
    print("AGGREGATED COMPONENT-SIZE SUMMARY")
    print("=" * 80)

    threshold_values = [50, 100, 250, 500, 1000, 2000, 5000, 10000]

    for k in classes:
        sizes = sorted(all_sizes[k], reverse=True)
        if not sizes:
            print(f"\nclass {k}: no components")
            continue

        arr_sizes = np.asarray(sizes, dtype=np.float64)
        print(f"\nclass {k}: {len(sizes)} total components across {len(files)} patients")
        print(
            f"  largest={int(arr_sizes[0]):,} | "
            f"median={np.median(arr_sizes):,.0f} | "
            f"mean={np.mean(arr_sizes):,.0f} | "
            f"p10={np.percentile(arr_sizes, 10):,.0f} | "
            f"p25={np.percentile(arr_sizes, 25):,.0f} | "
            f"p75={np.percentile(arr_sizes, 75):,.0f} | "
            f"p90={np.percentile(arr_sizes, 90):,.0f}"
        )

        print("  components removed by threshold (< threshold):")
        for threshold in threshold_values:
            removed = int(np.sum(arr_sizes < threshold))
            remaining = len(sizes) - removed
            removed_voxels = int(np.sum(arr_sizes[arr_sizes < threshold]))
            print(
                f"    < {threshold:>5,}: {removed:>4} components removed, "
                f"{remaining:>4} remain, {removed_voxels:>10,} voxels removed"
            )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["patient", "class", "component_rank", "voxels", "volume_mm3"])
            writer.writerows(csv_rows)
        print(f"\nSaved component-size CSV to {args.csv}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect 3D connected-component sizes in stitched segmentation predictions"
    )
    parser.add_argument(
        "--input_folder",
        type=Path,
        required=True,
        help="Folder containing stitched prediction .nii.gz files (preferably RAW predictions)",
    )
    parser.add_argument(
        "--num_classes",
        "-K",
        type=int,
        default=5,
        help="Number of segmentation classes, including background",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="Foreground classes to inspect; defaults to 1..K-1",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=[6, 18, 26],
        default=6,
        help="3D connectivity for connected components",
    )
    parser.add_argument(
        "--max_components",
        type=int,
        default=30,
        help="Maximum number of component sizes printed per patient/class",
    )
    parser.add_argument(
        "--show_mm3",
        action="store_true",
        help="Also print component volumes in mm^3",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV path to save every component size",
    )

    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    main(get_args())
