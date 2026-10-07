#!/usr/bin/env python3

"""3D post-processing of stitched segmentation predictions.

Run postprocess.py AFTER infer.py, and BEFORE metrics3d.py. To run different
postprocessing techniques on the same baseline, you only need to run infer.py
once. Make sure to change the output_folder.

The script operates on hard-label 3D NIfTI predictions, such as those produced
by infer.py + stitch.py. It can independently apply:

1. Largest connected component (LCC)
2. Small-component removal (remove 3D components smaller than a threshold)
3. 3D binary hole filling
4. 3D morphological closing followed by opening

Each technique can be restricted to selected classes. Class 0 is treated as
background and is always skipped, even if it is listed on the command line.

The processing order for a class selected for multiple techniques is:

    LCC -> small-component removal -> hole filling -> closing -> opening

Component-size thresholds are expressed in voxels (3D pixels).
For per-class minimum-size filtering, use e.g. `--min_size 1:500 2:1000 4:250`.

The NIfTI affine/header are preserved and output masks remain uint8 labels.
"""

import argparse
import warnings
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    binary_opening,
    distance_transform_edt,
    generate_binary_structure,
    label,
)

from segthor.utils import tqdm_


def connectivity_structure(connectivity: int) -> np.ndarray:
    """Return a 3D binary structuring element for 6/18/26-connectivity."""
    if connectivity not in {6, 18, 26}:
        raise ValueError(f"Unsupported connectivity: {connectivity}")

    rank = {6: 1, 18: 2, 26: 3}[connectivity]
    return generate_binary_structure(rank=3, connectivity=rank)


def largest_connected_component(mask: np.ndarray, structure: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of a binary 3D mask."""
    assert mask.dtype == bool, mask.dtype
    assert mask.ndim == 3, mask.shape

    if not mask.any():
        return mask.copy()

    labels, n_components = label(mask, structure=structure)
    if n_components <= 1:
        return mask.copy()

    counts = np.bincount(labels.ravel())
    largest_label = 1 + int(np.argmax(counts[1:]))
    return labels == largest_label


def keep_near_largest(
    mask: np.ndarray,
    structure: np.ndarray,
    max_distance_mm: float,
    spacing: tuple[float, float, float],
) -> tuple[np.ndarray, int, int]:
    """Keep the largest component and every component within max_distance_mm of it.

    LCC fails on tubular organs because a small gap in the prediction splits
    the organ and "keep the largest piece" then deletes the correctly
    segmented part beyond the gap (E021). A fragment across a gap lies right
    next to the main component, a stray false-positive blob lies far away:
    filtering by distance keeps the first and removes the second.
    Returns the filtered mask, number of input components, and number removed.
    """
    assert mask.dtype == bool, mask.dtype
    assert mask.ndim == 3, mask.shape
    assert max_distance_mm >= 0, max_distance_mm

    if not mask.any():
        return mask.copy(), 0, 0
    labels, n_components = label(mask, structure=structure)
    if n_components <= 1:
        return mask.copy(), n_components, 0

    counts = np.bincount(labels.ravel())
    largest = 1 + int(np.argmax(counts[1:]))
    # Distances only matter up to max_distance_mm: work in the bounding box of
    # the largest component grown by that margin, not the whole volume.
    margin = [int(np.ceil(max_distance_mm / s)) + 1 for s in spacing]
    coords = np.argwhere(labels == largest)
    lo = np.maximum(coords.min(0) - margin, 0)
    hi = np.minimum(coords.max(0) + margin + 1, mask.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    dist = distance_transform_edt(labels[box] != largest, sampling=spacing)

    near = np.unique(labels[box][(dist <= max_distance_mm) & (labels[box] > 0)])
    keep = np.isin(labels, near)
    return keep, n_components, n_components - len(near)


def remove_small_components(
    mask: np.ndarray,
    structure: np.ndarray,
    min_size: int,
) -> tuple[np.ndarray, int, int]:
    """Remove all connected components smaller than min_size voxels.

    Components with exactly min_size voxels are retained.
    Returns the filtered mask, number of input components, and number removed.
    """
    assert mask.dtype == bool, mask.dtype
    assert mask.ndim == 3, mask.shape
    assert min_size > 0, min_size

    if not mask.any():
        return mask.copy(), 0, 0

    labels, n_components = label(mask, structure=structure)
    if n_components == 0:
        return mask.copy(), 0, 0

    counts = np.bincount(labels.ravel())
    keep_labels = np.flatnonzero(counts >= min_size)
    keep_labels = keep_labels[keep_labels != 0]

    filtered = np.isin(labels, keep_labels)
    removed = n_components - len(keep_labels)
    return filtered, n_components, removed


def fill_holes(mask: np.ndarray, structure: np.ndarray) -> np.ndarray:
    """Fill enclosed 3D holes in a binary mask."""
    assert mask.dtype == bool, mask.dtype
    assert mask.ndim == 3, mask.shape
    return binary_fill_holes(mask, structure=structure)


def morphological_cleanup(
    mask: np.ndarray,
    structure: np.ndarray,
    closing_iterations: int,
    opening_iterations: int,
) -> np.ndarray:
    """Apply 3D binary closing and then opening."""
    assert mask.dtype == bool, mask.dtype
    assert mask.ndim == 3, mask.shape
    assert closing_iterations >= 0, closing_iterations
    assert opening_iterations >= 0, opening_iterations

    result = mask

    if closing_iterations > 0:
        result = binary_closing(
            result,
            structure=structure,
            iterations=closing_iterations,
        )

    if opening_iterations > 0:
        result = binary_opening(
            result,
            structure=structure,
            iterations=opening_iterations,
        )

    return result


def validate_classes(classes: list[int], K: int, name: str) -> list[int]:
    """Validate class ids and remove background class 0 from a technique list."""
    duplicates = sorted({k for k in classes if classes.count(k) > 1})
    if duplicates:
        raise ValueError(f"{name}: duplicate class ids: {duplicates}")

    invalid = sorted(set(classes) - set(range(K)))
    if invalid:
        raise ValueError(
            f"{name}: invalid class ids {invalid}; expected values in [0, {K - 1}]"
        )

    if 0 in classes:
        print(f">> {name}: ignoring class 0 (background); post-processing organs only")

    return [k for k in classes if k != 0]


def load_prediction(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Load a hard-label NIfTI prediction and keep its original NIfTI image."""
    img = nib.load(str(path))
    arr = np.asarray(img.dataobj)

    assert arr.ndim == 3, (path, arr.shape)
    assert np.issubdtype(arr.dtype, np.integer), (path, arr.dtype)

    if not np.array_equal(arr, arr.astype(np.uint8)):
        raise ValueError(f"{path}: labels cannot be represented safely as uint8")

    return arr.astype(np.uint8, copy=False), img


def save_prediction(path: Path, labels: np.ndarray, reference: nib.Nifti1Image) -> None:
    """Save a uint8 label NIfTI while preserving affine and header metadata."""
    labels = labels.astype(np.uint8, copy=False)

    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    output = nib.Nifti1Image(labels, reference.affine, header=header)
    output.set_data_dtype(np.uint8)
    nib.save(output, str(path))


def process_volume(
    path: Path,
    dest: Path,
    K: int,
    lcc_classes: list[int],
    hole_fill_classes: list[int],
    morph_classes: list[int],
    min_size_by_class: dict[int, int],
    connectivity: int,
    closing_iterations: int,
    opening_iterations: int,
    keep_near_by_class: dict[int, float] | None = None,
    min_ml_by_class: dict[int, float] | None = None,
) -> dict[str, object]:
    """Process one NIfTI volume and return a compact summary."""
    pred, reference = load_prediction(path)
    keep_near_by_class = keep_near_by_class or {}
    min_ml_by_class = min_ml_by_class or {}
    # Our predictions carry the CT header (stitch.py), so these are real mm
    spacing = tuple(float(z) for z in reference.header.get_zooms()[:3])

    unique = np.unique(pred)
    if not np.all(np.isin(unique, np.arange(K))):
        raise ValueError(
            f"{path}: found labels {unique.tolist()}, but num_classes={K}"
        )

    structure = connectivity_structure(connectivity)
    selected_classes = sorted(
        set(lcc_classes)
        | set(hole_fill_classes)
        | set(morph_classes)
        | set(min_size_by_class)
        | set(keep_near_by_class)
        | set(min_ml_by_class)
    )

    processed = pred.copy()
    changed = 0
    class_summaries: list[str] = []

    for k in selected_classes:
        original_mask = pred == k
        mask = original_mask.copy()

        before = int(mask.sum())

        if k in lcc_classes:
            mask = largest_connected_component(mask, structure)

        if k in keep_near_by_class:
            mask, n_near, n_far = keep_near_largest(mask, structure, keep_near_by_class[k], spacing)
            class_summaries.append(f"class {k}: keep-near {keep_near_by_class[k]:g} mm: "
                                   f"{n_near} components, {n_far} removed")

        if k in min_ml_by_class:
            # A size in ml, converted with this scan's voxel volume: a voxel
            # count is not the same size from one scan to the next (voxel
            # volumes differ ~4x in SegTHOR).
            voxel_ml = float(np.prod(spacing)) / 1000.0
            min_voxels = max(1, int(round(min_ml_by_class[k] / voxel_ml)))
            mask, n_ml, n_ml_removed = remove_small_components(mask, structure, min_size=min_voxels)
            class_summaries.append(f"class {k}: min {min_ml_by_class[k]:g} ml ({min_voxels} voxels): "
                                   f"{n_ml} components, {n_ml_removed} removed")

        if k in min_size_by_class:
            min_size = min_size_by_class[k]
            mask, n_components, n_removed = remove_small_components(
                mask,
                structure,
                min_size=min_size,
            )
        else:
            n_components, n_removed = 0, 0

        if k in hole_fill_classes:
            mask = fill_holes(mask, structure)

        if k in morph_classes:
            mask = morphological_cleanup(
                mask,
                structure,
                closing_iterations=closing_iterations,
                opening_iterations=opening_iterations,
            )

        after = int(mask.sum())
        class_changed = int(np.count_nonzero(mask != original_mask))
        changed += class_changed

        # First remove the original voxels of this class, then insert the
        # processed mask only into background. This prevents one organ's
        # post-processing from overwriting another organ.
        processed[processed == k] = 0
        can_write = mask & (processed == 0)
        processed[can_write] = k

        if k in min_size_by_class:
            min_size = min_size_by_class[k]
            class_summaries.append(
                f"class {k}: {before:,}->{after:,} voxels ({class_changed:,} changed); "
                f"small-component filter: {n_components} components, "
                f"{n_removed} removed (<{min_size:,} voxels)"
            )
        else:
            class_summaries.append(
                f"class {k}: {before:,}->{after:,} voxels ({class_changed:,} changed)"
            )

    dest.mkdir(parents=True, exist_ok=True)
    output_path = dest / path.name
    save_prediction(output_path, processed, reference)

    return {
        "input": path.name,
        "output": output_path.name,
        "changed": changed,
        "summary": class_summaries,
    }


def get_prediction_files(folder: Path) -> list[Path]:
    """Return all supported NIfTI prediction files in a folder."""
    files = sorted(folder.glob("*.nii.gz"))
    files += sorted(folder.glob("*.nii"))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No .nii or .nii.gz files found in {folder}")
    return files


def parse_min_size_specs(specs: list[str], K: int) -> dict[int, int]:
    """Parse CLASS:VOXELS specifications into a per-class threshold mapping."""
    result: dict[int, int] = {}

    for spec in specs:
        if ":" not in spec:
            raise ValueError(
                f"Invalid --min_size specification '{spec}'. "
                "Expected CLASS:VOXELS, e.g. 1:500."
            )

        class_text, size_text = spec.split(":", 1)
        try:
            class_id = int(class_text)
            min_size = int(size_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --min_size specification '{spec}'. "
                "CLASS and VOXELS must both be integers."
            ) from exc

        if class_id < 0 or class_id >= K:
            raise ValueError(
                f"--min_size: invalid class {class_id}; "
                f"expected values in [0, {K - 1}]"
            )
        if class_id == 0:
            print(">> --min_size: ignoring class 0 (background); post-processing organs only")
            continue
        if min_size <= 0:
            raise ValueError(
                f"--min_size: threshold for class {class_id} must be positive, got {min_size}"
            )
        if class_id in result:
            raise ValueError(
                f"--min_size: class {class_id} was specified more than once"
            )

        result[class_id] = min_size

    return result


def main(args: argparse.Namespace) -> None:
    assert args.input_folder.exists(), args.input_folder
    assert args.input_folder.is_dir(), args.input_folder

    files = get_prediction_files(args.input_folder)
    print(f">> Found {len(files)} NIfTI predictions in {args.input_folder}")

    lcc_classes = validate_classes(args.lcc, args.num_classes, "--lcc")
    keep_near_by_class: dict[int, float] = {}
    for spec in args.keep_near:
        k, mm = spec.split(":")
        assert 0 < int(k) < args.num_classes, f"--keep_near: invalid class in {spec}"
        assert int(k) not in lcc_classes, f"--keep_near: class {k} is also in --lcc"
        keep_near_by_class[int(k)] = float(mm)
    min_ml_by_class: dict[int, float] = {}
    for spec in args.min_ml:
        k, ml = spec.split(":")
        assert 0 < int(k) < args.num_classes, f"--min_ml: invalid class in {spec}"
        min_ml_by_class[int(k)] = float(ml)
    hole_fill_classes = validate_classes(args.hole_fill, args.num_classes, "--3dhf")
    morph_classes = validate_classes(args.morph, args.num_classes, "--morph")
    min_size_by_class = parse_min_size_specs(
        args.min_size,
        args.num_classes,
    )

    if not (lcc_classes or hole_fill_classes or morph_classes or min_size_by_class or keep_near_by_class
            or min_ml_by_class):
        raise ValueError(
            "No post-processing selected. Supply at least one of "
            "--lcc, --keep_near, --min_ml, --min_size, --3dhf or --morph."
        )

    if args.closing_iterations == 0 and args.opening_iterations == 0 and morph_classes:
        warnings.warn(
            "--morph was supplied but both --closing_iterations and "
            "--opening_iterations are 0; morphology will do nothing."
        )

    print(f">> LCC classes: {lcc_classes or 'none'}")
    if min_size_by_class:
        specs = ", ".join(
            f"class {k}={v:,} voxels" for k, v in sorted(min_size_by_class.items())
        )
        print(f">> Small-component filter: {specs}")
    else:
        print(">> Small-component filter: none")
    print(f">> 3D hole-fill classes: {hole_fill_classes or 'none'}")
    print(f">> Morphology classes: {morph_classes or 'none'}")
    print(f">> Connectivity: {args.connectivity}")
    print(
        f">> Morphology: closing={args.closing_iterations}, "
        f"opening={args.opening_iterations}"
    )

    pfun = partial(
        process_volume,
        dest=args.output_folder,
        K=args.num_classes,
        lcc_classes=lcc_classes,
        hole_fill_classes=hole_fill_classes,
        morph_classes=morph_classes,
        min_size_by_class=min_size_by_class,
        keep_near_by_class=keep_near_by_class,
        min_ml_by_class=min_ml_by_class,
        connectivity=args.connectivity,
        closing_iterations=args.closing_iterations,
        opening_iterations=args.opening_iterations,
    )

    match args.process:
        case 1:
            results = [pfun(path) for path in tqdm_(files, desc=">> Post-processing")]
        case -1:
            results = Pool().map(pfun, files)
        case p:
            if p <= 0:
                raise ValueError("--process must be 1, -1, or a positive number of workers")
            results = Pool(p).map(pfun, files)

    total_changed = 0
    for result in results:
        total_changed += int(result["changed"])
        print(f">> {result['input']}: {result['changed']:,} class-mask voxels changed")
        for summary in result["summary"]:  # type: ignore[union-attr]
            print(f"   {summary}")

    print(f">> Saved {len(results)} processed volumes to {args.output_folder}")
    print(f">> Total changed voxels across all processed volumes: {total_changed:,}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply selected 3D post-processing to stitched NIfTI segmentation predictions"
    )

    parser.add_argument(
        "--input_folder",
        type=Path,
        required=True,
        help="Folder containing predicted .nii.gz/.nii label volumes",
    )
    parser.add_argument(
        "--output_folder",
        type=Path,
        required=True,
        help="Folder where processed NIfTI volumes are written",
    )
    parser.add_argument(
        "--num_classes",
        "-K",
        type=int,
        default=5,
        help="Number of label classes, including background",
    )
    parser.add_argument(
        "--class_names",
        type=str,
        nargs="+",
        default=None,
        help="Optional class names, background first",
    )

    parser.add_argument(
        "--lcc",
        type=int,
        nargs="+",
        default=[],
        metavar="CLASS",
        help="Apply largest connected component to these classes",
    )
    parser.add_argument(
        "--keep_near",
        type=str,
        nargs="+",
        default=[],
        metavar="CLASS:MM",
        help=(
            "Keep the largest component plus every component within MM millimetres "
            "of it (gap-tolerant LCC), e.g. --keep_near 1:10 4:10"
        ),
    )
    parser.add_argument(
        "--min_ml",
        type=str,
        nargs="+",
        default=[],
        metavar="CLASS:ML",
        help="Remove components smaller than ML millilitres (spacing-aware), e.g. --min_ml 4:1",
    )
    parser.add_argument(
        "--min_size",
        type=str,
        nargs="+",
        default=[],
        metavar="CLASS:VOXELS",
        help=(
            "Remove connected components smaller than a class-specific threshold. "
            "Use one or more CLASS:VOXELS pairs, e.g. "
            "--min_size 1:500 2:1000 3:250 4:750"
        ),
    )
    parser.add_argument(
        "--3dhf",
        dest="hole_fill",
        type=int,
        nargs="+",
        default=[],
        metavar="CLASS",
        help="Apply 3D binary hole filling to these classes",
    )
    parser.add_argument(
        "--morph",
        type=int,
        nargs="+",
        default=[],
        metavar="CLASS",
        help="Apply 3D closing followed by opening to these classes",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=[6, 18, 26],
        default=6,
        help="3D neighbourhood used by connected components and morphology (default: 6)",
    )
    parser.add_argument(
        "--closing_iterations",
        type=int,
        default=1,
        help="Number of 3D binary-closing iterations when --morph is used (default: 1)",
    )
    parser.add_argument(
        "--opening_iterations",
        type=int,
        default=1,
        help="Number of 3D binary-opening iterations when --morph is used (default: 1)",
    )
    parser.add_argument(
        "--process",
        "-p",
        type=int,
        default=1,
        help="Number of worker processes (1: sequential, -1: all cores)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output folder",
    )

    args = parser.parse_args()

    if args.num_classes < 2:
        raise ValueError("--num_classes must be at least 2")
    if args.closing_iterations < 0 or args.opening_iterations < 0:
        raise ValueError("Morphology iterations cannot be negative")
    if args.class_names is not None and len(args.class_names) != args.num_classes:
        raise ValueError(
            f"Expected {args.num_classes} class names, got {len(args.class_names)}"
        )

    if args.output_folder.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output folder already exists: {args.output_folder}. "
            "Use --overwrite or choose a new output folder."
        )

    args.output_folder.mkdir(parents=True, exist_ok=True)

    print(args)
    return args


if __name__ == "__main__":
    main(get_args())
