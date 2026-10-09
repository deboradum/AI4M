from pathlib import Path
from functools import partial
from multiprocessing import Pool
import argparse
import json
import random
import warnings

import nibabel as nib
import numpy as np
from skimage.io import imsave
from skimage.transform import resize


LABELS = {
    "esophagus": 1,
    "heart": 2,
    "trachea": 3,
    "aorta": 4,
}

LUNG_LABELS = [
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
]


resize_ = partial(
    resize,
    mode="constant",
    preserve_range=True,
    anti_aliasing=False,
)


def norm_window(img, low=-1000, high=1000):
    img = img.astype(np.float32)
    img = np.clip(img, low, high)

    img = 255 * (img - low) / (high - low)

    return img.astype(np.uint8)


def load_subject_ids(path: Path):
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def build_gt(subject_dir: Path, ct_img):
    masks = {}

    for name in LABELS:
        mask_path = (
            subject_dir
            / "segmentations"
            / f"{name}.nii.gz"
        )

        mask_img = nib.load(str(mask_path))

        if mask_img.shape != ct_img.shape:
            raise RuntimeError(
                f"{subject_dir.name}: "
                f"{name} shape mismatch"
            )

        if not np.allclose(
            mask_img.affine,
            ct_img.affine,
            atol=1e-5,
        ):
            raise RuntimeError(
                f"{subject_dir.name}: "
                f"{name} affine mismatch"
            )

        masks[name] = (
            np.asarray(mask_img.dataobj) > 0
        )

    #
    # Count how many structures claim every voxel.
    #
    claims = np.zeros(
        ct_img.shape,
        dtype=np.uint8,
    )

    for mask in masks.values():
        claims += mask.astype(np.uint8)

    #
    # Build SegTHOR-style multiclass target.
    #
    gt = np.zeros(
        ct_img.shape,
        dtype=np.uint8,
    )

    for name, class_id in LABELS.items():
        #
        # Only use unambiguous foreground voxels.
        #
        valid = masks[name] & (claims == 1)
        gt[valid] = class_id

    overlap_voxels = int(
        (claims > 1).sum()
    )

    return gt, overlap_voxels


def slice_subject(
    subject_id,
    source_root,
    destination,
    shape,
    margin,
):
    subject_dir = source_root / subject_id

    ct_path = subject_dir / "ct.nii.gz"

    ct_img = nib.load(str(ct_path))
    ct = np.asarray(ct_img.dataobj)

    gt, overlap_voxels = build_gt(
        subject_dir,
        ct_img,
    )

    #
    # Determine thoracic z range from target organs.
    #
    # Use lungs to define the thoracic field of view.
    thorax = np.zeros(ct_img.shape, dtype=bool)

    for name in LUNG_LABELS:
        path = (
            subject_dir
            / "segmentations"
            / f"{name}.nii.gz"
        )

        if path.exists():
            lung_img = nib.load(str(path))

            if (
                lung_img.shape == ct_img.shape
                and np.allclose(
                    lung_img.affine,
                    ct_img.affine,
                    atol=1e-5,
                )
            ):
                thorax |= (
                    np.asarray(lung_img.dataobj) > 0
                )

    foreground_z = np.where(
        np.any(thorax, axis=(0, 1))
    )[0]

    if len(foreground_z) == 0:
        raise RuntimeError(
            f"{subject_id}: no lung foreground"
        )


    z_start = max(
        0,
        int(foreground_z.min()) - margin,
    )

    z_end = min(
        ct.shape[2] - 1,
        int(foreground_z.max()) + margin,
    )

    #
    # Same HU preprocessing as E_F04.
    #
    ct = norm_window(
        ct,
        -1000,
        1000,
    )

    img_dir = destination / "img"
    gt_dir = destination / "gt"

    img_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gt_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    n_slices = 0

    for z in range(
        z_start,
        z_end + 1,
    ):
        img_slice = resize_(
            ct[:, :, z],
            shape,
            order=1,
        ).astype(np.uint8)

        gt_slice = resize_(
            gt[:, :, z],
            shape,
            order=0,
        ).astype(np.uint8)

        assert set(
            np.unique(gt_slice)
        ).issubset(
            {0, 1, 2, 3, 4}
        )

        #
        # Match existing SegTHOR PNG encoding:
        #
        # 0 -> 0
        # 1 -> 63
        # 2 -> 126
        # 3 -> 189
        # 4 -> 252
        #
        gt_png = (
            gt_slice * 63
        ).astype(np.uint8)

        filename = (
            f"{subject_id}_{z:04d}.png"
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
            )

            imsave(
                str(
                    img_dir / filename
                ),
                img_slice,
            )

            imsave(
                str(
                    gt_dir / filename
                ),
                gt_png,
            )

        n_slices += 1

    return {
        "subject": subject_id,
        "slices": n_slices,
        "overlap_voxels_removed": overlap_voxels,
        "z_start": z_start,
        "z_end": z_end,
        "spacing": [
            float(x)
            for x in ct_img.header.get_zooms()[:3]
        ],
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--subjects",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--shape",
        nargs=2,
        type=int,
        default=[256, 256],
    )

    parser.add_argument(
        "--margin",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--val-count",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
    )

    parser.add_argument(
        "--process",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    if args.dest.exists():
        raise RuntimeError(
            f"Destination already exists: "
            f"{args.dest}"
        )

    subjects = load_subject_ids(
        args.subjects
    )

    rng = random.Random(
        args.seed
    )

    subjects = sorted(subjects)
    rng.shuffle(subjects)

    validation = sorted(
        subjects[:args.val_count]
    )

    training = sorted(
        subjects[args.val_count:]
    )

    print(
        f"Total subjects: {len(subjects)}"
    )
    print(
        f"Training:       {len(training)}"
    )
    print(
        f"Validation:     {len(validation)}"
    )

    args.dest.mkdir(
        parents=True
    )

    metadata = {}

    for split_name, split_ids in [
        ("train", training),
        ("val", validation),
    ]:
        print(
            f"\nProcessing {split_name}: "
            f"{len(split_ids)} subjects"
        )

        destination = (
            args.dest / split_name
        )

        function = partial(
            slice_subject,
            source_root=args.source,
            destination=destination,
            shape=tuple(args.shape),
            margin=args.margin,
        )

        if args.process == 1:
            results = [
                function(subject_id)
                for subject_id
                in split_ids
            ]
        else:
            workers = (
                None
                if args.process == -1
                else args.process
            )

            with Pool(workers) as pool:
                results = pool.map(
                    function,
                    split_ids,
                )

        for result in results:
            metadata[
                result["subject"]
            ] = result

            print(
                result["subject"],
                "slices=",
                result["slices"],
                "overlap_removed=",
                result[
                    "overlap_voxels_removed"
                ],
            )

    split = {
        "seed": args.seed,
        "training": training,
        "validation": validation,
    }

    (
        args.dest / "split.json"
    ).write_text(
        json.dumps(
            split,
            indent=2,
        )
    )

    (
        args.dest / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print("\nFinished.")
    print(
        f"Output: {args.dest}"
    )


if __name__ == "__main__":
    main()