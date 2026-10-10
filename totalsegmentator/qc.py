from pathlib import Path
import argparse
import itertools

import nibabel as nib
import numpy as np


STRUCTURES = [
    "esophagus",
    "heart",
    "trachea",
    "aorta",
]


def check_subject(subject: Path):
    ct_path = subject / "ct.nii.gz"

    if not ct_path.exists():
        return False, "missing CT", []

    ct_img = nib.load(str(ct_path))
    masks = {}
    warnings = []

    for name in STRUCTURES:
        mask_path = subject / "segmentations" / f"{name}.nii.gz"

        if not mask_path.exists():
            return False, f"missing {name}", []

        mask_img = nib.load(str(mask_path))

        if mask_img.shape != ct_img.shape:
            return False, (
                f"{name} shape mismatch: "
                f"{mask_img.shape} != {ct_img.shape}"
            ), []

        if not np.allclose(
            mask_img.affine,
            ct_img.affine,
            atol=1e-5,
        ):
            return False, f"{name} affine mismatch", []

        mask = np.asarray(mask_img.dataobj) > 0

        if not np.any(mask):
            return False, f"{name} mask empty", []

        masks[name] = mask

    # Overlaps are allowed.
    # They will later become IGNORE pixels during pretraining.
    for a, b in itertools.combinations(STRUCTURES, 2):
        overlap = int((masks[a] & masks[b]).sum())

        if overlap > 0:
            warnings.append(
                f"{a}/{b} overlap={overlap}"
            )

    return True, "OK", warnings


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("totalsegmentator"),
    )

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    subjects = sorted(
        p for p in args.root.iterdir()
        if p.is_dir() and (p / "ct.nii.gz").exists()
    )

    print(f"Found {len(subjects)} subjects\n")

    good = []
    excluded = []

    for i, subject in enumerate(subjects, 1):
        valid, reason, warnings = check_subject(subject)

        if valid:
            good.append(subject.name)

            if warnings:
                warning_text = "; ".join(warnings)
                print(
                    f"[{i:3d}/{len(subjects)}] "
                    f"{subject.name}: KEEP - "
                    f"{warning_text}"
                )
            else:
                print(
                    f"[{i:3d}/{len(subjects)}] "
                    f"{subject.name}: KEEP - OK"
                )

        else:
            excluded.append((subject.name, reason))

            print(
                f"[{i:3d}/{len(subjects)}] "
                f"{subject.name}: DROP - {reason}"
            )

    good_file = args.out_dir / "good_subjects.txt"
    excluded_file = args.out_dir / "excluded_subjects.txt"

    with good_file.open("w") as f:
        for subject in good:
            f.write(subject + "\n")

    with excluded_file.open("w") as f:
        for subject, reason in excluded:
            f.write(f"{subject}\t{reason}\n")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Total:    {len(subjects)}")
    print(f"Kept:     {len(good)}")
    print(f"Excluded: {len(excluded)}")

    print("\nExcluded patients:")

    for subject, reason in excluded:
        print(f"  {subject}: {reason}")

    print("\nWrote:")
    print(f"  {good_file}")
    print(f"  {excluded_file}")


if __name__ == "__main__":
    main()