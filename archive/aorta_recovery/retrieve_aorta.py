#!/usr/bin/env python3

# MIT License
#
# Recover the aorta label that is missing from data/segthor_part1.zip.
#
# In that archive every GT.nii.gz holds only the label values {0,1,2,3}: class 1
# is esophagus UNION aorta and there is no class 4. The aorta is not absent from
# the images, it is folded into class 1. Patient_07/GT2.nii.gz - the only file in
# the tree that separates the two organs, and the only reference available - gives
# 25373 esophagus + 89856 aorta = 115229 = the class-1 voxel count of GT.nii.gz.
#
# This script splits class 1 back into its two organs and writes a challenge-style
# 4-organ GT (0 background, 1 esophagus, 2 heart, 3 trachea, 4 aorta). The split
# runs a marker-controlled watershed on the distance transform of class 1: the
# aorta is seeded from the thick round core, the esophagus from the thin per-slice
# components, whichever basin ends up with the smaller median axial area is the
# esophagus, and anything whose inscribed ball is wider than the esophagus can be
# is forced into the aorta. A fixed threshold cannot separate the two organs in
# every patient (Patient_03 has a genuinely thin aorta), so the thin-component
# threshold is swept and only a result passing four plausibility gates is shipped;
# whatever fails is reported for human review, never silently.
#
# Validation: Dice 0.9994 (aorta) / 0.9980 (esophagus) against Patient_07/GT2.nii.gz.
# Patient_07 is written from GT2 itself, which is exact rather than derived, and its
# Dice scores are printed as the regression check for this script. The other 19
# patients have no reference labels and are validated by the gates plus the figures
# written with --figures (see AORTA_INSPECTION/ and its aorta-findings.md).
#
# The labels produced here are derived, not annotated. Said so in the report and in
# EXPERIMENTS.md whenever an experiment is trained on them.
#
# Nothing is ever written over data/SEGTHOR or data/segthor_part1: use --dest
# data/SEGTHOR_aorta (default) and slice it with slice_segthor.py --source_dir.

import os
import csv
import argparse
import warnings
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as nd_label
from skimage.segmentation import watershed

# Challenge class convention, as used by --class_names everywhere else in the repo.
CLASS_NAMES: list[str] = ["background", "esophagus", "heart", "trachea", "aorta"]
K: int = len(CLASS_NAMES)

# The thin-component threshold swept by split_patient: a per-slice component of
# class 1 smaller than this can only be the esophagus. Values in mm^2.
THIN_SWEEP: list[float] = [300.0, 200.0, 120.0]
# The aorta seed is the largest component of dt >= CORE_FRAC * dt.max().
CORE_FRAC: float = 0.6
# An inscribed ball wider than this is aorta by construction (the esophagus is at
# most about 20 mm across). In mm.
ESO_MAX_RADIUS: float = 10.0

# Plausibility gates: [min, max] voxels, median aorta area >= AREA_RATIO * median
# esophagus area, esophagus median area within ESO_MEDIAN_AREA (mm^2).
AORTA_VOXELS: tuple[int, int] = (35000, 260000)
ESO_VOXELS: tuple[int, int] = (5000, 40000)
AREA_RATIO: float = 1.3
ESO_MEDIAN_AREA: tuple[float, float] = (80.0, 350.0)

# Figure conventions, shared with visualize_labels.py.
CT_WINDOW: tuple[float, float] = (-200.0, 400.0)
AORTA_COLOUR: tuple[float, float, float] = (1.0, 0.85, 0.0)
ESO_COLOUR: tuple[float, float, float] = (1.0, 0.25, 0.25)
N_MONTAGE_SLICES: int = 5

REPORT_FIELDS: list[str] = ["patient", "thin_area", "aorta_voxels", "eso_voxels",
                            "med_aorta_mm2", "med_eso_mm2", "ratio",
                            "gate_aorta_voxels", "gate_eso_voxels",
                            "gate_aorta_median", "gate_eso_median", "verdict"]


def load_labels(path: Path) -> tuple[np.ndarray, tuple[float, float, float], nib.Nifti1Image]:
    nib_obj = nib.load(str(path))
    arr = np.asarray(nib_obj.dataobj)
    assert arr.ndim == 3, (path, arr.shape)
    assert np.issubdtype(arr.dtype, np.integer), (path, arr.dtype)
    spacing = tuple(float(z) for z in nib_obj.header.get_zooms()[:3])
    return arr, spacing, nib_obj


def median_slice_area(region: np.ndarray, spacing: tuple[float, float, float]) -> float:
    # Median area (mm^2) of the non-empty axial slices of a binary region. The aorta
    # is a fat round vessel and the esophagus a thin tube, so this separates the two
    # anatomically even when a watershed basin has the wrong shape.
    areas = region.sum(axis=(0, 1)) * spacing[0] * spacing[1]
    areas = areas[areas > 0]
    return float(np.median(areas)) if areas.size else 0.0


def dice(a: np.ndarray, b: np.ndarray) -> float:
    denominator = int(a.sum()) + int(b.sum())
    return 1.0 if denominator == 0 else float(2 * int((a & b).sum()) / denominator)


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, n = nd_label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == counts.argmax()


def gates(aorta: np.ndarray, eso: np.ndarray,
          spacing: tuple[float, float, float]) -> dict[str, float | bool]:
    med_aorta = median_slice_area(aorta, spacing)
    med_eso = median_slice_area(eso, spacing)
    n_aorta = int(aorta.sum())
    n_eso = int(eso.sum())

    return {"aorta_voxels": n_aorta,
            "eso_voxels": n_eso,
            "med_aorta_mm2": med_aorta,
            "med_eso_mm2": med_eso,
            "ratio": med_aorta / med_eso if med_eso > 0 else float("inf"),
            "gate_aorta_voxels": AORTA_VOXELS[0] <= n_aorta <= AORTA_VOXELS[1],
            "gate_eso_voxels": ESO_VOXELS[0] <= n_eso <= ESO_VOXELS[1],
            "gate_aorta_median": med_aorta >= AREA_RATIO * med_eso,
            "gate_eso_median": ESO_MEDIAN_AREA[0] <= med_eso <= ESO_MEDIAN_AREA[1]}


def passes(check: dict) -> bool:
    return all(bool(v) for k, v in check.items() if k.startswith("gate_"))


def split_once(m: np.ndarray, dt: np.ndarray, thin_area: float,
               spacing: tuple[float, float, float],
               trace: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    # trace, when given, is filled with the intermediates of this call so a figure
    # script can draw the shipped pipeline instead of a copy of it.
    # Esophagus candidate: in a slice where class 1 is a single blob the two organs
    # touch, so only slices holding several components can seed the esophagus
    # directly. Everything small enough there can only be the thin tube.
    eso_cand = np.zeros_like(m, dtype=bool)
    for z in range(m.shape[2]):
        comps, n = nd_label(m[:, :, z])
        if n > 1:  # organs apart in this slice
            for c in range(1, n + 1):
                comp = comps == c
                if comp.sum() * spacing[0] * spacing[1] <= thin_area:
                    eso_cand[:, :, z] |= comp

    # Aorta seed: the round thick core of the folded class.
    aorta_seed = largest_component(dt >= CORE_FRAC * dt.max())

    eso_seed = markers = regions = None
    if eso_cand.any():
        # Every qualifying component seeds the esophagus, as aorta-findings.md
        # documents. Seeding only the largest component silently drops any seed
        # stretch that no other component connects to, and the watershed then hands
        # those slices to the aorta: Patient_15 lost its esophagus below z=62 that way
        # (37 of the 113 slices class 1 occupies), Patient_14 below z=89 and
        # Patient_19 below z=101.
        eso_seed = eso_cand
        markers = np.zeros(m.shape, dtype=np.int8)
        markers[aorta_seed] = 1
        markers[eso_seed] = 2
        regions = watershed(-dt, markers=markers, mask=m)
        r_aorta, r_eso = regions == 1, regions == 2
    else:  # never observed on this data: fall back to the inscribed-ball rule alone
        r_eso = m & (dt <= ESO_MAX_RADIUS)
        r_aorta = m & ~r_eso

    if trace is not None:
        trace.update(dt=dt, thin_area=thin_area, eso_cand=eso_cand, aorta_seed=aorta_seed,
                     eso_seed=eso_seed, markers=markers, regions=regions,
                     basins_aorta=r_aorta, basins_eso=r_eso)

    # Role assignment: the smaller median axial area is the esophagus (the aorta is
    # 2.5x to 7.6x larger in every clean case).
    swapped = median_slice_area(r_eso, spacing) > median_slice_area(r_aorta, spacing)
    if swapped:
        r_aorta, r_eso = r_eso, r_aorta
    if trace is not None:
        trace.update(swapped=swapped, roles_aorta=r_aorta, roles_eso=r_eso)

    # Hard rule: a class-1 voxel whose inscribed ball is wider than the esophagus can
    # be is aorta, whatever the watershed said.
    eso = r_eso & (dt <= ESO_MAX_RADIUS)
    aorta = m & ~eso

    if trace is not None:
        trace.update(aorta=aorta, eso=eso)

    return aorta, eso


def split_patient(m: np.ndarray, spacing: tuple[float, float, float],
                  trace: dict | None = None) \
        -> tuple[float, np.ndarray, np.ndarray, dict, str]:
    # Sweep the thin-component threshold and keep the first result passing all four
    # gates; if none does, keep the first and flag the patient for human review.
    # trace is cleared per sweep value so it always holds the shipped threshold.
    dt = distance_transform_edt(m, sampling=spacing)

    thin_area = THIN_SWEEP[0]
    aorta = eso = np.zeros_like(m, dtype=bool)
    check: dict = {}
    ok = False
    for thin_area in THIN_SWEEP:
        if trace is not None:
            trace.clear()
        aorta, eso = split_once(m, dt, thin_area, spacing, trace)
        check = gates(aorta, eso, spacing)
        ok = passes(check)
        if ok:
            break

    return thin_area, aorta, eso, check, "ok" if ok else "REVIEW"


def compose(gt: np.ndarray, aorta: np.ndarray, eso: np.ndarray) -> np.ndarray:
    # Challenge-style label map: only class 1 is rewritten, classes 2 and 3 are
    # copied from the source GT and the background is whatever is left.
    arr = np.zeros(gt.shape, dtype=np.uint8)
    arr[eso] = 1
    arr[gt == 2] = 2
    arr[gt == 3] = 3
    arr[aorta] = 4
    return arr


def write_patient(dest: Path, src: Path, id_: str, arr: np.ndarray,
                  ct_nib: nib.Nifti1Image) -> Path:
    out_dir = dest / "train" / id_
    out_dir.mkdir(parents=True, exist_ok=True)

    out_nib = nib.nifti1.Nifti1Image(arr.astype(np.uint8), affine=ct_nib.affine, header=ct_nib.header)
    out_nib.set_data_dtype(np.uint8)  # geometry from the CT, never GT's identity affine
    gt_path = out_dir / "GT.nii.gz"
    nib.save(out_nib, str(gt_path))

    # The CT is symlinked, not copied: slice_segthor.py only reads it back, and this
    # keeps --dest a few hundred kilobytes instead of ~800 MB.
    link = out_dir / f"{id_}.nii.gz"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath((src / id_ / f"{id_}.nii.gz").resolve(), out_dir))

    return gt_path


def shared_slices(aorta: np.ndarray, eso: np.ndarray, n: int) -> list[int]:
    # Axial slices where both organs are visible, evenly spaced from the first to the
    # last of them: every panel of a montage then shows both contours.
    both = np.where(aorta.any(axis=(0, 1)) & eso.any(axis=(0, 1)))[0]
    if both.size == 0:
        both = np.where((aorta | eso).any(axis=(0, 1)))[0]
    if both.size == 0:
        return list(range(min(n, aorta.shape[2])))
    picks = np.unique(np.linspace(0, both.size - 1, min(n, both.size)).astype(int))
    return [int(both[p]) for p in picks]


def panel(ax, ct: np.ndarray, z: int, masks: list[tuple[np.ndarray, tuple]],
          title: str = "") -> None:
    ax.imshow(ct[:, :, z].T, cmap="gray", vmin=CT_WINDOW[0], vmax=CT_WINDOW[1],
              origin="lower")
    for mask, colour in masks:
        ax.contour(mask[:, :, z].T.astype(float), levels=[0.5], colors=[colour], linewidths=0.7)
    if title:
        ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()


def montage(path: Path, ct: np.ndarray, zs: list[int], rows: list[tuple[str, list]],
            title: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), len(zs), figsize=(3 * len(zs), 3.2 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (label, masks) in enumerate(rows):
        for c, z in enumerate(zs):
            panel(axes[r, c], ct, z, masks, f"{label} | axial z={z}")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def render_montage(out_dir: Path, id_: str, ct: np.ndarray, merged: np.ndarray,
                   aorta: np.ndarray, eso: np.ndarray, gt2: np.ndarray | None) -> list[Path]:
    zs = shared_slices(aorta, eso, N_MONTAGE_SLICES)
    rows = [("merged class 1", [(merged, AORTA_COLOUR)]),
            ("recovered split", [(aorta, AORTA_COLOUR), (eso, ESO_COLOUR)])]

    written = [montage(out_dir / f"{id_}_axial_merged_vs_split.png", ct, zs, rows,
                       f"{id_} - folded class 1 vs recovered aorta (yellow) and esophagus (red)")]
    if gt2 is not None:
        written.append(montage(out_dir / f"{id_}_axial_vs_GT2.png", ct, zs,
                               rows + [("reference GT2", [(gt2 == 4, AORTA_COLOUR),
                                                          (gt2 == 1, ESO_COLOUR)])],
                               f"{id_} - recovered split against the reference GT2"))
    return written


def render_overview(out_dir: Path, records: list[dict]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = 5
    nrows = int(np.ceil(len(records) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.2 * nrows))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, rec in zip(axes.ravel(), records):
        zs = shared_slices(rec["aorta"], rec["eso"], 1)
        panel(ax, rec["ct"], zs[0], [(rec["aorta"], AORTA_COLOUR), (rec["eso"], ESO_COLOUR)],
              f"{rec['id']} z={zs[0]}")
    fig.suptitle("recovered aorta (yellow) / esophagus (red), one axial slice per patient")
    fig.tight_layout()

    path = out_dir / "00_overview_all_patients.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def render_figures(out_dir: Path, records: list[dict]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [p for r in records
               for p in render_montage(out_dir, r["id"], r["ct"], r["merged"], r["aorta"],
                                       r["eso"], r["gt2"])]
    written.append(render_overview(out_dir, records))
    return written


def print_report(rows: list[dict]) -> None:
    header = f"{'patient':12s}{'thin':>6s}{'aorta_vx':>10s}{'eso_vx':>9s}" \
             f"{'med_aorta':>11s}{'med_eso':>9s}{'ratio':>7s}  verdict"
    print("\n" + header)
    for r in rows:
        if r["verdict"] == "excluded":
            print(f"{r['patient']:12s}{'-':>6s}{'-':>10s}{'-':>9s}{'-':>11s}{'-':>9s}{'-':>7s}  excluded")
            continue
        print(f"{r['patient']:12s}{r['thin_area']:6.0f}{r['aorta_voxels']:10d}{r['eso_voxels']:9d}"
              f"{r['med_aorta_mm2']:11.1f}{r['med_eso_mm2']:9.1f}{r['ratio']:7.2f}  {r['verdict']}")


def main(args: argparse.Namespace) -> None:
    src: Path = args.src
    dest: Path = args.dest
    excluded: set[str] = set(args.exclude or [])

    forbidden = {Path("data/SEGTHOR").resolve(), Path("data/segthor_part1").resolve()}
    assert dest.resolve() not in forbidden, f"--dest must not be {dest}: it holds the source data"
    assert dest.resolve() != src.resolve(), f"--dest {dest} would overwrite --src {src}"

    assert src.is_dir(), f"No such source directory: {src}"
    ids: list[str] = sorted(p.name for p in src.iterdir()
                            if p.is_dir() and (p / "GT.nii.gz").exists())
    assert len(ids) > 0, f"No <Patient_XX>/GT.nii.gz under {src}"
    print(f"Recovering the aorta for {len(ids)} patients from {src} into {dest}")

    gt2_path: Path | None = args.gt2
    gt2_id: str | None = None
    gt2_arr: np.ndarray | None = None
    if gt2_path is not None:
        assert gt2_path.exists(), f"No such --gt2 reference: {gt2_path}"
        gt2_id = gt2_path.parent.name

    rows: list[dict] = []
    records: list[dict] = []
    for id_ in ids:
        if id_ in excluded:
            print(f"{id_}: excluded, nothing written")
            rows.append({"patient": id_, **{f: "" for f in REPORT_FIELDS[1:-1]}, "verdict": "excluded"})
            continue

        ct, spacing, ct_nib = load_labels(src / id_ / f"{id_}.nii.gz")
        gt, gt_spacing, _ = load_labels(src / id_ / "GT.nii.gz")
        assert gt.shape == ct.shape, (id_, gt.shape, ct.shape)
        values = set(int(v) for v in np.unique(gt))
        assert values <= set(range(K)), f"{id_}: GT holds label values {sorted(values)}"

        m = gt == 1
        assert m.any(), f"{id_}: no class-1 voxels, nothing to split"
        thin_area, aorta, eso, check, verdict = split_patient(m, spacing)

        if not np.allclose(spacing, gt_spacing):
            warnings.warn(f"{id_}: splitting in CT spacing {spacing}; GT header says {gt_spacing} (ignored)")

        if id_ == gt2_id:
            assert gt2_arr is None
            gt2_arr, gt2_spacing, _ = load_labels(gt2_path)
            assert gt2_arr.shape == gt.shape, (id_, gt2_arr.shape, gt.shape)
            d_aorta = dice(aorta, gt2_arr == 4)
            d_eso = dice(eso, gt2_arr == 1)
            print(f"\n{id_} Dice vs GT2: aorta {d_aorta:.4f}, esophagus {d_eso:.4f}"
                  f"  (GT2 spacing {gt2_spacing})")
            assert d_aorta >= 0.99 and d_eso >= 0.99, (id_, d_aorta, d_eso)
            # The reference is exact where the split is derived: write it as is.
            arr = gt2_arr.astype(np.uint8)
        else:
            arr = compose(gt, aorta, eso)

        gt_path = write_patient(dest, src, id_, arr, ct_nib)
        rows.append({"patient": id_, "thin_area": thin_area, **check, "verdict": verdict})
        records.append({"id": id_, "ct": ct, "merged": m, "aorta": aorta, "eso": eso,
                        "gt2": gt2_arr if id_ == gt2_id else None})
        print(f"{id_}: thin_area={thin_area:.0f} mm^2, aorta={check['aorta_voxels']} vx, "
              f"esophagus={check['eso_voxels']} vx -> {verdict} ({gt_path})")

    print_report(rows)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r[k] for k in REPORT_FIELDS})
        print(f"\nWrote {args.report}")

    if args.figures is not None:
        written = render_figures(args.figures, records)
        print(f"\nWrote {len(written)} figures to {args.figures}: "
              f"{[p.name for p in written[:3]]} ...")

    failed = [r["patient"] for r in rows if r["verdict"] != "ok"]
    if failed:
        print(f"\n{len(failed)} patient(s) not ok: {failed}"
              f"{' (excluded)' if any(r['verdict'] == 'excluded' for r in rows) else ''}")
        raise SystemExit(1)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Recover the aorta label folded into class 1 by data/segthor_part1.zip',
        epilog="Exit code 0 when every patient passes the four gates, 1 when any row is REVIEW or excluded.")
    parser.add_argument('--src', type=Path, default=Path("data/segthor_part1/train"),
                        help="Folder holding <Patient_XX>/ with GT.nii.gz and <id>.nii.gz")
    parser.add_argument('--dest', type=Path, default=Path("data/SEGTHOR_aorta"),
                        help="Output folder, laid out as <dest>/train/<Patient_XX>/; never data/SEGTHOR")
    parser.add_argument('--gt2', type=Path, default=None,
                        help="Optional reference that separates the organs (Patient_XX/GT2.nii.gz). "
                             "Its patient is written from that file and its Dice is asserted >= 0.99")
    parser.add_argument('--report', type=Path, default=None,
                        help="CSV of the per-patient split and gates (default <dest>/split_report.csv)")
    parser.add_argument('--figures', type=Path, default=None,
                        help="Folder for the axial montages, the overview, and the --gt2 reference row")
    parser.add_argument('--exclude', type=str, nargs='+', default=None,
                        help="Patient ids to skip; they are listed as excluded and force exit code 1")

    args = parser.parse_args()
    if args.report is None:
        args.report = args.dest / "split_report.csv"
    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
