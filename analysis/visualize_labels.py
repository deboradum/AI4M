#!/usr/bin/env python3

# MIT License
#
# Draw a CT slice with organ contours over it, so a label file can be looked at
# instead of only counted. Written for the SegTHOR aorta problem: with the folded
# segthor_part1 label, class 1 = esophagus UNION aorta, and only a picture shows
# that the folded blob is two organs.
#
#   ./ai4mi/bin/python analysis/visualize_labels.py \
#       --ct data/segthor_part1/train/Patient_07/Patient_07.nii.gz \
#       --gt data/SEGTHOR_aorta/train/Patient_07/GT.nii.gz --out /tmp/gt.png
#
# Label maps are NIfTI (.nii.gz, via nibabel) or the 2D .png slices made by
# slice_segthor.py (via skimage.io.imread, decoded with stitch.label_scale). The
# layout is one row per axial/coronal/sagittal plane and N slices per row, at the
# fixed fractions chosen by --n; give --gt2 to put a second label map next to the
# first on the same slices and compare them directly.
#
# Colours: esophagus (class 1) red, aorta (class 4) yellow, other organs cycled
# from tab10. --split_class C draws class C per 3D connected component (tab10,
# legend entries C.a, C.b, ... by size) when it holds several: a diagnostic for a
# folded label, where one class covers two organs. A class holding a single
# component keeps its fixed colour.

import sys
import argparse
from pathlib import Path

import numpy as np
from skimage.io import imread
from skimage.transform import resize

import matplotlib
matplotlib.use("Agg")  # headless: never try to open a window
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CLASS_NAMES: list[str] = ["background", "esophagus", "heart", "trachea", "aorta"]
PLANE_AXIS: dict[str, int] = {"axial": 2, "coronal": 1, "sagittal": 0}
PLANE_FRACTIONS: tuple[float, float] = (0.3, 0.7)  # deterministic slice placement
AORTA_COLOUR: tuple[float, float, float] = (1.0, 0.85, 0.0)
ESO_COLOUR: tuple[float, float, float] = (1.0, 0.25, 0.25)
LABEL_SUFFIXES: tuple[str, ...] = (".nii.gz", ".nii", ".png")


def class_colour(k: int, K: int) -> tuple:
    # Anatomically fixed for the two organs that the fold merges, cycled otherwise.
    if k == 1:
        return ESO_COLOUR
    if k == 4:
        return AORTA_COLOUR
    return plt.get_cmap("tab10")((k - 1) % 10)


def load_label(path: Path, K: int) -> tuple[np.ndarray, str]:
    assert path.exists(), f"No such file: {path}"
    if path.name.endswith((".nii.gz", ".nii")):
        import nibabel as nib
        arr = np.asarray(nib.load(str(path)).dataobj)
        assert arr.ndim == 3, (path, arr.shape)
        return arr.astype(np.int32), path.stem
    raw = imread(path)
    assert raw.ndim == 2, (path, raw.shape)
    from stitch import label_scale
    return np.rint(raw / label_scale(K)).astype(np.int32), path.stem


def load_ct(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    if path is None:
        return None, None
    assert path.exists(), f"No such file: {path}"
    if path.name.endswith((".nii.gz", ".nii")):
        import nibabel as nib
        arr = np.asarray(nib.load(str(path)).dataobj)
        assert arr.ndim == 3, (path, arr.shape)
        return arr, arr.shape
    return imread(path), None  # 2D background


def counts(arr: np.ndarray, K: int) -> np.ndarray:
    return np.bincount(arr.ravel(), minlength=K)[:K]


def check_values(arr: np.ndarray, K: int, path: Path) -> None:
    values = sorted(int(v) for v in np.unique(arr))
    if values and values[-1] >= K:
        print(f"{path}: observed label values {values} >= K={K}")
        sys.exit(2)


def crop_in_plane(img: np.ndarray, c: int) -> np.ndarray:
    # Centre-crop N pixels per side, like viewer/viewer.py; an axis shorter than 2N+1
    # is left alone, so the coronal/sagittal rows still work when a 512-wide crop
    # would eat the whole 179-slice z axis.
    if c <= 0:
        return img
    for axis in range(img.ndim):
        if img.shape[axis] > 2 * c:
            sl = [slice(None)] * img.ndim
            sl[axis] = slice(c, -c)
            img = img[tuple(sl)]
    return img


def plane_slice(volume: np.ndarray, plane: str, index: int, c: int = 0) -> np.ndarray:
    axis = PLANE_AXIS[plane]
    sl = [slice(None)] * volume.ndim
    sl[axis] = index
    return crop_in_plane(volume[tuple(sl)], c).T  # rows vertical, origin at the bottom


def components(mask: np.ndarray) -> list[np.ndarray]:
    from scipy.ndimage import label as nd_label
    labels, n = nd_label(mask)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return [labels == lab for lab in np.argsort(sizes)[::-1] if sizes[lab] > 0]


def build_split(arr: np.ndarray, split_class: int) -> tuple[np.ndarray, list] | None:
    # Connected components of the class that holds several. The legend and the
    # drawn contours must agree, so both come from this one labelling; a class with
    # a single component keeps its fixed colour.
    comps = components(arr == split_class)
    if len(comps) <= 1:
        return None
    label_map = np.zeros(arr.shape, dtype=np.int32)
    for i, comp in enumerate(comps, start=1):
        label_map[comp] = i
    cmap = plt.get_cmap("tab10")
    return label_map, [cmap(i % 10) for i in range(len(comps))]


def legend_handles(arr: np.ndarray, K: int, class_names: list[str], split_class: int,
                   split: tuple[np.ndarray, list] | None) -> list:
    total = counts(arr, K)
    handles: list[Line2D] = []
    for k in range(1, K):
        if split is not None and k == split_class:
            for i, colour in enumerate(split[1]):
                n_vox = int((split[0] == i + 1).sum())
                letter = "abcdefghij"[i] if i < 10 else str(i)
                handles.append(Line2D([], [], color=colour, label=f"{k}.{letter} ({n_vox} voxels)"))
            continue
        handles.append(Line2D([], [], color=class_colour(k, K),
                              label=f"{class_names[k]} ({int(total[k])} voxels)"))
    return handles


def draw(ax, ct, arr, plane: str, index: int, args, K: int, split_class: int,
         split: tuple[np.ndarray, list] | None, title: str) -> None:
    slice_arr = plane_slice(arr, plane, index, args.crop) if arr.ndim == 3 \
        else crop_in_plane(arr, args.crop)

    if ct is not None:
        background = plane_slice(ct, plane, index, args.crop) if ct.ndim == 3 \
            else crop_in_plane(ct, args.crop)
        if background.shape != slice_arr.shape:  # a 256x256 slice over a 512x512 CT
            background = resize(background, slice_arr.shape, order=0, preserve_range=True,
                                anti_aliasing=False)
        ax.imshow(background, cmap="gray", vmin=args.window[0], vmax=args.window[1],
                  origin="lower")

    for k in range(1, K):
        if k == split_class and split is not None:
            lab_slice = plane_slice(split[0], plane, index, args.crop) if split[0].ndim == 3 \
                else crop_in_plane(split[0], args.crop)
            for i, colour in enumerate(split[1]):
                comp = lab_slice == i + 1
                if comp.any():
                    ax.contour(comp.astype(float), levels=[0.5], colors=[colour], linewidths=0.7)
            continue
        mask = slice_arr == k
        if mask.any():
            ax.contour(mask.astype(float), levels=[0.5], colors=[class_colour(k, K)], linewidths=0.7)

    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])


def png_slice_index(path: Path, ct: np.ndarray | None) -> int:
    # A slice_segthor.py slice is named <id>_<zzzz>.png: reuse stitch.py's own
    # parsing so the CT background lines up with the label slice.
    if ct is None or ct.ndim != 3:
        return 0
    from stitch import get_z
    try:
        z = get_z(path)
    except ValueError:
        return 0
    assert 0 <= z < ct.shape[2], (path, z, ct.shape)
    return z


def figure_title(arr: np.ndarray, stem: str, K: int) -> str:
    observed = sorted(int(v) for v in np.unique(arr) if 0 <= int(v) < K)
    missing = [c for c in range(1, K) if c not in observed]
    return f"{stem} | observed {observed} | missing {missing}"


def main(args: argparse.Namespace) -> None:
    K: int = args.K
    class_names: list[str] = args.class_names or [f"class{k}" for k in range(K)]
    assert len(class_names) == K, (class_names, K)
    assert 1 <= args.split_class < K, args.split_class

    if args.gt is None:
        raise FileNotFoundError("--gt is required: no label volume given")
    if not args.gt.exists():
        raise FileNotFoundError(f"No such --gt: {args.gt}")

    label_maps: list[tuple[str, np.ndarray]] = [("GT", load_label(args.gt, K)[0])]
    label_paths: list[Path] = [args.gt]
    if args.gt2 is not None:
        label_maps.append(("GT2", load_label(args.gt2, K)[0]))
        label_paths.append(args.gt2)
    for (name, arr), path in zip(label_maps, label_paths):
        check_values(arr, K, path)

    shapes = [arr.shape for _, arr in label_maps]
    for a, b in zip(shapes, shapes[1:]):
        assert a == b, (args.gt, shapes[0], args.gt2, shapes[1])

    ct, _ = load_ct(args.ct)
    if ct is not None and ct.ndim == 3:
        # A 3D label must sit on the CT grid; a 2D slice is just drawn on top at its
        # own resolution (draw() resamples the background for it).
        for (name, arr), path in zip(label_maps, label_paths):
            if arr.ndim == 3:
                assert ct.shape == arr.shape, (args.ct, ct.shape, path, arr.shape)

    two_d = label_maps[0][1].ndim == 2  # .png slices: one panel per label map
    planes: list[str] = [] if two_d else (args.planes or ["axial"])
    for plane in planes:
        assert plane in PLANE_AXIS, (plane, sorted(PLANE_AXIS))
        axis = PLANE_AXIS[plane]
        for name, arr in label_maps:
            assert 0 <= axis < arr.ndim, (name, plane, arr.shape)

    splits = [build_split(arr, args.split_class) for _, arr in label_maps]

    ncols = (len(label_maps) * args.n) if not two_d else len(label_maps)
    nrows = len(planes) if not two_d else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.2 * nrows), squeeze=False)

    for r, plane in enumerate(planes):
        axis = PLANE_AXIS[plane]
        # Fixed fractions of the axis, not of the patient: --n slices, always the
        # same ones for the same volume shape.
        dim = label_maps[0][1].shape[axis]
        idx = np.unique((np.linspace(PLANE_FRACTIONS[0], PLANE_FRACTIONS[1], args.n) * (dim - 1)).astype(int))
        for g, (name, arr) in enumerate(label_maps):
            # The title says which classes the file holds: the whole point of the
            # figure is to see "missing []" or "missing [4]" at a glance.
            title = figure_title(arr, label_paths[g].stem, K)
            for c, i in enumerate(idx):
                col = g * args.n + c
                ax = axes[r, col]
                draw(ax, ct, arr, plane, int(i), args, K, args.split_class,
                     splits[g], f"{title}\n{plane} {i}" if r == 0 else f"{plane} {i}")
                if col == 0:
                    ax.set_ylabel(plane, fontsize=8)
                if r == 0 and c == 0:
                    ax.legend(handles=legend_handles(arr, K, class_names, args.split_class, splits[g]),
                              fontsize=6, loc="lower left", framealpha=0.6)

    if two_d:
        for g, (name, arr) in enumerate(label_maps):
            ax = axes[0, g]
            index = png_slice_index(label_paths[g], ct)
            draw(ax, ct, arr, "axial", index, args, K, args.split_class, splits[g],
                 figure_title(arr, label_paths[g].stem, K))
            ax.legend(handles=legend_handles(arr, K, class_names, args.split_class, splits[g]),
                      fontsize=6, loc="lower left", framealpha=0.6)

    if len(label_maps) > 1 and not two_d:  # column-group headers
        for g, (name, _) in enumerate(label_maps):
            x = (g * args.n + args.n / 2) / ncols
            fig.text(x, 0.99, name, ha="center", va="top", fontsize=12)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Wrote {args.out}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Draw CT slices with label contours over them (a way to see a label file)')
    parser.add_argument('--ct', type=Path, default=None,
                        help="CT volume (.nii.gz) or image (.png) used as the greyscale background")
    parser.add_argument('--gt', type=Path, default=None,
                        help="Label map (.nii.gz or .png) to draw; required")
    parser.add_argument('--gt2', type=Path, default=None,
                        help="Second label map drawn on the same slices, in its own columns")
    parser.add_argument('--out', type=Path, required=True, help="Output .png figure")
    parser.add_argument('--planes', type=str, nargs='+', default=["axial", "coronal", "sagittal"],
                        help="Planes to draw, one row each: axial (axis 2), coronal (1), sagittal (0)")
    parser.add_argument('--n', type=int, default=4,
                        help="Slices per row, at fixed fractions linspace(0.3, 0.7, n) of that axis")
    parser.add_argument('--window', type=float, nargs=2, default=[-200.0, 400.0],
                        help="Hounsfield window for the background, low high")
    parser.add_argument('--crop', type=int, default=0,
                        help="Centre-crop N pixels per side, like viewer/viewer.py (128 at 512 keeps "
                             "only the mediastinum)")
    parser.add_argument('--split_class', type=int, default=1,
                        help="Class drawn per connected component when it holds several "
                             "(tab10, legend C.a, C.b, ...); a single component keeps its colour")
    parser.add_argument('--K', type=int, default=5, help="Number of classes, background included")
    parser.add_argument('--class_names', type=str, nargs='+', default=None,
                        help="K names, background first (default: background esophagus heart trachea aorta)")

    args = parser.parse_args()
    print(args)

    return args


if __name__ == "__main__":
    main(get_args())