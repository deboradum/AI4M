# SegTHOR aorta label: what happened, and how the label was recovered

## The problem in one paragraph

Every experiment in this repository (E001-E016) trained on a ground truth in which the aorta has no class of its own. The archive the group holds, `data/segthor_part1.zip` (sha256 `6c203831b5554245acfb50fa0c415179cbe7f1d342a82a6a400d5e2ab0e2088c`, all 63 entries dated 2026-09-02), ships 20 patients under `data/segthor_part1/train/Patient_01..20/`, each with a CT (`<id>.nii.gz`, int32, e.g. 512x512x229, spacing 0.9765625/0.9765625/2.0 mm) and a repackaged `GT.nii.gz` (uint8, identity affine, zooms (1,1,1)). Every `GT.nii.gz` contains only the label values {0,1,2,3} - 1 esophagus, 2 heart, 3 trachea - and there is no 4, which is the aorta's class in the original challenge. The aorta is not missing from the images; it is labelled, but folded into class 1, so in this archive class 1 = esophagus union aorta. `data/segthor_part1/train/Patient_07/GT2.nii.gz`, the only extra file in the archive and referenced nowhere in the repository, still separates the two structures; it both proves the fold and provides the single reference pair available for validating a recovery.

## Evidence

| Quantity | Value |
| --- | --- |
| Archive | `data/segthor_part1.zip`, sha256 `6c203831b5554245acfb50fa0c415179cbe7f1d342a82a6a400d5e2ab0e2088c`, all 63 entries dated 2026-09-02 |
| Contents | 20 patients under `data/segthor_part1/train/Patient_01..20/`, each with `<id>.nii.gz` and `GT.nii.gz` |
| CT | int32, e.g. 512x512x229, spacing 0.9765625/0.9765625/2.0 mm |
| GT | uint8, identity affine, zooms (1,1,1) - a repackaged file |
| GT label values | {0,1,2,3} in every file (1 esophagus, 2 heart, 3 trachea); there is no 4 |
| Patient_07 class-1 voxels | 115229 |
| GT2 label values | {0,1,2,3,4}; carries the CT's affine and zooms, i.e. the original challenge-style file |
| Confusion GT (rows) vs GT2 (cols) | diagonal except row 1: 1 -> 1 = 25373, 1 -> 4 = 89856; 25373 + 89856 = 115229 |
| Voxel-identical classes | background 46427909, class 2 = 365861, class 3 = 14777 |
| Aorta (Patient_07, GT2) | 89856 voxels, slices 113-142, mean HU 33.8 |
| Esophagus (Patient_07, GT2) | 25373 voxels, mean HU -2.7 |
| Class-1 topology (Patient_07) | one 3D connected component; the two structures touch |
| Split accuracy vs GT2 | Dice 0.9994 aorta, 0.9980 esophagus |
| Archive checksum history | changed 2026-09-08 (commit `628fa20`), `6c203831...` -> `db9e4bb0...`: the course replaced the archive after this copy was downloaded |
| Local zip vs extracted tree | byte-identical, so the extracted `data/segthor_part1/` is the fold-in version and every experiment E001-E016 trained on it |
| Downstream symptom | configs use `K=5`; `main.py` encodes 5 classes into PNGs with `img / 63` and averages `log_dice_val[e, :, 1:].mean()` for model selection, so the empty class 4 enters the average with Dice exactly 1.0 |
| Class-4 inflation | `results/E014_hu_wide/dice_val.npy`: 0.8631 reported vs 0.8174 over the three classes the GT actually contains, a 0.0457 inflation; class 4 has 0/915 slices with GT |
| 3D tables | `metrics3d.py` returns NaN when a class is absent from both GT and prediction, so the 3D tables show `aorta: n/a` |

## Why this was invisible so far

The fold is invisible to every check the repository performs, and the two checks that could have caught it point the other way.

- `main.py` encodes the five configured classes into PNGs with `img / 63`, so an empty class 4 becomes an all-zero channel rather than an error.
- Model selection averages `log_dice_val[e, :, 1:].mean()`, which includes the empty class 4. Empty-vs-empty scores exactly 1.0, so the phantom class raises the selection metric instead of exposing itself: on `results/E014_hu_wide/dice_val.npy`, 0.8631 reported against 0.8174 over the three classes the GT actually contains, a 0.0457 inflation, with class 4 present in 0/915 validation slices.
- `metrics3d.py` is honest - it returns NaN when a class is absent from both GT and prediction - so the 3D tables show `aorta: n/a`. What is wrong is the reason attached to that `n/a`: the repository notes at `TEAM_WORKFLOW.md:180` and `E016_handoff.md:132` both state "Aorta is absent in segthor_part1". It is not absent; it is merged.
- Reading the GT directly does not contradict the note either: a class-1 file looks like a plausible esophagus label, and only its voxel count (115229 on Patient_07) is anomalous. Without `GT2.nii.gz`, which the repository never references, nothing in the tree separates the two organs.

## Recovery strategy A: geometric split (attempts and outcomes)

| Version | Change | Result |
| --- | --- | --- |
| v1 | 3D distance transform `dt`; aorta seed = largest component of `dt >= 0.6*max(dt)`; esophagus seed = largest component of `skeleton & (dt < otsu(skeleton dt))`; then `skimage.segmentation.watershed(-dt, markers, mask=class1)` | Patient_07: Dice 0.9969 aorta / 0.9893 esophagus (cross-validated). But on Patient_12 the esophagus came out as 54 voxels and on Patient_14 as 55 - the big structure's flood wins and swallows the thin tube. v1 is rejected. |
| v2 | Seed the esophagus from per-slice structure instead: in each axial slice the esophagus appears as its own small component (area <= 300 mm^2) whenever class 1 is not a single blob that slice | Patient_12 gained 24285 esophagus voxels, Patient_14 8216, Patient_08 15030; Patient_07 accuracy unchanged. Two flaws remained: the role assignment (which basin is the aorta) was never checked, and nothing prevented a fat blob from ending up in the esophagus region (Patient_11: esophagus region had a 17.8 mm inscribed radius, thicker than the aorta's 16.7 mm). |
| v3 | Two repairs: (a) role assignment by median per-slice area - whichever of the two regions has the smaller median axial area is the esophagus (the aorta is 2.5x-7.6x larger in every clean case); (b) hard rule that any class-1 voxel whose inscribed ball radius exceeds 10.0 mm is aorta (esophagus diameter is at most about 20 mm) | Fixed Patient_11, but flipped Patient_03 wrongly: Patient_03's aorta is genuinely thin (max inscribed radius 12.7 mm, the smallest of all 20), so fixed thresholds cannot separate its two thin tubes. |
| final | Sweep the thin-component threshold over `[300, 200, 120]` mm^2, take the first value whose result passes four plausibility gates, and report the rest for human review | All 20 patients pass once Patient_03 is re-run at 200 mm^2. Patient_07 accuracy is unchanged at every sweep value (Dice 0.9994 aorta / 0.9980 esophagus). |

## Reproducing this, and checking a label file

Three repository scripts carry this work; all run under `./ai4mi/bin/python`.

```bash
# 1. Write the 4-organ GT (20 patients, gates + Patient_07 Dice 0.9994/0.9980)
./ai4mi/bin/python retrieve_aorta.py --src data/segthor_part1/train --dest data/SEGTHOR_aorta \
    --gt2 data/segthor_part1/train/Patient_07/GT2.nii.gz --figures AORTA_INSPECTION

# 2. Check what a label file holds - the answer to "the aorta label is not there"
./ai4mi/bin/python check_labels.py --nifti data/SEGTHOR_aorta
./ai4mi/bin/python check_labels.py --nifti data/segthor_part1 --components
./ai4mi/bin/python check_labels.py --diff data/segthor_part1/train/Patient_02/GT.nii.gz \
    data/SEGTHOR_aorta/train/Patient_02/GT.nii.gz
./ai4mi/bin/python check_labels.py --dice_npy results/E014_hu_wide/dice_val.npy

# 3. See it
./ai4mi/bin/python visualize_labels.py --ct data/segthor_part1/train/Patient_07/Patient_07.nii.gz \
    --gt data/SEGTHOR_aorta/train/Patient_07/GT.nii.gz \
    --gt2 data/segthor_part1/train/Patient_07/GT2.nii.gz \
    --out /tmp/p07.png --planes axial coronal sagittal --n 4 --crop 128
```

`check_labels.py --nifti data/SEGTHOR_aorta` exits `0` with `class4 PRESENT in 20/20 files (min 43431 voxels)` and `class1 PRESENT in 20/20 files (min 8216 voxels)` - the direct answer to "the aorta label is not there". On the folded tree, `--nifti data/segthor_part1` prints one row per file (`class4=0` for all 20 `GT.nii.gz`, `class4=89856` for the single `GT2.nii.gz`) and, because `GT2` is scanned too, reports `class4 PRESENT in 1/21 files`; add `--components` to see per file that class 1 is one folded component (`class4: absent`). Sliced datasets are the unambiguous case: `--png data/SEGTHOR` exits `1` with `class4: 0/3732 slices` and raw `.png` values `[0, 63, 126, 189]`. `--diff` shows that only class 1 was rewritten (`B=1: A=1 19402`, `B=4: A=1 85951`, every other row diagonal), and `--dice_npy` prints the phantom-class inflation of model selection (`main.py:321` mean 0.8631 against 0.8174 over the classes the GT holds).

`data/SEGTHOR_aorta/` is written with the CT's affine and spacing, holds a `split_report.csv` with the per-patient gates, and symlinks each CT back into `data/segthor_part1/train/`, so `slice_segthor.py --source_dir data/SEGTHOR_aorta --dest_dir …` works without copying ~800 MB.

## The algorithm that works

1. Load class 1 of `GT.nii.gz` as a 3D boolean mask; the CT supplies spacing and HU.
2. Compute the Euclidean distance transform `dt` of that mask, in millimetres.
3. Build the aorta seed: the largest connected component of `dt >= 0.6 * max(dt)`, the round thick core of the folded class.
4. Build the esophagus seed per axial slice: in any slice where class 1 is not a single blob, every component with area <= `thin_area` mm^2 is an esophagus seed. Slices where the two organs touch contribute nothing here.
5. Watershed `-dt` from those two markers, masked to class 1, so every class-1 voxel is assigned to one of the two regions.
6. Repair (a), role assignment: whichever region has the smaller median per-slice area is the esophagus.
7. Repair (b), hard rule: any class-1 voxel whose inscribed ball radius exceeds 10.0 mm is aorta.
8. Sweep `thin_area` over `[300, 200, 120]` mm^2 and keep the first result that passes all four gates in the next section; anything that fails is reported for human review rather than shipped.

```python
def split_class1(gt, thin_area, spacing):
    c1 = (gt == 1)
    dt = distance_transform_edt(c1, sampling=spacing)          # millimetres

    aorta_seed = largest_component(dt >= 0.6 * dt.max())

    eso_seed = zeros_like(c1)
    for z in range(c1.shape[2]):
        comps = label_components(c1[:, :, z])                  # 2D, per axial slice
        if len(comps) > 1:                                     # organs apart in this slice
            for comp in comps:
                if area_mm2(comp, spacing) <= thin_area:
                    eso_seed[:, :, z] |= comp

    markers = zeros(c1.shape, int8)
    markers[aorta_seed] = 1
    markers[eso_seed] = 2
    regions = watershed(-dt, markers=markers, mask=c1)         # skimage.segmentation.watershed
    aorta, eso = regions == 1, regions == 2

    # (a) role assignment: the smaller median axial area is the esophagus
    if median_slice_area(eso, spacing) > median_slice_area(aorta, spacing):
        aorta, eso = eso, aorta

    # (b) hard rule: anything fat is aorta (the esophagus is at most ~20 mm across)
    eso = eso & (dt <= 10.0)                                   # millimetres
    aorta = c1 & ~eso

    return aorta, eso


def recover_aorta(gt, spacing):
    for thin_area in (300, 200, 120):                          # mm^2
        aorta, eso = split_class1(gt, thin_area, spacing)
        if passes_gates(aorta, eso, spacing):
            return aorta, eso, thin_area
    flag_for_human_review(gt)                                  # never shipped silently
```

## Per-patient result and plausibility gates

The four gates are: aorta voxels in [35000, 260000]; esophagus voxels in [5000, 40000]; median aorta slice area >= 1.3 x median esophagus slice area; esophagus median slice area in [80, 350] mm^2.

| Patient | thin_area (mm^2) | Aorta voxels | Esophagus voxels | Median aorta slice (mm^2) | Median esophagus slice (mm^2) | Ratio | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Patient_01 | 300 | 106556 | 32372 | 606 | 234 | 2.6 | ok |
| Patient_02 | 300 | 85951 | 19402 | 513 | 195 | 2.6 | ok |
| Patient_03 | 200 | 43737 | 9517 | 261 | 152 | 1.7 | ok (thin aorta) |
| Patient_04 | 300 | 149574 | 18782 | 1117 | 155 | 7.2 | ok |
| Patient_05 | 300 | 144062 | 15964 | 562 | 220 | 2.5 | ok |
| Patient_06 | 300 | 119931 | 19061 | 841 | 176 | 4.8 | ok |
| Patient_07 | 300 | 89891 | 25338 | 501 | 145 | 3.5 | ok |
| Patient_08 | 300 | 84543 | 15030 | 628 | 138 | 4.5 | ok |
| Patient_09 | 300 | 59517 | 16086 | 432 | 152 | 2.8 | ok |
| Patient_10 | 300 | 108732 | 27236 | 560 | 187 | 3.0 | ok |
| Patient_11 | 300 | 46814 | 19744 | 549 | 185 | 3.0 | ok |
| Patient_12 | 300 | 149731 | 24285 | 718 | 192 | 3.7 | ok |
| Patient_13 | 300 | 133173 | 23687 | 807 | 189 | 4.3 | ok |
| Patient_14 | 300 | 80230 | 8216 | 530 | 125 | 4.2 | ok |
| Patient_15 | 300 | 250470 | 9459 | 1384 | 183 | 7.6 | ok (outlier aorta) |
| Patient_16 | 300 | 68521 | 17686 | 499 | 185 | 2.7 | ok |
| Patient_17 | 300 | 58507 | 13112 | 436 | 122 | 3.6 | ok |
| Patient_18 | 300 | 97062 | 24184 | 591 | 195 | 3.0 | ok |
| Patient_19 | 300 | 76766 | 11034 | 417 | 143 | 2.9 | ok |
| Patient_20 | 300 | 43431 | 10519 | 566 | 224 | 2.5 | ok |

## What is still uncertain

- Accuracy is measured on one patient only. Only Patient_07 has reference labels (`GT2.nii.gz`), so Dice 0.9994 / 0.9980 is a single-patient number; the other 19 are validated by the geometry gates plus visual review, not by Dice.
- Patient_03: ratio 1.7 against 2.5-7.6 for everyone else, with a genuinely thin aorta; the seed thresholds barely separate its two tubes. Lower confidence than the rest.
- Patient_15: aorta 250470 voxels and median axial area 1384 mm^2 (about 42 mm across) - an outlier. Either a genuinely wide aorta or another structure inside the folded class 1; it needs a look at the rendered slice.
- Eight of twenty patients have a per-slice second component for part of their length only; for the slices where the two organs touch, the split is decided by the watershed, not by direct observation.
- 17 of 20 patients have class 1 as a single 3D connected component, so 3D connected-component analysis alone cannot separate the organs (it does resolve Patient_02, 09 and 16 exactly).
- The recovered labels are derived, not annotated.
- Whether the course's replacement archive (checksum `db9e4bb0...`) ships a real aorta class is still unchecked: the Sharepoint link at `readme.md:126` answers `HTTP/2 401` with `x-msdavext_error: 917656; Toegang geweigerd` ("access denied: browse to the website first and select automatic sign-in"), i.e. 0 bytes without an authenticated UvA browser session. Re-run `check_labels.py --nifti <extracted replacement> --components` from a logged-in browser download to settle it; if that archive turns out to have a populated class 4, it beats these derived labels and this recovery becomes only a validator.

## Strategies considered and not used

- 3D connected components of class 1: only works for the 3 patients where the organs do not touch.
- Per-slice component tracking across z (assign the small component to the esophagus, propagate through the slices where they merge): more moving parts than the seed + watershed rule, and no reference to validate it beyond Patient_07.
- TotalSegmentator or another pretrained model to produce aorta pseudo-labels (`full_plan.md` section 3.2): a real alternative, but it answers a different research question (pseudo-labelling), needs its own licence check, and does not recover the annotation already sitting in the file.
- Taking the aorta from `Patient_07/GT2.nii.gz` alone: one patient is not a training set.
- Manual delineation in 3D Slicer: accurate, but 20 patients of hand contouring, and the patient count makes it unaffordable as a first pass.
- Waiting for the course's part-2 archive: pinned in `full_plan.md` as the plan for the aorta; this work supersedes that as far as a recoverable label goes.

## Figures

Rendered galleries for this recovery, written next to this file into `AORTA_INSPECTION/`.

| File | Content |
| --- | --- |
| `00_overview_all_patients.png` | one axial slice per patient, all 20 |
| `<Patient_XX>_axial_merged_vs_split.png` (20 files) | 2 rows: merged class-1 contour on top, recovered yellow-aorta and red-esophagus contours below |
| `Patient_07_axial_vs_GT2.png` | 3 rows: merged, recovered split, reference GT2 |
| `Patient_07_3d.png` | 3D rendering for the reference patient |
| `<Patient_XX>_3d.png` for Patient_03, Patient_12, Patient_14, Patient_15 | 3D renderings for the four cases flagged above |
