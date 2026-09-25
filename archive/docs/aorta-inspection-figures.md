# AORTA_INSPECTION - recovered aorta for SegTHOR part-1

`GT.nii.gz` in `data/segthor_part1/train/Patient_XX/` labels the aorta, but folded into class 1, so class 1 = esophagus UNION aorta. These figures show the folded label and the recovered split side by side. Read `aorta-findings.md` in this folder for the full evidence, the algorithm and the per-patient table.

Validation: the only patient with reference labels is Patient_07 (`GT2.nii.gz`, which separates the two organs); the recovery scores Dice 0.9994 aorta / 0.9980 esophagus against it. The other 19 patients have no reference labels and are judged by eye plus four geometric plausibility gates.

Colour key: yellow = aorta, red = esophagus; in the 3D merged panels the whole folded class 1 is gray.

## Figures

- `00_overview_all_patients.png` - One axial slice per patient, all 20, recovered contours only. Fastest sanity pass: yellow must follow the round vessel lateral to the spine, red must stay a thin tube.
- `Patient_01_axial_merged_vs_split.png` - Top row = folded class 1 as one contour; bottom row = recovered split. thin_area=300 mm2, aorta=106556 vx, esophagus=32372 vx.
- `Patient_02_axial_merged_vs_split.png` - Top row = folded class 1 as one contour; bottom row = recovered split. thin_area=300 mm2, aorta=85951 vx, esophagus=19402 vx.
- `Patient_03_3d.png` - 3D surfaces, merged (gray) vs recovered split (yellow + red). aorta=43737 vx, esophagus=9517 vx.
- `Patient_03_axial_merged_vs_split.png` - Top row = folded class 1 as one contour; bottom row = recovered split. thin_area=200 mm2, aorta=43737 vx, esophagus=9517 vx.
- `Patient_04_axial_merged_vs_split.png` - Top row = folded class 1 as one contour; bottom row = recovered split. thin_area=300 mm2, aorta=149574 vx, esophagus=18782 vx.
- `Patient_05_axial_merged_vs_split.png` - Top row = folded class 1 as one contour; bottom row = recovered split. thin_area=300 mm2, aorta=144062 vx, esophagus=15964 vx.
- `Patient_06_axial_merged_vs_split.png` - Top row = folded class 1 as one contour; bottom row = recovered split. thin_area=300 mm2, aorta=119931 vx, esophagus=19061 vx.
- `Patient_07_3d.png` - 3D surfaces: folded class 1 (gray), recovered split, reference GT2.
- `Patient_07_axial_merged_vs_split.png` - Top row = folded class 1 as one contour; bottom row = recovered split. thin_area=300 mm2, aorta=89891 vx, esophagus=25338 vx.
- `Patient_07_axial_vs_GT2.png` - The validation figure. Three rows on the same five slices: folded class 1, recovered split, reference GT2. Row 2 and row 3 should be the same picture.
- `Patient_08_axial_merged_vs_split.png` - Top row = folded class 1 as one contour; bottom row = recovered split. thin_area=300 mm2, aorta=84543 vx, esophagus=15030 vx.
- `Patient_09_axial_merged_vs_split.png` - Top row = folded class 1 as one contour; bottom row = recovered split. thin_area=300 mm2, aorta=59517 vx, esophagus=16086 vx.
- `Patient_10_axial_merged_vs_split.png` - Top row = folded class 1 as one contour; bottom row = recovered split. thin_area=300 mm2, aorta=108732 vx, esophagus=27236 vx.
- `Patient_11_axial_merged_vs_split.png` - Top row = folded class 1 as one contour; bottom row = recovered split. thin_area=300 mm2, aorta=46814 vx, esophagus=19744 vx.
- `Patient_12_3d.png` - 3D surfaces, merged (gray) vs recovered split (yellow + red). aorta=149731 vx, esophagus=24285 vx.
- `Patient_12_axial_merged_vs_split.png` - Top row = folded class 1 as one contour; bottom row = recovered split. thin_area=300 mm2, aorta=149731 vx, esophagus=24285 vx.
- `Patient_13_axial_merged_vs_split.png` - Top row = folded class 1 as one contour; bottom row = recovered split. thin_area=300 mm2, aorta=133173 vx, esophagus=23687 vx.
- `Patient_14_3d.png` - 3D surfaces, merged (gray) vs recovered split (yellow + red). aorta=80230 vx, esophagus=8216 vx.
- `Patient_14_axial_merged_vs_split.png` - Top row = folded class 1 as one contour; bottom row = recovered split. thin_area=300 mm2, aorta=80230 vx, esophagus=8216 vx.
- `Patient_15_3d.png` - 3D surfaces, merged (gray) vs recovered split (yellow + red). aorta=250470 vx, esophagus=9459 vx.
- `Patient_15_axial_merged_vs_split.png` - Top row = folded class 1 as one contour; bottom row = recovered split. thin_area=300 mm2, aorta=250470 vx, esophagus=9459 vx.
- `Patient_16_axial_merged_vs_split.png` - Top row = folded class 1 as one contour; bottom row = recovered split. thin_area=300 mm2, aorta=68521 vx, esophagus=17686 vx.
- `Patient_17_axial_merged_vs_split.png` - Top row = folded class 1 as one contour; bottom row = recovered split. thin_area=300 mm2, aorta=58507 vx, esophagus=13112 vx.
- `Patient_18_axial_merged_vs_split.png` - Top row = folded class 1 as one contour; bottom row = recovered split. thin_area=300 mm2, aorta=97062 vx, esophagus=24184 vx.
- `Patient_19_axial_merged_vs_split.png` - Top row = folded class 1 as one contour; bottom row = recovered split. thin_area=300 mm2, aorta=76766 vx, esophagus=11034 vx.
- `Patient_20_axial_merged_vs_split.png` - Top row = folded class 1 as one contour; bottom row = recovered split. thin_area=300 mm2, aorta=43431 vx, esophagus=10519 vx.

## Things to check

- Does the yellow contour follow the round, contrast-bright vessel next to the spine, and does it never wrap the thin tube?
- Does the red contour stay a thin tube along the whole mediastinum, with no fat blob in it?
- Patient_03: thinnest aorta of the 20 (median axial area 261 mm^2, aorta/esophagus area ratio 1.7 against 2.5-7.6 elsewhere).
- Patient_15: widest aorta (median axial area 1384 mm^2, about 42 mm across, 250470 voxels) and the esophagus is visible in only part of the shared z-range - an outlier worth a look.
- Patient_12 and Patient_14: their esophagus nearly disappears if you only look at 3D connected components, which is what the first attempt got wrong.

## Where these files live

These figures sit in `<repo>/AORTA_INSPECTION/`, next to the findings document, and the folder is listed in `.gitignore`: they are inspection artifacts, not part of the submission bundle. `retrieve_aorta.py --figures AORTA_INSPECTION` regenerates the axial montages, the overview and the Patient_07 GT2 comparison from the repository script, so none of them needs this session to be rebuilt. The 3D renderings were made by hand during the investigation and are not regenerated by the script.
