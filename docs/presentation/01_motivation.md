# 1. Motivation: why these changes and not others

Rubric: *"Why did you try specific improvements? What motivated those ones and not
others?"* Every change below starts from something we **observed** in the data
or in a model's errors. Each one comes with the alternatives we considered and
why we did not pick them. Cohort numbers are measured from the NIfTI files
(`configs/splits/segthor_full_32_8.json`, all 40 training patients).

## What the data looks like

| Property | Measured |
|---|---|
| Scans | 40 labelled (32 train / 8 val, stratified split) + 20 unlabelled test |
| Volume | 512 x 512 in-plane, 147-284 slices |
| Spacing | in-plane 0.896-1.367 mm (varies ~1.5x between patients), slices 2.0-2.5 mm |
| Contrast agent | 12 / 40 scans (aorta median > 100 HU) |
| Organ volume (ml) | heart 438-1829, aorta 95-465, esophagus 31-170, trachea 22-73 |
| GT header | identity affine (1 x 1 x 1 mm), wrong; metrics take spacing from the CT |

## Decision chain

Each step answers the problem the previous one exposed.

| # | Observation | Change | Why this and not an alternative |
|---|---|---|---|
| 1 | Per-patient min-max normalisation gives each scan a different contrast, and 12/40 scans have contrast agent | **Fixed HU window** -1000..1000 (team, E_F01) | A soft-tissue window (-200..300) saturates the air-filled trachea: E014 trachea 0.614 vs 0.653 with the wide window. E_F01 0.795 vs E_F02 (-1000..600) 0.784 foreground Dice |
| 2 | In-plane spacing varies 1.5x, so the same organ is a different number of pixels in each patient | **Resample to 1.0 x 1.0 x 2.5 mm** (team, E_F03) | Scale augmentation could cover some of it, but resampling removes the variation at the source. E_F03 0.801, best single preprocessing change. Combined with the window for all final runs |
| 3 | After the input fixes, the weak organs are the **esophagus (0.52)** and **trachea (0.74)**. Both are thin tubes along z, hard to see in one axial slice and continuous across slices | **3D U-Net** (E_F07) | *2.5D* (3 slices) gives only ±2.5 mm of context and was worse on ENet (E008); E_F05 re-tests it with the U-Net. *A bigger 2D network* (E010 large ENet) gained only +0.02. *Transformers* need more than 32 training volumes or pre-training. *Ensembling* is discouraged by the brief. *nnU-Net* is forbidden |
| 4 | 3D needs the voxels to mean the same physical size in every patient, or a 3x3x3 kernel covers different anatomy each time | 3D trained on the **resampled** data | This is also why resampling matters more for 3D than for the 2D runs |
| 5 | A full volume does not fit a GPU batch | Patches of 256 x 256 x 80 (full slice width, 200 mm along z) + sliding-window inference | A smaller in-plane patch would cut organs at the side; 80 slices covers most of the thorax |
| 6 | Forcing 1/3 of patches to contain an organ, by drawing a random organ voxel, mostly picks the heart (it has ~20x the esophagus's voxels) | **Draw the organ class first**, then a voxel | Gives each organ an equal share of forced patches; idea from nnU-Net, cited |
| 7 | A 3D batch holds only 2 patches, too few for BatchNorm statistics | InstanceNorm | Standard for small-batch 3D |
| 8 | Augmentation: the thorax is not left-right symmetric; elastic deformation collapsed the thin trachea (E015: trachea Dice 0.081) | Rotation, scale, intensity only; **no flips, no elastic** | Flips would teach anatomy that does not exist; elastic was measured to hurt (E016) |
| 9 | The 3D model's worst errors: a heart predicted far below the heart (Patient_37, HD95 160 mm) and a stray trachea piece (Patient_17, 103 mm) | **Keep the largest connected component for heart and trachea** | Not for esophagus/aorta: those are tubes that break into pieces, and "largest piece" deletes real parts beyond a gap (team E021 measured it) |
| 10 | Leftover aorta pieces: on validation all under 0.4 ml; real fragments across gaps are 9-21 ml | **Remove aorta pieces under 1 ml** (size in ml, per scan) | A distance rule was tried first and failed: false blobs sit at every distance (3-143 mm). A voxel count is not the same size from scan to scan |

## Considered and not done

| Option | Why not |
|---|---|
| Native 512 x 512 resolution | 4x the compute; the team's E009 (512 on ENet) was worse but confounded by other changes |
| Topology-aware loss (Skeleton Recall, clDice) | Directly targets the remaining failure (gaps in tubes) but needs another 7 h run; listed as the next step, see [05](05_critical_analysis.md) |
| 5-fold cross-validation | 5 x 7 h of GPU; splits prepared (`scripts/make_folds.py`), not run |
| Test-time flipping | The network never saw flipped anatomy (no flips in training), so flipped inputs are out of distribution |
| Removing small esophagus pieces | Measured: they sit 5-13 mm from the true esophagus inside gaps; removing them raised HD95. See [04](04_postprocessing.md) |
