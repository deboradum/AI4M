AI for Medical Imaging · UvA · Fall 2026 · group project
SegTHOR Project Plan
A segmentation pipeline for esophagus, heart, trachea and (once part-2 data lands) aorta on thoracic CT, that measurably beats the course baseline, with a reason for every change and its cost in training time, inference time and code.

Esophagus
0.49
3D Dice, baseline
Heart
0.81
3D Dice, baseline
Trachea
0.55
3D Dice, baseline
Aorta
n/a
no labels in part 1
E000, untouched course code. ENet (8 kernels), cross-entropy, 256×256 slices, 25 epochs in 12.5 min on one Quadro RTX 6000. Mean over validation patients 01, 11, 15, 17, 19, computed at 256×256. The 0.828 the training log prints is inflated by empty-vs-empty slices and a phantom aorta class scoring 1.0.

Principles
Phase 0 · Infrastructure
Phase 1 · Cheap 2D wins
Phase 2 · 3D context
Phase 3 · Pretrained & foundation models
Report & submission
Ownership
Principles for every run
One change per run, always against a named parent row in EXPERIMENTS.md. The row is written before the run starts.
Hypothesis first. State the expected effect and the reason, then check whether the reasoning held.
3D metrics on the original 512×512×Z grid are the only numbers that count: Dice plus a boundary metric (HD95 or NSD), per patient, using the CT header spacing. The GT headers carry an identity affine and must never be used for spacing.
Cost is a result. Training time, inference time per patient, parameter count. A +0.01 Dice for 10× compute is a finding, not a win.
Explain with evidence: per-class and per-patient deltas plus a viewer screenshot of a slice where the change is visible. "It scored higher" is not an explanation.
Eight GPUs means up to eight experiments in parallel with CUDA_VISIBLE_DEVICES=n. Use them.
Phase 0Infrastructure week 1 · must land before any experiment counts
Pure coding work, no research. Split it across the group.

0.1 Own slicing script (assignment 01)
Rewrite slice_segthor.py from scratch, keeping the output layout so main.py still works.

Read spacing and affine from the CT header.
Add --window LOW HIGH for Hounsfield clipping. The baseline does a global min-max per patient, so a patient with a 3071 HU implant gets its soft tissue squeezed into a narrower band than everyone else.
Add --keep_res (no resize) and --crop (center crop on the mediastinum).
Write split.json next to spacing.pkl so the fold is versioned.
Handle the test set, which has no GT.
0.2 Own stitching script (assignment 03)
Must accept volumes with absent classes. The provided script asserts all five and currently fails on every patient.
Use the CT affine and header so volumes overlay correctly in 3D Slicer.
Nearest-neighbour resize back to 512×512 (or undo the crop). Labels 0..4 as uint8, not ×63.
Generic over class count and filename regex. Works for val and test.
0.3 3D metrics script
python metrics.py --pred volumes/E007 --gt "data/segthor_part1/train/{id}/GT.nii.gz" \
 --spacing_from "data/segthor_part1/train/{id}/{id}.nii.gz" --dest results/E007/metrics
Per patient, per class: Dice, HD95 (mm), ASSD (mm), NSD at 2 mm tolerance. These follow the Metrics Reloaded recommendations for semantic segmentation; be ready to cite it.
Class absent in both GT and prediction: NaN, never 1.0.
Writes the submission format: one .npz per metric mapping Patient_XX to a K-length array.
Prints a per-patient table plus a mean row that pastes straight into the log.
First use is E001: re-evaluate E000 on the original grid. That row becomes the official baseline.
0.4 Make main.py configurable and reproducible
Flags for learning rate, batch size, epochs, --loss {ce,wce,dice,ce_dice,focal}, --model {enet,enet_large,unet2d,…}, --aug, --num_workers, --seed, --scheduler, --in_slices for 2.5D, --exp_id.
Dump config.json with all arguments, the git commit and the class list into the run folder.
Seed Python, NumPy, torch and the DataLoader workers.
Save last.pt every epoch so a run can resume. Log wall-clock per epoch.
Keep the 2D Dice printout, add "Dice on non-empty slices", and exclude class 4 from the model-selection mean while it is absent.
0.5 Inference script
infer.py --weights results/E007/bestweights.pt --split val|test --dest volumes/E007 runs the whole chain: slice if needed, forward pass, stitch. Needed for the test submission and for measuring inference time per patient honestly.

0.6 Experiment hygiene
EXPERIMENTS.md is the source of truth. Row first, run second.
Run folders are named by experiment ID: results/E007_wce/.
Results stay out of git, but commit config.json copies under experiments/E007/ so graders can see them in the bundle.
Phase 1Cheap improvements to the 2D pipeline weeks 2–3
Each item is a 15-minute run. Run them as a controlled ablation off E001. Expected effects are written down so the reasoning can be checked, not just the score.

# Change Why it should help Cost What to check

1.1 Class-weighted CE (inverse frequency, capped) Foreground is under 3 % of pixels. Plain CE predicted no esophagus for 9 epochs and no trachea for 13. Weighting removes the collapse. none Do small classes appear from epoch 1? Does background precision drop?
1.2 CE + soft Dice loss Optimises overlap directly and is invariant to class size. The nnU-Net default. none Compare HD95, not just Dice. Dice loss can give blobby edges.
1.3 Focal loss Down-weights easy background pixels. none Usually close to 1.1. Run only if 1.1 and 1.2 disagree.
1.4 HU windowing: [−1000, 1000] and soft tissue [−200, 300] A fixed physical window gives every patient the same contrast. The esophagus is soft tissue and is being crushed by the global min-max. reslice, minutes Esophagus Dice specifically.
1.5 Augmentation: rotation ±15°, scale 0.9–1.1, intensity shift, light elastic. No horizontal flip. Train Dice 0.95 vs val 0.82 by epoch 16 is overfitting on 15 patients. Flips are excluded because the thorax is not left-right symmetric: a flipped image is anatomically wrong. ~1.3× per epoch Train/val gap. Does 50 epochs with aug beat 25 without?
1.6 Longer schedule: 50 and 100 epochs, cosine LR decay Val loss was still noisy. A decaying LR usually adds a point or two for free. 2–4× Keep only if the best epoch moves late.
1.7 Largest connected component per class, in 3D Each organ is a single connected structure. Stray false positives cost Dice and wreck HD95. Zero training cost, fully explainable. inference only HD95 should drop sharply. Check Patient_19 trachea (3D Dice 0.08).
1.8 2.5D input: slices z−1, z, z+1 as three channels Trachea and esophagus are tubes along z. A single slice cannot tell a thin dark circle from noise. Adjacent slices give context at 2D cost. ~1× Trachea Dice and z-continuity in the 3D viewer.
1.9 Resolution: full 512, or a 256 crop around the mediastinum At 256 the trachea is a few pixels wide and nearest-neighbour label resize erodes it. A crop keeps resolution without 4× compute. 512: 4× · crop: 1× Trachea and esophagus.
1.10 Capacity: ENet at paper size (16 kernels, factor 4); a plain 2D U-Net The baseline ENet is deliberately shrunk. 2–4× Does capacity help once loss and aug are fixed, or just overfit?
1.11 Test-time augmentation, seed ensembling Cheap variance reduction. inference × N Report inference time.
Order: 1.1 and 1.2 first (largest expected gain), then 1.4, then 1.5. Combine the winners into a "best 2D" run and evaluate it on all four folds. That model is the reference every later phase must beat.

Deliverable: an ablation table (each change alone, then cumulative), a training-curve figure showing the class collapse disappearing, two viewer figures.

Phase 23D context weeks 4–5
2.1 Patch-based 3D U-Net (MONAI DynUNet or UNet)
New dataset class that loads NIfTI directly, resamples to a common spacing such as 1.0 × 1.0 × 2.5 mm, applies the HU window, and samples patches around 160×160×64 with foreground oversampling.
Sliding-window inference with Gaussian blending, which MONAI provides.
Why: organs are continuous along z and a 2D model has no way to enforce that. Expect the gain on trachea and esophagus continuity and on HD95 more than on heart Dice.
Cost: minutes per epoch on a 24 GB card at these patch sizes. Inference is seconds per patient. Report both.
Compare against the best 2D model and against 2.5D. If 2.5D gets most of the gain at a fraction of the cost, that is the more interesting finding.
2.2 nnU-Net v2 as a reference ceiling
Run once with defaults, 2D and 3D full-res configurations. It self-configures preprocessing, patch size, augmentation and CE + Dice loss, which is exactly the recipe Phase 1 assembles by hand.
Use it as an upper bound and a sanity check, not as "our model". Its default 1000 epochs is many hours; cap it and say so.
Explanation angle: which of its automatic decisions match ours, and which differ.
2.3 Cross-validation
Four folds of five patients via the slicer's --fold. Report mean ± std across folds for the final 2D and 3D models. With five validation patients a single split is too noisy to rank models that differ by one or two points, and the fixed split already holds two of the three unusual-spacing patients (11 and 15).

Phase 3Pretrained and foundation models weeks 5–7 · pick one or two
Only worth doing once Phases 1 and 2 are logged and explained. Each is a study with a question, not just a run.

3.1 Pretrained 2D encoder (segmentation_models_pytorch)
U-Net with an ImageNet-pretrained ResNet34 or EfficientNet encoder, the single CT channel replicated to three. Question: does natural-image pretraining help with 15 training CTs? Same pipeline as Phase 1, so it is cheap. Compare convergence speed as well as final Dice.

3.2 TotalSegmentator study
TotalSegmentator is an nnU-Net trained on over a thousand CTs and already outputs esophagus, trachea, heart and aorta.

Zero-shot: run it on the validation CTs, map its labels to ours, score with our metrics. This is the ceiling a large-data model reaches on this annotation style.
Distil: use its predictions as pseudo-labels for the aorta we lack, or as an auxiliary target. Question: does training on our 15 patients plus pseudo-labels beat our labels alone?
Check the weights licence and confirm with course staff that external pretrained models are acceptable before building on it.
3.3 MedSAM and the SAM family
MedSAM is SAM (ViT-B) fine-tuned on 1.5 M medical image-mask pairs. It is prompt-based, one bounding box per object per slice, on 1024×1024 RGB input, so it is not a drop-in for an automatic pipeline. Three honest ways to use it:

Oracle upper bound: prompt with GT boxes. Not a usable system, but it shows how good the mask decoder is on these organs.
Automatic prompts: derive boxes from our own best model, then let MedSAM refine the mask. Question: does refinement improve HD95 over our model alone, and at what cost? Roughly a second per slice on GPU, so minutes per patient.
Fine-tune the mask decoder on our slices with box prompts. Small compute, but still needs a box at inference.
SAM-Med3D and SegVol are the 3D variants if volume prompts become interesting. Worth a paragraph in the report either way.

3.4 Transformer 3D nets (MONAI UNETR, SwinUNETR)
Pretrained SwinUNETR weights exist for CT. Heavier than a 3D U-Net. Only interesting if the 3D U-Net plateaus and we can show the gain is not just parameter count.

3.5 Weak and partial supervision
The baseline already has --mode partial, which leaves the heart unsupervised. When part-2 data arrives with aorta labels, a natural study is to train with one organ unlabelled and recover it with pseudo-labels or a foundation model. It ties the project to the course material and is cheap once the pipeline exists.

What the final report needs from each phase
One results table: rows are models, columns are per-class Dice and HD95 as mean ± std over folds, parameter count, training time, inference time per patient.
For every improvement we keep: the hypothesis, the ablation row that confirms it, one figure.
For every idea we drop: the row that shows it did not help and one sentence on why.
Submission checklist from the course readme, all inside group-XX/: git bundle, bestmodel-group-XX.pkl, test predictions as NIfTI, validation predictions and validation GT as NIfTI, one .npz per metric.

Suggested ownership
Adjust to group size. Each area owns its Phase 0 item first.

Area Deliverables
Data 0.1 slicing · 1.4 windowing · 1.9 resolution · 2.1 3D dataset
Evaluation 0.2 stitching · 0.3 metrics · 0.5 inference · 1.7 post-processing · 2.3 cross-validation
Training 0.4 main.py refactor · 1.1–1.3 losses · 1.5–1.6 augmentation and schedule · 1.8 2.5D
Models 1.10 U-Net · 2.1 3D U-Net · 2.2 nnU-Net · Phase 3
Legend for class colours used above: esophagus (label 1) · heart (2) · trachea (3) · aorta (4). Mirrors PLAN.md and EXPERIMENTS.md in the repo.
