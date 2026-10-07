# 2. Method: the 3D U-Net (E_F07)

Goal: isolate the effect of going from 2D/2.5D to 3D. So E_F07 keeps everything
E_F05/E_F06 use that carries over to 3D, and changes only what 3D forces.

## Kept from E_F05/E_F06

| | Value |
|---|---|
| Data | `SEGTHOR_FULL_huwide_resampled`: HU window -1000..1000, resampled to 1.0 x 1.0 x 2.5 mm, 256 x 256 in-plane, same stratified 32/8 split |
| Loss | cross-entropy + soft Dice over all 5 classes (same formula as `CEDiceLoss`) |
| Optimizer | AdamW, lr 5e-4, weight decay 1e-4, cosine decay to 0 |
| Augmentation | rotation +-15 deg, scale 0.9-1.1, intensity shift +-0.1, no elastic, **no flips** (the thorax is not left-right symmetric) |
| Model selection | checkpoint with the best validation Dice |
| Evaluation | the same `stitch.py` + `metrics3d.py` path, on the original CT grid |

The volumes are built by stacking the exact 2D slices the 2D runs train on, and
3D predictions are written back as slices, so preprocessing and evaluation are
shared code, not re-implementations.

## Changed because the model is 3D

| | Value | Why |
|---|---|---|
| Model | 3D U-Net, 4 levels, base 32 channels (capped at 320), 3x3x3 convs, InstanceNorm, PReLU; 16.5 M parameters | Same layout as the team's 2D U-Net; InstanceNorm because a 3D batch holds only 2 patches |
| Input | random 256 x 256 x 80 patches (full slice width, 80 slices = 200 mm) | A full volume does not fit; the full in-plane view keeps all organs in context |
| Batch / epoch | 2 patches; an "epoch" = 250 iterations; 300 epochs, 5 warm-up | Slice epochs don't exist in 3D |
| Patch sampling | 1/3 of patches forced to contain an organ, **organ class drawn first, then a voxel** | Drawing a voxel directly follows voxel counts, so the heart (the largest organ) got most forced patches; class-first gives each organ an equal share |
| Precision | bf16 autocast, loss in fp32 | ~2x faster; softmax/log are not bf16-safe |
| Validation | every 5 epochs, sliding window (50 % overlap, Gaussian weighting) over whole volumes, 3D Dice | Selects the checkpoint on the actual 3D task |

Class-first sampling, Gaussian sliding-window weighting, InstanceNorm and the
320-channel cap are ideas from nnU-Net (Isensee et al. 2021). The course forbids
the nnU-Net framework; none of its code is used, and the slides should cite the
ideas.

## Engineering that made it practical

- **Speed.** 3D trains at ~0.3 s per step (78 s per epoch on an A100). It
  processes ~17x more voxels per second than the 2D pipeline. We did not profile
  the 2D pipeline; the likely causes are per-batch `torch.unique` asserts in the
  loss/metric helpers, PNG decoding on CPU, and saving every validation slice
  each epoch. On 3D tensors
  those asserts alone cost 2.5 s per step, so `train3d.py` uses the identical
  CE + Dice formula without them (verified equal to 7 digits).
- **Data on the GPU.** All training volumes live on the GPU (~0.4 GB); patch
  cutting and augmentation run there (5 ms per batch).
- **Resumable.** Model, optimizer, schedule, RNG state and W&B run id are saved
  atomically every epoch; `--resume` continues the same run after a crash.
- **Inference** takes ~2.5 s per patient (sliding window + stitching), and
  predictions land on the exact CT grid (shape and affine checked on all 20 test
  scans).

## How it was run

```bash
scripts/train_eval_3d.sh E_F07_unet3d_adamw_cosine   # train, infer on val, stitch, 3D metrics
```

Logged to W&B: https://wandb.ai/ai4miFrancesco/AI4M/runs/qzwj6s46

## Not the same as the 2D runs (state it on the slide)

- Selection metric: 3D Dice on whole volumes vs per-slice 2D Dice in `main.py`.
- Non-deterministic cuDNN kernels for speed: the seed fixes initialisation and
  sampling, not bit-exact results.
