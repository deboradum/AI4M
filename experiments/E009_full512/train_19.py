import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from main import img_transform, gt_transform
from dataset import SliceDataset
from ENet import ENet
from losses import CrossEntropy
from utils import dice_coef, probs2one_hot, probs2class, save_images


def write_json(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Submit the GPU batch job.")

    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("Submit through Slurm.")

    dest = Path("results") / f"E009_full512_{job_id}"
    dest.mkdir(parents=True, exist_ok=False)

    seed = 123
    epochs = 25
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator().manual_seed(seed)

    loaders = {}
    patients = {}
    for split in ["train", "val"]:
        dataset = SliceDataset(
            split,
            Path("data/SEGTHOR_512"),
            img_transform=img_transform,
            gt_transform=partial(gt_transform, 5),
        )
        if not len(dataset):
            raise ValueError(f"No {split} slices.")

        patients[split] = sorted({
            image.stem.rsplit("_", 1)[0]
            for image, _ in dataset.files
        })

        # Verify that resolution changed without changing the patient split.
        reference = sorted({
            image.stem.rsplit("_", 1)[0]
            for image in (Path("data/SEGTHOR") / split / "img").glob("*.png")
        })
        if patients[split] != reference:
            raise ValueError(f"{split} patients differ from the 256 baseline.")

        loaders[split] = DataLoader(
            dataset,
            batch_size=8,
            shuffle=(split == "train"),
            num_workers=5,
            worker_init_fn=seed_worker,
            generator=generator,
        )

    if set(patients["train"]) & set(patients["val"]):
        raise ValueError("Training and validation share patients.")

    net = ENet(1, 5, kernels=8, factor=2)
    net.init_weights()
    net.cuda()

    optimizer = torch.optim.Adam(
        net.parameters(), lr=0.0005, betas=(0.9, 0.999)
    )
    loss_fn = CrossEntropy(idk=list(range(5)))

    config = {
        "exp_id": "E009_full512",
        "status": "training",
        "change": "512x512 inputs instead of 256x256",
        "hypothesis": "Retaining resolution improves small-organ segmentation.",
        "model": "ENet",
        "kernels": 8,
        "factor": 2,
        "in_slices": 1,
        "resolution": [512, 512],
        "dataset": "data/SEGTHOR_512",
        "patients": patients,
        "epochs": epochs,
        "seed": seed,
        "batch_size": 8,
        "loss": "CrossEntropy",
        "optimizer": "Adam",
        "learning_rate": 0.0005,
        "scheduler": None,
        "augmentation": False,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "job_id": job_id,
        "parameter_count": sum(p.numel() for p in net.parameters()),
        "comparison_note": (
            "Compare against a 25-epoch, constant-LR, seed-123 baseline. "
            "Personal E000 was unseeded; E006 also changed the schedule."
        ),
        "metrics_3d": None,
    }
    write_json(dest / "config.json", config)

    history = []
    best = -1.0
    best_epoch = -1
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()

    for epoch in range(epochs):
        row = {"epoch": epoch, "learning_rate": 0.0005}

        for split, loader in loaders.items():
            training = split == "train"
            net.train(training)
            losses = []
            dices = []
            torch.cuda.synchronize()
            phase_start = time.perf_counter()

            with torch.set_grad_enabled(training):
                for index, batch in enumerate(loader):
                    images = batch["images"].cuda()
                    labels = batch["gts"].cuda()

                    if images.shape[1:] != (1, 512, 512):
                        raise ValueError(f"Wrong input shape: {images.shape}")

                    if training:
                        optimizer.zero_grad()

                    probabilities = net(images).softmax(dim=1)
                    loss = loss_fn(probabilities, labels)

                    with torch.no_grad():
                        dices.append(dice_coef(
                            probs2one_hot(probabilities), labels
                        ).cpu())

                    if training:
                        loss.backward()
                        optimizer.step()

                    losses.append(loss.item())
                    if index % 50 == 0:
                        print(
                            f"Epoch {epoch + 1}/{epochs} {split}: "
                            f"batch {index + 1}/{len(loader)}",
                            flush=True,
                        )

            torch.cuda.synchronize()
            row[f"{split}_seconds"] = time.perf_counter() - phase_start
            row[f"{split}_loss"] = float(np.mean(losses))
            per_class = torch.cat(dices).mean(dim=0)
            row[f"{split}_dice_per_class"] = per_class.tolist()
            row[f"{split}_dice_2d"] = per_class[1:].mean().item()

        if row["val_dice_2d"] > best:
            best = row["val_dice_2d"]
            best_epoch = epoch
            torch.save(net.state_dict(), dest / "bestweights.pt")
            torch.save(net, dest / "bestmodel.pkl")
            (dest / "best_epoch.txt").write_text(
                f"Epoch {epoch}: validation Dice {best:.6f}\n"
            )

        history.append(row)
        temporary = dest / "last.pt.tmp"
        torch.save({
            "epoch": epoch,
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_epoch": best_epoch,
            "best_dice": best,
            "history": history,
        }, temporary)
        temporary.replace(dest / "last.pt")

        stats = {
            "epochs_completed": epoch + 1,
            "best_epoch": best_epoch,
            "best_validation_dice_2d": best,
            "total_minutes": (time.perf_counter() - started) / 60,
            "peak_gpu_memory_mb":
                torch.cuda.max_memory_allocated() / 2**20,
            "history": history,
        }
        write_json(dest / "stats.json", stats)
        print(
            f"Epoch {epoch + 1}: train {row['train_seconds']:.1f}s, "
            f"val {row['val_seconds']:.1f}s, "
            f"Dice {row['val_dice_2d']:.4f}",
            flush=True,
        )

    config["results"] = stats
    config["status"] = "training_complete"
    write_json(dest / "config.json", config)

    # Save predictions only once, using the best weights.
    net.load_state_dict(torch.load(
        dest / "bestweights.pt", map_location="cuda", weights_only=True
    ))
    net.eval()
    torch.cuda.synchronize()
    prediction_start = time.perf_counter()

    with torch.no_grad():
        for batch in loaders["val"]:
            predictions = probs2class(
                net(batch["images"].cuda()).softmax(dim=1)
            )
            save_images(
                predictions * 63, batch["stems"], dest / "best_epoch/val"
            )

    torch.cuda.synchronize()
    config["prediction_seconds_including_png_writes"] = (
        time.perf_counter() - prediction_start
    )
    write_json(dest / "config.json", config)

    subprocess.run([
        sys.executable, "evaluation_tools/stitch.py",
        "--data_folder", str(dest / "best_epoch/val"),
        "--dest_folder", str(dest / "volumes"),
        "--source_scan_pattern",
        "data/segthor_part1/train/{id_}/{id_}.nii.gz",
        "--num_classes", "5", "--process", "1",
    ], check=True)

    subprocess.run([
        sys.executable, "evaluation_tools/metrics3d.py",
        "--pred_folder", str(dest / "volumes"),
        "--gt_pattern", "data/segthor_part1/train/{id_}/GT.nii.gz",
        "--scan_pattern", "data/segthor_part1/train/{id_}/{id_}.nii.gz",
        "--class_names",
        "background", "esophagus", "heart", "trachea", "aorta",
        "--dest", str(dest / "metrics3d"),
        "--process", "1",
    ], check=True)

    with (dest / "metrics3d/metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))

    summary = {}
    for metric in ["dice", "iou", "hd95", "assd", "nsd"]:
        means = {}
        for organ in ["esophagus", "heart", "trachea", "aorta"]:
            values = [
                float(row[metric]) for row in rows
                if row["class_name"] == organ
            ]
            values = [v for v in values if math.isfinite(v)]
            means[organ] = sum(values) / len(values) if values else None
        available = [v for v in means.values() if v is not None]
        means["foreground_mean"] = (
            sum(available) / len(available) if available else None
        )
        summary[metric] = means

    config["metrics_3d"] = summary
    config["status"] = "complete"
    write_json(dest / "config.json", config)
    print(f"TRAINING AND EVALUATION COMPLETE: {dest}", flush=True)


if __name__ == "__main__":
    main()
