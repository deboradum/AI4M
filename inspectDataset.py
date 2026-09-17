import os
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path

def analyze_segthor_dataset(data_path):
    """
    Analyzes the SEGTHOR dataset to extract volume, spacing, HU statistics,
    and class imbalances which are vital for medical image preprocessing.
    """
    data_dir = Path(data_path)

    patient_dirs = [d for d in data_dir.iterdir() if d.is_dir() and "Patient" in d.name]

    if not patient_dirs:
        print(f"No patient directories found in {data_path}. Ensure it points to the train/test splits.")
        return

    organs = {
        1: "Esophagus",
        2: "Heart",
        3: "Trachea",
        4: "Aorta"
    }

    stats = {
        "Patient": [],
        "Slices (Z)": [],
        "Height (Y)": [],
        "Width (X)": [],
        "Spacing X (mm)": [],
        "Spacing Y (mm)": [],
        "Spacing Z (mm)": []
    }

    # Track Hounsfield Unit (HU) statistics
    hu_stats = {org: {"min": [], "max": [], "mean": [], "std": []} for org in organs.values()}

    # Track Class distributions
    class_volumes = {org: [] for org in organs.values()}

    print(f"Starting analysis on {len(patient_dirs)} patients. This may take a moment...")

    for p_dir in patient_dirs:
        ct_file = p_dir / f"{p_dir.name}.nii.gz"
        gt_file = p_dir / "GT.nii.gz"

        if not ct_file.exists() or not gt_file.exists():
            continue

        # Load NIfTI files
        ct_img = nib.load(str(ct_file))
        gt_img = nib.load(str(gt_file))

        ct_data = ct_img.get_fdata()
        gt_data = gt_img.get_fdata()

        # Geometry & Physical Spacing
        header = ct_img.header
        spacing = header.get_zooms()
        dims = ct_data.shape

        stats["Patient"].append(p_dir.name)
        stats["Width (X)"].append(dims[0])
        stats["Height (Y)"].append(dims[1])
        stats["Slices (Z)"].append(dims[2] if len(dims) == 3 else 1)
        stats["Spacing X (mm)"].append(spacing[0])
        stats["Spacing Y (mm)"].append(spacing[1])
        stats["Spacing Z (mm)"].append(spacing[2] if len(spacing) >= 3 else 0.0)

        # Analyze Foreground Classes (HU and Volume)
        for class_idx, org_name in organs.items():
            mask = (gt_data == class_idx)
            voxels = np.sum(mask)
            class_volumes[org_name].append(voxels)

            if voxels > 0:
                intensities = ct_data[mask]
                hu_stats[org_name]["mean"].append(np.mean(intensities))
                hu_stats[org_name]["std"].append(np.std(intensities))
                # Use percentiles instead of absolute min/max to ignore outliers/artifacts
                hu_stats[org_name]["min"].append(np.percentile(intensities, 1))
                hu_stats[org_name]["max"].append(np.percentile(intensities, 99))

    # Aggregate & Output
    df_geometry = pd.DataFrame(stats)
    print("\n--- Geometry & Spacing Statistics ---")
    print(df_geometry.describe().round(3).T[['min', 'mean', 'max']])

    print("\n--- Class Voxel Volumes (Imbalance Check) ---")
    total_voxels = df_geometry["Slices (Z)"] * df_geometry["Height (Y)"] * df_geometry["Width (X)"]
    avg_total_voxels = total_voxels.mean()

    for org in organs.values():
        avg_v = np.mean(class_volumes[org])
        perc = (avg_v / avg_total_voxels) * 100
        print(f"{org:>10}: {avg_v:10.0f} avg voxels ({perc:.4f}% of volume)")

    print("\n--- Hounsfield Units (HU) Intensity Profiles ---")
    for org in organs.values():
        o_mean = np.mean(hu_stats[org]["mean"])
        o_std = np.mean(hu_stats[org]["std"])
        o_min = np.mean(hu_stats[org]["min"])
        o_max = np.mean(hu_stats[org]["max"])
        print(f"{org:>10}: Mean {o_mean:6.1f} | Std {o_std:5.1f} | 1st Pct {o_min:6.1f} | 99th Pct {o_max:6.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/segthor_part1/train", help="Path to the dataset directory containing Patient_XX folders.")
    args = parser.parse_args()

    analyze_segthor_dataset(args.data_path)
