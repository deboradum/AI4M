import os
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
import warnings

import seaborn as sns
import matplotlib.pyplot as plt


def generate_eda_plots(df_geometry, class_volumes, hu_samples, global_hu_samples, organs):
    sns.set_theme(style="whitegrid")
    active_organs = [org for org in organs.values() if len(hu_samples[org]) > 0]

    # --- Spatial Resolution Variance ---
    plt.figure(figsize=(7, 6))
    sns.kdeplot(data=df_geometry, x="Spacing X (mm)", y="Spacing Z (mm)", fill=True, cmap="Blues")
    sns.scatterplot(data=df_geometry, x="Spacing X (mm)", y="Spacing Z (mm)", color="black", alpha=0.5)
    plt.title("Spatial Spacing Variance Across Patients")
    plt.xlabel("In-Plane Spacing (X/Y mm)")
    plt.ylabel("Slice Thickness (Z mm)")
    plt.tight_layout()
    plt.savefig("segthor_eda_spacing.png", dpi=300)
    plt.close()

    # --- Class Imbalance (Linear Scale & Decimal %) ---
    fig, ax = plt.subplots(figsize=(8, 5))

    vol_data = {org: class_volumes[org] for org in active_organs}
    df_vols = pd.DataFrame(vol_data).melt(var_name="Organ", value_name="Voxel Count")

    sns.violinplot(
        data=df_vols,
        x="Voxel Count",
        y="Organ",
        hue="Organ",
        palette="viridis",
        inner="quartile",
        linewidth=1.5,
        legend=False,
        ax=ax
    )

    ax.set_xlabel("Voxel Count per Scan")
    ax.set_ylabel("")

    total_voxels = df_geometry["Slices (Z)"] * df_geometry["Height (Y)"] * df_geometry["Width (X)"]
    avg_total_voxels = total_voxels.mean()

    def voxels_to_pct(v):
        return (v / avg_total_voxels) * 100

    def pct_to_voxels(p):
        return (p / 100) * avg_total_voxels

    secax = ax.secondary_xaxis('top', functions=(voxels_to_pct, pct_to_voxels))
    secax.set_xlabel("Percentage of Average Total Scan Volume (%)")
    secax.ticklabel_format(style='plain')

    plt.title("Voxel Volume Distribution Across Patients", y=1.2)
    plt.tight_layout()
    plt.savefig("segthor_eda_volume.png", dpi=300)
    plt.close()

    # --- HU Intensity Distribution (All Organs) ---
    plt.figure(figsize=(9, 6))
    for org in active_organs:
        sns.kdeplot(hu_samples[org], label=org, fill=True, alpha=0.4, linewidth=2)
    plt.title("Hounsfield Unit (HU) Distributions per Organ")
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Density")
    plt.xlim(-1100, 300)
    plt.legend()
    plt.tight_layout()
    plt.savefig("segthor_eda_hu_distributions.png", dpi=300)
    plt.close()

    # --- HU Intensity Distribution (Zoomed: Heart, Esophagus, Aorta) ---
    plt.figure(figsize=(9, 6))
    soft_tissue_organs = ["Esophagus", "Heart", "Aorta"]
    for org in soft_tissue_organs:
        if org in active_organs:
            sns.kdeplot(hu_samples[org], label=org, fill=True, alpha=0.4, linewidth=2)
    plt.title("HU Distributions (Zoomed: Soft Tissues Only)")
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Density")
    # Soft tissues usually reside between -100 and +300 HU (accounting for contrast agents)
    plt.xlim(-100, 300)
    plt.legend()
    plt.tight_layout()
    plt.savefig("segthor_eda_hu_distributions_zoomed.png", dpi=300)
    plt.close()

    # --- The Normalization Problem ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)

    all_organ_samples = np.concatenate([hu_samples[org] for org in active_organs])
    global_min, global_max = np.min(global_hu_samples), np.max(global_hu_samples)

    sns.kdeplot(global_hu_samples, ax=axes[0], color="gray", fill=True, alpha=0.3, label="Entire CT Scan")
    sns.kdeplot(all_organ_samples, ax=axes[0], color="red", fill=True, alpha=0.7, label="Target Organs")
    axes[0].set_title("Raw CT Scale: Organs occupy a tiny fraction of the total dynamic range")
    axes[0].set_xlabel("Raw Hounsfield Units (HU)")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    normalized_global = (global_hu_samples - global_min) / (global_max - global_min)
    normalized_organs = (all_organ_samples - global_min) / (global_max - global_min)

    sns.kdeplot(normalized_global, ax=axes[1], color="gray", fill=True, alpha=0.3)
    sns.kdeplot(normalized_organs, ax=axes[1], color="red", fill=True, alpha=0.7)
    axes[1].set_title("Min-Max Normalization: Organ contrast is completely compressed")
    axes[1].set_xlabel("Normalized Pixel Value [0, 1]")
    axes[1].set_ylabel("Density")
    axes[1].set_xlim(0, 1)

    window_min, window_max = -1000, 200
    clipped_global = np.clip(global_hu_samples, window_min, window_max)
    clipped_organs = np.clip(all_organ_samples, window_min, window_max)

    windowed_global = (clipped_global - window_min) / (window_max - window_min)
    windowed_organs = (clipped_organs - window_min) / (window_max - window_min)

    sns.kdeplot(windowed_global, ax=axes[2], color="gray", fill=True, alpha=0.3)
    sns.kdeplot(windowed_organs, ax=axes[2], color="red", fill=True, alpha=0.7)
    axes[2].set_title(f"Windowed Normalization (Clipped {window_min} to {window_max} HU): Organ contrast is maximized")
    axes[2].set_xlabel("Normalized Pixel Value [0, 1]")
    axes[2].set_ylabel("Density")
    axes[2].set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig("segthor_eda_normalization_issue.png", dpi=300)
    plt.close()

    print("- segthor_eda_spacing.png")
    print("- segthor_eda_volume.png")
    print("- segthor_eda_hu_distributions.png")
    print("- segthor_eda_hu_distributions_zoomed.png")
    print("- segthor_eda_normalization_issue.png")


def analyze_segthor_dataset(data_path):
    data_dir = Path(data_path)
    patient_dirs = [d for d in data_dir.iterdir() if d.is_dir() and "Patient" in d.name]

    if not patient_dirs:
        print(f"No patient directories found in {data_path}.")
        return

    organs = {1: "Esophagus", 2: "Heart", 3: "Trachea", 4: "Aorta"}

    stats = {
        "Patient": [], "Slices (Z)": [], "Height (Y)": [], "Width (X)": [],
        "Spacing X (mm)": [], "Spacing Y (mm)": [], "Spacing Z (mm)": []
    }

    hu_samples = {org: [] for org in organs.values()}
    global_hu_samples = []
    class_volumes = {org: [] for org in organs.values()}

    print(f"Starting analysis on {len(patient_dirs)} patients. Sampling voxels for KDE plots...")

    for p_dir in patient_dirs:
        ct_file = p_dir / f"{p_dir.name}.nii.gz"
        gt_file = p_dir / "GT.nii.gz"

        if not ct_file.exists() or not gt_file.exists():
            continue

        ct_img = nib.load(str(ct_file))
        gt_img = nib.load(str(gt_file))
        ct_data = ct_img.get_fdata()
        gt_data = gt_img.get_fdata()

        spacing = ct_img.header.get_zooms()
        dims = ct_data.shape
        stats["Patient"].append(p_dir.name)
        stats["Width (X)"].append(dims[0])
        stats["Height (Y)"].append(dims[1])
        stats["Slices (Z)"].append(dims[2] if len(dims) == 3 else 1)
        stats["Spacing X (mm)"].append(spacing[0])
        stats["Spacing Y (mm)"].append(spacing[1])
        stats["Spacing Z (mm)"].append(spacing[2] if len(spacing) >= 3 else 0.0)

        valid_bg = ct_data[(ct_data > -1050) & (ct_data < 3000)]
        if len(valid_bg) > 0:
            global_hu_samples.extend(np.random.choice(valid_bg, min(5000, len(valid_bg)), replace=False))

        for class_idx, org_name in organs.items():
            mask = (gt_data == class_idx)
            voxels = np.sum(mask)
            class_volumes[org_name].append(voxels)

            if voxels > 0:
                intensities = ct_data[mask]
                hu_samples[org_name].extend(np.random.choice(intensities, min(5000, len(intensities)), replace=False))

    df_geometry = pd.DataFrame(stats)

    print("\n--- Geometry & Spacing Statistics ---")
    geom_summary = df_geometry.describe().round(3).T
    geom_summary.rename(columns={'50%': 'median'}, inplace=True)
    print(geom_summary[['min', 'median', 'mean', 'max']])

    print("\n--- Class Voxel Volumes (Imbalance Check) ---")
    total_voxels = df_geometry["Slices (Z)"] * df_geometry["Height (Y)"] * df_geometry["Width (X)"]
    avg_total_voxels = total_voxels.mean()

    for org in organs.values():
        avg_v = np.mean(class_volumes[org])
        perc = (avg_v / avg_total_voxels) * 100
        print(f"{org:>10}: {avg_v:10.0f} avg voxels ({perc:.4f}% of volume)")

    print("\n--- Hounsfield Units (HU) Intensity Profiles ---")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for org in organs.values():
            if len(hu_samples[org]) > 0:
                o_mean = np.mean(hu_samples[org])
                o_std = np.std(hu_samples[org])
                o_min = np.percentile(hu_samples[org], 1)
                o_max = np.percentile(hu_samples[org], 99)
                print(f"{org:>10}: Mean {o_mean:6.1f} | Std {o_std:5.1f} | 1st Pct {o_min:6.1f} | 99th Pct {o_max:6.1f}")
            else:
                print(f"{org:>10}: No data found for this class.")

    print("\n--- Generating Seaborn Plots ---")
    generate_eda_plots(df_geometry, class_volumes, hu_samples, global_hu_samples, organs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/segthor_part1/train", help="Path to the dataset directory containing Patient_XX folders.")
    args = parser.parse_args()

    analyze_segthor_dataset(args.data_path)
