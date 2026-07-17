"""
Merge Datasets — Combines real Materials Project data with synthetic data.

Creates a hybrid dataset:
  - 100% of real data (from real_materials.csv)
  - ~30% sample of synthetic data (from mp_dataset.csv)
  - Added derived features: is_metal, stability_score

Usage:
    python merge_datasets.py
"""

import os
import re
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")

REAL_PATH = os.path.join(DATASET_DIR, "real_materials.csv")
SYNTHETIC_PATH = os.path.join(DATASET_DIR, "mp_dataset.csv")
OUTPUT_PATH = os.path.join(DATASET_DIR, "final_dataset.csv")

SYNTHETIC_FRACTION = 0.30  # Use 30% of synthetic data


def count_atoms_in_formula(formula: str) -> int:
    """Count total atoms in a chemical formula."""
    tokens = re.findall(r'([A-Z][a-z]?)(\d*)', str(formula))
    total = 0
    for _, count in tokens:
        total += int(count) if count else 1
    return total


def merge_datasets():
    """Merge real and synthetic datasets into a hybrid training set."""
    print("=" * 60)
    print(" Merging Real + Synthetic Datasets")
    print("=" * 60)

    # ---- Load real data ----
    if not os.path.exists(REAL_PATH):
        print(f"[ERROR] Real dataset not found: {REAL_PATH}")
        print("  Run fetch_mp_data.py first!")
        return

    real_df = pd.read_csv(REAL_PATH)
    print(f"\n[Load] Real dataset: {len(real_df)} compounds")

    # ---- Load synthetic data ----
    if not os.path.exists(SYNTHETIC_PATH):
        print(f"[ERROR] Synthetic dataset not found: {SYNTHETIC_PATH}")
        print("  Run download_datasets.py first!")
        return

    synthetic_df = pd.read_csv(SYNTHETIC_PATH)
    print(f"[Load] Synthetic dataset: {len(synthetic_df)} compounds")

    # ---- Sample synthetic data ----
    synthetic_sample = synthetic_df.sample(
        frac=SYNTHETIC_FRACTION,
        random_state=42
    )
    print(f"[Sample] Using {len(synthetic_sample)} synthetic compounds "
          f"({SYNTHETIC_FRACTION*100:.0f}% of {len(synthetic_df)})")

    # ---- Ensure common columns ----
    # Both datasets should have these columns from their respective generators
    required_cols = [
        "formula", "formation_energy_eV_per_atom", "band_gap_eV",
        "density_g_cm3", "is_stable"
    ]

    for col in required_cols:
        if col not in real_df.columns:
            print(f"[WARN] Missing column in real data: {col}")
        if col not in synthetic_df.columns:
            print(f"[WARN] Missing column in synthetic data: {col}")

    # Tag the source
    real_df["source"] = "real"
    synthetic_sample["source"] = "synthetic"

    # ---- Merge ----
    # Use only the common columns + source
    common_cols = [c for c in required_cols if c in real_df.columns and c in synthetic_sample.columns]
    common_cols.append("source")

    # Add optional columns if they exist in both
    for opt_col in ["n_elements", "n_atoms", "crystal_system"]:
        if opt_col in real_df.columns and opt_col in synthetic_sample.columns:
            common_cols.append(opt_col)

    final_df = pd.concat([
        real_df[common_cols],
        synthetic_sample[[c for c in common_cols if c in synthetic_sample.columns]]
    ], ignore_index=True)

    # ---- Remove duplicates by formula (keep real data) ----
    before_dedup = len(final_df)
    final_df = final_df.drop_duplicates(subset="formula", keep="first")
    print(f"[Dedup] Removed {before_dedup - len(final_df)} duplicate formulas")

    # ---- Add derived features ----
    print("[Features] Adding derived features...")

    # Ensure n_atoms column exists
    if "n_atoms" not in final_df.columns:
        final_df["n_atoms"] = final_df["formula"].apply(count_atoms_in_formula)

    # Is metal (band gap < 0.1 eV)
    final_df["is_metal"] = (final_df["band_gap_eV"] < 0.1).astype(int)

    # Stability score (negative of formation energy — more negative = more stable)
    final_df["stability_score"] = -final_df["formation_energy_eV_per_atom"]

    # ---- Final cleanup ----
    final_df = final_df.dropna(subset=["formation_energy_eV_per_atom", "band_gap_eV"])

    # ---- Save ----
    os.makedirs(DATASET_DIR, exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)

    # ---- Report ----
    n_real = (final_df["source"] == "real").sum()
    n_synth = (final_df["source"] == "synthetic").sum()

    print(f"\n{'=' * 60}")
    print(f" Final Hybrid Dataset: {len(final_df)} compounds")
    print(f"{'=' * 60}")
    print(f"  Real data:      {n_real:6d} ({100*n_real/len(final_df):.1f}%)")
    print(f"  Synthetic data: {n_synth:6d} ({100*n_synth/len(final_df):.1f}%)")
    print(f"  Saved to: {OUTPUT_PATH}")
    print(f"\nColumn summary:")
    print(f"  {list(final_df.columns)}")
    print(f"\nDescriptive statistics:")
    print(final_df.describe().round(4).to_string())
    print(f"\nSample entries:")
    print(final_df.head(10).to_string())


if __name__ == "__main__":
    merge_datasets()
