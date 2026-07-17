"""
Learning Service — Self-learning loop for material discovery.

Stores newly discovered materials in a CSV so they can be used
for model retraining, improving predictions over time.
"""

import os
import csv
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DISCOVERIES_PATH = os.path.join(BASE_DIR, "Dataset", "discovered_materials.csv")

COLUMNS = [
    "timestamp", "compound", "formation_energy", "band_gap_eV",
    "quantum_score", "stability_score", "novelty_score",
    "material_class", "primary_applications",
]

_initialized = False


def _ensure_file():
    """Create the discoveries CSV if it doesn't exist."""
    global _initialized
    if _initialized:
        return
    if not os.path.exists(DISCOVERIES_PATH):
        os.makedirs(os.path.dirname(DISCOVERIES_PATH), exist_ok=True)
        with open(DISCOVERIES_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
        print(f"[LearningService] Created {DISCOVERIES_PATH}")
    _initialized = True


def store_discovery(material: dict):
    """
    Store a discovered material for future model retraining.

    Args:
        material: dict with compound, formation_energy, band_gap_eV, etc.
    """
    _ensure_file()
    try:
        apps = material.get("predicted_applications", {})
        row = [
            datetime.now().isoformat(),
            material.get("compound", ""),
            material.get("formation_energy", ""),
            material.get("band_gap_eV", ""),
            material.get("quantum_score", ""),
            material.get("stability_score", ""),
            material.get("novelty_score", ""),
            apps.get("material_class", "") if isinstance(apps, dict) else "",
            "|".join(apps.get("primary_applications", [])) if isinstance(apps, dict) else "",
        ]
        with open(DISCOVERIES_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        print(f"[LearningService] Failed to store discovery: {e}")


def get_discovery_count() -> int:
    """Get the number of stored discoveries."""
    _ensure_file()
    try:
        with open(DISCOVERIES_PATH, "r") as f:
            return sum(1 for _ in f) - 1  # subtract header
    except Exception:
        return 0


def get_discoveries() -> list[dict]:
    """Load all stored discoveries."""
    _ensure_file()
    try:
        import pandas as pd
        df = pd.read_csv(DISCOVERIES_PATH)
        return df.to_dict(orient="records")
    except Exception:
        return []


# -----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Discovery count: {get_discovery_count()}")
    print(f"Path: {DISCOVERIES_PATH}")
