"""
Candidate Service — Filters real alloys from the dataset based on
user-specified minimum tensile strength and melting point.
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ALLOY_PATH = os.path.join(BASE_DIR, "Dataset", "Alloy Properties", "alloys.csv")

# Module-level cache
_alloy_df: pd.DataFrame | None = None


def _load_alloys() -> pd.DataFrame:
    """Load and prepare the alloy dataset (cached)."""
    global _alloy_df
    if _alloy_df is not None:
        return _alloy_df

    df = pd.read_csv(ALLOY_PATH)

    # Rename for easier access
    df = df.rename(columns={
        "Tensile Strength: Ultimate (UTS) (psi)": "UTS",
        "Melting Completion (Liquidus)": "MeltingPoint",
    })

    # Convert to numeric, drop invalid rows
    df["UTS"] = pd.to_numeric(df["UTS"], errors="coerce")
    df["MeltingPoint"] = pd.to_numeric(df["MeltingPoint"], errors="coerce")
    df = df.dropna(subset=["UTS", "MeltingPoint"])

    _alloy_df = df
    print(f"[CandidateService] Loaded {len(df)} alloys from dataset")
    return df


def filter_alloys(
    min_strength: float = 600,
    min_melting_point: float = 1400,
    top_n: int = 5,
) -> list[dict]:
    """
    Filter alloy candidates by minimum strength and melting point.

    Returns:
        List of dicts with keys: Alloy, UTS, MeltingPoint
    """
    df = _load_alloys()

    candidates = df[
        (df["UTS"] >= min_strength) &
        (df["MeltingPoint"] >= min_melting_point)
    ]

    # Sort by strength descending
    candidates = candidates.sort_values(by="UTS", ascending=False)

    # Select top results
    top = candidates.head(top_n)

    return top[["Alloy", "UTS", "MeltingPoint"]].to_dict(orient="records")


# -----------------------------------------------------------------------
if __name__ == "__main__":
    results = filter_alloys(600, 1400, 5)
    print(f"Found {len(results)} alloy candidates:")
    for r in results:
        print(f"  {r['Alloy']}: UTS={r['UTS']}, MP={r['MeltingPoint']}")
