"""
Element Loader — Reads all element property CSVs into a unified lookup.

Loads 11 CSV files from Dataset/Elements Properties/ and merges them
into a single dictionary keyed by element symbol.
"""

import os
import pandas as pd

# -----------------------------------------------------------------------
# Path to the element properties directory (relative to Backend/)
# -----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ELEMENTS_DIR = os.path.join(BASE_DIR, "Dataset", "Elements Properties")

# CSV filenames covering all element categories
ELEMENT_FILES = [
    "Actinides.csv",
    "Alkali_Metals.csv",
    "Earth_Alkaline_Metals.csv",
    "Halogens.csv",
    "lanthanides.csv",
    "Noble_Gases.csv",
    "Nonmetals.csv",
    "Semi_Metals.csv",
    "Transition_Metals.csv",
    "Unknown_Properties.csv",
    "Weak_Metals.csv",
]

# Columns we care about (standardised names from the CSVs)
KEEP_COLUMNS = [
    "Symbol",
    "Element",
    "Atomic_number",
    "Atomic_weight",
    "Density",
    "Melting_point",
    "Boiling_point",
    "Specific_heat_capacity",
    "Electro_negativity",
]

# -----------------------------------------------------------------------
# Module-level cache so we only read CSVs once
# -----------------------------------------------------------------------
_elements_cache: dict | None = None


def _read_csv_safe(filepath: str) -> pd.DataFrame:
    """Read a CSV and rename columns to a canonical form."""
    df = pd.read_csv(filepath)

    # The CSVs use slightly varying column names; normalise them
    rename_map = {}
    for col in df.columns:
        col_lower = col.strip().lower()
        if "symbol" in col_lower:
            rename_map[col] = "Symbol"
        elif col_lower == "element":
            rename_map[col] = "Element"
        elif "atomic_number" in col_lower or col_lower == "atomic_number":
            rename_map[col] = "Atomic_number"
        elif "atomic_weight" in col_lower or "weight" in col_lower:
            rename_map[col] = "Atomic_weight"
        elif col_lower == "density":
            rename_map[col] = "Density"
        elif "melting" in col_lower:
            rename_map[col] = "Melting_point"
        elif "boiling" in col_lower:
            rename_map[col] = "Boiling_point"
        elif "specific_heat" in col_lower or "heat_capacity" in col_lower:
            rename_map[col] = "Specific_heat_capacity"
        elif "electro_negativity" in col_lower or "negativity" in col_lower:
            rename_map[col] = "Electro_negativity"

    df = df.rename(columns=rename_map)
    return df


def load_all_elements() -> dict:
    """
    Load every element CSV and return a dictionary:
        { "Fe": { "Element": "Iron", "Atomic_weight": 55.845, ... }, ... }
    """
    global _elements_cache
    if _elements_cache is not None:
        return _elements_cache

    all_frames = []
    for fname in ELEMENT_FILES:
        fpath = os.path.join(ELEMENTS_DIR, fname)
        if os.path.exists(fpath):
            df = _read_csv_safe(fpath)
            all_frames.append(df)

    if not all_frames:
        raise FileNotFoundError(
            f"No element CSV files found in {ELEMENTS_DIR}"
        )

    combined = pd.concat(all_frames, ignore_index=True)

    # Keep only the columns we need (if they exist)
    available = [c for c in KEEP_COLUMNS if c in combined.columns]
    combined = combined[available].copy()

    # Convert numeric columns, coercing errors to NaN
    for col in ["Atomic_weight", "Density", "Melting_point", "Boiling_point",
                 "Specific_heat_capacity", "Electro_negativity"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Drop rows without a symbol
    combined = combined.dropna(subset=["Symbol"])

    # Build lookup dict keyed by symbol
    lookup = {}
    for _, row in combined.iterrows():
        sym = str(row["Symbol"]).strip()
        if sym:
            lookup[sym] = row.to_dict()

    _elements_cache = lookup
    print(f"[ElementLoader] Loaded {len(lookup)} elements from {len(all_frames)} CSV files")
    return lookup


def get_element(symbol: str) -> dict | None:
    """Get properties for a single element by its symbol (e.g. 'Fe')."""
    elements = load_all_elements()
    return elements.get(symbol)


def list_all_symbols() -> list[str]:
    """Return a sorted list of all loaded element symbols."""
    elements = load_all_elements()
    return sorted(elements.keys())


# -----------------------------------------------------------------------
# Quick self-test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    elems = load_all_elements()
    print(f"Total elements: {len(elems)}")
    for sym in ["H", "C", "O", "Fe", "Al", "Si", "Li", "Na"]:
        e = get_element(sym)
        if e:
            print(f"  {sym}: mass={e.get('Atomic_weight')}, EN={e.get('Electro_negativity')}")
        else:
            print(f"  {sym}: NOT FOUND")
