"""
Fetch Real Materials Data — Downloads compounds from the Materials Project REST API.

Uses the Materials Project API v2 directly via HTTP requests (no mp-api library needed).

Downloads real DFT-computed material properties for training the GNN:
  - Formation energy (eV/atom)
  - Band gap (eV)
  - Density (g/cm³)
  - Stability flag

Usage:
    python fetch_mp_data.py
"""

import os
import sys
import re
import json
import time
import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY = os.getenv("MATERIALS_PROJECT_API_KEY", "EEkYhPYOuCqoC6XeAV2HPFBy6UILZAOs")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "Dataset")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "real_materials.csv")

# Materials Project API v2 endpoint
MP_API_URL = "https://api.materialsproject.org/materials/summary/"

# Fields to request (underscore prefix for MP API v2)
FIELDS = "material_id,formula_pretty,formation_energy_per_atom,band_gap,energy_above_hull,density,is_stable"

# Element groups to query (each query filters by "contains this element")
ELEMENT_QUERIES = [
    "O",    # Oxides
    "F",    # Fluorides
    "Cl",   # Chlorides
    "N",    # Nitrides
    "C",    # Carbides
    "S",    # Sulfides
    "Br",   # Bromides
    "Se",   # Selenides
    "Te",   # Tellurides
    "P",    # Phosphides
    "I",    # Iodides
]


# ---------------------------------------------------------------------------
# API Fetching
# ---------------------------------------------------------------------------

def fetch_materials_for_element(element: str, api_key: str, max_results: int = 5000) -> list:
    """Fetch materials containing a given element from MP API v2."""
    headers = {
        "X-API-KEY": api_key,
        "accept": "application/json",
    }
    params = {
        "elements": element,
        "_fields": FIELDS,
        "_limit": min(max_results, 1000),
        "_skip": 0,
    }

    all_data = []
    retries = 3

    while True:
        for attempt in range(retries):
            try:
                resp = requests.get(MP_API_URL, headers=headers, params=params, timeout=60)
                if resp.status_code == 200:
                    break
                elif resp.status_code == 429:
                    # Rate limited
                    wait = 5 * (attempt + 1)
                    print(f"    Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"    HTTP {resp.status_code}: {resp.text[:200]}")
                    if attempt < retries - 1:
                        time.sleep(2)
            except requests.exceptions.RequestException as e:
                print(f"    Request error: {e}")
                if attempt < retries - 1:
                    time.sleep(2)
        else:
            # All retries failed
            print(f"    Failed after {retries} retries")
            break

        try:
            data = resp.json()
        except json.JSONDecodeError:
            print(f"    Invalid JSON response")
            break

        docs = data.get("data", [])
        if not docs:
            break

        all_data.extend(docs)

        # Check if there are more pages
        if len(docs) < params["_limit"]:
            break
        if len(all_data) >= max_results:
            break

        params["_skip"] += len(docs)
        time.sleep(0.5)  # Be respectful of rate limits

    return all_data


def fetch_all_materials() -> pd.DataFrame:
    """Download real materials data from the Materials Project API."""
    print(f"[Fetch] API Key: {API_KEY[:8]}...{API_KEY[-4:]}")

    all_records = []
    seen_formulas = set()

    for element in ELEMENT_QUERIES:
        print(f"\n[Fetch] Querying compounds with element: {element}")
        docs = fetch_materials_for_element(element, API_KEY, max_results=5000)

        count = 0
        for doc in docs:
            formula = doc.get("formula_pretty", "")
            if not formula or formula in seen_formulas:
                continue
            seen_formulas.add(formula)

            fe = doc.get("formation_energy_per_atom")
            bg = doc.get("band_gap")
            ehull = doc.get("energy_above_hull")
            dens = doc.get("density")
            is_stable = doc.get("is_stable")

            if fe is None or bg is None:
                continue

            all_records.append({
                "formula": formula,
                "formation_energy_eV_per_atom": fe,
                "band_gap_eV": bg,
                "e_above_hull": ehull,   # ← PRIMARY thermodynamic stability metric
                "density_g_cm3": dens if dens is not None else 0.0,
                "is_stable": int(is_stable) if is_stable is not None else 0,
            })
            count += 1

        print(f"  → Got {count} unique compounds (total so far: {len(all_records)})")

    df = pd.DataFrame(all_records)
    print(f"\n[Fetch] Total raw records: {len(df)}")
    return df


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and filter the fetched data for physically realistic values."""
    before = len(df)

    # Drop rows with missing values
    df = df.dropna()
    print(f"[Clean] Dropped {before - len(df)} rows with NaN values")

    before = len(df)

    # Filter to physically realistic ranges
    df = df[
        (df["formation_energy_eV_per_atom"] > -5) &
        (df["formation_energy_eV_per_atom"] < 2) &
        (df["band_gap_eV"] >= 0) &
        (df["band_gap_eV"] <= 10) &
        (df["density_g_cm3"] > 0) &
        (df["density_g_cm3"] < 25)
    ]
    print(f"[Clean] Filtered out {before - len(df)} rows outside realistic ranges")

    # Add derived columns to match existing dataset schema
    df["n_elements"] = df["formula"].apply(_count_elements)
    df["n_atoms"]    = df["formula"].apply(_count_atoms)
    df["crystal_system"] = "unknown"  # Not available in summary endpoint
    df["data_source"] = "real"        # Explicit data provenance label

    ehull_count = df["e_above_hull"].notna().sum()
    print(f"[Clean] e_above_hull values: {ehull_count}/{len(df)} ({100*ehull_count/len(df):.1f}%)")

    # Reorder columns (e_above_hull included now)
    df = df[[
        "formula", "n_elements", "n_atoms",
        "formation_energy_eV_per_atom", "e_above_hull", "band_gap_eV",
        "density_g_cm3", "is_stable", "crystal_system", "data_source"
    ]]

    print(f"[Clean] Final clean dataset: {len(df)} compounds")
    return df


def _count_elements(formula: str) -> int:
    """Count unique elements in a formula."""
    return len(set(re.findall(r'[A-Z][a-z]?', formula)))


def _count_atoms(formula: str) -> int:
    """Count total atoms in a formula."""
    tokens = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    total = 0
    for _, count in tokens:
        total += int(count) if count else 1
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print(" Fetching Real Materials Data from Materials Project")
    print("=" * 60)

    # Fetch
    df = fetch_all_materials()
    if df.empty:
        print("[ERROR] No data fetched. Check your API key and network.")
        sys.exit(1)

    # Clean
    df = clean_data(df)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'=' * 60}")
    print(f" Saved {len(df)} real compounds to {OUTPUT_FILE}")
    print(f"{'=' * 60}")
    print(f"\nDataset summary:")
    print(df.describe().round(4).to_string())
    print(f"\nSample entries:")
    print(df.head(10).to_string())


if __name__ == "__main__":
    main()
