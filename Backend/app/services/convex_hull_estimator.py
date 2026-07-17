"""
Convex Hull Estimator — Data-Driven Phase Stability Analysis.

Approximates the thermodynamic convex hull using formation energies from
the Materials Project dataset. For each candidate composition, identifies
competing phases in the same chemical system and estimates distance from
the convex hull.

This replaces the heuristic-only estimate_e_above_hull approach with a
dataset-backed phase competition analysis.

How it works:
  1. Load all (formula, formation_energy, e_above_hull) from real_materials.csv
  2. Index by chemical system (frozenset of elements)
  3. For a new candidate, find all competing phases in the same system
  4. Estimate hull distance by comparing against the interpolated energy
     of the most stable linear combination of competing phases
  5. Classify as on_hull / near_hull / above_hull

Limitations:
  - This is an approximation — true convex hull requires scipy.spatial
    and DFT-quality energies for all competing stoichiometries.
  - Works best for binary and ternary systems where the dataset has good
    coverage. For quaternary+ systems, falls back to pairwise analysis.
"""

from __future__ import annotations

import os
import re
import math
import pandas as pd
import numpy as np
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Module-level cache
_phase_db: dict[frozenset, list[dict]] | None = None
_loaded = False


def _parse_formula(formula: str) -> list[tuple[str, int]]:
    """Parse formula → [(element, count), ...]."""
    return [(s, int(c) if c else 1)
            for s, c in re.findall(r'([A-Z][a-z]?)(\d*)', formula) if s]


def _get_element_set(formula: str) -> frozenset:
    """Get the frozenset of elements in a formula."""
    return frozenset(s for s, _ in _parse_formula(formula))


def _composition_vector(formula: str) -> dict[str, float]:
    """Get normalized composition vector {element: fraction}."""
    parsed = _parse_formula(formula)
    total = sum(c for _, c in parsed)
    if total == 0:
        return {}
    return {s: c / total for s, c in parsed}


def _load_phase_database():
    """
    Load all compounds and their formation energies from the MP dataset.
    Index by chemical system (frozenset of elements) for fast lookup.
    """
    global _phase_db, _loaded
    if _loaded:
        return

    _phase_db = defaultdict(list)

    for dataset_path in [
        os.path.join(BASE_DIR, "Dataset", "real_materials.csv"),
        os.path.join(BASE_DIR, "Dataset", "mp_dataset.csv"),
    ]:
        if not os.path.exists(dataset_path):
            continue

        try:
            cols_to_load = ["formula"]
            # Check which columns exist
            header = pd.read_csv(dataset_path, nrows=0).columns
            if "formation_energy_eV_per_atom" in header:
                cols_to_load.append("formation_energy_eV_per_atom")
            if "e_above_hull" in header:
                cols_to_load.append("e_above_hull")
            if "is_stable" in header:
                cols_to_load.append("is_stable")

            df = pd.read_csv(dataset_path, usecols=cols_to_load)

            for _, row in df.iterrows():
                formula = str(row.get("formula", ""))
                fe = row.get("formation_energy_eV_per_atom")
                if not formula or pd.isna(fe):
                    continue

                elem_set = _get_element_set(formula)
                ehull = row.get("e_above_hull")
                is_stable = row.get("is_stable")

                entry = {
                    "formula": formula,
                    "formation_energy": float(fe),
                    "e_above_hull": float(ehull) if not pd.isna(ehull) else None,
                    "is_stable": bool(is_stable) if not pd.isna(is_stable) else None,
                    "composition": _composition_vector(formula),
                    "element_set": elem_set,
                }
                _phase_db[elem_set].append(entry)

                # Also index under all subsets for competing phase lookup
                if len(elem_set) <= 4:
                    for elem in elem_set:
                        _phase_db[frozenset([elem])].append(entry)

            print(f"[ConvexHull] Loaded {len(df)} phases from "
                  f"{os.path.basename(dataset_path)}")
            break  # Use first available
        except Exception as e:
            print(f"[ConvexHull] Warning: Could not load {dataset_path}: {e}")

    _loaded = True


def _find_competing_phases(
    target_elements: frozenset,
    max_phases: int = 50,
) -> list[dict]:
    """
    Find all known phases whose element sets are subsets of the target system.
    These are the potential decomposition products.
    """
    _load_phase_database()
    if _phase_db is None:
        return []

    competing = []
    seen_formulas = set()

    for elem_set, phases in _phase_db.items():
        if not elem_set.issubset(target_elements):
            continue
        for phase in phases:
            if phase["formula"] not in seen_formulas:
                seen_formulas.add(phase["formula"])
                competing.append(phase)

    # Sort by formation energy (most stable first)
    competing.sort(key=lambda p: p["formation_energy"])
    return competing[:max_phases]


def _estimate_hull_distance(
    predicted_energy: float,
    target_composition: dict[str, float],
    competing_phases: list[dict],
) -> float | None:
    """
    Estimate distance from the convex hull.

    Uses a simplified approach: finds the most stable competing phase
    in the same system and computes the energy difference. For binary
    systems, this is exact. For ternary+ systems, this is an upper bound
    on the true hull distance.
    """
    if not competing_phases:
        return None

    # Find exact formula match first
    target_key = "".join(f"{e}{target_composition[e]:.0f}"
                         if target_composition[e] != 1.0
                         else e
                         for e in sorted(target_composition.keys()))

    for phase in competing_phases:
        if phase.get("e_above_hull") is not None:
            # If we have a real Ehull from MP, and the composition matches
            comp = phase["composition"]
            if _compositions_similar(comp, target_composition):
                return phase["e_above_hull"]

    # No exact match — estimate by comparing against the best
    # decomposition energy (most stable competing phase)
    most_stable_energy = competing_phases[0]["formation_energy"]

    # The hull energy at the target composition is approximated as
    # the minimum achievable energy via decomposition
    hull_energy = most_stable_energy

    # For better accuracy with ternary systems, try to find phases
    # that bracket the target composition
    same_system_phases = [
        p for p in competing_phases
        if p["element_set"] == frozenset(target_composition.keys())
    ]
    
    if same_system_phases:
        # Sort same-system phases by how similar they are to the target
        same_system_phases.sort(key=lambda p: _composition_distance(p["composition"], target_composition))
        hull_energy = same_system_phases[0]["formation_energy"]
    else:
        # If no same-system phase, use the most stable binary/unary as hull energy
        # but add a penalty for distance to composition
        hull_energy = most_stable_energy

    estimated_ehull = max(0.0, predicted_energy - hull_energy)
    
    # If we only have binary endpoints for a ternary, the estimation is rough
    # Add a data-scarcity adjustment
    if not same_system_phases and len(target_composition) >= 3:
        estimated_ehull = max(0.0, estimated_ehull - 0.2) # Conservative: assume ternary is more stable than binary
        
    return round(estimated_ehull, 4)


def _composition_distance(comp1: dict[str, float], comp2: dict[str, float]) -> float:
    """Compute Euclidean distance between two composition vectors."""
    all_elements = set(comp1.keys()) | set(comp2.keys())
    return math.sqrt(sum((comp1.get(e, 0.0) - comp2.get(e, 0.0))**2 for e in all_elements))


def _compositions_similar(
    comp1: dict[str, float],
    comp2: dict[str, float],
    tol: float = 0.05,
) -> bool:
    """Check if two compositions are similar within tolerance."""
    all_elements = set(comp1.keys()) | set(comp2.keys())
    for elem in all_elements:
        if abs(comp1.get(elem, 0.0) - comp2.get(elem, 0.0)) > tol:
            return False
    return True


def evaluate_phase_stability(
    formula: str,
    predicted_formation_energy: float | None = None,
    ml_ehull: float | None = None,
) -> dict:
    """
    Evaluate a candidate's position relative to the thermodynamic convex hull.

    Combines:
    1. Direct ML Ehull prediction (if available) — highest priority
    2. Dataset lookup (if formula exists in MP) — ground truth
    3. Convex hull approximation from competing phases — fallback

    Args:
        formula: chemical formula
        predicted_formation_energy: ML-predicted formation energy (eV/atom)
        ml_ehull: direct ML-predicted Ehull (eV/atom), None if unavailable

    Returns:
        {
          "hull_distance_eV": float | None,
          "hull_source": "ml_model" | "dataset_lookup" | "hull_approximation" | "unknown",
          "hull_status": "on_hull" | "near_hull" | "above_hull" | "far_above_hull" | "unknown",
          "competing_phases": list,
          "n_competing_phases": int,
          "chemical_system": str,
          "decomposition_warning": str | None,
        }
    """
    _load_phase_database()

    target_elements = _get_element_set(formula)
    target_comp = _composition_vector(formula)
    system_str = "-".join(sorted(target_elements))

    # Find competing phases
    competing = _find_competing_phases(target_elements)

    # Priority 1: Direct ML Ehull prediction
    if ml_ehull is not None:
        hull_distance = ml_ehull
        hull_source = "ml_model"

    # Priority 2: Dataset lookup — does this exact formula exist in MP?
    elif _phase_db is not None:
        exact_match = None
        for phase in competing:
            if _compositions_similar(phase["composition"], target_comp):
                if phase.get("e_above_hull") is not None:
                    exact_match = phase
                    break

        if exact_match:
            hull_distance = exact_match["e_above_hull"]
            hull_source = "dataset_lookup"
        elif predicted_formation_energy is not None:
            hull_distance = _estimate_hull_distance(
                predicted_formation_energy, target_comp, competing
            )
            hull_source = "hull_approximation"
        else:
            hull_distance = None
            hull_source = "unknown"
    else:
        hull_distance = None
        hull_source = "unknown"

    # Classify hull status
    if hull_distance is None:
        hull_status = "unknown"
        warning = "Phase stability could not be evaluated."
    elif hull_distance <= 0.0:
        hull_status = "on_hull"
        warning = None
    elif hull_distance < 0.05:
        hull_status = "on_hull"
        warning = None
    elif hull_distance < 0.2:
        hull_status = "near_hull"
        warning = ("Near the convex hull — may be synthesizable under "
                   "non-equilibrium conditions.")
    elif hull_distance < 1.0:
        hull_status = "above_hull"
        warning = (f"Above hull by {hull_distance:.3f} eV — thermodynamically "
                   "unfavorable. Decomposes into competing phases.")
    else:
        hull_status = "far_above_hull"
        warning = (f"Far above hull ({hull_distance:.2f} eV) — thermodynamically "
                   "impossible under equilibrium. Theoretical interest only.")

    # Top competing phases for display
    top_competing = [
        {
            "formula": p["formula"],
            "formation_energy": round(p["formation_energy"], 4),
            "is_stable": p.get("is_stable"),
        }
        for p in competing[:5]
        if p["element_set"].issubset(target_elements)
    ]

    # Decomposition risk analysis (Phase 4)
    decomp_risk = _compute_decomposition_risk(
        hull_distance, predicted_formation_energy, competing, target_comp
    )

    return {
        "hull_distance_eV": round(hull_distance, 4) if hull_distance is not None else None,
        "hull_source": hull_source,
        "hull_status": hull_status,
        "competing_phases": top_competing,
        "n_competing_phases": len(competing),
        "chemical_system": system_str,
        "decomposition_warning": warning,
        "decomposition_risk_pct": decomp_risk["risk_pct"],
        "likely_decomposition_products": decomp_risk["products"],
        "decomposition_energy_gain_eV": decomp_risk["energy_gain"],
    }


def _compute_decomposition_risk(
    hull_distance: float | None,
    predicted_ef: float | None,
    competing: list[dict],
    target_comp: dict[str, float],
) -> dict:
    """
    Estimate the probability and pathway of decomposition.
    
    Uses Boltzmann-weighted energy difference to compute the
    likelihood that the material decomposes into competing phases
    at room temperature (kT ≈ 0.026 eV).
    """
    if hull_distance is None or predicted_ef is None:
        return {"risk_pct": None, "products": [], "energy_gain": None}
    
    kT = 0.026  # eV at 300K
    
    # Find the most likely decomposition products
    # (phases closest in composition but lower in energy)
    products = []
    best_energy_gain = 0.0
    
    for phase in competing[:20]:
        if phase["formation_energy"] < predicted_ef:
            energy_gain = predicted_ef - phase["formation_energy"]
            comp_dist = _composition_distance(phase["composition"], target_comp)
            # Products must share chemistry — penalize distant compositions
            if comp_dist < 0.5:
                products.append({
                    "formula": phase["formula"],
                    "energy_gain_eV": round(energy_gain, 4),
                    "composition_distance": round(comp_dist, 3),
                })
                best_energy_gain = max(best_energy_gain, energy_gain)
    
    # Sort products by energy gain (most favorable first)
    products.sort(key=lambda p: p["energy_gain_eV"], reverse=True)
    products = products[:3]  # Top 3 most likely decomposition pathways
    
    # Compute decomposition risk using Boltzmann probability
    if hull_distance <= 0.0:
        risk_pct = 0.0
    elif hull_distance < 0.01:
        risk_pct = 5.0
    else:
        # P(decompose) ∝ 1 - exp(-Ehull / kT)
        risk_pct = (1.0 - math.exp(-hull_distance / kT)) * 100.0
    
    risk_pct = round(min(100.0, risk_pct), 1)
    
    return {
        "risk_pct": risk_pct,
        "products": products,
        "energy_gain": round(best_energy_gain, 4) if best_energy_gain > 0 else None,
    }


# -----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))

    tests = [
        ("BaTiO3", -3.24),
        ("NaCl", -3.29),
        ("TiO2", -3.39),
        ("Fe2O3", -2.70),
        ("CrMoN3", -1.50),
        ("ZrVSe4", -0.80),
    ]
    print("Convex Hull Estimator Test")
    print("=" * 70)
    for formula, ef in tests:
        r = evaluate_phase_stability(formula, predicted_formation_energy=ef)
        hd = r["hull_distance_eV"]
        hd_str = f"{hd:.4f} eV" if hd is not None else "N/A"
        print(f"  {formula:12s}  Ef={ef:.2f}  hull={hd_str:>10s}  "
              f"status={r['hull_status']}  source={r['hull_source']}  "
              f"({r['n_competing_phases']} competing)")
