"""
Stability Analyzer — Approximate Energy Above Hull.

Provides fast thermodynamic stability estimation without full DFT
by using a local database of reference formation energies from
the Materials Project.

The energy above the convex hull (e_above_hull) indicates how
thermodynamically stable a compound is relative to its decomposition
products:
  - e_above_hull = 0   → on the hull (stable)
  - e_above_hull < 0.05 → metastable (potentially synthesizable)
  - e_above_hull > 0.1  → likely unstable (decomposes)
"""

from __future__ import annotations

import re


# -----------------------------------------------------------------------
# Reference Formation Energies (eV/atom) from Materials Project
# These represent the convex hull endpoints for common systems
# -----------------------------------------------------------------------
REFERENCE_ENERGIES: dict[str, float] = {
    # Binary oxides
    "Li2O": -2.013, "Na2O": -1.560, "K2O": -1.340, "MgO": -3.099,
    "CaO": -3.360, "SrO": -3.140, "BaO": -2.885, "Al2O3": -3.457,
    "TiO2": -3.393, "Ti2O3": -3.320, "VO2": -2.870, "V2O5": -2.590,
    "Cr2O3": -3.580, "MnO": -2.610, "MnO2": -2.520, "Fe2O3": -2.700,
    "FeO": -2.210, "CoO": -1.950, "NiO": -1.610, "CuO": -1.150,
    "Cu2O": -0.610, "ZnO": -1.780, "Ga2O3": -2.830, "GeO2": -2.320,
    "SiO2": -3.188, "SnO2": -2.260, "In2O3": -2.430, "ZrO2": -3.710,
    "HfO2": -3.810, "Nb2O5": -3.140, "MoO3": -2.470, "WO3": -2.680,
    "Y2O3": -3.750, "La2O3": -3.860, "CeO2": -3.690, "Ta2O5": -3.380,
    "PbO": -1.250, "Bi2O3": -1.720,

    # Binary halides
    "NaCl": -3.293, "KCl": -3.060, "LiCl": -2.940, "LiF": -3.720,
    "NaF": -3.460, "KF": -3.310, "CaF2": -3.940, "MgF2": -3.680,
    "BaF2": -3.580, "AlF3": -3.880, "TiF4": -3.240, "CuCl": -0.880,

    # Ternary perovskites
    "BaTiO3": -3.244, "SrTiO3": -3.385, "CaTiO3": -3.380,
    "BaZrO3": -3.420, "SrVO3": -2.810, "LaAlO3": -3.750,
    "KNbO3": -2.940, "NaNbO3": -2.710, "BiFeO3": -2.150,
    "LaMnO3": -3.060, "LaCoO3": -2.480,

    # Spinels and garnets
    "MgAl2O4": -3.440, "CoFe2O4": -2.360, "NiFe2O4": -2.290,
    "Fe3O4": -2.510, "ZnFe2O4": -2.180, "MnFe2O4": -2.350,

    # Olivines and battery materials
    "LiFePO4": -3.590, "LiCoO2": -2.460, "LiMnO2": -2.750,
    "LiNiO2": -2.100,

    # Sulfides and selenides
    "ZnS": -1.120, "CdS": -0.830, "PbS": -0.630, "MoS2": -1.340,
    "WS2": -1.340, "FeS2": -0.810, "Bi2S3": -0.620,
    "CdSe": -0.620, "PbSe": -0.530, "Bi2Se3": -0.480,
    "Cu2S": -0.350, "CdTe": -0.380,

    # Nitrides
    "BN": -2.360, "Si3N4": -2.370, "AlN": -3.270, "GaN": -1.240,
    "TiN": -3.420, "ZrN": -3.570, "VN": -2.580, "CrN": -2.100,

    # Carbides
    "SiC": -0.720, "TiC": -1.950, "WC": -0.480, "ZrC": -2.080,
    "NbC": -1.720,

    # Simple elements (reference: 0 by convention)
}

# Elemental reference energies (per atom)
ELEMENTAL_ENERGIES: dict[str, float] = {
    "H": 0.0, "Li": 0.0, "Be": 0.0, "B": 0.0, "C": 0.0,
    "N": 0.0, "O": 0.0, "F": 0.0, "Na": 0.0, "Mg": 0.0,
    "Al": 0.0, "Si": 0.0, "P": 0.0, "S": 0.0, "Cl": 0.0,
    "K": 0.0, "Ca": 0.0, "Sc": 0.0, "Ti": 0.0, "V": 0.0,
    "Cr": 0.0, "Mn": 0.0, "Fe": 0.0, "Co": 0.0, "Ni": 0.0,
    "Cu": 0.0, "Zn": 0.0, "Ga": 0.0, "Ge": 0.0, "As": 0.0,
    "Se": 0.0, "Br": 0.0, "Sr": 0.0, "Y": 0.0, "Zr": 0.0,
    "Nb": 0.0, "Mo": 0.0, "Ru": 0.0, "Pd": 0.0, "Ag": 0.0,
    "Cd": 0.0, "In": 0.0, "Sn": 0.0, "Sb": 0.0, "Te": 0.0,
    "I": 0.0, "Ba": 0.0, "La": 0.0, "Ce": 0.0, "Hf": 0.0,
    "Ta": 0.0, "W": 0.0, "Pt": 0.0, "Au": 0.0, "Pb": 0.0,
    "Bi": 0.0,
}


def _parse_formula(formula: str) -> list[tuple[str, int]]:
    """Parse formula → [(element, count), ...]."""
    return [(s, int(c) if c else 1)
            for s, c in re.findall(r'([A-Z][a-z]?)(\d*)', formula) if s]


def estimate_e_above_hull(
    formula: str,
    predicted_formation_energy: float | None = None,
) -> dict:
    """
    Estimate the energy above the convex hull for a compound.

    Uses a simplified approach:  compare the predicted formation energy
    against known reference compounds in the same chemical system.

    Args:
        formula: chemical formula (e.g. "BaTiO3")
        predicted_formation_energy: GNN-predicted eV/atom

    Returns:
        dict with e_above_hull estimate, stability classification, and
        competing phase information
    """
    parsed = _parse_formula(formula)
    if not parsed or predicted_formation_energy is None:
        return {
            "e_above_hull_eV": None,
            "stability": "unknown",
            "competing_phases": [],
        }

    elements = [s for s, _ in parsed]
    total_atoms = sum(c for _, c in parsed)

    # Find known compounds in the same chemical system
    competing = []
    for ref_formula, ref_energy in REFERENCE_ENERGIES.items():
        ref_parsed = _parse_formula(ref_formula)
        ref_elements = {s for s, _ in ref_parsed}
        # Competing phase if it's a subset of our element system
        if ref_elements.issubset(set(elements)):
            competing.append({
                "formula": ref_formula,
                "formation_energy_eV": ref_energy,
            })

    # Check if this exact formula is in the database
    exact_match = REFERENCE_ENERGIES.get(formula)

    # Estimate e_above_hull
    if exact_match is not None:
        # Known compound — compare predicted vs reference
        e_above_hull = predicted_formation_energy - exact_match
    elif competing:
        # Not in database — compare against most stable competing phase
        most_stable = min(c["formation_energy_eV"] for c in competing)
        # Approximate: e_above_hull ≈ predicted_energy - interpolated_hull
        e_above_hull = predicted_formation_energy - most_stable
        # Adjust for the fact that the hull is typically below any single reference
        e_above_hull = max(0, e_above_hull + 0.1)  # Small positive offset
    else:
        # No competing phases known — can't estimate
        e_above_hull = None

    # -------------------------------------------------------------------
    # Classify stability using research-grade Ehull tiers
    # Based on Materials Project convention and thermodynamic principles
    # -------------------------------------------------------------------
    if e_above_hull is None:
        stability = "unknown"
        stability_tier = 0       # 0 = unknown
        stability_label = "Unknown (insufficient data)"
        stability_warning = None
    elif e_above_hull <= 0.0:
        stability = "stable (on hull)"
        stability_tier = 1       # best
        stability_label = "Thermodynamically Stable"
        stability_warning = None
    elif e_above_hull < 0.05:
        stability = "thermodynamically_stable"
        stability_tier = 1
        stability_label = "Thermodynamically Stable"
        stability_warning = None
    elif e_above_hull < 0.2:
        stability = "metastable"
        stability_tier = 2
        stability_label = "Metastable"
        stability_warning = ("Metastable phase — may be synthesizable under "
                             "non-equilibrium conditions (e.g., thin films, "
                             "high pressure).")
    elif e_above_hull < 0.5:
        stability = "high_energy_metastable"
        stability_tier = 3
        stability_label = "High-energy Metastable"
        stability_warning = ("High-energy metastable phase — unlikely to form "
                             "spontaneously. Requires kinetic trapping or "
                             "epitaxial stabilization.")
    elif e_above_hull < 1.0:
        stability = "unstable"
        stability_tier = 4
        stability_label = "Unstable"
        stability_warning = ("Thermodynamically unstable — decomposes into "
                             "competing phases under equilibrium. Not suitable "
                             "for bulk applications.")
    else:
        stability = "highly_unstable"
        stability_tier = 5       # worst
        stability_label = "Highly Unstable"
        stability_warning = ("Highly unstable (Ehull > 1.0 eV) — this phase "
                             "will not form under any equilibrium conditions. "
                             "Theoretical interest only.")

    return {
        "e_above_hull_eV": round(e_above_hull, 4) if e_above_hull is not None else None,
        "stability": stability,
        "stability_tier": stability_tier,
        "stability_label": stability_label,
        "stability_warning": stability_warning,
        "competing_phases": sorted(competing, key=lambda x: x["formation_energy_eV"])[:5],
        "reference_available": exact_match is not None,
        "formula": formula,
    }


# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("Stability Analyzer Test")
    print("=" * 60)

    tests = [
        ("TiO2", -3.39),
        ("NaCl", -3.29),
        ("BaTiO3", -3.24),
        ("Fe2O3", -2.70),
        ("LiCoO2", -2.45),
        ("CrMoN3", -1.50),
        ("AlTiO3", -3.30),
    ]

    for formula, energy in tests:
        result = estimate_e_above_hull(formula, energy)
        hull = result["e_above_hull_eV"]
        hull_str = f"{hull:.4f} eV" if hull is not None else "N/A"
        print(f"  {formula:12s} E_f={energy:.2f}  e_hull={hull_str:>10s}  "
              f"→ {result['stability']}")
        if result["competing_phases"]:
            top = result["competing_phases"][0]
            print(f"{'':14s} Most stable competing: {top['formula']} "
                  f"({top['formation_energy_eV']:.3f} eV)")
