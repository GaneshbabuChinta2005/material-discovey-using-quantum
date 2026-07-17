"""
Feature Engine — Advanced feature generation from chemical formulas.

Generates ~35 features per compound using real elemental properties:
 - Composition statistics (mean, std, min, max of EN, mass, radius, IE, EA, valence)
 - Cross-interaction terms (EN × radius, IE × EA, etc.)
 - Periodic table group/period indicators
 - Ionic character, size mismatch, density proxy
 - Band gap estimator features (EN range, polarity index)

Used by both the discovery pipeline (for novel prediction) and
the training pipeline (for building the model).
"""

import re
import math
import numpy as np
from app.utils.element_loader import load_all_elements

# -----------------------------------------------------------------------
# Property tables
# -----------------------------------------------------------------------
IONIZATION_ENERGY = {
    "H": 13.6, "He": 24.6, "Li": 5.4, "Be": 9.3, "B": 8.3, "C": 11.3,
    "N": 14.5, "O": 13.6, "F": 17.4, "Ne": 21.6, "Na": 5.1, "Mg": 7.6,
    "Al": 6.0, "Si": 8.2, "P": 10.5, "S": 10.4, "Cl": 13.0, "Ar": 15.8,
    "K": 4.3, "Ca": 6.1, "Sc": 6.6, "Ti": 6.8, "V": 6.7, "Cr": 6.8,
    "Mn": 7.4, "Fe": 7.9, "Co": 7.9, "Ni": 7.6, "Cu": 7.7, "Zn": 9.4,
    "Ga": 6.0, "Ge": 7.9, "As": 9.8, "Se": 9.8, "Br": 11.8, "Kr": 14.0,
    "Rb": 4.2, "Sr": 5.7, "Y": 6.2, "Zr": 6.6, "Nb": 6.8, "Mo": 7.1,
    "Ru": 7.4, "Rh": 7.5, "Pd": 8.3, "Ag": 7.6, "Cd": 9.0, "In": 5.8,
    "Sn": 7.3, "Sb": 8.6, "Te": 9.0, "I": 10.5, "Cs": 3.9, "Ba": 5.2,
    "La": 5.6, "Ce": 5.5, "Hf": 6.8, "Ta": 7.5, "W": 7.9, "Re": 7.8,
    "Os": 8.4, "Ir": 9.1, "Pt": 9.0, "Au": 9.2, "Pb": 7.4, "Bi": 7.3,
    "Nd": 5.5, "Pr": 5.5, "Sm": 5.6, "Eu": 5.7, "Gd": 6.1, "Tb": 5.9,
    "Dy": 5.9, "Ho": 6.0, "Er": 6.1, "Tm": 6.2, "Yb": 6.3, "Lu": 5.4,
    "Tl": 6.1, "Po": 8.4, "At": 9.3, "Fr": 4.1, "Ra": 5.3,
    "Ac": 5.4, "Th": 6.3, "Pa": 5.9, "U": 6.2, "Np": 6.3, "Pu": 6.0,
}

ELECTRON_AFFINITY = {
    "H": 0.75, "Li": 0.62, "Be": 0.0, "B": 0.28, "C": 1.26, "N": 0.0,
    "O": 1.46, "F": 3.40, "Na": 0.55, "Mg": 0.0, "Al": 0.43, "Si": 1.39,
    "P": 0.75, "S": 2.08, "Cl": 3.61, "K": 0.50, "Ca": 0.02, "Sc": 0.19,
    "Ti": 0.08, "V": 0.53, "Cr": 0.67, "Mn": 0.0, "Fe": 0.15, "Co": 0.66,
    "Ni": 1.16, "Cu": 1.24, "Zn": 0.0, "Ga": 0.43, "Ge": 1.23,
    "As": 0.80, "Se": 2.02, "Br": 3.36, "Rb": 0.49, "Sr": 0.05,
    "Y": 0.31, "Zr": 0.43, "Nb": 0.92, "Mo": 0.75, "Ru": 1.05,
    "Rh": 1.14, "Pd": 0.56, "Ag": 1.30, "Cd": 0.0, "In": 0.40,
    "Sn": 1.11, "Sb": 1.05, "Te": 1.97, "I": 3.06, "Cs": 0.47,
    "Ba": 0.14, "La": 0.47, "Ce": 0.55, "Hf": 0.0, "Ta": 0.32,
    "W": 0.82, "Re": 0.15, "Pt": 2.13, "Au": 2.31, "Pb": 0.36, "Bi": 0.95,
    "Nd": 0.0, "Pr": 0.0, "Sm": 0.0, "Eu": 0.0, "Gd": 0.0, "Tb": 0.0,
    "Dy": 0.0, "Ho": 0.0, "Er": 0.0, "Tm": 1.03, "Yb": -0.02, "Lu": 0.34,
    "Tl": 0.38, "Po": 1.9, "At": 2.8,
}

ATOMIC_RADIUS = {
    "H": 0.53, "He": 0.31, "Li": 1.67, "Be": 1.12, "B": 0.87, "C": 0.67,
    "N": 0.56, "O": 0.48, "F": 0.42, "Ne": 0.38, "Na": 1.90, "Mg": 1.45,
    "Al": 1.18, "Si": 1.11, "P": 0.98, "S": 0.88, "Cl": 0.79, "Ar": 0.71,
    "K": 2.43, "Ca": 1.94, "Sc": 1.84, "Ti": 1.76, "V": 1.71, "Cr": 1.66,
    "Mn": 1.61, "Fe": 1.56, "Co": 1.52, "Ni": 1.49, "Cu": 1.45, "Zn": 1.42,
    "Ga": 1.36, "Ge": 1.25, "As": 1.14, "Se": 1.03, "Br": 0.94,
    "Rb": 2.65, "Sr": 2.19, "Y": 2.12, "Zr": 2.06, "Nb": 1.98, "Mo": 1.90,
    "Ru": 1.78, "Rh": 1.73, "Pd": 1.69, "Ag": 1.65, "Cd": 1.61,
    "In": 1.56, "Sn": 1.45, "Sb": 1.33, "Te": 1.23, "I": 1.15,
    "Cs": 2.98, "Ba": 2.53, "La": 2.26, "Ce": 2.10, "Hf": 2.08,
    "Ta": 2.00, "W": 1.93, "Re": 1.88, "Os": 1.85, "Ir": 1.80,
    "Pt": 1.77, "Au": 1.74, "Pb": 1.54, "Bi": 1.43,
    "Nd": 2.01, "Pr": 2.03, "Sm": 1.98, "Eu": 1.98, "Gd": 1.96,
    "Tb": 1.94, "Dy": 1.92, "Ho": 1.92, "Er": 1.89, "Tm": 1.90,
    "Yb": 1.87, "Lu": 1.87, "Tl": 1.56, "Po": 1.40, "At": 1.27,
}

VALENCE_ELECTRONS = {
    "H": 1, "He": 2, "Li": 1, "Be": 2, "B": 3, "C": 4,
    "N": 5, "O": 6, "F": 7, "Ne": 8, "Na": 1, "Mg": 2,
    "Al": 3, "Si": 4, "P": 5, "S": 6, "Cl": 7, "Ar": 8,
    "K": 1, "Ca": 2, "Sc": 3, "Ti": 4, "V": 5, "Cr": 6,
    "Mn": 7, "Fe": 8, "Co": 9, "Ni": 10, "Cu": 11, "Zn": 12,
    "Ga": 3, "Ge": 4, "As": 5, "Se": 6, "Br": 7, "Kr": 8,
    "Rb": 1, "Sr": 2, "Y": 3, "Zr": 4, "Nb": 5, "Mo": 6,
    "Ru": 8, "Rh": 9, "Pd": 10, "Ag": 11, "Cd": 12,
    "In": 3, "Sn": 4, "Sb": 5, "Te": 6, "I": 7,
    "Cs": 1, "Ba": 2, "La": 3, "Ce": 4, "Hf": 4, "Ta": 5,
    "W": 6, "Re": 7, "Os": 8, "Ir": 9, "Pt": 10, "Au": 11,
    "Pb": 4, "Bi": 5,
    "Nd": 4, "Pr": 3, "Sm": 4, "Eu": 3, "Gd": 4, "Tb": 3,
    "Dy": 4, "Ho": 3, "Er": 4, "Tm": 3, "Yb": 2, "Lu": 3,
    "Tl": 3, "Po": 6, "At": 7,
}

# Periodic table period numbers
PERIOD = {
    "H": 1, "He": 1,
    "Li": 2, "Be": 2, "B": 2, "C": 2, "N": 2, "O": 2, "F": 2, "Ne": 2,
    "Na": 3, "Mg": 3, "Al": 3, "Si": 3, "P": 3, "S": 3, "Cl": 3, "Ar": 3,
    "K": 4, "Ca": 4, "Sc": 4, "Ti": 4, "V": 4, "Cr": 4, "Mn": 4,
    "Fe": 4, "Co": 4, "Ni": 4, "Cu": 4, "Zn": 4, "Ga": 4, "Ge": 4,
    "As": 4, "Se": 4, "Br": 4, "Kr": 4,
    "Rb": 5, "Sr": 5, "Y": 5, "Zr": 5, "Nb": 5, "Mo": 5, "Ru": 5,
    "Rh": 5, "Pd": 5, "Ag": 5, "Cd": 5, "In": 5, "Sn": 5, "Sb": 5,
    "Te": 5, "I": 5, "Cs": 6, "Ba": 6, "La": 6, "Ce": 6,
    "Nd": 6, "Pr": 6, "Sm": 6, "Eu": 6, "Gd": 6, "Tb": 6,
    "Dy": 6, "Ho": 6, "Er": 6, "Tm": 6, "Yb": 6, "Lu": 6,
    "Hf": 6, "Ta": 6, "W": 6, "Re": 6, "Os": 6, "Ir": 6,
    "Pt": 6, "Au": 6, "Pb": 6, "Bi": 6, "Tl": 6, "Po": 6, "At": 6,
}

# Mendeleev number (ordering on periodic table for machine learning)
MENDELEEV_NO = {
    "H": 92, "He": 98, "Li": 1, "Be": 67, "B": 72, "C": 77, "N": 82, "O": 87, "F": 93, "Ne": 99,
    "Na": 2, "Mg": 68, "Al": 73, "Si": 78, "P": 83, "S": 88, "Cl": 94, "Ar": 100,
    "K": 3, "Ca": 7, "Sc": 11, "Ti": 43, "V": 44, "Cr": 45, "Mn": 46,
    "Fe": 47, "Co": 48, "Ni": 49, "Cu": 50, "Zn": 51, "Ga": 74, "Ge": 79,
    "As": 84, "Se": 89, "Br": 95, "Kr": 101,
    "Rb": 4, "Sr": 8, "Y": 12, "Zr": 52, "Nb": 53, "Mo": 54, "Ru": 55,
    "Rh": 56, "Pd": 57, "Ag": 58, "Cd": 59, "In": 75, "Sn": 80, "Sb": 85,
    "Te": 90, "I": 96, "Cs": 5, "Ba": 9,
    "La": 13, "Ce": 14, "Pr": 15, "Nd": 16, "Sm": 17, "Eu": 18, "Gd": 19,
    "Tb": 20, "Dy": 21, "Ho": 22, "Er": 23, "Tm": 24, "Yb": 25, "Lu": 26,
    "Hf": 60, "Ta": 61, "W": 62, "Re": 63, "Os": 64, "Ir": 65,
    "Pt": 66, "Au": 69, "Pb": 81, "Bi": 86, "Tl": 76, "Po": 91, "At": 97,
}

# Module-level cache
_elems = None

FEATURE_NAMES = [
    # Basic composition (2)
    "n_atoms",
    "n_elements",
    # Electronegativity (5)
    "en_mean",
    "en_std",
    "en_diff",
    "en_min",
    "en_max",
    # Atomic mass (3)
    "mass_mean",
    "mass_std",
    "mass_total",
    # Atomic radius (4)
    "radius_mean",
    "radius_std",
    "radius_diff",
    "radius_max",
    # Ionization energy (3)
    "ie_mean",
    "ie_std",
    "ie_diff",
    # Electron affinity (3)
    "ea_mean",
    "ea_std",
    "ea_diff",
    # Valence (3)
    "valence_mean",
    "valence_std",
    "valence_total",
    # Derived / interaction features (9)
    "ionic_character",
    "size_mismatch",
    "avg_density_proxy",
    "en_x_radius",         # cross: electronegativity × radius
    "ie_x_ea",             # cross: ionization energy × electron affinity
    "polarity_index",      # (en_max - en_min) / en_mean
    "packing_efficiency",  # radius_min / radius_max
    "period_mean",         # average period number
    "mendeleev_range",     # max - min Mendeleev number (chemical distance)
    # Nonlinear features (3)
    "en_diff_sq",          # en_diff squared (Pauling formula)
    "mass_ratio",          # max_mass / min_mass
    "valence_per_atom",    # total_valence / n_atoms
]


def _load():
    global _elems
    if _elems is None:
        _elems = load_all_elements()
    return _elems


def _get(sym: str, prop: str, default=0.0) -> float:
    """Get property from loaded element data."""
    elems = _load()
    info = elems.get(sym, {})
    val = info.get(prop, default)
    try:
        return float(val) if val else default
    except (ValueError, TypeError):
        return default


def _parse(formula: str) -> list[tuple[str, int]]:
    """Parse formula → [(symbol, count), ...]"""
    parts = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    return [(s, int(c) if c else 1) for s, c in parts if s]


def _weighted_stats(values: list[float], weights: list[int]):
    """Compute weighted mean, std, min, max."""
    expanded = []
    for v, w in zip(values, weights):
        expanded.extend([v] * w)
    arr = np.array(expanded, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())


def compound_to_features(formula: str) -> list[float] | None:
    """
    Generate feature vector from a chemical formula.

    Returns:
        List of 35 floats matching FEATURE_NAMES, or None if formula is invalid.
    """
    parsed = _parse(formula)
    if not parsed:
        return None

    syms = [s for s, _ in parsed]
    counts = [c for _, c in parsed]
    total_atoms = sum(counts)
    n_elements = len(syms)

    if total_atoms == 0:
        return None

    # Gather per-element properties
    en_vals = [_get(s, "Electronegativity") or 2.0 for s in syms]
    mass_vals = [_get(s, "Atomic_weight") or 50.0 for s in syms]
    radius_vals = [ATOMIC_RADIUS.get(s, 1.5) for s in syms]
    ie_vals = [IONIZATION_ENERGY.get(s, 7.0) for s in syms]
    ea_vals = [ELECTRON_AFFINITY.get(s, 0.5) for s in syms]
    val_vals = [VALENCE_ELECTRONS.get(s, 4) for s in syms]
    period_vals = [PERIOD.get(s, 4) for s in syms]
    mend_vals = [MENDELEEV_NO.get(s, 50) for s in syms]

    # Weighted statistics
    en_mean, en_std, en_min, en_max = _weighted_stats(en_vals, counts)
    mass_mean, mass_std, mass_min, mass_max = _weighted_stats(mass_vals, counts)
    radius_mean, radius_std, r_min, r_max = _weighted_stats(radius_vals, counts)
    ie_mean, ie_std, ie_min, ie_max = _weighted_stats(ie_vals, counts)
    ea_mean, ea_std, ea_min, ea_max = _weighted_stats(ea_vals, counts)
    val_mean, val_std, _, _ = _weighted_stats(val_vals, counts)
    period_mean, _, _, _ = _weighted_stats(period_vals, counts)

    # Basic diffs
    en_diff = en_max - en_min
    radius_diff = r_max - r_min
    ie_diff = ie_max - ie_min
    ea_diff = ea_max - ea_min
    mass_total = sum(m * c for m, c in zip(mass_vals, counts))
    valence_total = sum(v * c for v, c in zip(val_vals, counts))

    # Ionic character (Pauling formula)
    ionic_character = 1.0 - math.exp(-0.25 * en_diff * en_diff) if en_diff > 0 else 0.0

    # Size mismatch (normalized)
    size_mismatch = sum(
        abs(r - radius_mean) * c for r, c in zip(radius_vals, counts)
    ) / total_atoms

    # Density proxy
    avg_density_proxy = mass_total / (
        sum(r ** 3 * c for r, c in zip(radius_vals, counts)) + 1e-6
    )

    # Cross-interaction features
    en_x_radius = en_mean * radius_mean
    ie_x_ea = ie_mean * (ea_mean + 0.01)

    # Polarity index
    polarity_index = en_diff / (en_mean + 1e-6)

    # Packing efficiency
    packing_efficiency = (r_min / r_max) if r_max > 0 else 1.0

    # Mendeleev range (chemical distance)
    mend_range = float(max(mend_vals) - min(mend_vals)) if len(mend_vals) > 1 else 0.0

    # Nonlinear features
    en_diff_sq = en_diff ** 2
    mass_ratio = (mass_max / mass_min) if mass_min > 0 else 1.0
    valence_per_atom = valence_total / total_atoms

    return [
        float(total_atoms),
        float(n_elements),
        round(en_mean, 5),
        round(en_std, 5),
        round(en_diff, 5),
        round(en_min, 5),
        round(en_max, 5),
        round(mass_mean, 5),
        round(mass_std, 5),
        round(mass_total, 5),
        round(radius_mean, 5),
        round(radius_std, 5),
        round(radius_diff, 5),
        round(r_max, 5),
        round(ie_mean, 5),
        round(ie_std, 5),
        round(ie_diff, 5),
        round(ea_mean, 5),
        round(ea_std, 5),
        round(ea_diff, 5),
        round(val_mean, 5),
        round(val_std, 5),
        round(valence_total, 5),
        round(ionic_character, 5),
        round(size_mismatch, 5),
        round(avg_density_proxy, 5),
        round(en_x_radius, 5),
        round(ie_x_ea, 5),
        round(polarity_index, 5),
        round(packing_efficiency, 5),
        round(period_mean, 5),
        round(mend_range, 5),
        round(en_diff_sq, 5),
        round(mass_ratio, 5),
        round(valence_per_atom, 5),
    ]


def features_from_dataframe_row(row) -> list[float]:
    """Generate features from a dataset row (for training)."""
    return compound_to_features(row["formula"])


# -----------------------------------------------------------------------
if __name__ == "__main__":
    tests = ["H2O", "Al2O3", "TiO2", "LiCoO2", "Fe2O3", "NaCl"]
    for f in tests:
        feats = compound_to_features(f)
        if feats:
            print(f"\n{f}: ({len(feats)} features)")
            for name, val in zip(FEATURE_NAMES, feats):
                print(f"  {name:22s} = {val}")
