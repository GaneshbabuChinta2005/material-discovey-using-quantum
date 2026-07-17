"""
Dataset Generator — Creates a comprehensive inorganic materials dataset
derived from element properties and known crystal structure patterns.

Since the Materials Project API requires registration, this generates a
high-quality synthetic dataset based on real elemental data from our CSV files,
combined with empirical rules for formation energy, band gap, and stability.

The resulting dataset covers:
  - ~50K inorganic compounds (binary + ternary)
  - Formation energy (eV/atom) from electronegativity differences
  - Band gap (eV) from element orbital energies
  - Density, crystal system, stability predictions
"""

import os
import sys
import math
import random
import itertools
import pandas as pd
import numpy as np

TARGET_MIN_COMPOUNDS = 50000
TARGET_MAX_COMPOUNDS = 70000
ANION_PRIORITY = ["O", "F", "Cl"]

METAL_ELEMENTS = {
    "Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd",
    "La", "Ce", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
    "Al", "Ga", "In", "Sn", "Pb", "Bi"
}

NONMETAL_ELEMENTS = {
    "H", "C", "N", "O", "F", "P", "S", "Cl", "Se", "Br", "I", "Te"
}

# Ensure we can import from app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.utils.element_loader import load_all_elements

# -----------------------------------------------------------------------
# Empirical property tables
# -----------------------------------------------------------------------

# Common oxidation states for generating valid compounds
OXIDATION_STATES = {
    "H":  [1, -1],    "Li": [1],      "Be": [2],      "B":  [3],
    "C":  [4, -4, 2], "N":  [-3, 3, 5],"O": [-2],      "F":  [-1],
    "Na": [1],        "Mg": [2],       "Al": [3],      "Si": [4, -4],
    "P":  [-3, 3, 5], "S":  [-2, 4, 6],"Cl": [-1, 1],  "K":  [1],
    "Ca": [2],        "Sc": [3],       "Ti": [2, 3, 4],"V":  [2, 3, 4, 5],
    "Cr": [2, 3, 6],  "Mn": [2, 3, 4, 7],"Fe":[2, 3],  "Co": [2, 3],
    "Ni": [2, 3],     "Cu": [1, 2],    "Zn": [2],      "Ga": [3],
    "Ge": [2, 4],     "As": [-3, 3, 5],"Se": [-2, 4, 6],"Br":[-1],
    "Rb": [1],        "Sr": [2],       "Y":  [3],      "Zr": [4],
    "Nb": [3, 5],     "Mo": [2, 3, 4, 5, 6],"Ru":[2, 3, 4],
    "Rh": [3],        "Pd": [2, 4],    "Ag": [1],      "Cd": [2],
    "In": [3],        "Sn": [2, 4],    "Sb": [-3, 3, 5],
    "Te": [-2, 4, 6], "I":  [-1, 1],   "Cs": [1],      "Ba": [2],
    "La": [3],        "Ce": [3, 4],    "Hf": [4],      "Ta": [5],
    "W":  [2, 4, 6],  "Re": [2, 4, 7], "Os": [2, 3, 4],
    "Ir": [3, 4],     "Pt": [2, 4],    "Au": [1, 3],   "Pb": [2, 4],
    "Bi": [3, 5],
}


def get_positive_state(sym: str) -> int:
    states = [s for s in OXIDATION_STATES.get(sym, []) if s > 0]
    if states:
        return states[0]
    return 1


def get_negative_state(sym: str) -> int:
    states = [s for s in OXIDATION_STATES.get(sym, []) if s < 0]
    if states:
        return states[0]
    return -1

# Approximate ionization energies (eV) for feature engineering
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
}

# Approximate electron affinity (eV)
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
}

# Atomic radii (Angstroms)
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
}

CRYSTAL_SYSTEMS = [
    "cubic", "hexagonal", "tetragonal", "orthorhombic",
    "monoclinic", "triclinic", "trigonal"
]


def get_elem_prop(sym: str, elems: dict, prop: str, default=0.0):
    """Get element property from loaded CSV data."""
    info = elems.get(sym, {})
    val = info.get(prop, default)
    try:
        return float(val) if val else default
    except (ValueError, TypeError):
        return default


def estimate_formation_energy(syms: list[str], counts: list[int],
                               elems: dict) -> float:
    """
    Estimate formation energy (eV/atom) using Miedema-inspired model:
    ΔHf ≈ f(electronegativity_diff, atomic_radius_mismatch, electron_density)
    """
    total_atoms = sum(counts)
    if total_atoms == 0 or len(syms) < 2:
        return 0.0

    # Weighted average properties
    en_vals = [get_elem_prop(s, elems, "Electronegativity") for s in syms]
    rad_vals = [ATOMIC_RADIUS.get(s, 1.5) for s in syms]
    ie_vals = [IONIZATION_ENERGY.get(s, 7.0) for s in syms]

    # Electronegativity difference (drives ionic bonding → negative ΔHf)
    en_diff = 0.0
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            en_diff += abs(en_vals[i] - en_vals[j]) * counts[i] * counts[j]
    en_diff /= (total_atoms * total_atoms)

    # Size mismatch (destabilizes)
    r_avg = sum(r * c for r, c in zip(rad_vals, counts)) / total_atoms
    size_mismatch = sum(
        abs(r - r_avg) * c for r, c in zip(rad_vals, counts)
    ) / total_atoms

    # Simplified Miedema-inspired formation energy model
    # Electronegativity difference drives negative formation energy (ionic bonding)
    # Size mismatch destabilizes slightly
    energy = -1.8 * en_diff - 0.2 * size_mismatch

    # Add small controlled noise for realism
    energy += random.gauss(0, 0.04)

    return round(max(-5.0, min(2.0, energy)), 4)


def estimate_band_gap(syms: list[str], counts: list[int],
                       elems: dict) -> float:
    """
    Estimate band gap (eV) based on element electronegativities and types.

    Rules:
    - Metals (all transition metals): ~0 eV
    - Metal + nonmetal: depends on EN difference (ionic → wide gap)
    - Nonmetal + nonmetal: moderate gap
    """
    total_atoms = sum(counts)
    en_vals = [get_elem_prop(s, elems, "Electronegativity") for s in syms]

    # Classify elements
    has_metal = any(s in METAL_ELEMENTS for s in syms)
    has_nonmetal = any(s in NONMETAL_ELEMENTS for s in syms)
    all_metals = all(s in METAL_ELEMENTS for s in syms)

    # Electronegativity difference
    en_diff = max(en_vals) - min(en_vals) if en_vals else 0.0

    # Deterministic band gap rules based on element types
    if all_metals:
        # Pure metals have zero or near-zero band gap
        gap = 0.0
    elif has_metal and has_nonmetal:
        # Ionic/covalent compounds: gap scales with EN difference
        gap = en_diff * 2.0
    else:
        # Nonmetal–nonmetal compounds: moderate gap
        gap = en_diff * 1.2

    # Small noise for realism
    gap += random.gauss(0, 0.03)

    return round(max(0.0, min(10.0, gap)), 3)


def estimate_density(syms: list[str], counts: list[int], elems: dict) -> float:
    """Estimate density from atomic masses and radii."""
    total_atoms = sum(counts)
    total_mass = sum(
        get_elem_prop(s, elems, "Atomic_weight") * c
        for s, c in zip(syms, counts)
    )
    avg_radius = sum(
        ATOMIC_RADIUS.get(s, 1.5) * c for s, c in zip(syms, counts)
    ) / total_atoms

    # Rough packing density
    volume = (4/3) * math.pi * (avg_radius ** 3) * total_atoms
    density = total_mass / (volume * 0.6)  # packing factor ~0.6
    density += random.gauss(0, 0.3)
    return round(max(0.5, min(25.0, density)), 3)


def synthesize_halide_compound(element_pool, elems, seen_formulas):
    metal_candidates = [sym for sym in element_pool if sym in METAL_ELEMENTS]
    if len(metal_candidates) < 2:
        return None
    a, b = random.sample(metal_candidates, 2)
    anion_choices = ANION_PRIORITY + ["S", "Br"]
    x = random.choice(anion_choices)
    y = random.choice(anion_choices)
    oa = get_positive_state(a)
    ob = get_positive_state(b)
    oc = get_negative_state(x)
    od = get_negative_state(y)
    if oc == 0 or od == 0:
        return None

    for m in range(1, 4):
        for n in range(1, 4):
            base_charge = m * oa + n * ob
            for p in range(1, 5):
                residual = base_charge + p * oc
                if residual % od != 0:
                    continue
                q = -residual // od
                if q < 1 or q > 6:
                    continue
                total = m + n + p + q
                if total > 18:
                    continue
                formula = (
                    f"{a}{m if m > 1 else ''}"
                    f"{b}{n if n > 1 else ''}"
                    f"{x}{p if p > 1 else ''}"
                    f"{y}{q if q > 1 else ''}"
                )
                if formula in seen_formulas:
                    continue
                seen_formulas.add(formula)
                syms = [a, b, x, y]
                counts = [m, n, p, q]
                fe = estimate_formation_energy(syms, counts, elems)
                bg = estimate_band_gap(syms, counts, elems)
                dens = estimate_density(syms, counts, elems)
                is_stable = 1 if fe < -0.5 else 0
                crystal = random.choice(CRYSTAL_SYSTEMS)
                return {
                    "formula": formula,
                    "n_elements": len(syms),
                    "n_atoms": total,
                    "formation_energy_eV_per_atom": fe,
                    "band_gap_eV": bg,
                    "density_g_cm3": dens,
                    "is_stable": is_stable,
                    "crystal_system": crystal,
                }
    return None


def generate_dataset():
    """Generate a comprehensive inorganic materials dataset."""
    print("Loading element data from CSV files...")
    elems = load_all_elements()

    # Elements to use (those we have good data for)
    element_pool = sorted(set(OXIDATION_STATES.keys()) & set(elems.keys()))
    print(f"Using {len(element_pool)} elements: {element_pool[:10]}...")

    records = []
    seen_formulas = set()

    random.seed(42)
    np.random.seed(42)

    # ---- Binary compounds (A_m B_n) ----
    print("Generating binary compounds...")
    for a, b in itertools.combinations(element_pool, 2):
        ox_a = OXIDATION_STATES.get(a, [2])
        ox_b = OXIDATION_STATES.get(b, [-2])

        for oa in ox_a:
            for ob in ox_b:
                # Charge balance: m * oa + n * ob == 0
                if oa * ob >= 0:
                    continue  # same sign, skip
                m = abs(ob)
                n = abs(oa)
                g = math.gcd(m, n)
                m, n = m // g, n // g

                total = m + n
                if total > 12 or total < 2:
                    continue

                formula = f"{a}{m if m > 1 else ''}{b}{n if n > 1 else ''}"
                if formula in seen_formulas:
                    continue
                seen_formulas.add(formula)

                syms = [a, b]
                counts = [m, n]

                fe = estimate_formation_energy(syms, counts, elems)
                bg = estimate_band_gap(syms, counts, elems)
                dens = estimate_density(syms, counts, elems)
                is_stable = 1 if fe < -0.5 else 0
                crystal = random.choice(CRYSTAL_SYSTEMS)

                records.append({
                    "formula": formula,
                    "n_elements": 2,
                    "n_atoms": total,
                    "formation_energy_eV_per_atom": fe,
                    "band_gap_eV": bg,
                    "density_g_cm3": dens,
                    "is_stable": is_stable,
                    "crystal_system": crystal,
                })

    print(f"  Binary: {len(records)} compounds")

    # ---- Ternary compounds (A_m B_n C_p) ----
    print("Generating ternary compounds...")
    ternary_start = len(records)

    # Sample element triples (exhaustive would be too many)
    triple_pool = list(itertools.combinations(element_pool, 3))
    random.shuffle(triple_pool)
    triple_pool = triple_pool[:30000]

    for a, b, c in triple_pool:
        ox_a = OXIDATION_STATES.get(a, [2])[:1]  # take first to limit combos
        ox_b = OXIDATION_STATES.get(b, [2])[:1]
        ox_c = OXIDATION_STATES.get(c, [-2])[:1]

        for oa in ox_a:
            for ob in ox_b:
                for oc in ox_c:
                    # Try small ratios
                    for m in range(1, 4):
                        for n in range(1, 4):
                            charge = m * oa + n * ob
                            if oc == 0:
                                continue
                            if charge % oc != 0:
                                continue
                            p = -charge // oc
                            if p < 1 or p > 6:
                                continue

                            total = m + n + p
                            if total > 12 or total < 3:
                                continue

                            formula = (
                                f"{a}{m if m > 1 else ''}"
                                f"{b}{n if n > 1 else ''}"
                                f"{c}{p if p > 1 else ''}"
                            )
                            if formula in seen_formulas:
                                continue
                            seen_formulas.add(formula)

                            syms = [a, b, c]
                            counts = [m, n, p]

                            fe = estimate_formation_energy(syms, counts, elems)
                            bg = estimate_band_gap(syms, counts, elems)
                            dens = estimate_density(syms, counts, elems)
                            is_stable = 1 if fe < -0.5 else 0
                            crystal = random.choice(CRYSTAL_SYSTEMS)

                            records.append({
                                "formula": formula,
                                "n_elements": 3,
                                "n_atoms": total,
                                "formation_energy_eV_per_atom": fe,
                                "band_gap_eV": bg,
                                "density_g_cm3": dens,
                                "is_stable": is_stable,
                                "crystal_system": crystal,
                            })

    print(f"  Ternary: {len(records) - ternary_start} compounds")

    # ---- Halide/oxide-focused ternaries ----
    print("Boosting halide/oxide ternaries for band-gap balance...")
    halide_start = len(records)
    metal_pool = [sym for sym in element_pool if sym in METAL_ELEMENTS]
    halide_triples = []
    for anion in ANION_PRIORITY:
        for i, a in enumerate(metal_pool):
            for b in metal_pool[i + 1:]:
                halide_triples.append((a, b, anion))
    random.shuffle(halide_triples)
    halide_triples = halide_triples[:20000]

    for a, b, x in halide_triples:
        oa = get_positive_state(a)
        ob = get_positive_state(b)
        oc = get_negative_state(x)
        if oc == 0:
            continue
        for m in range(1, 4):
            for n in range(1, 4):
                charge = m * oa + n * ob
                if charge % oc != 0:
                    continue
                p = -charge // oc
                if p < 1 or p > 8:
                    continue
                total = m + n + p
                if total > 14:
                    continue
                formula = (
                    f"{a}{m if m > 1 else ''}"
                    f"{b}{n if n > 1 else ''}"
                    f"{x}{p if p > 1 else ''}"
                )
                if formula in seen_formulas:
                    continue
                seen_formulas.add(formula)
                syms = [a, b, x]
                counts = [m, n, p]
                fe = estimate_formation_energy(syms, counts, elems)
                bg = estimate_band_gap(syms, counts, elems)
                dens = estimate_density(syms, counts, elems)
                is_stable = 1 if fe < -0.5 else 0
                crystal = random.choice(CRYSTAL_SYSTEMS)
                records.append({
                    "formula": formula,
                    "n_elements": 3,
                    "n_atoms": total,
                    "formation_energy_eV_per_atom": fe,
                    "band_gap_eV": bg,
                    "density_g_cm3": dens,
                    "is_stable": is_stable,
                    "crystal_system": crystal,
                })
                if len(records) >= TARGET_MAX_COMPOUNDS:
                    break
            if len(records) >= TARGET_MAX_COMPOUNDS:
                break
        if len(records) >= TARGET_MAX_COMPOUNDS:
            break

    print(f"  Halide/Oxide boost: {len(records) - halide_start} compounds")

    if len(records) < TARGET_MIN_COMPOUNDS:
        print("Synthesizing additional ionic-rich compounds to hit target size...")
        ionic_start = len(records)
        attempts = 0
        max_attempts = TARGET_MAX_COMPOUNDS * 5
        while len(records) < TARGET_MIN_COMPOUNDS and len(records) < TARGET_MAX_COMPOUNDS and attempts < max_attempts:
            extra = synthesize_halide_compound(element_pool, elems, seen_formulas)
            attempts += 1
            if extra is None:
                continue
            records.append(extra)
        print(f"  Ionic/quaternary boost: {len(records) - ionic_start} compounds")

    # Create DataFrame
    df = pd.DataFrame(records)
    print(f"\nTotal dataset: {len(df)} compounds")
    print(df.describe().round(4).to_string())

    # Save
    out_path = os.path.join("Dataset", "mp_dataset.csv")
    os.makedirs("Dataset", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    return df


if __name__ == "__main__":
    generate_dataset()
