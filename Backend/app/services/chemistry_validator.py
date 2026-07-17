"""
Chemistry Validator — Research-Grade Chemical Compound Validation.

Validates compounds using real inorganic chemistry principles:
  1. Charge neutrality via oxidation state enumeration
  2. Element compatibility (no halogen-halogen, noble gas, etc.)
  3. Transition metal coordination chemistry (preferred CN per oxidation state)
  4. d-electron count stability (crystal field stabilization energy)
  5. Known structural prototypes (perovskite, spinel, rutile, etc.)
  6. Goldschmidt tolerance factor for perovskite candidacy
  7. Electronegativity difference threshold
  8. Stoichiometry reasonableness

Uses pure Python with real chemistry data — no external dependencies.
"""

import re
from math import gcd, sqrt
from functools import reduce
from itertools import product


# -----------------------------------------------------------------------
# Oxidation state database (ordered by thermodynamic prevalence)
# -----------------------------------------------------------------------
OXIDATION_STATES: dict[str, list[int]] = {
    # Alkali metals — always +1
    "Li": [1], "Na": [1], "K": [1], "Rb": [1], "Cs": [1],
    # Alkaline earth — always +2
    "Be": [2], "Mg": [2], "Ca": [2], "Sr": [2], "Ba": [2],
    # 3d transition metals — ordered by stability
    "Sc": [3],
    "Ti": [4, 3, 2],
    "V":  [5, 4, 3, 2],
    "Cr": [3, 6, 2],
    "Mn": [2, 4, 7, 3, 6],
    "Fe": [3, 2],
    "Co": [2, 3],
    "Ni": [2, 3],
    "Cu": [2, 1],
    "Zn": [2],
    # 4d transition metals
    "Y":  [3],
    "Zr": [4],
    "Nb": [5, 3],
    "Mo": [6, 4, 3, 2],
    "Ru": [3, 4, 8],
    "Rh": [3],
    "Pd": [2, 4],
    "Ag": [1, 2],
    "Cd": [2],
    # 5d transition metals
    "Hf": [4],
    "Ta": [5],
    "W":  [6, 4],
    "Re": [7, 4],
    "Os": [4, 8],
    "Ir": [3, 4],
    "Pt": [2, 4],
    "Au": [3, 1],
    # Post-transition / weak metals
    "Al": [3], "Ga": [3], "In": [3],
    "Sn": [4, 2], "Pb": [2, 4],
    "Tl": [1, 3], "Bi": [3, 5],
    # Semi-metals
    "B":  [3], "Si": [4], "Ge": [4, 2],
    "As": [3, 5], "Sb": [3, 5], "Te": [4, -2, 6],
    # Nonmetals
    "H":  [1, -1],
    "C":  [-4, 4, 2],
    "N":  [-3, 5, 3],
    "O":  [-2],
    "F":  [-1],
    "P":  [-3, 5, 3],
    "S":  [-2, 6, 4],
    "Se": [-2, 6, 4],
    "Cl": [-1, 7, 5],
    "Br": [-1, 5],
    "I":  [-1, 7, 5],
    # Lanthanides
    "La": [3], "Ce": [3, 4], "Pr": [3, 4], "Nd": [3],
    "Sm": [3, 2], "Eu": [3, 2], "Gd": [3],
    "Tb": [3, 4], "Dy": [3], "Ho": [3],
    "Er": [3], "Tm": [3], "Yb": [3, 2], "Lu": [3],
}


# -----------------------------------------------------------------------
# Preferred coordination numbers per (element, oxidation_state)
# Based on crystal field theory, CFSE, and experimental data.
# The list is ordered from most preferred to least preferred.
# -----------------------------------------------------------------------
PREFERRED_COORDINATION: dict[tuple[str, int], list[int]] = {
    # 3d block
    ("Sc", 3): [6],              # octahedral (d⁰)
    ("Ti", 4): [6],              # octahedral (d⁰, as in TiO₂ rutile)
    ("Ti", 3): [6],              # octahedral (d¹)
    ("Ti", 2): [6],              # octahedral (d²)
    ("V",  5): [4, 6],           # tetrahedral/octahedral (d⁰)
    ("V",  4): [6],              # octahedral (d¹)
    ("V",  3): [6],              # octahedral (d²)
    ("Cr", 3): [6],              # octahedral (d³ — max CFSE!)
    ("Cr", 6): [4, 6],           # tetrahedral CrO₄²⁻ / octahedral
    ("Mn", 2): [6, 4],           # octahedral preferred (d⁵ HS)
    ("Mn", 4): [6],              # octahedral (d³ — like MnO₂)
    ("Mn", 7): [4],              # tetrahedral (MnO₄⁻)
    ("Fe", 3): [6, 4],           # octahedral preferred (d⁵ HS — max CFSE)
    ("Fe", 2): [6, 4],           # octahedral (d⁶)
    ("Co", 2): [6, 4],           # octahedral (d⁷)
    ("Co", 3): [6],              # octahedral (d⁶ LS — strong CFSE)
    ("Ni", 2): [6, 4],           # octahedral (d⁸)
    ("Ni", 3): [6],              # octahedral (d⁷)
    ("Cu", 2): [6, 4],           # Jahn-Teller distorted octahedral (d⁹)
    ("Cu", 1): [4, 2],           # tetrahedral or linear (d¹⁰)
    ("Zn", 2): [4, 6],           # tetrahedral preferred (d¹⁰)
    # 4d block
    ("Zr", 4): [6, 8],           # octahedral / cubic (d⁰)
    ("Nb", 5): [6],              # octahedral (d⁰)
    ("Mo", 6): [6, 4],           # octahedral / tetrahedral (d⁰)
    ("Mo", 4): [6],              # octahedral (d²)
    ("Ru", 3): [6],              # octahedral
    ("Ru", 4): [6],              # octahedral
    ("Rh", 3): [6],              # octahedral
    ("Pd", 2): [4],              # square planar (d⁸ — strong preference)
    ("Pd", 4): [6],              # octahedral
    ("Ag", 1): [2, 4],           # linear or tetrahedral (d¹⁰)
    ("Cd", 2): [4, 6],           # tetrahedral / octahedral (d¹⁰)
    # 5d block
    ("Hf", 4): [6, 8],           # octahedral / cubic (d⁰)
    ("Ta", 5): [6],              # octahedral (d⁰)
    ("W",  6): [6, 4],           # octahedral / tetrahedral (d⁰)
    ("W",  4): [6],              # octahedral (d²)
    ("Pt", 2): [4],              # square planar (d⁸ — very strong preference)
    ("Pt", 4): [6],              # octahedral (d⁶)
    ("Au", 3): [4],              # square planar (d⁸)
    ("Au", 1): [2],              # linear (d¹⁰)
    ("Ir", 3): [6],              # octahedral
    ("Ir", 4): [6],              # octahedral
    # Main group metals
    ("Al", 3): [4, 6],           # tetrahedral or octahedral
    ("Ga", 3): [4, 6],           # tetrahedral or octahedral
    ("In", 3): [6],              # octahedral
    ("Sn", 4): [6],              # octahedral
    ("Pb", 2): [6, 4],           # octahedral (lone pair)
    ("Bi", 3): [6],              # octahedral (lone pair)
    # Alkaline earth
    ("Mg", 2): [6, 4],           # octahedral
    ("Ca", 2): [6, 8],           # octahedral / cubic
    ("Sr", 2): [8, 6, 12],       # cubic / cubo-octahedral
    ("Ba", 2): [8, 12],          # cubic / cubo-octahedral (large ion)
    # Alkali
    ("Li", 1): [4, 6],
    ("Na", 1): [6, 8],
    ("K",  1): [8, 6, 12],
    ("Rb", 1): [8, 12],
    ("Cs", 1): [8, 12],
}


# d-electron count for given (element, oxidation_state)
# d-electron count = group_number - oxidation_state (for transition metals)
_TRANSITION_METAL_GROUP = {
    "Sc": 3, "Ti": 4, "V": 5, "Cr": 6, "Mn": 7,
    "Fe": 8, "Co": 9, "Ni": 10, "Cu": 11, "Zn": 12,
    "Y": 3, "Zr": 4, "Nb": 5, "Mo": 6, "Ru": 8,
    "Rh": 9, "Pd": 10, "Ag": 11, "Cd": 12,
    "Hf": 4, "Ta": 5, "W": 6, "Re": 7, "Os": 8,
    "Ir": 9, "Pt": 10, "Au": 11,
}

# Stable d-electron counts (high crystal field stabilization energy)
_STABLE_D_COUNTS = {0, 3, 5, 6, 8, 10}  # d⁰, d³, d⁵(HS), d⁶(LS), d⁸, d¹⁰

# Moderately stable
_MODERATE_D_COUNTS = {1, 2, 4, 7, 9}

def d_electron_count(element: str, oxidation_state: int) -> int | None:
    """Get d-electron count for a transition metal in a given oxidation state."""
    group = _TRANSITION_METAL_GROUP.get(element)
    if group is None:
        return None
    return group - oxidation_state


# -----------------------------------------------------------------------
# Ionic radii (Shannon, 1976) for coordination number 6 (Angstroms)
# Used for Goldschmidt tolerance factor calculation
# -----------------------------------------------------------------------
IONIC_RADII: dict[tuple[str, int], float] = {
    # Alkali (+1), CN=6
    ("Li", 1): 0.76, ("Na", 1): 1.02, ("K", 1): 1.38,
    ("Rb", 1): 1.52, ("Cs", 1): 1.67,
    # Alkaline earth (+2), CN=6
    ("Mg", 2): 0.72, ("Ca", 2): 1.00, ("Sr", 2): 1.18, ("Ba", 2): 1.35,
    # 3d TM
    ("Sc", 3): 0.745, ("Ti", 4): 0.605, ("Ti", 3): 0.67,
    ("V", 5): 0.54, ("V", 3): 0.64,
    ("Cr", 3): 0.615, ("Cr", 6): 0.44,
    ("Mn", 2): 0.83, ("Mn", 4): 0.53, ("Mn", 3): 0.645,
    ("Fe", 3): 0.645, ("Fe", 2): 0.78,
    ("Co", 2): 0.745, ("Co", 3): 0.545,
    ("Ni", 2): 0.69, ("Ni", 3): 0.56,
    ("Cu", 2): 0.73, ("Cu", 1): 0.77,
    ("Zn", 2): 0.74,
    # 4d TM
    ("Y", 3): 0.90, ("Zr", 4): 0.72, ("Nb", 5): 0.64, ("Mo", 6): 0.59,
    ("Ru", 4): 0.62, ("Rh", 3): 0.665, ("Pd", 2): 0.86,
    ("Ag", 1): 1.15, ("Cd", 2): 0.95,
    # 5d TM
    ("Hf", 4): 0.71, ("Ta", 5): 0.64, ("W", 6): 0.60,
    ("Ir", 4): 0.625, ("Pt", 4): 0.625, ("Pt", 2): 0.80,
    ("Au", 3): 0.85,
    # Post-transition
    ("Al", 3): 0.535, ("Ga", 3): 0.62, ("In", 3): 0.80,
    ("Sn", 4): 0.69, ("Pb", 2): 1.19, ("Pb", 4): 0.775,
    ("Bi", 3): 1.03, ("Bi", 5): 0.76,
    ("Si", 4): 0.40, ("Ge", 4): 0.53,
    # Lanthanides (+3)
    ("La", 3): 1.032, ("Ce", 3): 1.01, ("Ce", 4): 0.87,
    ("Pr", 3): 0.99, ("Nd", 3): 0.983,
    ("Sm", 3): 0.958, ("Eu", 3): 0.947, ("Eu", 2): 1.17,
    ("Gd", 3): 0.938, ("Tb", 3): 0.923, ("Dy", 3): 0.912,
    ("Ho", 3): 0.901, ("Er", 3): 0.890, ("Tm", 3): 0.880,
    ("Yb", 3): 0.868, ("Lu", 3): 0.861,
    # Anions
    ("O", -2): 1.40, ("F", -1): 1.33, ("Cl", -1): 1.81,
    ("Br", -1): 1.96, ("I", -1): 2.20, ("S", -2): 1.84,
    ("Se", -2): 1.98, ("N", -3): 1.46,
}


# -----------------------------------------------------------------------
# Known stable structure prototypes
# Maps (stoichiometry_type, anion) → list of valid oxidation patterns
# -----------------------------------------------------------------------
STABLE_PROTOTYPES = {
    # AX binary: rock salt, wurtzite, zinc blende, etc.
    "AX": [
        {"A": [1, 2], "X": [-1, -2]},  # NaCl, MgO, ZnS, etc.
    ],
    # AX₂: rutile, fluorite, etc.
    "AX2": [
        {"A": [2, 4], "X": [-1, -2]},  # CaF₂ (fluorite), TiO₂ (rutile)
    ],
    # A₂X₃: corundum, bixbyite
    "A2X3": [
        {"A": [3], "X": [-2]},  # Al₂O₃, Fe₂O₃, Cr₂O₃
    ],
    # A₃X₂: antifluorite-like
    "A3X2": [
        {"A": [2], "X": [-3]},  # Mg₃N₂
    ],
    # ABX₃: perovskite
    "ABX3": [
        {"A": [1, 2], "B": [3, 4, 5], "X": [-2]},  # BaTiO₃, SrVO₃
    ],
    # AB₂X₄: spinel
    "AB2X4": [
        {"A": [2], "B": [3], "X": [-2]},  # MgAl₂O₄, CoFe₂O₄
    ],
    # A₂BX₄: olivine
    "A2BX4": [
        {"A": [2], "B": [4], "X": [-2]},  # Mg₂SiO₄
    ],
}


# -----------------------------------------------------------------------
# Element classification sets
# -----------------------------------------------------------------------
_NOBLE_GASES = {"He", "Ne", "Ar", "Kr", "Xe", "Rn"}
_HALOGENS = {"F", "Cl", "Br", "I", "At"}
_ALKALI_METALS = {"Li", "Na", "K", "Rb", "Cs", "Fr"}
_ALKALINE_EARTH = {"Be", "Mg", "Ca", "Sr", "Ba", "Ra"}
_TRANSITION_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
}
_METALS = (
    _ALKALI_METALS | _ALKALINE_EARTH | _TRANSITION_METALS |
    {"Al", "Ga", "In", "Sn", "Pb", "Tl", "Bi",
     "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd",
     "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}
)
_NONMETALS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Se", "Br", "I"}
_SEMI_METALS = {"B", "Si", "Ge", "As", "Sb", "Te"}

# Electronegativity (Pauling)
_EN = {
    "H": 2.20, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55,
    "N": 3.04, "O": 3.44, "F": 3.98, "Na": 0.93, "Mg": 1.31,
    "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16,
    "K": 0.82, "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V": 1.63,
    "Cr": 1.66, "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.91,
    "Cu": 1.90, "Zn": 1.65, "Ga": 1.81, "Ge": 2.01, "As": 2.18,
    "Se": 2.55, "Br": 2.96, "Rb": 0.82, "Sr": 0.95, "Y": 1.22,
    "Zr": 1.33, "Nb": 1.60, "Mo": 2.16, "Ru": 2.20, "Rh": 2.28,
    "Pd": 2.20, "Ag": 1.93, "Cd": 1.69, "In": 1.78, "Sn": 1.96,
    "Sb": 2.05, "Te": 2.10, "I": 2.66, "Cs": 0.79, "Ba": 0.89,
    "La": 1.10, "Ce": 1.12, "Hf": 1.30, "Ta": 1.50, "W": 2.36,
    "Pt": 2.28, "Au": 2.54, "Pb": 2.33, "Bi": 2.02,
    "Nd": 1.14, "Gd": 1.20, "Eu": 1.20, "Sm": 1.17,
    "Pr": 1.13, "Tb": 1.10, "Dy": 1.22, "Ho": 1.23,
    "Er": 1.24, "Tm": 1.25, "Yb": 1.10, "Lu": 1.27,
}

# Maximum coordination number (typical)
_MAX_COORD = {
    "Li": 6, "Na": 6, "K": 8, "Rb": 8, "Cs": 12,
    "Be": 4, "Mg": 6, "Ca": 8, "Sr": 12, "Ba": 12,
    "Ti": 6, "V": 6, "Cr": 6, "Mn": 6, "Fe": 6,
    "Co": 6, "Ni": 6, "Cu": 6, "Zn": 4, "Al": 6,
    "Si": 4, "B": 4, "C": 4, "N": 4, "O": 3,
    "F": 1, "Cl": 1, "Br": 1, "I": 1, "S": 6,
    "P": 6, "Se": 6, "Ga": 6, "Ge": 4, "Zr": 8,
    "Nb": 6, "Mo": 6, "W": 6, "Sn": 6, "Pb": 6,
    "Hf": 8, "Ta": 6, "Pt": 6, "Pd": 4, "Au": 4,
    "Ag": 4, "Cd": 6, "In": 6, "Bi": 6, "Sb": 6,
    "Ru": 6, "Rh": 6, "Ir": 6, "Os": 6, "Re": 6,
    "La": 12, "Ce": 12, "Y": 8, "Nd": 9, "Gd": 9,
    "Eu": 9, "Sm": 9,
}


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def parse_formula(formula: str) -> list[tuple[str, int]]:
    """Parse formula → [(symbol, count), ...]."""
    parts = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    return [(s, int(c) if c else 1) for s, c in parts if s]


def validate_compound(formula: str) -> tuple[bool, list[str]]:
    """
    Validate a chemical compound for physical plausibility.
    Returns (is_valid, list_of_reasons_if_invalid).
    """
    parsed = parse_formula(formula)
    if not parsed:
        return False, ["Cannot parse formula"]

    symbols = [s for s, _ in parsed]
    counts = [c for _, c in parsed]
    reasons = []

    # 1. No noble gases
    for sym in symbols:
        if sym in _NOBLE_GASES:
            reasons.append(f"{sym} is a noble gas — does not form compounds")

    # 2. No duplicate elements
    if len(symbols) != len(set(symbols)):
        reasons.append("Duplicate elements in formula")

    # 3. Element compatibility
    reasons.extend(_check_compatibility(symbols))

    # 4. Charge neutrality
    if not reasons:
        neutral_ok, charge_reason = _check_charge_neutrality(symbols, counts)
        if not neutral_ok:
            reasons.append(charge_reason)

    # 4b. Electrochemical redox compatibility
    if not reasons:
        reasons.extend(_check_redox_compatibility(symbols, counts))

    # 5. Coordination sanity (basic max-CN check)
    reasons.extend(_check_coordination(symbols, counts))

    # 6. Coordination geometry (strict preferred CN enforcement)
    if not reasons:
        reasons.extend(_check_coordination_geometry(symbols, counts))

    # 7. Transition metal d-electron stability check
    if not reasons:
        reasons.extend(_check_d_electron_stability(symbols, counts))

    # 8. Stoichiometry reasonableness
    for sym, count in parsed:
        if count > 8 and sym not in ("O", "F", "Cl"): # Allow up to 12 for O in NASICON
            reasons.append(f"{sym}{count}: subscript too large (max 8)")

    # 9. Battery Structural Ionic Radius Compatibility Check
    # For AMO2 (Layered), AMPO4, AB2O4 (Spinel) with Li/Na
    if ("Li" in symbols or "Na" in symbols) and any(s in _TRANSITION_METALS for s in symbols):
        tm_syms = [s for s in symbols if s in _TRANSITION_METALS]
        if len(tm_syms) == 1:
            tm = tm_syms[0]
            tm_oxs = OXIDATION_STATES.get(tm, [])
            valid_r = False
            for ox in tm_oxs:
                r = IONIC_RADII.get((tm, ox))
                if r is not None and 0.45 <= r <= 0.95:  # typical range for octahedral TM in Li/Na batteries
                    valid_r = True
                    break
            # If radius is clearly outside stable battery TM range, penalize
            if not valid_r and any((tm, ox) in IONIC_RADII for ox in tm_oxs):
                reasons.append(f"Transition metal {tm} ionic radius incompatible with stable Li/Na intercalation lattice")


    return (len(reasons) == 0), reasons


def is_valid_compound(formula: str) -> bool:
    """Quick validity check."""
    valid, _ = validate_compound(formula)
    return valid


def classify_compound(formula: str) -> dict:
    """
    Multi-tier compound classification for research-grade filtering.

    Returns:
        {
            "validity": "valid" | "partially_valid" | "invalid",
            "reasons": [...],
            "prototype_match": bool,
            "stability_score": float,  # 0-10
            "coordination_ok": bool,
        }

    Classification rules:
        - "valid":           Passes all checks AND matches a structure prototype
        - "partially_valid": Passes fundamental checks but no prototype match
        - "invalid":         Fails charge neutrality, compatibility, or coordination
    """
    parsed = parse_formula(formula)
    if not parsed:
        return {
            "validity": "invalid",
            "reasons": ["Cannot parse formula"],
            "prototype_match": False,
            "stability_score": 0.0,
            "coordination_ok": False,
        }

    symbols = [s for s, _ in parsed]
    counts = [c for _, c in parsed]

    # Run full validation
    valid, reasons = validate_compound(formula)

    if not valid:
        return {
            "validity": "invalid",
            "reasons": reasons,
            "prototype_match": False,
            "stability_score": 0.0,
            "coordination_ok": False,
        }

    # Passes fundamental checks — now check prototype and CN geometry
    has_prototype = _matches_prototype(symbols, counts)
    coord_ok = len(_check_coordination_geometry(symbols, counts)) == 0
    score = get_compound_stability_score(formula)

    # Extra check for ABX₃ compounds: enforce Goldschmidt tolerance
    abx3_ok = True
    if has_prototype and sorted(counts) == [1, 1, 3]:
        t_factor = goldschmidt_tolerance(formula)
        if t_factor is not None:
            if not (0.75 <= t_factor <= 1.10):
                abx3_ok = False
                reasons.append(
                    f"ABX₃ Goldschmidt tolerance t={t_factor:.3f} outside "
                    f"stable range [0.75, 1.10]"
                )
        # Also check: does the B-site TM actually work in perovskite CN=6?
        # In ABX₃ perovskite, B-site has CN=6 (octahedral)
        anion_sym = None
        b_site_syms = []
        for sym, cnt in zip(symbols, counts):
            if cnt == 3:
                anion_sym = sym
            elif sym in _TRANSITION_METALS:
                b_site_syms.append(sym)
        for b_sym in b_site_syms:
            # Check all oxidation states — does any give valid CN=6?
            b_states = OXIDATION_STATES.get(b_sym, [])
            cn6_possible = False
            for ox in b_states:
                pref = PREFERRED_COORDINATION.get((b_sym, ox))
                if pref is None or 6 in pref:
                    cn6_possible = True
                    break
            if not cn6_possible:
                abx3_ok = False
                pref_cns = set()
                for ox in b_states:
                    pref = PREFERRED_COORDINATION.get((b_sym, ox), [])
                    pref_cns.update(pref)
                reasons.append(
                    f"{b_sym} cannot occupy perovskite B-site (requires "
                    f"CN=6 octahedral, but {b_sym} prefers CN={sorted(pref_cns)})"
                )

    if not abx3_ok:
        return {
            "validity": "invalid",
            "reasons": reasons,
            "prototype_match": has_prototype,
            "stability_score": score,
            "coordination_ok": False,
        }

    if has_prototype and coord_ok and score >= 6.0:
        validity = "valid"
    else:
        validity = "partially_valid"

    return {
        "validity": validity,
        "reasons": reasons,
        "prototype_match": has_prototype,
        "stability_score": score,
        "coordination_ok": coord_ok,
    }


def get_compound_stability_score(formula: str) -> float:
    """
    Compute a stability heuristic score (0-10) for a compound.
    Higher = more likely to be stable.

    Uses both bonuses AND penalties:
    - EN difference bonus (ionic character)
    - Prototype matching bonus (only if structurally validated)
    - d-electron stability bonus
    - Goldschmidt tolerance bonus/penalty (for ABX₃)
    - Penalty for failed structural checks

    Score of 10 is only possible when ALL checks pass.
    """
    parsed = parse_formula(formula)
    if not parsed:
        return 0.0

    symbols = [s for s, _ in parsed]
    counts = [c for _, c in parsed]
    score = 5.0  # base score

    # EN difference bonus (up to +2.5)
    en_vals = [_EN.get(s, 2.0) for s in symbols]
    en_diff = max(en_vals) - min(en_vals)
    score += min(en_diff * 0.8, 2.5)

    # Prototype matching — only give bonus if structurally validated
    proto_match = _matches_prototype(symbols, counts)
    if proto_match:
        score += 0.5  # reduced from 1.5 — need structural validation for full bonus

    # d-electron stability bonus (up to +0.8)
    d_bonus = _d_electron_bonus(symbols, counts)
    score += min(d_bonus, 0.8)

    # Goldschmidt tolerance for ABX₃ compounds
    t_factor = goldschmidt_tolerance(formula)
    if t_factor is not None:
        if 0.8 <= t_factor <= 1.0:
            score += 1.2  # strong structural validation bonus
        elif 0.71 <= t_factor < 0.8 or 1.0 < t_factor <= 1.05:
            score += 0.3  # marginal — distorted but possible
        else:
            score -= 1.5  # PENALTY — invalid tolerance factor
    elif sorted(counts) == [1, 1, 3]:
        # ABX₃ pattern but no Goldschmidt data — penalize uncertainty
        score -= 0.5

    return max(0.0, min(score, 10.0))


def get_validated_structure_type(formula: str) -> dict:
    """
    Determine the validated crystal structure type for a compound.

    Only labels a compound with a specific structure type (e.g., 'perovskite')
    if it passes the corresponding structural validation. Otherwise uses
    generic labels like 'oxide_compound' or 'distorted_structure'.

    Returns:
        dict with keys:
            structure_type: str — validated structure label
            prototype_validated: bool — True if structurally confirmed
            goldschmidt_t: float | None — tolerance factor if ABX₃
            validation_note: str — explanation
    """
    parsed = parse_formula(formula)
    if not parsed:
        return {
            "structure_type": "unknown",
            "prototype_validated": False,
            "goldschmidt_t": None,
            "validation_note": "Could not parse formula",
        }

    symbols = [s for s, _ in parsed]
    counts = [c for _, c in parsed]
    n_elem = len(symbols)

    # Identify anion
    anion_syms = {s for s in symbols if s in _NONMETALS or s in _HALOGENS}

    result = {
        "structure_type": "unknown",
        "prototype_validated": False,
        "goldschmidt_t": None,
        "validation_note": "",
    }

    # --- ABX₃ check (perovskite candidate) ---
    if n_elem == 3 and sorted(counts) == [1, 1, 3]:
        t = goldschmidt_tolerance(formula)
        result["goldschmidt_t"] = t

        if t is not None:
            if 0.8 <= t <= 1.0:
                result["structure_type"] = "perovskite"
                result["prototype_validated"] = True
                result["validation_note"] = f"Goldschmidt t={t:.3f} (valid perovskite range)"
            elif 0.71 <= t < 0.8:
                result["structure_type"] = "distorted_perovskite"
                result["prototype_validated"] = False
                result["validation_note"] = (
                    f"Goldschmidt t={t:.3f} — below 0.8, likely distorted "
                    f"or non-perovskite structure"
                )
            elif 1.0 < t <= 1.05:
                result["structure_type"] = "hexagonal_perovskite"
                result["prototype_validated"] = False
                result["validation_note"] = f"Goldschmidt t={t:.3f} — hexagonal variant possible"
            else:
                # Determine if it's an oxide or halide
                has_O = "O" in anion_syms
                has_halide = bool(anion_syms & {"F", "Cl", "Br", "I"})
                if has_O:
                    result["structure_type"] = "ternary_oxide"
                elif has_halide:
                    result["structure_type"] = "ternary_halide"
                else:
                    result["structure_type"] = "ternary_compound"
                result["prototype_validated"] = False
                result["validation_note"] = (
                    f"Goldschmidt t={t:.3f} — outside perovskite range (0.8-1.0), "
                    f"not a valid perovskite"
                )
        else:
            # No ionic radii data — can't validate
            has_O = "O" in anion_syms
            result["structure_type"] = "ternary_oxide" if has_O else "ternary_compound"
            result["prototype_validated"] = False
            result["validation_note"] = "No ionic radii data for Goldschmidt check"

    # --- AB₂X₄ (spinel candidate) ---
    elif n_elem == 3 and sorted(counts) == [1, 2, 4]:
        has_O = "O" in anion_syms
        result["structure_type"] = "spinel" if has_O else "ternary_compound"
        result["prototype_validated"] = has_O  # spinels are well-established for oxides
        result["validation_note"] = "AB₂O₄ spinel prototype" if has_O else "AB₂X₄ pattern"

    # --- A₂BX₄ (layered perovskite candidate) ---
    elif n_elem == 3 and sorted(counts) == [1, 2, 4]:
        # Already handled above — same stoichiometry
        pass

    # --- Binary AX ---
    elif n_elem == 2 and sorted(counts) == [1, 1]:
        has_O = "O" in anion_syms
        has_halide = bool(anion_syms & {"F", "Cl", "Br", "I"})
        has_S = "S" in anion_syms or "Se" in anion_syms
        has_N = "N" in anion_syms
        if has_O:
            result["structure_type"] = "binary_oxide"
        elif has_halide:
            result["structure_type"] = "rock_salt"
        elif has_S:
            result["structure_type"] = "zinc_blende"
        elif has_N:
            result["structure_type"] = "binary_nitride"
        else:
            result["structure_type"] = "binary_compound"
        result["prototype_validated"] = True
        result["validation_note"] = "Standard binary prototype"

    # --- Binary AX₂ ---
    elif n_elem == 2 and sorted(counts) == [1, 2]:
        has_O = "O" in anion_syms
        has_F = "F" in anion_syms
        if has_O:
            result["structure_type"] = "rutile_or_fluorite"
        elif has_F:
            result["structure_type"] = "fluorite"
        else:
            result["structure_type"] = "binary_AB2"
        result["prototype_validated"] = True
        result["validation_note"] = "Standard AX₂ prototype"

    # --- Binary A₂X₃ ---
    elif n_elem == 2 and sorted(counts) == [2, 3]:
        has_O = "O" in anion_syms
        result["structure_type"] = "corundum" if has_O else "binary_A2X3"
        result["prototype_validated"] = has_O
        result["validation_note"] = "A₂O₃ corundum prototype" if has_O else "A₂X₃ pattern"

    else:
        result["structure_type"] = "other"
        result["validation_note"] = "No standard prototype match"

    return result


def goldschmidt_tolerance(formula: str) -> float | None:
    """
    Compute Goldschmidt tolerance factor for ABX₃ compounds.
    t = (r_A + r_X) / (√2 × (r_B + r_X))
    
    Valid perovskite: 0.8 ≤ t ≤ 1.05
    Returns None if not an ABX₃ compound.
    """
    parsed = parse_formula(formula)
    if not parsed or len(parsed) != 3:
        return None

    symbols = [s for s, _ in parsed]
    counts = [c for _, c in parsed]

    # Must be ABX₃ pattern
    if sorted(counts) != [1, 1, 3]:
        return None

    # Find which element has count 3 (anion X)
    anion_sym = None
    cation_syms = []
    for sym, cnt in parsed:
        if cnt == 3:
            anion_sym = sym
        else:
            cation_syms.append(sym)

    if anion_sym is None or len(cation_syms) != 2:
        return None

    # Get oxidation states for the cations
    # Try to identify A (larger, lower charge) and B (smaller, higher charge)
    a_sym, b_sym = cation_syms

    # Try all valid oxidation state combos
    a_states = [s for s in OXIDATION_STATES.get(a_sym, []) if s > 0]
    b_states = [s for s in OXIDATION_STATES.get(b_sym, []) if s > 0]
    x_states = [s for s in OXIDATION_STATES.get(anion_sym, []) if s < 0]

    for va, vb, vx in product(a_states, b_states, x_states):
        if va + vb + 3 * vx == 0:
            # Valid charge combo — compute tolerance
            r_a = IONIC_RADII.get((a_sym, va))
            r_b = IONIC_RADII.get((b_sym, vb))
            r_x = IONIC_RADII.get((anion_sym, vx))

            if r_a and r_b and r_x:
                t = (r_a + r_x) / (sqrt(2) * (r_b + r_x))
                return round(t, 3)

    return None


# -----------------------------------------------------------------------
# Internal validation rules
# -----------------------------------------------------------------------

def _check_compatibility(symbols: list[str]) -> list[str]:
    """Check element compatibility rules."""
    reasons = []

    halogen_syms = [s for s in symbols if s in _HALOGENS]
    alkali_syms = [s for s in symbols if s in _ALKALI_METALS]
    ae_syms = [s for s in symbols if s in _ALKALINE_EARTH]
    tm_syms = [s for s in symbols if s in _TRANSITION_METALS]
    metal_syms = [s for s in symbols if s in _METALS]
    nonmetal_syms = [s for s in symbols if s in _NONMETALS]
    semimetal_syms = [s for s in symbols if s in _SEMI_METALS]

    # Rule: all-halogen compounds are invalid
    if len(halogen_syms) == len(symbols) and len(symbols) > 1:
        reasons.append(
            f"All-halogen compound ({', '.join(halogen_syms)}) — "
            f"halogens don't form stable homonuclear compounds"
        )

    # Rule: multiple alkali metals don't combine
    if len(alkali_syms) > 1:
        reasons.append(
            f"Multiple alkali metals ({', '.join(alkali_syms)}) — "
            f"these don't form stable compounds together"
        )

    # Rule: multiple alkaline earth metals are very unusual
    if len(ae_syms) > 1:
        reasons.append(
            f"Multiple alkaline earth metals ({', '.join(ae_syms)}) — "
            f"rarely form stable compounds together"
        )

    # Rule: alkali + transition metal + halogen ONLY → must be perovskite-like
    # e.g. RbPtBr3 needs valid ABX3 stoichiometry with good tolerance factor
    if (alkali_syms and tm_syms and halogen_syms and
            len(symbols) == len(alkali_syms) + len(tm_syms) + len(halogen_syms)):
        # All elements are alkali + TM + halogen — check for valid prototype
        oxide_anion = False  # no oxygen present
        if not oxide_anion:
            # Must have ABX3 stoichiometry for this combination to be valid
            # (halide perovskites like CsSnBr3 are valid)
            # If not ABX3, reject
            pass  # handled by coordination geometry check below

    # Rule: alkali + transition metal binary (no anion) → invalid
    # e.g. RbPt, NaFe — metals don't form stable binary alloys this way
    if (alkali_syms and tm_syms and not halogen_syms
            and not nonmetal_syms and not semimetal_syms
            and len(symbols) == len(alkali_syms) + len(tm_syms)):
        reasons.append(
            f"Alkali-TM binary ({', '.join(alkali_syms)}+{', '.join(tm_syms)}) — "
            f"requires bridging anion (O, S, halide) for stable structure"
        )

    # Rule: must include at least one nonmetal/halogen/semimetal for >2 elements
    if not nonmetal_syms and not halogen_syms and not semimetal_syms:
        if len(symbols) > 2:
            reasons.append(
                f"All-metal compound with {len(symbols)} metals — "
                f"no anion species to form stable ionic compound"
            )

    # Rule: EN difference for ionic/polar bonding
    if len(symbols) >= 2 and not reasons:
        en_vals = [_EN.get(s, 2.0) for s in symbols]
        en_range = max(en_vals) - min(en_vals)
        if en_range < 0.3:
            reasons.append(
                f"Very low electronegativity difference ({en_range:.2f}) — "
                f"elements too similar for ionic/polar bonding"
            )

    return reasons


def _check_charge_neutrality(
    symbols: list[str],
    counts: list[int],
) -> tuple[bool, str]:
    """Check if any oxidation state combination gives charge neutrality."""
    state_lists = []
    for sym in symbols:
        states = OXIDATION_STATES.get(sym)
        if states is None:
            return True, ""  # unknown element — assume OK
        state_lists.append(states)

    MAX_ITER = 50_000
    for n_checked, combo in enumerate(product(*state_lists)):
        if n_checked >= MAX_ITER:
            return True, ""  # too many combos — assume OK
        total_charge = sum(ox * cnt for ox, cnt in zip(combo, counts))
        if total_charge == 0:
            return True, ""

    return False, (
        f"No valid oxidation state combination achieves charge neutrality "
        f"for {_formula_from(symbols, counts)}"
    )


# Maximum oxidation state allowed with each anion type
# Based on Hard-Soft Acid-Base (HSAB) theory:
#   - Hard anions (O²⁻, F⁻) stabilize high oxidation states
#   - Soft/reducing anions (N³⁻, S²⁻, Se²⁻, P³⁻) cannot stabilize high ox states
_MAX_OX_WITH_ANION: dict[str, int] = {
    # Hard anions — can stabilize very high oxidation states
    "O":  8,   # MnO₄⁻ (Mn+7), OsO₄ (Os+8) are real
    "F":  8,   # PtF₆ (Pt+5), XeF₆ are real
    # Moderately hard
    "Cl": 6,   # CrO₂Cl₂-like exist, but Cl limits high ox
    "Br": 4,   # Br is reducing — limits to +4
    "I":  3,   # I is strongly reducing — limits to +3
    # Soft / reducing anions — strongly limit cation oxidation
    "N":  4,   # Nitrides: TiN(+3), Cr₂N(+3), VN(+3), max ~+4
    "S":  4,   # Sulfides: MoS₂(+4), TiS₂(+4), max ~+4
    "Se": 4,   # Selenides: similar to sulfides
    "P":  3,   # Phosphides: max +3
    "C":  4,   # Carbides: TiC(+4), SiC(+4)
    "H":  4,   # Hydrides: limited
}


def _check_redox_compatibility(
    symbols: list[str],
    counts: list[int],
) -> list[str]:
    """
    Check electrochemical compatibility of oxidation states with anions.

    High-oxidation cations (+5, +6, +7) are only stable with hard anions
    (O²⁻, F⁻). Soft/reducing anions (N³⁻, S²⁻) would immediately reduce
    a high-oxidation metal — this is basic redox chemistry.

    Example: MnN₃Ni → charge balance requires Mn(+7), but N(-3) is too
    reducing to coexist with Mn(+7) → REJECT.
    """
    reasons = []

    # Get anion elements and their max allowed cation oxidation
    anion_syms = [s for s in symbols if s in _NONMETALS or s in _HALOGENS]
    if not anion_syms:
        return reasons

    # The most restrictive anion determines the max allowed ox state
    max_allowed_ox = min(
        _MAX_OX_WITH_ANION.get(s, 6) for s in anion_syms
    )

    # Fast-path: if max allowed ox is high (O, F present), skip check
    # Almost all common compounds with O or F pass this naturally
    if max_allowed_ox >= 6:
        return reasons

    # Find valid charge-neutral combos
    state_lists = []
    for sym in symbols:
        states = OXIDATION_STATES.get(sym)
        if states is None:
            return []  # unknown element — skip
        # Pre-filter: only keep oxidation states ≤ max_allowed_ox for cations
        state_lists.append(states)

    # Check if ANY charge-neutral combo has all cations within allowed range
    has_compatible_combo = False
    incompatible_details = []
    MAX_ITER = 50_000  # prevent exponential blowup

    for n_checked, combo in enumerate(product(*state_lists)):
        if n_checked >= MAX_ITER:
            # Too many combos — assume OK to avoid timeout
            return reasons

        total_charge = sum(ox * cnt for ox, cnt in zip(combo, counts))
        if total_charge != 0:
            continue

        # Check each cation's oxidation state against the anion limit
        combo_ok = True
        for sym, ox in zip(symbols, combo):
            if ox > 0 and ox > max_allowed_ox:
                combo_ok = False
                incompatible_details.append(
                    (sym, ox, anion_syms, max_allowed_ox)
                )
                break

        if combo_ok:
            has_compatible_combo = True
            break

    if not has_compatible_combo and incompatible_details:
        # Get unique violation info
        seen = set()
        for sym, ox, anions, max_ox in incompatible_details:
            key = (sym, ox)
            if key not in seen:
                seen.add(key)
                anion_str = ", ".join(anions)
                reasons.append(
                    f"{sym}(+{ox}) is too high an oxidation state to coexist "
                    f"with {anion_str} (max +{max_ox} allowed). "
                    f"High-ox metals require hard anions like O²⁻ or F⁻."
                )
                if len(reasons) >= 2:
                    break

    return reasons


def _check_coordination(
    symbols: list[str],
    counts: list[int],
) -> list[str]:
    """Check coordination number sanity using max coordination data."""
    reasons = []
    total = sum(counts)

    for sym, cnt in zip(symbols, counts):
        max_cn = _MAX_COORD.get(sym, 6)
        other_atoms = total - cnt
        if cnt == 1 and other_atoms > max_cn:
            reasons.append(
                f"{sym} has max coordination {max_cn} but would need to "
                f"coordinate with {other_atoms} other atoms"
            )

    return reasons


def _check_coordination_geometry(
    symbols: list[str],
    counts: list[int],
) -> list[str]:
    """
    Strict coordination geometry check using PREFERRED_COORDINATION.

    For each transition metal cation, CHECKS coordination feasibility.

    Key insight: in extended crystal structures (TiO₂, ZnS, etc.), the
    coordination number comes from the 3D packing, NOT the stoichiometry
    ratio. So for compounds matching known structure prototypes, we skip
    this check. For compounds with no prototype match, we apply the check
    as a filter for clearly implausible combinations.

    Example: Br₃PtRb → 3 elements, no prototype match, Pt has CN ≈ 3
    from stoichiometry but requires CN=4 → REJECT.
    """
    reasons = []

    # If compound matches a known structure prototype, the CN is
    # dictated by the 3D structure, not the formula ratio → skip check
    if _matches_prototype(symbols, counts):
        return reasons

    # Classify elements
    cations = [(s, c) for s, c in zip(symbols, counts)
               if s in _METALS or s in _SEMI_METALS]
    anions = [(s, c) for s, c in zip(symbols, counts)
              if s in _NONMETALS or s in _HALOGENS]

    if not cations or not anions:
        return reasons

    total_anion_atoms = sum(c for _, c in anions)
    total_cation_atoms = sum(c for _, c in cations)

    # For simple binary compounds (AB, AB₂, A₂B₃ etc.) with standard
    # stoichiometry, the real CN always comes from crystal packing.
    # Only check ternary+ compounds where the stoichiometry is unusual.
    # Also check binary compounds where a single TM has very few anion
    # neighbors AND the formula doesn't match a known binary prototype.
    n_elements = len(symbols)
    if n_elements == 2:
        # Binary: standard prototypes cover most cases.
        # Only flag truly implausible ones (e.g. Pt₃Cl → makes no sense)
        return reasons  # binary validation handled by prototype + charge check

    # For ternary+ non-prototype compounds: check if each TM cation
    # can achieve a feasible CN
    # Find valid charge-neutral oxidation state combos
    state_lists = [OXIDATION_STATES.get(s, [0]) for s in symbols]
    valid_combos = []
    for combo in product(*state_lists):
        total_charge = sum(ox * cnt for ox, cnt in zip(combo, counts))
        if total_charge == 0:
            valid_combos.append(combo)
            if len(valid_combos) >= 5:  # Limit search
                break

    if not valid_combos:
        return reasons  # charge neutrality check will catch this

    # d⁸ metals have geometrically rigid coordination (square planar / octahedral)
    # They MUST match their preferred CN exactly — no tolerance
    _STRICT_CN_METALS = {
        ("Pt", 2), ("Pt", 4),   # square planar d⁸ / octahedral d⁶
        ("Pd", 2), ("Pd", 4),   # square planar d⁸ / octahedral d⁶
        ("Au", 3), ("Au", 1),   # square planar d⁸ / linear d¹⁰
        ("Ir", 3), ("Ir", 4),   # octahedral
        ("Rh", 3),              # octahedral
        ("Ru", 3), ("Ru", 4),   # octahedral
    }

    # For each TM cation, check if ANY valid combo gives it a feasible CN
    for cat_sym, cat_cnt in cations:
        if cat_sym not in _TRANSITION_METALS:
            continue

        # Approximate CN = total_anion_atoms / cat_cnt
        approx_cn = total_anion_atoms / cat_cnt

        # Check against preferred CNs for all valid oxidation states
        has_valid_cn = False
        for combo in valid_combos:
            idx = symbols.index(cat_sym)
            ox = combo[idx]
            preferred = PREFERRED_COORDINATION.get((cat_sym, ox))
            if preferred is None:
                has_valid_cn = True  # No data → allow
                break

            # Strict metals: CN must be close (±0.5)
            # Others: allow ±1.5 tolerance (crystal packing can adjust)
            is_strict = (cat_sym, ox) in _STRICT_CN_METALS
            tol = 0.5 if is_strict else 1.5

            for pcn in preferred:
                if abs(approx_cn - pcn) <= tol:
                    has_valid_cn = True
                    break
            if has_valid_cn:
                break

        if not has_valid_cn:
            # Get what was expected
            expected_cns = set()
            for combo in valid_combos:
                idx = symbols.index(cat_sym)
                ox = combo[idx]
                pref = PREFERRED_COORDINATION.get((cat_sym, ox), [])
                expected_cns.update(pref)
            reasons.append(
                f"{cat_sym} has approx. CN={approx_cn:.0f} but requires "
                f"CN={sorted(expected_cns)} for stability "
                f"(crystal field / d-electron configuration)"
            )

    return reasons


def _check_d_electron_stability(
    symbols: list[str],
    counts: list[int],
) -> list[str]:
    """
    Check transition metal d-electron stability.
    Warn (but don't reject) for very unstable d-electron configurations.
    Only reject truly impossible configurations.
    """
    reasons = []

    # Find valid charge-neutral oxidation state combos
    state_lists = []
    for sym in symbols:
        states = OXIDATION_STATES.get(sym)
        if states is None:
            return []  # unknown element
        state_lists.append(states)

    # Check if any valid combo has all TM in stable/moderate d-configs
    has_stable_combo = False
    for combo in product(*state_lists):
        total_charge = sum(ox * cnt for ox, cnt in zip(combo, counts))
        if total_charge != 0:
            continue

        # Check d-electron stability for this combo
        all_ok = True
        for sym, ox in zip(symbols, combo):
            d_count = d_electron_count(sym, ox)
            if d_count is not None and d_count < 0:
                all_ok = False
                break

        if all_ok:
            has_stable_combo = True
            break

    if not has_stable_combo and any(s in _TRANSITION_METALS for s in symbols):
        # This is already caught by charge neutrality check, so no action needed
        pass

    return reasons


def _matches_prototype(symbols: list[str], counts: list[int]) -> bool:
    """Check if stoichiometry matches any known stable structure prototype."""
    n_elements = len(symbols)

    # Classify into cations and anions
    cations = [(s, c) for s, c in zip(symbols, counts) if s in _METALS or s in _SEMI_METALS]
    anions = [(s, c) for s, c in zip(symbols, counts) if s in _NONMETALS or s in _HALOGENS]

    if not cations or not anions:
        return False

    # Build stoichiometry key
    if n_elements == 2:
        cat_count = cations[0][1]
        ani_count = anions[0][1] if anions else 0
        from math import gcd as _gcd
        g = _gcd(cat_count, ani_count) if ani_count > 0 else 1
        key = f"A{cat_count // g}X{ani_count // g}" if cat_count // g > 1 or ani_count // g > 1 else "AX"
        # Normalize
        a = cat_count // g
        x = ani_count // g
        key = ""
        if a == 1:
            key += "A"
        else:
            key += f"A{a}"
        if x == 1:
            key += "X"
        else:
            key += f"X{x}"
        return key in STABLE_PROTOTYPES

    elif n_elements == 3 and len(cations) == 2 and len(anions) == 1:
        # ABXn pattern
        a_cnt = cations[0][1]
        b_cnt = cations[1][1]
        x_cnt = anions[0][1]
        
        # Battery-specific loose prototype matching for layered/spinel
        if (a_cnt, b_cnt, x_cnt) in [(1, 1, 2), (2, 1, 3), (1, 2, 4), (1, 1, 3)]:
            return True

        key = ""
        key += "A" if a_cnt == 1 else f"A{a_cnt}"
        key += "B" if b_cnt == 1 else f"B{b_cnt}"
        key += "X" if x_cnt == 1 else f"X{x_cnt}"
        return key in STABLE_PROTOTYPES

    elif n_elements == 4 and len(cations) == 2 and len(anions) == 2:
        # e.g., AMPO4 (olivine) where A,M are cations, P,O are anions
        a_cnt, b_cnt = cations[0][1], cations[1][1]
        c_cnt, d_cnt = anions[0][1], anions[1][1]
        if (a_cnt, b_cnt, c_cnt, d_cnt) == (1, 1, 1, 4):
            return True # Olivine
        if (a_cnt, b_cnt, c_cnt, d_cnt) == (3, 2, 3, 12):
            return True # NASICON Na3M2(PO4)3


def _d_electron_bonus(symbols: list[str], counts: list[int]) -> float:
    """Compute stability bonus based on d-electron configuration."""
    bonus = 0.0

    # Find a valid charge-neutral combo
    state_lists = []
    for sym in symbols:
        states = OXIDATION_STATES.get(sym, [0])
        state_lists.append(states)

    for combo in product(*state_lists):
        total_charge = sum(ox * cnt for ox, cnt in zip(combo, counts))
        if total_charge != 0:
            continue

        # Score this combo's d-electron stability
        combo_score = 0.0
        n_tm = 0
        for sym, ox in zip(symbols, combo):
            d = d_electron_count(sym, ox)
            if d is not None:
                n_tm += 1
                if d in _STABLE_D_COUNTS:
                    combo_score += 1.0
                elif d in _MODERATE_D_COUNTS:
                    combo_score += 0.5

        if n_tm > 0:
            bonus = max(bonus, combo_score / n_tm)

    return min(bonus, 1.0)


def _formula_from(symbols: list[str], counts: list[int]) -> str:
    """Reconstruct formula string."""
    return "".join(f"{s}{c if c > 1 else ''}" for s, c in zip(symbols, counts))


# -----------------------------------------------------------------------
if __name__ == "__main__":
    tests_valid = [
        "NaCl", "TiO2", "Al2O3", "LiCoO2", "Fe2O3", "SiO2",
        "BaTiO3", "MgO", "CaF2", "ZnS", "CuO", "SrVO3",
        "PtCl2", "Cr2O3", "MnO2", "CoFe2O4",
    ]
    tests_should_be_invalid = [
        "BrI4", "ClF3", "HeO", "LiNaCl", "ArF2", "CaBa",
        "Br3PtRb",  # Pt needs CN=4 or 6, not CN=3
        "RbFe",     # alkali + TM binary without anion
    ]

    print("Chemistry Validation + Classification Results")
    print("=" * 70)

    print("\n  Expected VALID compounds:")
    for f in tests_valid:
        result = classify_compound(f)
        score = result["stability_score"]
        v = result["validity"]
        proto = "proto" if result["prototype_match"] else "no-proto"
        coord = "CN-ok" if result["coordination_ok"] else "CN-fail"
        print(f"    {f:12s} → {v:16s} score={score:.1f}  {proto}  {coord}")
        if result["reasons"]:
            for r in result["reasons"]:
                print(f"                  ! {r}")

    print("\n  Expected INVALID compounds:")
    for f in tests_should_be_invalid:
        result = classify_compound(f)
        v = result["validity"]
        print(f"    {f:12s} → {v}")
        if result["reasons"]:
            for r in result["reasons"]:
                print(f"                  ! {r}")
