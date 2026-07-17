"""
Compound Generator — Research-Grade Chemistry-Aware Generation.

Generates chemically valid, stability-biased compounds using:
  1. Real oxidation states with charge neutrality
  2. Stability-weighted template selection
  3. Preferred oxidation states (most stable d-electron configs first)
  4. Structural prototypes (perovskite, spinel, rutile, corundum, etc.)
  5. Stability pre-scoring from chemistry_validator
  6. All output validated through chemistry_validator

Crystal structure templates include:
  - Perovskite ABO₃ (BaTiO₃, SrVO₃)
  - Spinel AB₂O₄ (MgAl₂O₄, CoFe₂O₄)
  - Rutile AO₂ (TiO₂, MnO₂)
  - Corundum A₂O₃ (Al₂O₃, Fe₂O₃)
  - Rock salt AO (MgO, NaCl)
  - Fluorite AX₂ (CaF₂)
  - Delafossite ABO₂ (CuFeO₂)
  - Olivine A₂BO₄ (Mg₂SiO₄)
  - Wurtzite AX (ZnS, GaN)
  - Layered AMX₂ (NaCoO₂)
"""

import random
from math import gcd
from itertools import product as itertools_product
from app.services.chemistry_validator import (
    OXIDATION_STATES,
    PREFERRED_COORDINATION,
    IONIC_RADII,
    is_valid_compound,
    validate_compound,
    get_compound_stability_score,
    d_electron_count,
)


# -----------------------------------------------------------------------
# Element pools organized by chemistry role
# -----------------------------------------------------------------------
CATION_ELEMENTS = {
    "alkali": ["Li", "Na", "K", "Rb", "Cs"],
    "alkaline": ["Mg", "Ca", "Sr", "Ba"],
    "transition_3d": ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"],
    "transition_4d": ["Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag"],
    "transition_5d": ["Hf", "Ta", "W", "Pt", "Au", "Ir"],
    "post_transition": ["Al", "Ga", "In", "Sn", "Pb", "Bi"],
    "semimetal": ["B", "Si", "Ge"],
    "rare_earth": ["La", "Ce", "Nd", "Y", "Gd", "Eu", "Sm"],
}

ANION_ELEMENTS = {
    "oxide": ["O"],
    "halide": ["F", "Cl", "Br", "I"],
    "chalcogenide": ["S", "Se"],
    "nitride_phosphide": ["N", "P"],
}

ALL_CATIONS = [e for group in CATION_ELEMENTS.values() for e in group]
ALL_ANIONS = [e for group in ANION_ELEMENTS.values() for e in group]

# Stability-biased d-electron counts (prefer these oxidation states)
_STABLE_D_COUNTS = {0, 3, 5, 6, 8, 10}


# -----------------------------------------------------------------------
# Structural templates with stability weighting
# Higher weight = generated more frequently (represents commonly stable types)
# -----------------------------------------------------------------------
STRUCTURAL_TEMPLATES = {
    # Binary templates
    "rock_salt": {
        "pattern": "{A}{X}",
        "cation_val": [1, 2],
        "anion_val": [-1, -2],
        "weight": 15,
        "description": "AX rock salt (NaCl, MgO)",
    },
    "fluorite": {
        "pattern": "{A}{X}2",
        "cation_val": [4],
        "anion_val": [-2],
        "weight": 10,
        "description": "AX₂ fluorite (CaF₂, ZrO₂)",
    },
    "rutile": {
        "pattern": "{A}{X}2",
        "cation_val": [4],
        "anion_val": [-2],
        "weight": 12,
        "description": "AX₂ rutile (TiO₂, MnO₂)",
    },
    "corundum": {
        "pattern": "{A}2{X}3",
        "cation_val": [3],
        "anion_val": [-2],
        "weight": 12,
        "description": "A₂X₃ corundum (Al₂O₃, Fe₂O₃, Cr₂O₃)",
    },
    "antifluorite": {
        "pattern": "{A}3{X}2",
        "cation_val": [2],
        "anion_val": [-3],
        "weight": 5,
        "description": "A₃X₂ (Mg₃N₂)",
    },
    "wurtzite": {
        "pattern": "{A}{X}",
        "cation_val": [2, 3],
        "anion_val": [-2, -3],
        "weight": 8,
        "description": "AX wurtzite (ZnS, GaN)",
    },
    # Ternary templates
    "perovskite": {
        "pattern": "{A}{B}{X}3",
        "cation_val_A": [1, 2],
        "cation_val_B": [3, 4, 5],
        "anion_val": [-2],
        "weight": 20,
        "description": "ABO₃ perovskite (BaTiO₃, SrVO₃)",
    },
    "spinel": {
        "pattern": "{A}{B}2{X}4",
        "cation_val_A": [2],
        "cation_val_B": [3],
        "anion_val": [-2],
        "weight": 15,
        "description": "AB₂O₄ spinel (MgAl₂O₄)",
    },
    "ilmenite": {
        "pattern": "{A}{B}{X}3",
        "cation_val_A": [2],
        "cation_val_B": [4],
        "anion_val": [-2],
        "weight": 10,
        "description": "ABO₃ ilmenite (FeTiO₃)",
    },
    "delafossite": {
        "pattern": "{A}{B}{X}2",
        "cation_val_A": [1],
        "cation_val_B": [3],
        "anion_val": [-2],
        "weight": 8,
        "description": "ABO₂ delafossite (CuFeO₂)",
    },
    "olivine": {
        "pattern": "{A}2{B}{X}4",
        "cation_val_A": [2],
        "cation_val_B": [4],
        "anion_val": [-2],
        "weight": 10,
        "description": "A₂BO₄ olivine (Mg₂SiO₄)",
    },
    "layered": {
        "pattern": "{A}{B}{X}2",
        "cation_val_A": [1],
        "cation_val_B": [3],
        "anion_val": [-2],
        "weight": 10,
        "description": "ABX₂ layered (NaCoO₂, LiFeO₂)",
    },
    "garnet": {
        "pattern": "{A}3{B}2{X}3",
        "cation_val_A": [2],
        "cation_val_B": [3],
        "anion_val": [-2],
        "weight": 5,
        "description": "A₃B₂X₃ garnet-like",
    },
}


def _format_formula(parts: list[tuple[str, int]]) -> str:
    """Format [(sym, count), ...] into a formula string."""
    return "".join(f"{sym}{count if count > 1 else ''}" for sym, count in parts)


def _preferred_oxidation_state(element: str, as_cation: bool = True) -> list[int]:
    """
    Get oxidation states sorted by thermodynamic stability.
    Prefers d-electron configurations with high CFSE.
    """
    states = OXIDATION_STATES.get(element, [])
    if as_cation:
        states = [s for s in states if s > 0]
    else:
        states = [s for s in states if s < 0]

    # Sort by d-electron stability
    def stability_key(ox):
        d = d_electron_count(element, ox)
        if d is not None and d in _STABLE_D_COUNTS:
            return 0  # most stable
        return 1  # less stable

    states.sort(key=stability_key)
    return states


def _balance_binary(cat: str, ani: str) -> str | None:
    """Generate charge-balanced binary compound, preferring stable oxidation states."""
    cat_states = _preferred_oxidation_state(cat, as_cation=True)
    ani_states = _preferred_oxidation_state(ani, as_cation=False)

    for cv in cat_states:
        for av in ani_states:
            abs_av = abs(av)
            g = gcd(cv, abs_av)
            n_cat = abs_av // g
            n_ani = cv // g
            if n_cat > 4 or n_ani > 6:
                continue
            formula = _format_formula([(cat, n_cat), (ani, n_ani)])
            if is_valid_compound(formula):
                return formula

    return None


def _balance_ternary(cat1: str, cat2: str, ani: str) -> str | None:
    """Generate charge-balanced ternary compound, preferring stable states."""
    cat1_states = _preferred_oxidation_state(cat1, as_cation=True)
    cat2_states = _preferred_oxidation_state(cat2, as_cation=True)
    ani_states = _preferred_oxidation_state(ani, as_cation=False)

    for v1, v2, va in itertools_product(cat1_states, cat2_states, ani_states):
        abs_va = abs(va)
        total_positive = v1 + v2
        if total_positive % abs_va == 0:
            n_ani = total_positive // abs_va
            if n_ani <= 8:
                formula = _format_formula([(cat1, 1), (cat2, 1), (ani, n_ani)])
                if is_valid_compound(formula):
                    return formula

    return None


def _weighted_template_choice() -> str:
    """Choose a template weighted by stability probability."""
    names = list(STRUCTURAL_TEMPLATES.keys())
    weights = [STRUCTURAL_TEMPLATES[n]["weight"] for n in names]
    total = sum(weights)
    probs = [w / total for w in weights]
    return random.choices(names, weights=probs, k=1)[0]


def generate_from_template(template_name: str | None = None) -> str | None:
    """Generate a compound from a structural template."""
    if template_name is None:
        template_name = _weighted_template_choice()

    tmpl = STRUCTURAL_TEMPLATES.get(template_name)
    if tmpl is None:
        return None

    pattern = tmpl["pattern"]
    anion_val = tmpl["anion_val"]

    ani_pool = [a for a in ALL_ANIONS
                if any(v in OXIDATION_STATES.get(a, []) for v in anion_val)]
    if not ani_pool:
        return None
    ani = random.choice(ani_pool)

    if "{B}" in pattern:
        cat_A_vals = tmpl.get("cation_val_A", tmpl.get("cation_val", []))
        cat_B_vals = tmpl.get("cation_val_B", tmpl.get("cation_val", []))

        pool_A = [c for c in ALL_CATIONS
                  if any(v in OXIDATION_STATES.get(c, []) for v in cat_A_vals)]
        pool_B = [c for c in ALL_CATIONS
                  if any(v in OXIDATION_STATES.get(c, []) for v in cat_B_vals)]

        if not pool_A or not pool_B:
            return None

        cat_A = random.choice(pool_A)
        cat_B = random.choice([b for b in pool_B if b != cat_A] or pool_B)
        formula = pattern.format(A=cat_A, B=cat_B, X=ani)
    else:
        cat_vals = tmpl.get("cation_val", [])
        pool = [c for c in ALL_CATIONS
                if any(v in OXIDATION_STATES.get(c, []) for v in cat_vals)]
        if not pool:
            return None
        cat = random.choice(pool)
        formula = pattern.format(A=cat, X=ani)

    if is_valid_compound(formula):
        return formula
    return None


def generate_binary_compound() -> str | None:
    """Generate a valid binary compound with stability-preferred oxidation states."""
    cat = random.choice(ALL_CATIONS)
    ani = random.choice(ALL_ANIONS)
    return _balance_binary(cat, ani)


def generate_ternary_compound() -> str | None:
    """Generate a valid ternary compound with stability-preferred states."""
    cat1, cat2 = random.sample(ALL_CATIONS, 2)
    ani = random.choice(ALL_ANIONS)
    return _balance_ternary(cat1, cat2, ani)


def generate_compounds(
    count: int = 10,
    include_ternary: bool = True,
    include_templates: bool = True,
    stability_threshold: float = 5.0,
) -> list[str]:
    """
    Generate unique, chemically valid, stability-biased compounds.

    Generates more compounds than requested, then sorts by stability score
    and returns the most stable ones.

    Args:
        count: number of compounds to return
        include_ternary: include ternary compounds
        include_templates: use structural templates
        stability_threshold: minimum stability score (0-10)

    Returns:
        List of validated formula strings, sorted by stability score
    """
    # Generate 3x to have selection pool, then sort by stability
    target_pool = count * 3
    compounds: dict[str, float] = {}  # formula -> stability_score
    max_attempts = target_pool * 20
    attempts = 0

    while len(compounds) < target_pool and attempts < max_attempts:
        attempts += 1
        formula = None

        r = random.random()
        if include_templates and r < 0.45:
            formula = generate_from_template()
        elif include_ternary and r < 0.75:
            formula = generate_ternary_compound()
        else:
            formula = generate_binary_compound()

        if formula and formula not in compounds and is_valid_compound(formula):
            score = get_compound_stability_score(formula)
            if score >= stability_threshold:
                compounds[formula] = score

    # Sort by stability score (highest first)
    sorted_compounds = sorted(compounds.items(), key=lambda x: x[1], reverse=True)

    return [f for f, _ in sorted_compounds[:count]]


# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("Research-Grade Compound Generation")
    print("=" * 60)

    print("\nBinary compounds (stability-biased):")
    for _ in range(6):
        c = generate_binary_compound()
        if c:
            score = get_compound_stability_score(c)
            print(f"  {c:12s}  score={score:.1f}")

    print("\nTernary compounds (stability-biased):")
    for _ in range(6):
        c = generate_ternary_compound()
        if c:
            score = get_compound_stability_score(c)
            print(f"  {c:12s}  score={score:.1f}")

    print("\nTemplate compounds:")
    for name in list(STRUCTURAL_TEMPLATES.keys())[:8]:
        c = generate_from_template(name)
        if c:
            score = get_compound_stability_score(c)
            desc = STRUCTURAL_TEMPLATES[name]["description"]
            print(f"  {name:14s} -> {c:12s}  score={score:.1f}  ({desc})")

    print(f"\nStability-sorted batch of 15:")
    for c in generate_compounds(15, stability_threshold=5.0):
        score = get_compound_stability_score(c)
        print(f"  {c:14s}  score={score:.1f}")
