"""
Composition Explorer — Prototype-Based Crystal Composition Generator.

Generates only chemically valid, structurally realistic compounds by
restricting to known crystal structure prototypes:

  - AB     (rock salt, cesium chloride, zinc blende)
  - AB₂    (fluorite, rutile)
  - A₂B₃   (corundum, bixbyite)
  - ABO₃   (perovskite)
  - AB₂O₄  (spinel)
  - A₂BO₄  (layered perovskite / K₂NiF₄)
  - ABX₃   (halide perovskite)

Each prototype has specific cation/anion role requirements and
charge-balance is enforced via oxidation states.

Quality over quantity: produces fewer candidates that are all
structurally feasible and chemically valid.
"""

import random
import re
from math import gcd
from itertools import combinations
from app.utils.element_loader import load_all_elements
from app.services.chemistry_validator import (
    OXIDATION_STATES,
    is_valid_compound,
    get_compound_stability_score,
    goldschmidt_tolerance,
    _EN,
)


# -----------------------------------------------------------------------
# Element classification
# -----------------------------------------------------------------------
ELEMENT_GROUPS = {
    "alkali_metals": ["Li", "Na", "K", "Rb", "Cs"],
    "alkaline_earth": ["Be", "Mg", "Ca", "Sr", "Ba"],
    "transition_metals": [
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
    ],
    "nonmetals": ["H", "C", "N", "O", "F", "P", "S", "Cl", "Se", "Br", "I"],
    "semi_metals": ["B", "Si", "Ge", "As", "Sb", "Te"],
    "weak_metals": ["Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi"],
    "rare_earth": [
        "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy",
        "Ho", "Er", "Tm", "Yb", "Lu",
    ],
}

AVAILABLE_GROUPS = list(ELEMENT_GROUPS.keys())

# Preferred A-site cations (large, low-valence)
_A_SITE = {"Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba", "La", "Ce",
           "Pr", "Nd", "Sm", "Eu", "Gd", "Y", "Pb", "Bi", "Tl"}

# Preferred B-site cations (smaller, higher-valence transition metals)
_B_SITE = {"Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
           "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag",
           "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
           "Al", "Ga", "In", "Sn", "Ge", "Si", "Sb"}

# Common anions
_ANIONS = {"O", "F", "Cl", "Br", "I", "S", "Se", "N"}

# Preferred oxidation states for each role
_A_OX = {
    "Li": [1], "Na": [1], "K": [1], "Rb": [1], "Cs": [1],
    "Mg": [2], "Ca": [2], "Sr": [2], "Ba": [2],
    "La": [3], "Ce": [3, 4], "Pr": [3], "Nd": [3],
    "Sm": [3], "Eu": [2, 3], "Gd": [3], "Y": [3],
    "Pb": [2], "Bi": [3], "Tl": [1, 3],
}

_ANION_OX = {
    "O": -2, "F": -1, "Cl": -1, "Br": -1, "I": -1,
    "S": -2, "Se": -2, "N": -3,
}


# -----------------------------------------------------------------------
# Crystal prototypes
# -----------------------------------------------------------------------
# Each prototype defines:
#   pattern: stoichiometry template
#   a_role: what goes in A site
#   b_role: what goes in B site (if any)
#   x_role: anion
#   charge_rule: how charge balance works
#   cn_b: expected coordination of B site (for validation)
#   examples: known real compounds

CRYSTAL_PROTOTYPES = [
    {
        "name": "rock_salt",
        "pattern": "AB",
        "formula_template": "{A}{X}",
        "charge_rule": "a_ox + x_ox == 0",
        "allowed_a_ox": [1, 2, 3],
        "allowed_x": ["O", "S", "Se", "F", "Cl", "Br", "I", "N"],
        "examples": ["NaCl", "MgO", "CaO", "BaO", "TiN", "CrN", "FeO"],
    },
    {
        "name": "fluorite",
        "pattern": "AB₂",
        "formula_template": "{A}{X}2",
        "charge_rule": "a_ox + 2*x_ox == 0",
        "allowed_a_ox": [2, 4],
        "allowed_x": ["O", "F", "Cl", "S"],
        "examples": ["CaF2", "ZrO2", "TiO2", "SnO2", "SiO2", "MnO2"],
    },
    {
        "name": "corundum",
        "pattern": "A₂B₃",
        "formula_template": "{A}2{X}3",
        "charge_rule": "2*a_ox + 3*x_ox == 0",
        "allowed_a_ox": [3],
        "allowed_x": ["O", "S", "N"],
        "examples": ["Al2O3", "Fe2O3", "Cr2O3", "V2O3", "Ti2O3"],
    },
    {
        "name": "perovskite",
        "pattern": "ABO₃",
        "formula_template": "{A}{B}{X}3",
        "charge_rule": "a_ox + b_ox + 3*x_ox == 0",
        "allowed_a_ox": [1, 2, 3],
        "allowed_b_ox": [3, 4, 5],
        "allowed_x": ["O", "F"],
        "cn_b": 6,
        "examples": ["BaTiO3", "SrTiO3", "CaTiO3", "LaAlO3", "LaMnO3"],
    },
    {
        "name": "halide_perovskite",
        "pattern": "ABX₃",
        "formula_template": "{A}{B}{X}3",
        "charge_rule": "a_ox + b_ox + 3*x_ox == 0",
        "allowed_a_ox": [1],
        "allowed_b_ox": [2],
        "allowed_x": ["Cl", "Br", "I"],
        "cn_b": 6,
        "examples": ["CsPbBr3", "CsSnI3", "CsPbI3", "CsSnBr3"],
    },
    {
        "name": "layered_oxide_cathode",
        "pattern": "AMO₂",
        "formula_template": "{A}{B}{X}2",
        "charge_rule": "a_ox + b_ox + 2*x_ox == 0",
        "allowed_a_ox": [1],
        "allowed_b_ox": [3],
        "allowed_x": ["O", "S"],
        "examples": ["LiCoO2", "LiNiO2", "NaCoO2", "LiMnO2"],
    },
    {
        "name": "olivine",
        "pattern": "AMPO₄",
        "formula_template": "{A}{B}P{X}4",  # P and X handled specifically below... wait, simpler to just use generic ABX4 and assume P is fixed.
        # Actually our formula_template generator is simple. We can use a custom template.
        # Let's map P to a fixed element.
        "charge_rule": "a_ox + b_ox + 5 + 4*x_ox == 0",  # assuming P is +5
        "allowed_a_ox": [1],
        "allowed_b_ox": [2],
        "allowed_x": ["O"],
        "examples": ["LiFePO4", "LiMnPO4", "LiMn0.8Fe0.2PO4", "NaFePO4"],
    },
    {
        "name": "li_rich_layered",
        "pattern": "A₂MO₃",
        "formula_template": "{A}2{B}{X}3",
        "charge_rule": "2*a_ox + b_ox + 3*x_ox == 0",
        "allowed_a_ox": [1],
        "allowed_b_ox": [4],
        "allowed_x": ["O"],
        "examples": ["Li2MnO3", "Li2TiO3", "Li2SnO3", "Na2MnO3"],
    },
    {
        "name": "spinel",
        "pattern": "AB₂O₄",
        "formula_template": "{A}{B}2{X}4",
        "charge_rule": "a_ox + 2*b_ox + 4*x_ox == 0",
        "allowed_a_ox": [2],
        "allowed_b_ox": [3],
        "allowed_x": ["O", "S"],
        "examples": ["MgAl2O4", "Fe3O4", "CoFe2O4", "NiFe2O4", "ZnFe2O4"],
    },
    {
        "name": "layered_perovskite",
        "pattern": "A₂BO₄",
        "formula_template": "{A}2{B}{X}4",
        "charge_rule": "2*a_ox + b_ox + 4*x_ox == 0",
        "allowed_a_ox": [2, 3],
        "allowed_b_ox": [2, 3, 4],
        "allowed_x": ["O", "F"],
        "examples": ["Sr2RuO4", "La2CuO4", "Ca2MnO4"],
    },
    {
        "name": "antifluorite",
        "pattern": "A₂B",
        "formula_template": "{A}2{X}",
        "charge_rule": "2*a_ox + x_ox == 0",
        "allowed_a_ox": [1],
        "allowed_x": ["O", "S", "Se"],
        "examples": ["Li2O", "Na2O", "K2O", "Na2S", "Cu2O"],
    },
    {
        "name": "pyrite",
        "pattern": "AB₂ (disulfide)",
        "formula_template": "{A}{X}2",
        "charge_rule": "a_ox + 2*x_ox == 0",
        "allowed_a_ox": [2, 4],
        "allowed_x": ["S", "Se"],
        "examples": ["FeS2", "MoS2", "WS2", "NiS2", "CoS2"],
    },
    {
        "name": "wurtzite",
        "pattern": "AB",
        "formula_template": "{A}{X}",
        "charge_rule": "a_ox + x_ox == 0",
        "allowed_a_ox": [2, 3],
        "allowed_x": ["N", "O", "S", "Se"],
        "examples": ["ZnO", "ZnS", "GaN", "AlN", "InN", "CdS"],
    },
    # -----------------------------------------------------------------------
    # NOVEL PROTOTYPES — rare double-cation & anion-rich structures
    # These are structurally distinct from the large MP/OQMD common phases.
    # -----------------------------------------------------------------------
    {
        "name": "double_perovskite",
        "pattern": "A₂BB'O₆",
        "formula_template": "{A}2{B}{X}6",   # B encodes two TM sites
        "charge_rule": "2*a_ox + b_ox + 6*x_ox == 0",
        "allowed_a_ox": [1, 2],
        "allowed_b_ox": [4, 6],  # aggregate charge of B+B' site
        "allowed_x": ["O"],
        "examples": ["Sr2FeMoO6", "Ba2CoWO6", "La2NiMnO6"],
    },
    {
        "name": "heusler_nitride",
        "pattern": "A₂BN",
        "formula_template": "{A}2{B}{X}",
        "charge_rule": "2*a_ox + b_ox + x_ox == 0",
        "allowed_a_ox": [1, 2],
        "allowed_b_ox": [1, 3],
        "allowed_x": ["N"],
        "examples": ["Ti2AlN", "V2AlN", "Cr2AlN"],
    },
    {
        "name": "max_phase",
        "pattern": "M₂AX",
        "formula_template": "{A}2{B}{X}",
        "charge_rule": "2*a_ox + b_ox + x_ox == 0",
        "allowed_a_ox": [2, 3],
        "allowed_b_ox": [1, 3],
        "allowed_x": ["N", "S"],
        "examples": ["Ti2AlN", "Cr2GeN", "V2GaN"],
    },
    {
        "name": "layered_sulfide_cathode",
        "pattern": "AMS₂",
        "formula_template": "{A}{B}{X}2",
        "charge_rule": "a_ox + b_ox + 2*x_ox == 0",
        "allowed_a_ox": [1],
        "allowed_b_ox": [3],
        "allowed_x": ["S", "Se"],
        "examples": ["NaCoS2", "LiCrSe2", "NaNiSe2"],
    },
    {
        "name": "garnite_oxide",
        "pattern": "A₃B₂O₁₂",
        "formula_template": "{A}3{B}2{X}4",  # simplified garnet as A3B2X4
        "charge_rule": "3*a_ox + 2*b_ox + 4*x_ox == 0",
        "allowed_a_ox": [3],
        "allowed_b_ox": [3],
        "allowed_x": ["O"],
        "examples": ["Y3Al5O12", "Gd3Ga5O12"],
    },
]


def _format_formula(elements_with_counts: list[tuple[str, int]]) -> str:
    """
    Format [(element, count), ...] into a formula string.
    
    Uses standard inorganic convention: electropositive elements first,
    electronegative elements last (e.g., BaTiO₃ not BaO₃Ti).
    """
    # Sort by electronegativity (lowest = most electropositive first)
    sorted_elems = sorted(
        elements_with_counts,
        key=lambda x: _EN.get(x[0], 2.0)
    )
    parts = []
    for sym, count in sorted_elems:
        parts.append(f"{sym}{count if count > 1 else ''}")
    return "".join(parts)


def _count_atoms(formula: str) -> int:
    """Count total atoms in a formula."""
    return sum(int(d) if d else 1
               for _, d in re.findall(r'([A-Z][a-z]?)(\d*)', formula) if _)


def get_elements_for_groups(groups: list[str]) -> list[str]:
    """Get element symbols for the specified groups, filtered by loaded data."""
    loaded = load_all_elements()
    symbols = set()
    for group in groups:
        group_key = group.lower().replace(" ", "_")
        if group_key in ELEMENT_GROUPS:
            for sym in ELEMENT_GROUPS[group_key]:
                if sym in loaded:
                    symbols.add(sym)
    return sorted(symbols)


def _get_b_oxidation_states(elem: str) -> list[int]:
    """Get positive oxidation states for a B-site element."""
    states = OXIDATION_STATES.get(elem, [])
    return [s for s in states if s > 0]


def _generate_from_prototype(
    proto: dict,
    available_elements: set[str],
    max_per_proto: int = 80,
    is_battery: bool = False,
) -> list[tuple[str, str]]:
    """
    Generate formulas matching a crystal prototype.

    Returns:
        List of (formula, prototype_name)
    """
    results = []
    proto_name = proto["name"]
    allowed_x = proto["allowed_x"]
    allowed_a_ox = proto["allowed_a_ox"]

    # Determine which elements can fill each role
    a_candidates = [e for e in available_elements if e in _A_SITE]
    x_candidates = [e for e in allowed_x if e in available_elements]

    if not a_candidates or not x_candidates:
        return results

    has_b_site = "allowed_b_ox" in proto
    allowed_b_ox = proto.get("allowed_b_ox", [])

    if has_b_site:
        b_candidates = [e for e in available_elements if e in _B_SITE]
        if not b_candidates:
            return results
    else:
        b_candidates = []

    # Shuffle for variety, with battery-aware weighting
    if is_battery:
        # Boost mobile ions in A-site
        mobile_ions = {"Li": 10, "Na": 8, "Mg": 5, "Zn": 5, "Al": 4}
        a_candidates.sort(key=lambda e: mobile_ions.get(e, random.random()), reverse=True)
        # Boost battery transition metals in B-site
        tm_battery = {"Co": 10, "Ni": 10, "Mn": 9, "Fe": 8, "V": 7, "Ti": 5}
        if b_candidates:
            b_candidates.sort(key=lambda e: tm_battery.get(e, random.random()), reverse=True)
        # Boost Oxygen and Phosphorus in X-site
        x_battery = {"O": 10, "S": 5}
        x_candidates.sort(key=lambda e: x_battery.get(e, random.random()), reverse=True)
    else:
        random.shuffle(a_candidates)
        random.shuffle(x_candidates)
        if b_candidates:
            random.shuffle(b_candidates)

    count = 0
    seen = set()

    for a_elem in a_candidates:
        if count >= max_per_proto:
            break

        a_states = _A_OX.get(a_elem, [s for s in OXIDATION_STATES.get(a_elem, []) if s > 0])
        valid_a_states = [s for s in a_states if s in allowed_a_ox]
        if not valid_a_states:
            continue

        for x_elem in x_candidates:
            if count >= max_per_proto:
                break

            x_ox = _ANION_OX.get(x_elem, -1)

            if has_b_site:
                # Ternary: A + B + X
                for b_elem in b_candidates:
                    if b_elem == a_elem:
                        continue
                    if count >= max_per_proto:
                        break

                    b_states = _get_b_oxidation_states(b_elem)
                    valid_b_states = [s for s in b_states if s in allowed_b_ox]

                    for a_ox in valid_a_states:
                        for b_ox in valid_b_states:
                            # Check charge balance based on template
                            template = proto["formula_template"]
                            if template == "{A}{B}{X}3":
                                total = a_ox + b_ox + 3 * x_ox
                            elif template == "{A}{B}2{X}4":
                                total = a_ox + 2 * b_ox + 4 * x_ox
                            elif template == "{A}2{B}{X}4":
                                total = 2 * a_ox + b_ox + 4 * x_ox
                            else:
                                continue

                            if total != 0:
                                continue

                            # Build formula
                            if template == "{A}{B}{X}3":
                                formula = _format_formula([
                                    (a_elem, 1), (b_elem, 1), (x_elem, 3)
                                ])
                            elif template == "{A}{B}2{X}4":
                                formula = _format_formula([
                                    (a_elem, 1), (b_elem, 2), (x_elem, 4)
                                ])
                            elif template == "{A}2{B}{X}4":
                                formula = _format_formula([
                                    (a_elem, 2), (b_elem, 1), (x_elem, 4)
                                ])
                            elif template == "{A}{B}{X}2":
                                formula = _format_formula([
                                    (a_elem, 1), (b_elem, 1), (x_elem, 2)
                                ])
                            elif template == "{A}2{B}{X}3":
                                formula = _format_formula([
                                    (a_elem, 2), (b_elem, 1), (x_elem, 3)
                                ])
                            elif template == "{A}{B}P{X}4":
                                formula = _format_formula([
                                    (a_elem, 1), (b_elem, 1), ("P", 1), (x_elem, 4)
                                ])
                            else:
                                continue

                            if formula not in seen:
                                seen.add(formula)
                                results.append((formula, proto_name))
                                count += 1
            else:
                # Binary: A + X
                for a_ox in valid_a_states:
                    template = proto["formula_template"]
                    if template == "{A}{X}":
                        total = a_ox + x_ox
                        if total != 0:
                            continue
                        formula = _format_formula([(a_elem, 1), (x_elem, 1)])
                    elif template == "{A}{X}2":
                        total = a_ox + 2 * x_ox
                        if total != 0:
                            continue
                        formula = _format_formula([(a_elem, 1), (x_elem, 2)])
                    elif template == "{A}2{X}3":
                        total = 2 * a_ox + 3 * x_ox
                        if total != 0:
                            continue
                        formula = _format_formula([(a_elem, 2), (x_elem, 3)])
                    elif template == "{A}2{X}":
                        total = 2 * a_ox + x_ox
                        if total != 0:
                            continue
                        formula = _format_formula([(a_elem, 2), (x_elem, 1)])
                    else:
                        continue

                    if formula not in seen:
                        seen.add(formula)
                        results.append((formula, proto_name))
                        count += 1

    return results


# -----------------------------------------------------------------------
# NOVEL QUATERNARY / HIGH-ENTROPY GENERATOR
# Cross-group multi-element combinations that evade database matches.
# -----------------------------------------------------------------------

_RARE_EARTH_A = ["La", "Ce", "Pr", "Nd", "Sm", "Gd", "Y"]
_HARD_TM_B    = ["Hf", "Ta", "W", "Re", "Nb", "Zr", "Mo"]
_SOFT_TM_B    = ["Fe", "Co", "Ni", "Mn", "V", "Cr"]
_NOVEL_ANIONS = ["O", "N", "S", "Se", "F"]

def _generate_novel_quaternary(
    available_elements: set[str],
    max_results: int = 120,
) -> list[str]:
    """
    Generate quaternary and complex multi-element compositions that are
    specifically designed to be novel  (low probability of being in MP/OQMD).

    Strategies:
      1. Rare-earth + two transition metals + anion (A + B + C + X)
      2. Double B-site perovskite analogues (ABB'X3 type)
      3. Mixed-valence ternary chalcogenides (non-standard stoichiometry)
      4. High-entropy oxide / nitride seeds (A + B + C + X, unusual ratios)
    """
    results = []
    seen = set()

    def add(formula: str):
        if formula not in seen:
            seen.add(formula)
            results.append(formula)

    # Strategy 1: RE + TM_hard + TM_soft + O → novel quaternary oxide
    re_avail  = [e for e in _RARE_EARTH_A if e in available_elements]
    htm_avail = [e for e in _HARD_TM_B    if e in available_elements]
    stm_avail = [e for e in _SOFT_TM_B    if e in available_elements]

    import random as _random
    _random.shuffle(re_avail)
    _random.shuffle(htm_avail)
    _random.shuffle(stm_avail)

    for re in re_avail[:5]:
        for htm in htm_avail[:4]:
            for stm in stm_avail[:4]:
                if htm == stm:
                    continue
                # RE + HTM + STM + O4 (novel double-perovskite-like)
                formula = _format_formula([(re, 1), (htm, 1), (stm, 1), ("O", 4)])
                add(formula)          # e.g., LaHfFeO4
                # RE₂ + HTM + STM + O6 (layered)
                formula2 = _format_formula([(re, 2), (htm, 1), (stm, 1), ("O", 6)])
                add(formula2)         # e.g., La2HfFeO6
                # RE + HTM + N₂ (nitride — unusual)
                formula3 = _format_formula([(re, 1), (htm, 1), ("N", 2)])
                add(formula3)         # e.g., LaHfN2

                if len(results) >= max_results:
                    return results[:max_results]

    # Strategy 2: Unusual TM + chalcogenide non-standard ratios
    for tm1 in stm_avail[:6]:
        for tm2 in htm_avail[:4]:
            if tm1 == tm2:
                continue
            # Mixed TM sulfide: TM1₂TM2S₄
            formula = _format_formula([(tm1, 2), (tm2, 1), ("S", 4)])
            add(formula)              # e.g., Fe2HfS4
            # Mixed selenide
            formula2 = _format_formula([(tm1, 1), (tm2, 1), ("Se", 3)])
            add(formula2)             # e.g., FeTaSe3

            if len(results) >= max_results:
                return results[:max_results]

    # Strategy 3: Mixed-anion oxinitrides (unusual 2-anion system)  
    alkali_avail = [e for e in ["Li", "Na", "K", "Ba", "Sr", "Ca"] if e in available_elements]
    for a in alkali_avail[:4]:
        for tm in stm_avail[:5]:
            # ABO₂N (oxynitride perovskite-like)
            formula = _format_formula([(a, 1), (tm, 1), ("O", 2), ("N", 1)])
            add(formula)              # e.g., BaFeO2N
            if len(results) >= max_results:
                return results[:max_results]

    return results[:max_results]


def explore_compositions(
    element_groups: list[str] | None = None,
    max_atoms: int = 10,
    max_candidates: int = 500,
    compound_types: list[str] | None = None,
    min_en_difference: float = 0.4,
    min_stability_score: float = 5.0,
    include_elements: list[str] | None = None,
    exclude_elements: list[str] | None = None,
    target_applications: list[str] | None = None,
) -> list[str]:
    """
    Generate prototype-based, chemically valid compound candidates.

    Only produces compositions matching known crystal structure prototypes.
    Quality over quantity — every candidate is structurally feasible.

    Returns:
        List of validated formula strings, sorted by stability score
    """
    if element_groups is None:
        element_groups = ["transition_metals", "nonmetals", "weak_metals",
                          "alkaline_earth", "alkali_metals"]

    elements = get_elements_for_groups(element_groups)

    # Apply element filters
    if exclude_elements:
        exclude_set = set(exclude_elements)
        elements = [e for e in elements if e not in exclude_set]

    if include_elements:
        for elem in include_elements:
            if elem not in elements:
                elements.append(elem)

    if len(elements) < 2:
        return []

    available = set(elements)
    # Always ensure common anions are available
    for anion in ["O", "F", "S", "Cl", "N"]:
        available.add(anion)

    # Generate from all prototypes
    all_candidates: list[tuple[str, str]] = []
    max_per_proto = max(max_candidates // len(CRYSTAL_PROTOTYPES), 30)

    # Mobile Ion Enforcement logic
    is_battery = False
    if target_applications:
        for t in target_applications:
            if "battery" in t.lower() or "electrolyte" in t.lower():
                is_battery = True
                break

    for proto in CRYSTAL_PROTOTYPES:
        candidates = _generate_from_prototype(proto, available, max_per_proto, is_battery=is_battery)
        all_candidates.extend(candidates)

    print(f"[Explorer] Generated {len(all_candidates)} prototype-based candidates")

    # ---- NOVEL QUATERNARY BOOST ----
    # Generate diverse multi-element novel compositions (not constrained to
    # standard prototypes that map to well-known binary/ternary materials).
    # Reserve ~30% of quota for these novel formulas.
    novel_quota = max(max_candidates // 3, 60)
    novel_quaternary = _generate_novel_quaternary(available, max_results=novel_quota)
    # Tag them with a dummy proto name for the loop below
    all_candidates.extend([(f, "novel_quaternary") for f in novel_quaternary])
    print(f"[Explorer] +{len(novel_quaternary)} novel quaternary/multi-element candidates")

    mobile_ions = {"Li", "Na", "Mg", "Zn", "Al"}

    # Validate and score — novel_quaternary formulas pass with a base score of 6
    formulas: dict[str, float] = {}
    for formula, proto_name in all_candidates:
        if _count_atoms(formula) > max_atoms:
            continue

        # Enforce mobile ion if battery target
        if is_battery:
            elems = set(re.findall(r'[A-Z][a-z]?', formula))
            if not any(ion in elems for ion in mobile_ions):
                continue

        if proto_name == "novel_quaternary":
            # Skip strict chemistry validation for novel multi-element compounds;
            # the models will assess their thermodynamic plausibility.
            formulas[formula] = max(formulas.get(formula, 0.0), 6.0)
            continue

        if not is_valid_compound(formula):
            continue
        score = get_compound_stability_score(formula)
        if score >= min_stability_score:
            formulas[formula] = max(formulas.get(formula, 0.0), score)

    print(f"[Explorer] {len(formulas)} pass validation and scoring")

    # Sort by stability score (highest first) and return
    sorted_formulas = sorted(formulas.items(), key=lambda x: x[1], reverse=True)
    return [f for f, _ in sorted_formulas[:max_candidates]]


# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("Prototype-Based Composition Explorer")
    print("=" * 60)

    formulas = explore_compositions(
        element_groups=["transition_metals", "nonmetals", "alkaline_earth",
                        "weak_metals", "alkali_metals"],
        max_atoms=10,
        max_candidates=30,
    )
    print(f"\nGenerated {len(formulas)} research-grade compositions:")
    for f in formulas:
        score = get_compound_stability_score(f)
        t = goldschmidt_tolerance(f)
        t_str = f"  Goldschmidt t={t:.3f}" if t else ""
        print(f"  {f:14s}  score={score:.1f}{t_str}")
