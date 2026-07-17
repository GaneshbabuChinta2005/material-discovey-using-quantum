"""
Chemical Naming Service — IUPAC-style names for inorganic compounds.

Generates systematic names:
  - Binary: "Sodium Chloride" (NaCl), "Titanium(IV) Oxide" (TiO₂)
  - Ternary: "Barium Titanate" (BaTiO₃), "Strontium Vanadate" (SrVO₃)
  - Uses Stock notation (Roman numerals) for transition metals
  - Appends structural prototype when known (e.g., "[Perovskite]")
"""

import re
from app.services.chemistry_validator import OXIDATION_STATES

# Full element names
ELEMENT_NAMES = {
    "H": "Hydrogen", "He": "Helium", "Li": "Lithium", "Be": "Beryllium",
    "B": "Boron", "C": "Carbon", "N": "Nitrogen", "O": "Oxygen",
    "F": "Fluorine", "Ne": "Neon", "Na": "Sodium", "Mg": "Magnesium",
    "Al": "Aluminium", "Si": "Silicon", "P": "Phosphorus", "S": "Sulfur",
    "Cl": "Chlorine", "Ar": "Argon", "K": "Potassium", "Ca": "Calcium",
    "Sc": "Scandium", "Ti": "Titanium", "V": "Vanadium", "Cr": "Chromium",
    "Mn": "Manganese", "Fe": "Iron", "Co": "Cobalt", "Ni": "Nickel",
    "Cu": "Copper", "Zn": "Zinc", "Ga": "Gallium", "Ge": "Germanium",
    "As": "Arsenic", "Se": "Selenium", "Br": "Bromine", "Kr": "Krypton",
    "Rb": "Rubidium", "Sr": "Strontium", "Y": "Yttrium", "Zr": "Zirconium",
    "Nb": "Niobium", "Mo": "Molybdenum", "Ru": "Ruthenium", "Rh": "Rhodium",
    "Pd": "Palladium", "Ag": "Silver", "Cd": "Cadmium", "In": "Indium",
    "Sn": "Tin", "Sb": "Antimony", "Te": "Tellurium", "I": "Iodine",
    "Cs": "Caesium", "Ba": "Barium", "La": "Lanthanum", "Ce": "Cerium",
    "Pr": "Praseodymium", "Nd": "Neodymium", "Sm": "Samarium", "Eu": "Europium",
    "Gd": "Gadolinium", "Tb": "Terbium", "Dy": "Dysprosium", "Ho": "Holmium",
    "Er": "Erbium", "Tm": "Thulium", "Yb": "Ytterbium", "Lu": "Lutetium",
    "Hf": "Hafnium", "Ta": "Tantalum", "W": "Tungsten", "Re": "Rhenium",
    "Os": "Osmium", "Ir": "Iridium", "Pt": "Platinum", "Au": "Gold",
    "Pb": "Lead", "Bi": "Bismuth",
}

# Anion naming (suffix for binary compounds)
ANION_SUFFIXES = {
    "O": "Oxide", "S": "Sulfide", "Se": "Selenide", "Te": "Telluride",
    "N": "Nitride", "P": "Phosphide", "As": "Arsenide",
    "F": "Fluoride", "Cl": "Chloride", "Br": "Bromide", "I": "Iodide",
    "C": "Carbide", "Si": "Silicide", "B": "Boride", "H": "Hydride",
}

# Ternary oxyanion naming (for ABO₃-type compounds, the B-site gets "-ate")
OXYANION_SUFFIXES = {
    "Ti": "Titanate", "V": "Vanadate", "Cr": "Chromate", "Mn": "Manganate",
    "Fe": "Ferrate", "Co": "Cobaltate", "Ni": "Nickelate", "Cu": "Cuprate",
    "Zn": "Zincate", "Al": "Aluminate", "Si": "Silicate", "Nb": "Niobate",
    "Mo": "Molybdate", "W": "Tungstate", "Ta": "Tantalate", "Zr": "Zirconate",
    "Sn": "Stannate", "Pb": "Plumbate", "Ru": "Ruthenate", "Ir": "Iridate",
    "Rh": "Rhodate", "Bi": "Bismuthate", "Sb": "Antimonate", "Ge": "Germanate",
    "P": "Phosphate", "As": "Arsenate", "B": "Borate", "S": "Sulfate",
}

# Roman numeral lookup
_ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V",
           6: "VI", 7: "VII", 8: "VIII"}

# Transition metals (need Stock notation)
_TM_SET = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
    "Sn", "Pb", "Bi", "Sb",
}

# Known compound common names
COMMON_NAMES = {
    "NaCl": "Sodium Chloride (Halite)",
    "MgO": "Magnesium Oxide (Periclase)",
    "TiO2": "Titanium(IV) Oxide (Rutile)",
    "Al2O3": "Aluminium Oxide (Corundum)",
    "Fe2O3": "Iron(III) Oxide (Hematite)",
    "SiO2": "Silicon Dioxide (Quartz)",
    "CaF2": "Calcium Fluoride (Fluorite)",
    "ZnS": "Zinc Sulfide (Sphalerite)",
    "BaTiO3": "Barium Titanate (Perovskite)",
    "SrTiO3": "Strontium Titanate (Perovskite)",
    "LiCoO2": "Lithium Cobalt(III) Oxide",
    "MgAl2O4": "Magnesium Aluminate (Spinel)",
    "CaCO3": "Calcium Carbonate (Calcite)",
    "KCl": "Potassium Chloride (Sylvite)",
    "CuO": "Copper(II) Oxide (Tenorite)",
    "Cu2O": "Copper(I) Oxide (Cuprite)",
    "FeS2": "Iron(II) Disulfide (Pyrite)",
    "PbS": "Lead(II) Sulfide (Galena)",
    "GaN": "Gallium Nitride",
    "ZnO": "Zinc Oxide (Wurtzite)",
    "Cr2O3": "Chromium(III) Oxide (Eskolaite)",
    "MnO2": "Manganese(IV) Oxide (Pyrolusite)",
    "WO3": "Tungsten(VI) Oxide",
    "V2O5": "Vanadium(V) Oxide",
    "HfO2": "Hafnium(IV) Oxide",
}

# Structural prototype patterns
_PEROVSKITE_RE = re.compile(r'^([A-Z][a-z]?)([A-Z][a-z]?)O3$')
_SPINEL_RE = re.compile(r'^([A-Z][a-z]?)([A-Z][a-z]?)2O4$')
_CORUNDUM_RE = re.compile(r'^([A-Z][a-z]?)2O3$')
_RUTILE_RE = re.compile(r'^([A-Z][a-z]?)O2$')


def _parse_formula(formula: str) -> list[tuple[str, int]]:
    """Parse a formula into [(element, count), ...]."""
    return [(sym, int(n) if n else 1)
            for sym, n in re.findall(r'([A-Z][a-z]?)(\d*)', formula) if sym]


def _infer_oxidation_state(cation: str, anion: str, n_cat: int, n_ani: int) -> int | None:
    """Infer the oxidation state of the cation in a binary compound."""
    ani_val = {
        "O": -2, "S": -2, "Se": -2, "Te": -2,
        "F": -1, "Cl": -1, "Br": -1, "I": -1,
        "N": -3, "P": -3,
    }.get(anion)
    if ani_val is None:
        return None
    total_neg = ani_val * n_ani
    if n_cat == 0:
        return None
    ox = -total_neg // n_cat
    if ox * n_cat + ani_val * n_ani == 0:
        return ox
    return None


def _needs_stock(element: str) -> bool:
    """Check if element needs Stock notation (Roman numerals)."""
    if element in _TM_SET:
        return True
    states = [s for s in OXIDATION_STATES.get(element, []) if s > 0]
    return len(states) > 1


def get_compound_name(formula: str) -> str:
    """
    Generate a systematic name for an inorganic compound.

    Examples:
        NaCl   → "Sodium Chloride"
        TiO2   → "Titanium(IV) Oxide"
        BaTiO3 → "Barium Titanate"
        Fe2O3  → "Iron(III) Oxide"
        LiCoO2 → "Lithium Cobalt(III) Oxide"
    """
    # Check common names first
    if formula in COMMON_NAMES:
        return COMMON_NAMES[formula]

    parsed = _parse_formula(formula)
    if not parsed:
        return formula

    # Identify cations and anions
    anion_set = set(ANION_SUFFIXES.keys())
    cations = [(sym, n) for sym, n in parsed if sym not in anion_set]
    anions = [(sym, n) for sym, n in parsed if sym in anion_set]

    # If no clear separation, try electronegativity
    if not anions:
        # Last element is usually the anion
        anions = [parsed[-1]]
        cations = parsed[:-1]

    if not cations:
        return formula

    # --- Binary compound: 1 cation + 1 anion ---
    if len(cations) == 1 and len(anions) == 1:
        cat_sym, n_cat = cations[0]
        ani_sym, n_ani = anions[0]
        cat_name = ELEMENT_NAMES.get(cat_sym, cat_sym)
        ani_name = ANION_SUFFIXES.get(ani_sym, ani_sym + "ide")

        if _needs_stock(cat_sym):
            ox = _infer_oxidation_state(cat_sym, ani_sym, n_cat, n_ani)
            if ox and ox in _ROMAN:
                cat_name = f"{cat_name}({_ROMAN[ox]})"

        return f"{cat_name} {ani_name}"

    # --- Ternary oxides: ABO₃, AB₂O₄, ABO₂ ---
    if len(cations) == 2 and len(anions) == 1 and anions[0][0] == "O":
        cat1_sym, n1 = cations[0]
        cat2_sym, n2 = cations[1]
        n_o = anions[0][1]

        cat1_name = ELEMENT_NAMES.get(cat1_sym, cat1_sym)

        # Try to use oxyanion suffix for the B-site cation
        if cat2_sym in OXYANION_SUFFIXES:
            suffix = OXYANION_SUFFIXES[cat2_sym]
            # Add structure prototype hint
            proto = ""
            if _PEROVSKITE_RE.match(formula):
                proto = " [Perovskite]"
            elif _SPINEL_RE.match(formula):
                proto = " [Spinel]"
            return f"{cat1_name} {suffix}{proto}"

        # Fallback: cation1 cation2 oxide
        cat2_name = ELEMENT_NAMES.get(cat2_sym, cat2_sym)
        ox2 = _infer_oxidation_state(cat2_sym, "O", n2, n_o - (n1 * 2 if n1 > 0 else 0))
        if ox2 and _needs_stock(cat2_sym) and ox2 in _ROMAN:
            cat2_name = f"{cat2_name}({_ROMAN[ox2]})"

        return f"{cat1_name} {cat2_name} Oxide"

    # --- Ternary non-oxides ---
    if len(cations) == 2 and len(anions) == 1:
        cat1_name = ELEMENT_NAMES.get(cations[0][0], cations[0][0])
        cat2_name = ELEMENT_NAMES.get(cations[1][0], cations[1][0])
        ani_name = ANION_SUFFIXES.get(anions[0][0], anions[0][0] + "ide")
        return f"{cat1_name} {cat2_name} {ani_name}"

    # Fallback: just join element names
    names = [ELEMENT_NAMES.get(sym, sym) for sym, _ in parsed]
    return " ".join(names)


# -----------------------------------------------------------------------
if __name__ == "__main__":
    test_formulas = [
        "NaCl", "TiO2", "Al2O3", "Fe2O3", "BaTiO3", "SrVO3",
        "LiCoO2", "MgAl2O4", "CaF2", "ZnS", "GaN", "MnO2",
        "PtCl2", "CuO", "Cr2O3", "KBr", "WO3", "HfO2",
        "BaCrO4", "SrIrO3", "NiMoS3", "CeTaO4",
    ]
    print("Chemical Naming Results")
    print("=" * 60)
    for f in test_formulas:
        name = get_compound_name(f)
        print(f"  {f:14s} → {name}")
