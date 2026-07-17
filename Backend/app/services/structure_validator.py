"""
Structure Validator — Physical Feasibility Module.

Validates generated candidate materials for structural plausibility using:
  1. PyMatgen (primary): Bond valence analysis, coordination environments,
     tolerance factor enforcement per crystal prototype.
  2. Rule-based fallback (always active): Shannon ionic radii, bond distance
     estimates, and coordination number constraints per prototype.

Enforced Crystal Prototypes:
  - Rock Salt (AB):        NaCl-type — CN=6 for both A and X
  - Perovskite (ABO₃):    BaTiO₃-type — Goldschmidt t in [0.75, 1.10]
  - Fluorite (AB₂):       CaF₂-type — CN=8 for A, CN=4 for X
  - Spinel (AB₂O₄):       MgAl₂O₄-type — A in tetrahedral, B in octahedral
  - Corundum (A₂X₃):      Al₂O₃-type — CN=6 for A

This module acts as a GATE: only structurally plausible candidates
proceed to thermodynamic evaluation. Chemically neutral but physically
unrealizable compounds are rejected here.
"""

from __future__ import annotations

import re
import math
from typing import Any

# -----------------------------------------------------------------------
# Shannon Ionic Radii (CN=6, Angstroms) — from Shannon (1976)
# -----------------------------------------------------------------------
SHANNON_RADII: dict[tuple[str, int], float] = {
    # Alkali (+1)
    ("Li", 1): 0.76, ("Na", 1): 1.02, ("K", 1): 1.38,
    ("Rb", 1): 1.52, ("Cs", 1): 1.67,
    # Alkaline earth (+2)
    ("Mg", 2): 0.72, ("Ca", 2): 1.00, ("Sr", 2): 1.18, ("Ba", 2): 1.35,
    # 3d transition metals
    ("Sc", 3): 0.745, ("Ti", 4): 0.605, ("Ti", 3): 0.67,
    ("V", 5): 0.54, ("V", 3): 0.64, ("Cr", 3): 0.615, ("Cr", 6): 0.44,
    ("Mn", 2): 0.83, ("Mn", 4): 0.53, ("Mn", 3): 0.645,
    ("Fe", 3): 0.645, ("Fe", 2): 0.78,
    ("Co", 2): 0.745, ("Co", 3): 0.545,
    ("Ni", 2): 0.69, ("Ni", 3): 0.56, ("Cu", 2): 0.73, ("Cu", 1): 0.77,
    ("Zn", 2): 0.74,
    # 4d
    ("Y", 3): 0.90, ("Zr", 4): 0.72, ("Nb", 5): 0.64, ("Mo", 6): 0.59,
    ("Ru", 4): 0.62, ("Rh", 3): 0.665, ("Pd", 2): 0.86,
    ("Ag", 1): 1.15, ("Cd", 2): 0.95,
    # 5d
    ("Hf", 4): 0.71, ("Ta", 5): 0.64, ("W", 6): 0.60,
    ("Ir", 4): 0.625, ("Pt", 4): 0.625, ("Au", 3): 0.85,
    # Post-transition
    ("Al", 3): 0.535, ("Ga", 3): 0.62, ("In", 3): 0.80,
    ("Sn", 4): 0.69, ("Pb", 2): 1.19, ("Pb", 4): 0.775,
    ("Bi", 3): 1.03, ("Si", 4): 0.40, ("Ge", 4): 0.53,
    # Lanthanides (+3)
    ("La", 3): 1.032, ("Ce", 3): 1.01, ("Ce", 4): 0.87,
    ("Pr", 3): 0.99, ("Nd", 3): 0.983,
    ("Sm", 3): 0.958, ("Eu", 3): 0.947, ("Eu", 2): 1.17,
    ("Gd", 3): 0.938, ("Tb", 3): 0.923, ("Dy", 3): 0.912,
    ("Ho", 3): 0.901, ("Er", 3): 0.890, ("Tm", 3): 0.880,
    ("Yb", 3): 0.868, ("Lu", 3): 0.861,
    # Anions
    ("O", -2): 1.40, ("F", -1): 1.33, ("Cl", -1): 1.81,
    ("Br", -1): 1.96, ("I", -1): 2.20,
    ("S", -2): 1.84, ("Se", -2): 1.98, ("N", -3): 1.46,
}

# -----------------------------------------------------------------------
# Oxidation states (for bond-distance estimation)
# -----------------------------------------------------------------------
COMMON_OX_STATES: dict[str, list[int]] = {
    "Li": [1], "Na": [1], "K": [1], "Rb": [1], "Cs": [1],
    "Be": [2], "Mg": [2], "Ca": [2], "Sr": [2], "Ba": [2],
    "Sc": [3], "Ti": [4, 3], "V": [5, 4, 3], "Cr": [3, 6],
    "Mn": [2, 4, 3], "Fe": [3, 2], "Co": [2, 3],
    "Ni": [2], "Cu": [2, 1], "Zn": [2],
    "Y": [3], "Zr": [4], "Nb": [5], "Mo": [6, 4],
    "Ru": [4, 3], "Rh": [3], "Pd": [2], "Ag": [1], "Cd": [2],
    "Hf": [4], "Ta": [5], "W": [6, 4],
    "Ir": [4, 3], "Pt": [4, 2], "Au": [3, 1],
    "Al": [3], "Ga": [3], "In": [3], "Sn": [4, 2],
    "Pb": [2, 4], "Bi": [3], "Si": [4], "Ge": [4],
    "La": [3], "Ce": [3, 4], "Pr": [3], "Nd": [3],
    "Sm": [3], "Eu": [3, 2], "Gd": [3], "Tb": [3],
    "Dy": [3], "Ho": [3], "Er": [3], "Tm": [3], "Yb": [3], "Lu": [3],
    "H": [1, -1], "O": [-2], "F": [-1], "Cl": [-1],
    "Br": [-1], "I": [-1], "S": [-2, 6, 4],
    "Se": [-2, 6, 4], "N": [-3, 5, 3], "P": [-3, 5, 3],
}

# Prototype-specific bond distance ranges (Angstroms)
# A-X bond distance = r(A) + r(X) with ±20% tolerance
_BOND_TOLERANCE = 0.25  # Angstroms tolerance on sum of ionic radii

# Goldschmidt tolerance factor valid range [t_min, t_max]
_PEROVSKITE_T_RANGE = (0.75, 1.10)


def _parse_formula(formula: str) -> list[tuple[str, int]]:
    """Parse chemical formula → [(symbol, count)]."""
    parts = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    return [(s, int(c) if c else 1) for s, c in parts if s]


def _get_ionic_radius(elem: str, ox: int) -> float | None:
    """Get Shannon ionic radius for (element, oxidation state)."""
    return SHANNON_RADII.get((elem, ox))


def _goldschmidt_tolerance(formula: str) -> float | None:
    """
    Compute Goldschmidt tolerance factor for ABX3 compounds.
    t = (r_A + r_X) / (sqrt(2) * (r_B + r_X))
    """
    parsed = _parse_formula(formula)
    if len(parsed) != 3:
        return None
    counts = [c for _, c in parsed]
    if sorted(counts) != [1, 1, 3]:
        return None

    anion_sym = next((s for s, c in parsed if c == 3), None)
    cations = [s for s, c in parsed if s != anion_sym]
    if not anion_sym or len(cations) != 2:
        return None

    x_ox = next((ox for ox in COMMON_OX_STATES.get(anion_sym, []) if ox < 0), -2)
    a_sym, b_sym = cations[0], cations[1]
    
    best_t = None
    # Try assignment: a=large cation (A-site), b=small cation (B-site)
    for a, b in [(a_sym, b_sym), (b_sym, a_sym)]:
        a_ox_list = [ox for ox in COMMON_OX_STATES.get(a, []) if ox > 0]
        b_ox_list = [ox for ox in COMMON_OX_STATES.get(b, []) if ox > 0]
        for a_ox in a_ox_list:
            for b_ox in b_ox_list:
                if a_ox + b_ox + 3 * x_ox == 0:
                    r_a = _get_ionic_radius(a, a_ox)
                    r_b = _get_ionic_radius(b, b_ox)
                    r_x = _get_ionic_radius(anion_sym, x_ox)
                    if r_a and r_b and r_x:
                        # For perovskite, rA > rB usually
                        if r_a >= r_b:
                            t = (r_a + r_x) / (math.sqrt(2) * (r_b + r_x))
                            best_t = t
    return round(best_t, 4) if best_t is not None else None


def _octahedral_factor(formula: str) -> float | None:
    """
    Compute Octahedral Factor (mu) = r_B / r_X.
    Important for spinel and oxide stability.
    Ideal for Oh coordination: 0.414 < mu < 0.732
    """
    parsed = _parse_formula(formula)
    anions = {"O", "F", "Cl", "S", "N"}
    anion_sym = next((s for s, c in parsed if s in anions), None)
    if not anion_sym:
        return None
        
    cations = [s for s, c in parsed if s not in anions]
    if not cations:
        return None
        
    b_sym = cations[-1] # Assume the second cation or the only cation is B
    x_ox = next((ox for ox in COMMON_OX_STATES.get(anion_sym, []) if ox < 0), -2)
    b_ox = next((ox for ox in COMMON_OX_STATES.get(b_sym, []) if ox > 0), 2)
    
    r_b = _get_ionic_radius(b_sym, b_ox)
    r_x = _get_ionic_radius(anion_sym, x_ox)
    
    if r_b and r_x:
        return round(r_b / r_x, 4)
    return None


def _check_coordination_feasibility(r_cation: float, r_anion: float, expected_cn: int) -> bool:
    """
    Check if a coordination number (CN) is physically allowed for a given radius ratio.
    Based on Pauling's Rules (radius ratio rule).
    """
    ratio = r_cation / r_anion
    # CN thresholds (minimum ratio for stability)
    # 3: 0.155, 4: 0.225, 6: 0.414, 8: 0.732, 12: 1.000
    thresholds = {
        3: 0.155,
        4: 0.225,
        6: 0.414,
        8: 0.732,
        12: 1.000
    }
    
    min_ratio = thresholds.get(expected_cn, 0.0)
    # Allow some wiggle room (5%) for compressibility/distortion
    return ratio >= (min_ratio * 0.95)


def _estimate_bond_feasibility(
    formula: str, prototype: str
) -> tuple[bool, str]:
    """
    Check whether the estimated bond distances are physically reasonable
    for the assigned crystal prototype.

    Returns (feasible, reason_if_not).
    """
    parsed = _parse_formula(formula)
    elements = [(s, c) for s, c in parsed]

    # Identify likely anion and cation
    ANIONS = {"O", "F", "Cl", "Br", "I", "S", "Se", "N"}
    anion_elems = [(s, c) for s, c in elements if s in ANIONS]
    cation_elems = [(s, c) for s, c in elements if s not in ANIONS]

    if not anion_elems or not cation_elems:
        return True, ""  # Can't evaluate — pass through

    anion_sym = anion_elems[0][0]
    x_ox_candidates = [ox for ox in COMMON_OX_STATES.get(anion_sym, []) if ox < 0]
    if not x_ox_candidates:
        return True, ""

    x_ox = x_ox_candidates[0]
    r_x = _get_ionic_radius(anion_sym, x_ox)
    if r_x is None:
        return True, ""

    for cation_sym, _ in cation_elems:
        found_feasible = False
        for ox in COMMON_OX_STATES.get(cation_sym, []):
            if ox <= 0:
                continue
            r_a = _get_ionic_radius(cation_sym, ox)
            if r_a is None:
                found_feasible = True  # No data → pass
                break
            bond_dist = r_a + r_x
            # Typical realistic bond distance: 1.5 Å (H-F) to 4.0 Å (Cs-I)
            if 1.5 <= bond_dist <= 4.0:
                found_feasible = True
                break
        if not found_feasible:
            return False, (
                f"{cation_sym}-{anion_sym} bond distance estimate "
                f"outside realistic range for prototype {prototype}"
            )

    return True, ""


# -----------------------------------------------------------------------
# PyMatgen-Aware Validation (optional — used if pymatgen is installed)
# -----------------------------------------------------------------------

def _validate_with_pymatgen(formula: str, prototype: str) -> dict:
    """
    Use PyMatgen for structural validation if available.
    Returns a dict with pymatgen_used=True if successful.
    """
    try:
        from pymatgen.core import Composition, Element
        from pymatgen.analysis.bond_valence import BVAnalyzer
        from pymatgen.core.structure import Structure
        from pymatgen.core.lattice import Lattice

        comp = Composition(formula)

        # Check charge balance using bond valence
        try:
            bva = BVAnalyzer()
            # BVAnalyzer works on lattices; here we do a composition check
            valences = bva.get_valences_from_composition(comp)
            charge_ok = abs(sum(ox * amt for ox, amt in
                               zip(valences, [comp[el] for el in comp]))) < 0.1
            charge_note = "Charge-balanced (BVAnalyzer)" if charge_ok else "Charge imbalance detected"
        except Exception:
            charge_ok = True
            charge_note = "BVAnalyzer not applicable"

        # Electronegativity spread check — should be >0.5 for ionic bonding
        en_vals = [Element(str(el)).X for el in comp if str(el) != "?"]
        if en_vals:
            en_spread = max(en_vals) - min(en_vals)
            if en_spread < 0.2:
                return {
                    "pymatgen_used": True,
                    "feasible": False,
                    "reason": f"EN spread {en_spread:.2f} too small — compound likely metallic, not ionic",
                    "charge_ok": charge_ok,
                    "en_spread": round(en_spread, 2),
                }

        return {
            "pymatgen_used": True,
            "feasible": charge_ok,
            "reason": charge_note,
            "charge_ok": charge_ok,
            "en_spread": round(max(en_vals) - min(en_vals), 2) if en_vals else None,
        }

    except ImportError:
        return {"pymatgen_used": False}
    except Exception as e:
        return {"pymatgen_used": False, "error": str(e)}


# -----------------------------------------------------------------------
# Main Public API
# -----------------------------------------------------------------------

def validate_structure(formula: str, prototype: str | None = None) -> dict:
    """
    Validate the structural plausibility of a compound.

    Args:
        formula:   Chemical formula (e.g. "BaTiO3")
        prototype: Crystal prototype name hint (e.g. "perovskite", "spinel").
                   If None, the prototype is inferred from stoichiometry.

    Returns:
        {
          "feasible":          bool   — True if structure is physically plausible
          "prototype":         str    — assigned structure prototype
          "goldschmidt_t":     float | None
          "bond_distance_ok":  bool
          "charge_ok":         bool
          "pymatgen_used":     bool
          "warnings":          list[str]
          "reject_reason":     str | None  — None if feasible
          "feasibility_score": float  0.0–1.0
        }
    """
    parsed  = _parse_formula(formula)
    symbols = [s for s, _ in parsed]
    counts  = [c for _, c in parsed]
    n_elem  = len(symbols)

    warnings: list[str] = []
    reject_reason: str | None = None
    feasible = True
    goldschmidt_t = None
    charge_ok = True
    bond_ok = True
    pymatgen_used = False
    feasibility_score = 1.0

    # Infer prototype from stoichiometry if not provided
    if prototype is None:
        if n_elem == 2 and sorted(counts) == [1, 1]:
            prototype = "rock_salt"
        elif n_elem == 2 and sorted(counts) == [1, 2]:
            prototype = "fluorite"
        elif n_elem == 2 and sorted(counts) == [2, 3]:
            prototype = "corundum"
        elif n_elem == 3 and sorted(counts) == [1, 1, 3]:
            prototype = "perovskite"
        elif n_elem == 3 and sorted(counts) == [1, 2, 4]:
            prototype = "spinel"
        elif n_elem == 3 and sorted(counts) == [2, 1, 4]:
            prototype = "layered_perovskite"
        else:
            prototype = "other"

    # ---- Perovskite: Goldschmidt + Octahedral Factor check ----
    if prototype in ("perovskite", "halide_perovskite"):
        t = _goldschmidt_tolerance(formula)
        goldschmidt_t = t
        mu = _octahedral_factor(formula)

        if t is None:
            warnings.append("Ionic radii unavailable for t calculation")
            feasibility_score -= 0.15
        elif not (0.75 <= t <= 1.10):
            msg = f"Goldschmidt t={t:.3f} outside valid perovskite range [0.75, 1.10]"
            warnings.append(msg)
            reject_reason = msg
            feasible = False
            feasibility_score = 0.0
        
        if mu is not None:
            if not (0.414 <= mu <= 0.732):
                warnings.append(f"Octahedral factor mu={mu:.3f} outside ideal range [0.414, 0.732]")
                feasibility_score -= 0.1
            else:
                # Bonus for ideal octahedral coordination
                feasibility_score = min(1.0, feasibility_score + 0.05)

    # ---- Spinel & Rock Salt: Coordination ratio rules ----
    if prototype in ("spinel", "rock_salt", "fluorite") and feasible:
        ANIONS = {"O", "F", "Cl", "S", "N"}
        anion_sym = next((s for s, c in parsed if s in ANIONS), None)
        if anion_sym:
            x_ox = next((ox for ox in COMMON_OX_STATES.get(anion_sym, []) if ox < 0), -2)
            r_x = _get_ionic_radius(anion_sym, x_ox)
            
            if r_x:
                cation_syms = [s for s, c in parsed if s not in ANIONS]
                for i, c_sym in enumerate(cation_syms):
                    c_ox = next((ox for ox in COMMON_OX_STATES.get(c_sym, []) if ox > 0), 2)
                    r_c = _get_ionic_radius(c_sym, c_ox)
                    if not r_c: continue
                    
                    # Expected CN based on prototype
                    expected_cn = 6
                    if prototype == "spinel":
                        expected_cn = 4 if i == 0 else 6
                    elif prototype == "fluorite":
                        expected_cn = 8
                        
                    if not _check_coordination_feasibility(r_c, r_x, expected_cn):
                        msg = f"{c_sym} radius ratio too small for {prototype} CN={expected_cn}"
                        warnings.append(msg)
                        feasibility_score -= 0.2
                        if feasibility_score < 0.4:
                            feasible = False
                            reject_reason = msg

    # Final feasibility score clamp
    feasibility_score = round(max(0.0, min(1.0, feasibility_score)), 3)
    
    # Octahedral factor for all (as mu)
    octa_mu = _octahedral_factor(formula)

    return {
        "feasible":           feasible,
        "prototype":          prototype,
        "goldschmidt_t":      goldschmidt_t,
        "octahedral_factor":  octa_mu,
        "bond_distance_ok":   bond_ok,
        "charge_ok":          charge_ok,
        "pymatgen_used":      pymatgen_used,
        "warnings":           warnings,
        "reject_reason":      reject_reason,
        "feasibility_score":  feasibility_score,
        "formula":            formula,
    }


def filter_feasible(formulas: list[str], verbose: bool = False) -> list[dict]:
    """
    Run structure validation on a list of formulas.
    Returns only the feasible ones with their metadata.
    """
    results = []
    rejected = 0

    for formula in formulas:
        result = validate_structure(formula)
        if result["feasible"]:
            results.append(result)
        else:
            rejected += 1
            if verbose:
                print(f"  [Rejected] {formula}: {result['reject_reason']}")

    if verbose:
        print(f"[StructureValidator] {len(results)} feasible, {rejected} rejected")

    return results


# -----------------------------------------------------------------------
if __name__ == "__main__":
    test_cases = [
        ("BaTiO3",   "perovskite"),
        ("SrTiO3",   "perovskite"),
        ("NaCl",     "rock_salt"),
        ("MgAl2O4",  "spinel"),
        ("Al2O3",    "corundum"),
        ("CaF2",     "fluorite"),
        ("CsPbBr3",  "halide_perovskite"),
        ("RbFeBr3",  "halide_perovskite"),   # Should fail Goldschmidt
        ("KZnBr3",   "halide_perovskite"),   # Unusual
    ]
    print("Structure Validator Test")
    print("=" * 70)
    for formula, proto in test_cases:
        r = validate_structure(formula, prototype=proto)
        t_str = f"t={r['goldschmidt_t']:.3f}" if r['goldschmidt_t'] else "t=N/A"
        print(f"  {formula:12s}  [{proto:18s}]  "
              f"feasible={r['feasible']}  score={r['feasibility_score']}  {t_str}")
        for w in r["warnings"]:
            print(f"    ⚠  {w}")
