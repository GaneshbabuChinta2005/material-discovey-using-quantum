"""
Physics Corrector — Domain-Aware Post-Processing of AI Predictions.

Corrects raw model outputs (band gap, formation energy) using
established materials science rules:

  1. Band gap clamping based on compound chemistry class
  2. Formation energy sanity using electronegativity-based estimates
  3. Confidence scoring based on how much correction was needed

References:
  - Shannon & Prewitt ionic radii
  - Duffy (1990) optical basicity → band gap correlation
  - Materials Project experimental data ranges
"""

import re
from typing import Any


# -----------------------------------------------------------------------
# Electronegativity (Pauling)
# -----------------------------------------------------------------------
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
}

# Element classifications
_TRANSITION_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
}
_HALOGENS = {"F", "Cl", "Br", "I"}
_ALKALI = {"Li", "Na", "K", "Rb", "Cs"}
_ALKALINE_EARTH = {"Be", "Mg", "Ca", "Sr", "Ba"}


# -----------------------------------------------------------------------
# Known band gap ranges from Materials Project / experimental data
# Format: {compound: (min_eV, max_eV, source)}
# -----------------------------------------------------------------------
REFERENCE_BANDGAPS: dict[str, tuple[float, float, str]] = {
    # Fluorides — wide-gap insulators
    "LiF":   (13.0, 14.5, "exp"),
    "NaF":   (11.0, 12.0, "exp"),
    "CaF2":  (11.0, 12.5, "exp"),
    "MgF2":  (10.5, 12.0, "exp"),
    "BaF2":  (9.0, 10.5, "exp"),
    "AlF3":  (10.0, 11.5, "exp"),
    # Oxides — large gap
    "MgO":   (7.2, 8.0, "exp"),
    "Al2O3": (8.0, 9.5, "exp"),
    "SiO2":  (8.5, 9.5, "exp"),
    "CaO":   (6.5, 7.5, "exp"),
    "BaO":   (3.5, 4.5, "exp"),
    "TiO2":  (3.0, 3.4, "exp"),
    "ZnO":   (3.2, 3.5, "exp"),
    "SnO2":  (3.5, 4.0, "exp"),
    "ZrO2":  (5.0, 6.0, "exp"),
    "HfO2":  (5.5, 6.5, "exp"),
    "CeO2":  (3.0, 3.5, "exp"),
    # Oxides — small gap / metallic
    "Fe2O3": (1.9, 2.3, "exp"),
    "Fe3O4": (0.0, 0.3, "exp"),  # half-metallic
    "CoO":   (2.3, 2.8, "exp"),
    "NiO":   (3.5, 4.3, "exp"),
    "CuO":   (1.2, 1.9, "exp"),
    "Cu2O":  (2.0, 2.5, "exp"),
    "MnO":   (3.5, 4.2, "exp"),
    "MnO2":  (1.3, 2.5, "exp"),
    "Cr2O3": (3.0, 3.5, "exp"),
    "V2O5":  (2.0, 2.8, "exp"),
    # Sulfides
    "ZnS":   (3.5, 3.8, "exp"),
    "CdS":   (2.3, 2.6, "exp"),
    "PbS":   (0.3, 0.5, "exp"),
    "MoS2":  (1.2, 1.9, "exp"),
    "CuS":   (0.0, 0.5, "exp"),  # metallic
    "FeS2":  (0.8, 1.2, "exp"),
    # Nitrides
    "BN":    (5.5, 6.5, "exp"),
    "AlN":   (5.9, 6.3, "exp"),
    "GaN":   (3.2, 3.5, "exp"),
    "Si3N4": (4.5, 5.5, "exp"),
    "TiN":   (0.0, 0.0, "exp"),  # metallic
    "CrN":   (0.0, 0.0, "exp"),  # metallic
    "VN":    (0.0, 0.0, "exp"),  # metallic
    # Chlorides
    "NaCl":  (8.5, 9.5, "exp"),
    "KCl":   (8.0, 9.0, "exp"),
    "CaCl2": (7.0, 8.5, "exp"),
    # Perovskites
    "BaTiO3": (3.0, 3.5, "exp"),
    "SrTiO3": (3.2, 3.4, "exp"),
    # Iodides — small gap (I⁻ is most polarizable halide)
    "NaI":   (5.5, 6.5, "exp"),
    "KI":    (5.8, 6.5, "exp"),
    "CsI":   (6.0, 6.5, "exp"),
    "PbI2":  (2.3, 2.6, "exp"),
    "BiI3":  (1.7, 2.0, "exp"),
    "CsPbI3": (1.7, 1.8, "exp"),  # halide perovskite
    "CsSnI3": (1.2, 1.4, "exp"),  # halide perovskite
    # Bromides
    "NaBr":  (7.0, 7.5, "exp"),
    "KBr":   (7.0, 7.5, "exp"),
    "CsPbBr3": (2.2, 2.4, "exp"),  # halide perovskite
}

# Known formation energies (eV/atom) from Materials Project
REFERENCE_ENERGIES: dict[str, tuple[float, float, str]] = {
    "TiO2":   (-3.4, -3.1, "MP"),
    "Al2O3":  (-3.5, -3.3, "MP"),
    "SiO2":   (-3.2, -3.0, "MP"),
    "MgO":    (-3.1, -2.9, "MP"),
    "Fe2O3":  (-2.7, -2.5, "MP"),
    "ZnO":    (-1.8, -1.6, "MP"),
    "BaTiO3": (-2.8, -2.5, "MP"),
    "NaCl":   (-2.1, -1.9, "MP"),
    "CaF2":   (-4.2, -3.9, "MP"),
    "ZnS":    (-1.2, -1.0, "MP"),
    "GaN":    (-1.3, -1.1, "MP"),
    "AlN":    (-1.6, -1.4, "MP"),
}


# -----------------------------------------------------------------------
# Band gap chemistry rules
# -----------------------------------------------------------------------

def _parse_formula(formula: str) -> list[tuple[str, int]]:
    """Parse formula → [(symbol, count), ...]."""
    parts = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    return [(s, int(c) if c else 1) for s, c in parts if s]


def _classify_compound(formula: str) -> dict[str, Any]:
    """Classify compound into chemistry categories for band gap rules."""
    parsed = _parse_formula(formula)
    if not parsed:
        return {"class": "unknown"}

    elements = [s for s, _ in parsed]
    counts = [c for _, c in parsed]
    elem_set = set(elements)
    total_atoms = sum(counts)

    # Fractions
    tm_count = sum(c for s, c in zip(elements, counts) if s in _TRANSITION_METALS)
    tm_fraction = tm_count / total_atoms if total_atoms > 0 else 0

    halogen_count = sum(c for s, c in zip(elements, counts) if s in _HALOGENS)
    has_F = "F" in elem_set
    has_O = "O" in elem_set
    has_N = "N" in elem_set
    has_S = "S" in elem_set or "Se" in elem_set

    # EN stats
    en_vals = [_EN.get(s, 2.0) for s in elements]
    en_diff = max(en_vals) - min(en_vals)

    result = {
        "elements": elements,
        "counts": counts,
        "tm_fraction": tm_fraction,
        "en_diff": en_diff,
        "has_F": has_F,
        "has_O": has_O,
        "has_N": has_N,
        "has_S": has_S,
    }

    # Classification
    if has_F and not has_O:
        if tm_fraction < 0.3:
            result["class"] = "alkali_fluoride"  # LiF, CaF2
        else:
            result["class"] = "tm_fluoride"  # TiF4, FeF3
    elif has_O and not has_F:
        if tm_fraction >= 0.4:
            result["class"] = "tm_oxide"  # Fe2O3, CoO
        elif tm_fraction > 0:
            result["class"] = "mixed_oxide"  # BaTiO3, LiCoO2
        else:
            result["class"] = "main_group_oxide"  # MgO, Al2O3, SiO2
    elif has_N and not has_O:
        if tm_fraction >= 0.5:
            result["class"] = "tm_nitride"  # TiN, CrN — metallic
        else:
            result["class"] = "main_group_nitride"  # BN, AlN, GaN
    elif has_S:
        result["class"] = "chalcogenide"  # ZnS, MoS2
    elif halogen_count > 0:
        has_I = "I" in elem_set
        has_Br = "Br" in elem_set
        has_Cl = "Cl" in elem_set
        # Halide perovskite check (ABX3 with TM/p-block + halide)
        if tm_fraction > 0 and sorted(counts) == [1, 1, 3]:
            result["class"] = "halide_perovskite"  # CsPbBr3, CsSnI3
        elif has_I and not has_F:
            result["class"] = "iodide"  # NaI, PbI2, BiI3
        elif has_Br and not has_F:
            result["class"] = "bromide"  # NaBr, KBr
        elif has_Cl and not has_F:
            result["class"] = "chloride"  # NaCl, KCl
        else:
            result["class"] = "halide"  # mixed or generic
    else:
        result["class"] = "intermetallic" if tm_fraction > 0.5 else "unknown"

    return result


# -----------------------------------------------------------------------
# Band gap correction rules
# -----------------------------------------------------------------------
_BANDGAP_RULES: dict[str, tuple[float, float]] = {
    # (min_eV, max_eV) — physics-based range for each class
    "alkali_fluoride":     (5.0, 14.0),  # Wide-gap insulators
    "tm_fluoride":         (2.0, 8.0),   # Somewhat reduced by TM d-bands
    "main_group_oxide":    (3.0, 10.0),  # MgO(7.8), Al2O3(8.8), SiO2(9.0)
    "mixed_oxide":         (1.5, 5.0),   # BaTiO3(3.2), LiCoO2(~2.5)
    "tm_oxide":            (0.0, 3.5),   # Fe2O3(2.1), CoO(2.5), Fe3O4(0)
    "main_group_nitride":  (3.0, 6.5),   # BN(6.0), AlN(6.1), GaN(3.4)
    "tm_nitride":          (0.0, 0.5),   # TiN, CrN, VN — metallic
    "chalcogenide":        (0.0, 4.0),   # ZnS(3.7), CdS(2.4), PbS(0.4)
    "iodide":              (1.0, 7.0),   # NaI(6.0), PbI2(2.5), BiI3(1.8)
    "bromide":             (2.0, 8.0),   # NaBr(7.2), KBr(7.2)
    "chloride":            (5.0, 10.0),  # NaCl(9.0), KCl(8.5)
    "halide":              (5.0, 10.0),  # generic halide fallback
    "halide_perovskite":   (1.0, 3.5),   # CsPbBr3(2.3), CsSnI3(1.3)
    "intermetallic":       (0.0, 0.5),   # Metals — zero gap
    "unknown":             (0.0, 10.0),  # No constraint
}


def correct_bandgap(
    raw_bandgap: float | None,
    formula: str,
) -> dict[str, Any]:
    """
    Apply physics-based correction to a raw band gap prediction.

    Returns:
        {
            "corrected_eV": float,
            "raw_eV": float,
            "was_corrected": bool,
            "correction_reason": str | None,
            "compound_class": str,
            "confidence": str,  # "high" / "moderate" / "low"
            "reference_range": (min, max) | None,
        }
    """
    if raw_bandgap is None:
        raw_bandgap = 0.0

    # Check for exact reference match — blend raw prediction within range
    ref = REFERENCE_BANDGAPS.get(formula)
    if ref is not None:
        ref_min, ref_max, _ = ref
        ref_mid = (ref_min + ref_max) / 2
        ref_range = ref_max - ref_min
        # Use raw prediction to pick a value within the reference range
        # This gives physically meaningful variability
        if ref_range > 0 and raw_bandgap is not None:
            # Clamp raw prediction into the reference range
            blended = max(ref_min, min(ref_max, raw_bandgap))
            # If raw is outside range, bias toward midpoint
            if raw_bandgap < ref_min or raw_bandgap > ref_max:
                blended = ref_mid + (raw_bandgap - ref_mid) * 0.1
                blended = max(ref_min, min(ref_max, blended))
        else:
            blended = ref_mid
        return {
            "corrected_eV": round(blended, 2),
            "raw_eV": round(raw_bandgap, 2),
            "was_corrected": True,
            "correction_reason": f"Reference range: {ref_min}-{ref_max} eV",
            "compound_class": "known",
            "confidence": "high",
            "reference_range": (ref_min, ref_max),
        }

    # Classify compound
    info = _classify_compound(formula)
    comp_class = info.get("class", "unknown")
    rule_range = _BANDGAP_RULES.get(comp_class, (0.0, 10.0))
    rule_min, rule_max = rule_range

    # Apply clamping
    corrected = raw_bandgap
    correction_reason = None
    was_corrected = False

    if raw_bandgap < rule_min:
        corrected = rule_min
        correction_reason = (
            f"{comp_class} compounds have band gap ≥ {rule_min} eV "
            f"(raw prediction {raw_bandgap:.2f} eV is too low)"
        )
        was_corrected = True
    elif raw_bandgap > rule_max:
        corrected = rule_max
        correction_reason = (
            f"{comp_class} compounds have band gap ≤ {rule_max} eV "
            f"(raw prediction {raw_bandgap:.2f} eV is too high)"
        )
        was_corrected = True

    # Ensure non-negative
    corrected = max(0.0, corrected)

    # Confidence based on correction magnitude
    if not was_corrected:
        confidence = "moderate"  # Model agreed with physics
    elif abs(raw_bandgap - corrected) < 1.0:
        confidence = "moderate"  # Small correction
    elif abs(raw_bandgap - corrected) < 3.0:
        confidence = "low"  # Significant correction
    else:
        confidence = "speculative"  # Huge correction — model is way off

    return {
        "corrected_eV": round(corrected, 2),
        "raw_eV": round(raw_bandgap, 2),
        "was_corrected": was_corrected,
        "correction_reason": correction_reason,
        "compound_class": comp_class,
        "confidence": confidence,
        "reference_range": None,
    }


def correct_formation_energy(
    raw_energy: float | None,
    formula: str,
) -> dict[str, Any]:
    """
    Apply physics-based correction to formation energy prediction.

    Returns:
        {
            "corrected_eV": float,
            "raw_eV": float,
            "was_corrected": bool,
            "correction_reason": str | None,
            "stability_class": str,
            "confidence": str,
        }
    """
    if raw_energy is None:
        raw_energy = 0.0

    # Check for reference match
    ref = REFERENCE_ENERGIES.get(formula)
    if ref is not None:
        ref_min, ref_max, source = ref
        ref_mid = (ref_min + ref_max) / 2
        return {
            "corrected_eV": round(ref_mid, 3),
            "raw_eV": round(raw_energy, 3),
            "was_corrected": True,
            "correction_reason": f"Reference value from {source}: {ref_min} to {ref_max} eV/atom",
            "stability_class": "stable" if ref_mid < -1.0 else "marginal",
            "confidence": "high",
        }

    # Physics-based correction using EN difference
    info = _classify_compound(formula)
    en_diff = info.get("en_diff", 1.0)

    # Estimate expected energy range from EN difference
    # Higher EN diff → more ionic → more negative formation energy
    expected_min = -en_diff * 1.5 - 0.5  # rough lower bound
    expected_max = -en_diff * 0.3         # rough upper bound

    corrected = raw_energy
    correction_reason = None
    was_corrected = False

    # If energy is positive, it's definitely wrong for a stable compound
    if raw_energy > 0:
        corrected = expected_max
        correction_reason = (
            f"Positive formation energy ({raw_energy:.2f} eV/atom) "
            f"indicates instability — corrected to {corrected:.2f} based on EN diff"
        )
        was_corrected = True

    # If energy is suspiciously shallow for a highly ionic compound
    if en_diff > 2.0 and raw_energy > -0.5:
        corrected = min(corrected, expected_max)
        correction_reason = (
            f"High EN diff ({en_diff:.1f}) suggests more negative energy "
            f"than {raw_energy:.2f} eV/atom"
        )
        was_corrected = True

    # Stability classification
    if corrected < -2.0:
        stability_class = "highly_stable"
    elif corrected < -1.0:
        stability_class = "stable"
    elif corrected < -0.3:
        stability_class = "marginal"
    else:
        stability_class = "unstable"

    # Confidence
    if not was_corrected:
        confidence = "moderate"
    elif abs(raw_energy - corrected) < 0.5:
        confidence = "moderate"
    else:
        confidence = "low"

    return {
        "corrected_eV": round(corrected, 3),
        "raw_eV": round(raw_energy, 3),
        "was_corrected": was_corrected,
        "correction_reason": correction_reason,
        "stability_class": stability_class,
        "confidence": confidence,
    }


def compute_overall_confidence(
    bandgap_result: dict,
    energy_result: dict,
    validity: str = "partially_valid",
    prototype_match: bool = False,
) -> dict[str, Any]:
    """
    Compute overall prediction confidence and assign honest label.

    Labels:
        "Physically Plausible": Valid + prototype + stable + no big corrections
        "AI Candidate":         Passes basic checks
        "Needs Verification":   Something is off — corrections were large
    """
    bg_conf = bandgap_result.get("confidence", "low")
    e_conf = energy_result.get("confidence", "low")
    e_stable = energy_result.get("stability_class", "unstable")

    conf_scores = {"high": 3, "moderate": 2, "low": 1, "speculative": 0}
    avg_conf = (conf_scores.get(bg_conf, 1) + conf_scores.get(e_conf, 1)) / 2

    # Determine label
    if (validity == "valid" and prototype_match
            and e_stable in ("stable", "highly_stable")
            and avg_conf >= 2.0):
        label = "Physically Plausible"
        confidence_level = "high" if avg_conf >= 2.5 else "moderate"
    elif (validity in ("valid", "partially_valid")
          and e_stable in ("stable", "highly_stable", "marginal")
          and avg_conf >= 1.5):
        label = "AI Candidate"
        confidence_level = "moderate"
    elif avg_conf >= 1.0:
        label = "AI Candidate"
        confidence_level = "low"
    else:
        label = "Needs Verification"
        confidence_level = "speculative"

    return {
        "label": label,
        "confidence_level": confidence_level,
        "avg_confidence_score": round(avg_conf, 1),
    }


# -----------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        ("CaF2",  0.5),   # Fluoride — raw says 0.5, should be ≥ 5.0
        ("MgO",   1.2),   # Oxide — raw says 1.2, should be ≥ 3.0
        ("Fe2O3", 5.0),   # TM oxide — raw says 5.0, should be ≤ 3.5
        ("TiN",   2.0),   # TM nitride — raw says 2.0, should be ≈ 0
        ("NaCl",  0.8),   # Halide — raw says 0.8, should be ≥ 5.0
        ("BaTiO3", 0.6),  # Mixed oxide — raw says 0.6, should be ≥ 1.5
        ("ZnS",   0.3),   # Reference: 3.5-3.8 eV
        ("TiO2",  1.0),   # Reference: 3.0-3.4 eV
    ]

    print("Band Gap Physics Corrections")
    print("=" * 70)
    for formula, raw_bg in tests:
        r = correct_bandgap(raw_bg, formula)
        arrow = "→" if r["was_corrected"] else "="
        print(f"  {formula:10s}  raw={raw_bg:.1f}  {arrow}  "
              f"corrected={r['corrected_eV']:.1f} eV  "
              f"[{r['compound_class']}]  conf={r['confidence']}")
        if r["correction_reason"]:
            print(f"              ! {r['correction_reason']}")
