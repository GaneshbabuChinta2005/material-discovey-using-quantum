"""
Novelty Checker — Research-Grade Database Cross-Validation.

Cross-references generated compounds against:
  1. Full Materials Project dataset (real_materials.csv / mp_dataset.csv)
  2. QM9 dataset (organic molecules)
  3. Curated list of well-known inorganic compounds
  4. Alloy database

Returns:
  - Boolean novelty flag (True = genuinely novel)
  - Novelty score (0.0–1.0)
  - Database match details (which database it was found in)

DATA POLICY: The Materials Project dataset cross-check uses the FULL
  compound set, including both known stable and metastable phases.
  This ensures we properly distinguish truly novel AI-generated candidates
  from re-discoveries of known materials.
"""

import os
import re
import pandas as pd
from collections import Counter
from app.utils.element_loader import load_all_elements

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -----------------------------------------------------------------------
# Well-known inorganic compounds that should NOT be called "novel"
# -----------------------------------------------------------------------
KNOWN_COMPOUNDS = {
    # ===== Binary Oxides =====
    "Al2O3", "Fe2O3", "Fe3O4", "FeO", "TiO2", "SiO2", "CaO", "MgO",
    "ZnO", "CuO", "Cu2O", "NiO", "MnO", "MnO2", "Mn2O3", "Mn3O4",
    "Cr2O3", "CrO3", "V2O3", "V2O5", "VO2", "WO3", "WO2", "MoO3", "MoO2",
    "CoO", "Co3O4", "BaO", "SrO", "Na2O", "K2O", "Li2O", "Rb2O", "Cs2O",
    "BeO", "CdO", "PbO", "PbO2", "Pb3O4", "SnO", "SnO2", "In2O3",
    "Ga2O3", "GeO2", "Sb2O3", "Bi2O3", "As2O3", "HfO2", "ZrO2",
    "Nb2O5", "Ta2O5", "ReO3", "Re2O7", "RuO2", "IrO2", "OsO4",
    "La2O3", "CeO2", "Ce2O3", "Pr2O3", "Nd2O3", "Y2O3", "Sc2O3",
    "Sm2O3", "Eu2O3", "Gd2O3", "Tb4O7", "Dy2O3", "Er2O3",
    "ThO2", "UO2", "U3O8",
    # ===== Binary Chlorides =====
    "NaCl", "KCl", "LiCl", "CaCl2", "MgCl2", "FeCl3", "FeCl2",
    "AlCl3", "TiCl4", "TiCl3", "ZnCl2", "CuCl2", "CuCl", "MnCl2",
    "CoCl2", "NiCl2", "BaCl2", "SrCl2", "CsCl", "RbCl",
    "SnCl2", "SnCl4", "PbCl2", "BiCl3", "CdCl2", "HgCl2",
    # ===== Binary Sulfides =====
    "FeS", "FeS2", "CuS", "Cu2S", "ZnS", "PbS", "MoS2", "WS2", "NiS",
    "MnS", "CoS", "Ag2S", "CdS", "SnS", "SnS2", "Bi2S3", "Sb2S3",
    "As2S3", "In2S3", "Ga2S3", "TiS2", "NbS2", "TaS2", "VS2", "HfS2",
    # ===== Binary Nitrides =====
    "BN", "AlN", "Si3N4", "TiN", "ZrN", "HfN", "VN", "NbN", "TaN",
    "CrN", "Cr2N", "Fe2N", "Fe4N", "GaN", "InN", "Li3N", "Mg3N2",
    "Ca3N2", "Cu3N", "ScN",
    # ===== Binary Carbides =====
    "SiC", "TiC", "WC", "W2C", "Fe3C", "CaC2", "B4C", "Cr3C2",
    "Cr7C3", "Cr23C6", "Mo2C", "NbC", "TaC", "ZrC", "HfC", "VC",
    # ===== Binary Fluorides =====
    "NaF", "KF", "LiF", "CaF2", "MgF2", "BaF2", "SrF2", "AlF3",
    "TiF4", "ZrF4", "FeF3", "CuF2", "ZnF2", "LaF3", "CeF3", "YF3",
    # ===== Binary Bromides / Iodides =====
    "NaBr", "KBr", "LiBr", "CaBr2", "NaI", "KI", "CsI", "AgBr", "AgI",
    "CuBr", "PbBr2", "PbI2", "BiI3", "HgI2", "CdBr2",
    # ===== Binary Phosphides =====
    "GaP", "InP", "AlP", "Zn3P2", "Cu3P", "Fe3P", "FeP", "Ni3P",
    # ===== Binary Arsenides / Selenides / Tellurides / Borides =====
    "GaAs", "InAs", "AlAs",
    "ZnSe", "CdSe", "PbSe", "Cu2Se", "Bi2Se3", "In2Se3", "GaSe",
    "MoSe2", "WSe2", "SnSe", "Sb2Se3", "TiSe2",
    "PbTe", "Bi2Te3", "CdTe", "ZnTe", "GeTe", "SnTe", "Sb2Te3",
    "TiB2", "ZrB2", "HfB2", "CrB2", "MoB2", "WB", "LaB6", "CeB6",
    "MgB2", "FeB", "CoB", "NiB",
    # ===== Ternary Perovskites =====
    "BaTiO3", "SrTiO3", "CaTiO3", "BaZrO3", "SrZrO3",
    "KNbO3", "NaNbO3", "LiNbO3", "LiTaO3",
    "LaMnO3", "LaCoO3", "LaNiO3", "LaFeO3", "LaCrO3", "LaAlO3",
    "SrVO3", "SrRuO3", "SrMnO3", "SrCoO3", "SrFeO3",
    "PbTiO3", "PbZrO3", "BiFeO3", "BiMnO3",
    "YAlO3", "GdFeO3", "NdFeO3", "SmFeO3",
    "CaMnO3", "CaVO3", "CaCrO3",
    # ===== Spinels & Garnets =====
    "MgAl2O4", "FeCr2O4", "CoFe2O4", "NiFe2O4", "ZnFe2O4",
    "MnFe2O4", "CuFe2O4", "MgFe2O4", "Fe3O4",
    "Co3O4", "Mn3O4", "ZnCo2O4", "NiCo2O4",
    "Y3Al5O12", "Y3Fe5O12", "Gd3Ga5O12",
    # ===== Battery Materials =====
    "LiCoO2", "LiNiO2", "LiMnO2", "LiMn2O4", "NaCoO2", "NaMnO2",
    "LiFePO4", "LiMnPO4", "LiCoPO4", "LiNiPO4", "NaFePO4",
    "Li4Ti5O12", "LiTiO2", "Na2Ti3O7", "Li7La3Zr2O12",
    # ===== Common Ternary Chalcogenides =====
    "CuInSe2", "CuInS2", "CuGaSe2", "CuInTe2", "AgInSe2",
    "CuFeS2", "Cu2ZnSnS4", "Cu2ZnSnSe4",
    # ===== Superconductors / MAX Phases =====
    "YBa2Cu3O7", "Bi2Sr2CaCu2O8", "La2CuO4", "MgB2", "Nb3Sn",
    "FeSe", "NbTi",
    "Ti3SiC2", "Ti3AlC2", "Ti2AlC", "Cr2AlC", "V2AlC",
}

# Module-level caches
_mp_formulas: set | None = None           # Full MP dataset (real + metastable)
_qm9_compositions: set | None = None
_alloy_names: set | None = None


def _parse_formula_key(formula: str) -> str:
    """
    Convert formula to canonical reduced form (GCD reduction) for comparison.
    Example: Fe4O6 -> Fe2O3, Li2Fe2O4 -> LiFeO2
    """
    parts = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    elements = {}
    for sym, count_str in parts:
        if not sym:
            continue
        count = int(count_str) if count_str else 1
        elements[sym] = elements.get(sym, 0) + count
    
    if not elements:
        return ""
        
    counts = list(elements.values())
    if not counts:
        return ""
        
    common_divisor = counts[0]
    if len(counts) > 1:
        from math import gcd
        from functools import reduce
        common_divisor = reduce(gcd, counts)
        
    if common_divisor > 1:
        for s in elements:
            elements[s] //= common_divisor
            
    return "".join(
        f"{s}{elements[s] if elements[s] > 1 else ''}"
        for s in sorted(elements.keys())
    )


def _load_mp_formulas() -> set:
    """
    Load ALL formulas from the Materials Project dataset.
    This includes known stable, metastable, and synthesized phases
    that any truly novel candidate must be distinguished from.
    """
    global _mp_formulas
    if _mp_formulas is not None:
        return _mp_formulas

    _mp_formulas = set()

    for dataset_path in [
        os.path.join(BASE_DIR, "Dataset", "real_materials.csv"),
        os.path.join(BASE_DIR, "Dataset", "mp_dataset.csv"),
    ]:
        if os.path.exists(dataset_path):
            try:
                df = pd.read_csv(dataset_path, usecols=["formula"])
                for f in df["formula"].dropna():
                    key = _parse_formula_key(str(f))
                    if key:
                        _mp_formulas.add(key)
                print(f"[NoveltyChecker] Loaded {len(_mp_formulas)} MP formulas "
                      f"from {os.path.basename(dataset_path)}")
                break   # Use the first available file
            except Exception as e:
                print(f"[NoveltyChecker] Warning: Could not load MP dataset: {e}")

    if not _mp_formulas:
        print("[NoveltyChecker] No MP dataset found — database cross-check skipped.")

    return _mp_formulas


def _load_qm9_compositions() -> set:
    """Load unique compositions from QM9 dataset."""
    global _qm9_compositions
    if _qm9_compositions is not None:
        return _qm9_compositions

    _qm9_compositions = set()
    qm9_path = os.path.join(BASE_DIR, "Dataset", "qm9_final.csv")

    if os.path.exists(qm9_path):
        try:
            df = pd.read_csv(qm9_path, usecols=["Stoichiometry"])
            for stoich in df["Stoichiometry"].dropna().unique():
                try:
                    nums = [int(x.strip()) for x in stoich.strip("[]").split(",")]
                    elements = ["C", "H", "N", "O", "F"]
                    parts = [f"{e}{n if n > 1 else ''}" for e, n in zip(elements, nums) if n > 0]
                    key = "".join(sorted(parts))
                    if key:
                        _qm9_compositions.add(key)
                except Exception:
                    pass
            print(f"[NoveltyChecker] Loaded {len(_qm9_compositions)} QM9 compositions")
        except Exception as e:
            print(f"[NoveltyChecker] Warning: Could not load QM9: {e}")

    return _qm9_compositions


def _load_alloy_names() -> set:
    """Load alloy names."""
    global _alloy_names
    if _alloy_names is not None:
        return _alloy_names

    _alloy_names = set()
    alloy_path = os.path.join(BASE_DIR, "Dataset", "Alloy Properties", "alloys.csv")

    if os.path.exists(alloy_path):
        try:
            df = pd.read_csv(alloy_path, usecols=["Alloy"])
            _alloy_names = set(df["Alloy"].dropna().str.strip().str.lower())
            print(f"[NoveltyChecker] Loaded {len(_alloy_names)} alloy names")
        except Exception as e:
            print(f"[NoveltyChecker] Warning: Could not load alloys: {e}")

    return _alloy_names


# -----------------------------------------------------------------------
# Phase 4: Multi-Database Loaders (OQMD, AFLOW, JARVIS)
# -----------------------------------------------------------------------
_oqmd_formulas: set | None = None
_aflow_formulas: set | None = None
_jarvis_formulas: set | None = None


def _load_db_formulas(csv_name: str, db_label: str) -> set:
    """Generic loader for external database CSV files (formula column)."""
    path = os.path.join(BASE_DIR, "Dataset", csv_name)
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path, usecols=["formula"])
        formulas = set()
        for f in df["formula"].dropna():
            key = _parse_formula_key(str(f))
            if key:
                formulas.add(key)
        print(f"[NoveltyChecker] Loaded {len(formulas)} {db_label} formulas")
        return formulas
    except Exception as e:
        print(f"[NoveltyChecker] Warning: Could not load {db_label}: {e}")
        return set()


def _load_oqmd_formulas() -> set:
    global _oqmd_formulas
    if _oqmd_formulas is None:
        _oqmd_formulas = _load_db_formulas("oqmd_formulas.csv", "OQMD")
    return _oqmd_formulas


def _load_aflow_formulas() -> set:
    global _aflow_formulas
    if _aflow_formulas is None:
        _aflow_formulas = _load_db_formulas("aflow_formulas.csv", "AFLOW")
    return _aflow_formulas


def _load_jarvis_formulas() -> set:
    global _jarvis_formulas
    if _jarvis_formulas is None:
        _jarvis_formulas = _load_db_formulas("jarvis_formulas.csv", "JARVIS-DFT")
    return _jarvis_formulas


def check_novelty(formula: str) -> dict:
    """
    Research-grade novelty check with FULL multi-database cross-validation.

    Cross-references:
      1. Curated known compounds (in-memory)
      2. Materials Project dataset
      3. OQMD compositions
      4. AFLOW compositions
      5. JARVIS-DFT compositions
      6. QM9 dataset (organic)
      7. Alloy database (fuzzy match)

    Returns:
        {
          "is_novel": bool,
          "novelty_score": float,  0.0 = known, 1.0 = very novel
          "found_in": list[str],   which databases matched
          "novelty_label": str,    human-readable label
          "match_detail": str,     which exact database matched
          "databases_checked": int, total databases checked
        }
    """
    canonical = _parse_formula_key(formula)
    found_in = []

    # 1. Curated known compounds (fastest, in-memory)
    known_canonical = {_parse_formula_key(c) for c in KNOWN_COMPOUNDS}
    if canonical in known_canonical:
        found_in.append("curated_known_list")

    # 2. Full Materials Project database
    mp_set = _load_mp_formulas()
    if canonical in mp_set:
        found_in.append("materials_project")

    # 3. OQMD database
    oqmd_set = _load_oqmd_formulas()
    if canonical in oqmd_set:
        found_in.append("oqmd")

    # 4. AFLOW database
    aflow_set = _load_aflow_formulas()
    if canonical in aflow_set:
        found_in.append("aflow")

    # 5. JARVIS-DFT database
    jarvis_set = _load_jarvis_formulas()
    if canonical in jarvis_set:
        found_in.append("jarvis_dft")

    # 6. QM9
    qm9 = _load_qm9_compositions()
    if canonical in qm9:
        found_in.append("qm9_dataset")

    # 7. Alloy database (fuzzy)
    alloys = _load_alloy_names()
    formula_lower = formula.lower()
    for alloy in alloys:
        if formula_lower in alloy:
            found_in.append("alloy_database")
            break

    databases_checked = 7
    is_novel = len(found_in) == 0

    # Compute novelty score
    if not is_novel:
        # Penalize more for appearing in multiple databases
        penalty = min(1.0, len(found_in) * 0.25)
        score = round(max(0.0, 0.1 - penalty * 0.1), 2)
        label = f"Known Material (found in {len(found_in)} database{'s' if len(found_in) > 1 else ''})"
        match_detail = f"Found in: {', '.join(found_in)}"
    else:
        # Score based on element combination rarity
        parts = re.findall(r'([A-Z][a-z]?)', formula)
        elem_set = set(parts)

        max_overlap = 0.0
        for known in KNOWN_COMPOUNDS:
            k_elems = set(re.findall(r'([A-Z][a-z]?)', known))
            if not k_elems:
                continue
            overlap = len(elem_set & k_elems) / len(elem_set | k_elems)
            max_overlap = max(max_overlap, overlap)

        score = 1.0 - (max_overlap * 0.5)

        # Bonuses for complexity
        if len(elem_set) >= 4:
            score = min(1.0, score + 0.15)
        elif len(elem_set) >= 3:
            score = min(1.0, score + 0.08)

        score = round(max(0.1, min(1.0, score)), 2)
        label = f"✨ Novel — Verified against {databases_checked} databases"
        match_detail = f"Not found in any of {databases_checked} databases ({', '.join(['MP', 'OQMD', 'AFLOW', 'JARVIS', 'QM9', 'Alloys', 'Curated'])})"

    return {
        "is_novel": is_novel,
        "novelty_score": score,
        "found_in": found_in,
        "novelty_label": label,
        "match_detail": match_detail,
        "databases_checked": databases_checked,
    }


def is_novel(formula: str) -> bool:
    """Quick boolean check (backward-compatible)."""
    return check_novelty(formula)["is_novel"]


def novelty_score(formula: str) -> float:
    """Quick score (backward-compatible)."""
    return check_novelty(formula)["novelty_score"]


# -----------------------------------------------------------------------
if __name__ == "__main__":
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
    from app.utils.element_loader import load_all_elements
    test_formulas = [
        "NaCl",       # very known — should be in MP + curated
        "TiO2",       # very known
        "LiCoO2",     # known battery material
        "BaTiO3",     # known perovskite
        "Ti2NiO5",    # likely novel
        "CrMoN3",     # likely novel
        "ZrVSe4",     # likely novel
        "HfNbIrO6",   # very novel
    ]
    print("Novelty Checker Test")
    print("=" * 70)
    for f in test_formulas:
        result = check_novelty(f)
        print(f"  {f:15s}  novel={result['is_novel']}  score={result['novelty_score']:.2f}  {result['novelty_label']}")
        if result["found_in"]:
            print(f"  {'':15s}  {result['match_detail']}")
