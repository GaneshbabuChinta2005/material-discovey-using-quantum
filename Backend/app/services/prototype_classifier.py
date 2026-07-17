"""
Prototype Classifier — ML-Based Crystal Structure Family Prediction.

Predicts the most likely crystal prototype family for a given composition
using physics-based descriptors and a trained ensemble classifier.

Supported Prototypes:
  - Perovskite (ABO₃)       — tolerance factor t ∈ [0.75, 1.10]
  - Spinel (AB₂X₄)          — tetrahedral A, octahedral B
  - Rock Salt (AX)           — NaCl-type, CN=6
  - Fluorite (AX₂)          — CaF₂-type, CN=8
  - Corundum (A₂X₃)         — α-Al₂O₃ type
  - Layered (Ruddlesden–Popper) — A_{n+1}B_nO_{3n+1}
  - Heusler (X₂YZ)          — full/half Heusler intermetallics
  - Other                    — unclassified

Features:
  1.  Goldschmidt tolerance factor (t)
  2.  Octahedral factor (μ)
  3.  Electronegativity spread (ΔEN)
  4.  Average ionic radius
  5.  Radius ratio (r_A / r_B)
  6.  Oxidation state balance
  7.  Number of elements
  8.  Stoichiometry signature

Output:
  - Predicted prototype label
  - Confidence score (0–1)
  - Top 3 prototype probabilities
"""

from __future__ import annotations

import os
import re
import math
import pickle
import numpy as np
from typing import Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "model", "prototype_classifier.pkl")

# -----------------------------------------------------------------------
# Element property tables
# -----------------------------------------------------------------------
ELECTRONEG = {
    "H": 2.20, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04,
    "O": 3.44, "F": 3.98, "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90,
    "P": 2.19, "S": 2.58, "Cl": 3.16, "K": 0.82, "Ca": 1.00, "Sc": 1.36,
    "Ti": 1.54, "V": 1.63, "Cr": 1.66, "Mn": 1.55, "Fe": 1.83, "Co": 1.88,
    "Ni": 1.91, "Cu": 1.90, "Zn": 1.65, "Ga": 1.81, "Ge": 2.01, "As": 2.18,
    "Se": 2.55, "Br": 2.96, "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33,
    "Nb": 1.60, "Mo": 2.16, "Ru": 2.20, "Rh": 2.28, "Pd": 2.20, "Ag": 1.93,
    "Cd": 1.69, "In": 1.78, "Sn": 1.96, "Sb": 2.05, "Te": 2.10, "I": 2.66,
    "Cs": 0.79, "Ba": 0.89, "La": 1.10, "Ce": 1.12, "Pr": 1.13, "Nd": 1.14,
    "Sm": 1.17, "Eu": 1.20, "Gd": 1.20, "Hf": 1.30, "Ta": 1.50, "W": 2.36,
    "Re": 1.90, "Os": 2.20, "Ir": 2.20, "Pt": 2.28, "Au": 2.54, "Pb": 2.33,
    "Bi": 2.02, "Tl": 1.62,
}

IONIC_RADII = {
    "Li": 0.76, "Na": 1.02, "K": 1.38, "Rb": 1.52, "Cs": 1.67,
    "Mg": 0.72, "Ca": 1.00, "Sr": 1.18, "Ba": 1.35,
    "Sc": 0.745, "Ti": 0.605, "V": 0.54, "Cr": 0.615, "Mn": 0.645,
    "Fe": 0.645, "Co": 0.545, "Ni": 0.69, "Cu": 0.73, "Zn": 0.74,
    "Y": 0.90, "Zr": 0.72, "Nb": 0.64, "Mo": 0.59, "Ru": 0.62,
    "Rh": 0.665, "Pd": 0.86, "Ag": 1.15, "Cd": 0.95,
    "Hf": 0.71, "Ta": 0.64, "W": 0.60, "Ir": 0.625, "Pt": 0.625,
    "La": 1.032, "Ce": 1.01, "Pr": 0.99, "Nd": 0.983, "Gd": 0.938,
    "Al": 0.535, "Ga": 0.62, "In": 0.80, "Sn": 0.69, "Pb": 1.19,
    "Bi": 1.03, "Sb": 0.76, "As": 0.58, "Ge": 0.53, "Si": 0.40,
    "O": 1.40, "S": 1.84, "Se": 1.98, "Te": 2.21,
    "F": 1.33, "Cl": 1.81, "Br": 1.96, "I": 2.20,
    "N": 1.46, "P": 2.12, "C": 2.60, "B": 0.27, "H": 0.00,
}

# Prototype labels
PROTOTYPES = [
    "perovskite", "spinel", "rock_salt", "fluorite", "corundum",
    "layered_rp", "heusler", "other"
]


def _parse_formula(formula: str) -> list[tuple[str, int]]:
    """Parse formula → [(element, count), ...]."""
    return [(s, int(c) if c else 1)
            for s, c in re.findall(r'([A-Z][a-z]?)(\d*)', formula) if s]


def _compute_features(formula: str) -> np.ndarray:
    """
    Extract 12 physics-based descriptors for prototype classification.
    """
    parsed = _parse_formula(formula)
    symbols = [s for s, _ in parsed]
    counts = [c for _, c in parsed]
    n_elem = len(symbols)
    total_atoms = sum(counts)

    # Electronegativity features
    ens = [ELECTRONEG.get(s, 1.5) for s in symbols]
    en_mean = np.mean(ens)
    en_spread = max(ens) - min(ens) if len(ens) > 1 else 0.0
    en_std = np.std(ens) if len(ens) > 1 else 0.0

    # Ionic radii features
    radii = [IONIC_RADII.get(s, 1.0) for s in symbols]
    r_mean = np.mean(radii)
    r_max = max(radii)
    r_min = min(radii)
    r_ratio = r_max / r_min if r_min > 0.01 else 10.0

    # Stoichiometry features
    sorted_counts = sorted(counts)
    stoich_signature = sum(c * (i + 1) for i, c in enumerate(sorted_counts))

    # Goldschmidt tolerance factor (for ternary ABX3-like)
    t_factor = 0.0
    mu_factor = 0.0
    if n_elem == 3 and sorted_counts == [1, 1, 3]:
        # ABO3 pattern
        cation_radii = sorted([IONIC_RADII.get(s, 1.0) for s, c in parsed if c <= 1], reverse=True)
        anion_r = IONIC_RADII.get([s for s, c in parsed if c == 3][0], 1.40) if any(c == 3 for _, c in parsed) else 1.40
        if len(cation_radii) >= 2:
            r_A, r_B = cation_radii[0], cation_radii[1]
            t_factor = (r_A + anion_r) / (math.sqrt(2) * (r_B + anion_r))
            mu_factor = r_B / anion_r if anion_r > 0 else 0
    elif n_elem == 3 and 4 in counts:
        # AB2X4 spinel-like
        r_A = min(radii[:2]) if len(radii) >= 2 else r_mean
        r_B = max(radii[:2]) if len(radii) >= 2 else r_mean
        mu_factor = r_A / r_B if r_B > 0 else 0

    # Metal/non-metal balance
    non_metals = {"O", "S", "Se", "Te", "N", "F", "Cl", "Br", "I", "P", "C", "H", "B"}
    n_nonmetal = sum(1 for s in symbols if s in non_metals)
    metal_frac = 1.0 - (n_nonmetal / n_elem) if n_elem > 0 else 0.5

    return np.array([
        n_elem,           # 0: number of elements
        total_atoms,      # 1: total atoms in formula unit
        en_mean,          # 2: mean electronegativity
        en_spread,        # 3: electronegativity spread
        en_std,           # 4: electronegativity std
        r_mean,           # 5: mean ionic radius
        r_ratio,          # 6: max/min radius ratio
        t_factor,         # 7: Goldschmidt tolerance factor
        mu_factor,        # 8: octahedral factor
        stoich_signature, # 9: stoichiometry signature
        metal_frac,       # 10: fraction of metallic elements
        len(sorted_counts), # 11: number of distinct counts
    ], dtype=np.float64)


# -----------------------------------------------------------------------
# Rule-based fallback classifier
# -----------------------------------------------------------------------
def _rule_based_classify(formula: str, features: np.ndarray) -> tuple[str, float, dict]:
    """
    Physics-based rule engine for prototype classification.
    Returns (label, confidence, probabilities_dict).
    """
    parsed = _parse_formula(formula)
    symbols = [s for s, _ in parsed]
    counts = [c for _, c in parsed]
    n_elem = len(symbols)
    sorted_counts = sorted(counts)

    probs = {p: 0.0 for p in PROTOTYPES}

    t = features[7]
    mu = features[8]

    # Perovskite: ABO3 (ternary, 1:1:3)
    if n_elem == 3 and sorted_counts == [1, 1, 3]:
        if t > 0.0:
            if 0.75 <= t <= 1.10:
                probs["perovskite"] = 0.85
                if 0.414 <= mu <= 0.732:
                    probs["perovskite"] = 0.95
            else:
                probs["perovskite"] = 0.3
                probs["other"] = 0.4
        else:
            probs["perovskite"] = 0.5
            probs["other"] = 0.3

    # Spinel: AB2X4 (ternary, 1:2:4)
    elif n_elem == 3 and sorted_counts == [1, 2, 4]:
        probs["spinel"] = 0.80
        if 0.35 <= mu <= 0.75:
            probs["spinel"] = 0.90

    # Rock salt: AX (binary, 1:1)
    elif n_elem == 2 and sorted_counts == [1, 1]:
        en_spread = features[3]
        if en_spread > 1.5:
            probs["rock_salt"] = 0.85
        else:
            probs["rock_salt"] = 0.45
            probs["heusler"] = 0.35  # Could be intermetallic

    # Fluorite: AX2 (binary, 1:2)
    elif n_elem == 2 and sorted_counts == [1, 2]:
        non_metals = {"O", "S", "Se", "Te", "N", "F", "Cl", "Br", "I"}
        has_nonmetal = any(s in non_metals for s in symbols)
        if has_nonmetal:
            probs["fluorite"] = 0.75
        else:
            probs["other"] = 0.5
            probs["heusler"] = 0.3

    # Corundum: A2X3 (binary, 2:3)
    elif n_elem == 2 and sorted_counts == [2, 3]:
        probs["corundum"] = 0.75
        probs["other"] = 0.15

    # Heusler: X2YZ (ternary, 1:1:2) or XYZ (1:1:1)
    elif n_elem == 3 and sorted_counts == [1, 1, 2]:
        metal_frac = features[10]
        if metal_frac > 0.8:
            probs["heusler"] = 0.80
        else:
            probs["layered_rp"] = 0.40
            probs["other"] = 0.35

    # Ruddlesden-Popper: A_{n+1}B_nO_{3n+1}
    elif n_elem == 3 and sorted_counts == [1, 2, 4]:
        # Already handled by spinel above
        pass
    elif n_elem == 3 and sorted_counts == [1, 2, 7]:
        probs["layered_rp"] = 0.70

    else:
        probs["other"] = 0.60
        # Assign small probabilities based on features
        if features[3] > 2.0:  # High EN spread → ionic
            probs["rock_salt"] = 0.15
        if features[10] > 0.8:  # Mostly metallic
            probs["heusler"] = 0.15

    # Normalize probabilities
    total = sum(probs.values())
    if total > 0:
        probs = {k: round(v / total, 3) for k, v in probs.items()}

    best = max(probs, key=probs.get)
    return best, probs[best], probs


# -----------------------------------------------------------------------
# ML-based classifier (loads trained model if available)
# -----------------------------------------------------------------------
_classifier = None
_classifier_loaded = False


def _load_classifier():
    global _classifier, _classifier_loaded
    if _classifier_loaded:
        return _classifier

    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                _classifier = pickle.load(f)
            print(f"[PrototypeClassifier] Loaded ML model from {MODEL_PATH}")
        except Exception as e:
            print(f"[PrototypeClassifier] Could not load model: {e}")
            _classifier = None
    else:
        print("[PrototypeClassifier] No trained model found — using rule-based engine")

    _classifier_loaded = True
    return _classifier


def classify_prototype(formula: str) -> dict:
    """
    Classify the most likely crystal prototype for a given formula.

    Returns:
        {
            "prototype": str,
            "confidence": float (0–1),
            "probabilities": dict[str, float],
            "method": "ml" | "rules",
            "features": dict (physics descriptors used),
        }
    """
    features = _compute_features(formula)
    clf = _load_classifier()

    if clf is not None:
        try:
            X = features.reshape(1, -1)
            pred_idx = clf.predict(X)[0]
            label = PROTOTYPES[int(pred_idx)]

            if hasattr(clf, "predict_proba"):
                prob_arr = clf.predict_proba(X)[0]
                probs = {PROTOTYPES[i]: round(float(p), 3) for i, p in enumerate(prob_arr)}
            else:
                probs = {p: (0.9 if p == label else 0.01) for p in PROTOTYPES}
                total = sum(probs.values())
                probs = {k: round(v / total, 3) for k, v in probs.items()}

            confidence = probs.get(label, 0.5)
            method = "ml"
        except Exception:
            label, confidence, probs = _rule_based_classify(formula, features)
            method = "rules"
    else:
        label, confidence, probs = _rule_based_classify(formula, features)
        method = "rules"

    # Top 3 for display
    top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]

    return {
        "prototype": label,
        "confidence": confidence,
        "probabilities": probs,
        "top3": [{"label": l, "prob": p} for l, p in top3],
        "method": method,
        "features": {
            "n_elements": int(features[0]),
            "total_atoms": int(features[1]),
            "en_spread": round(float(features[3]), 3),
            "r_mean": round(float(features[5]), 3),
            "r_ratio": round(float(features[6]), 3),
            "tolerance_factor": round(float(features[7]), 4),
            "octahedral_factor": round(float(features[8]), 4),
            "metal_fraction": round(float(features[10]), 3),
        },
    }


def train_classifier(save: bool = True) -> Any:
    """
    Train the prototype classifier on labeled data from the Materials Project.
    Uses composition-based features to predict crystal family.
    """
    from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
    from sklearn.model_selection import cross_val_score
    import pandas as pd

    print("[PrototypeClassifier] Training prototype classifier...")

    # Load MP dataset
    dataset_path = os.path.join(BASE_DIR, "Dataset", "real_materials.csv")
    if not os.path.exists(dataset_path):
        print("[PrototypeClassifier] No training data found")
        return None

    df = pd.read_csv(dataset_path)
    if "formula" not in df.columns:
        print("[PrototypeClassifier] Missing 'formula' column")
        return None

    # Generate labels using rules (semi-supervised)
    X_list, y_list = [], []
    for _, row in df.iterrows():
        formula = str(row.get("formula", ""))
        if not formula or formula == "nan":
            continue

        features = _compute_features(formula)
        label, conf, _ = _rule_based_classify(formula, features)

        if conf >= 0.6:  # Only use high-confidence rule labels
            label_idx = PROTOTYPES.index(label) if label in PROTOTYPES else len(PROTOTYPES) - 1
            X_list.append(features)
            y_list.append(label_idx)

    if len(X_list) < 100:
        print(f"[PrototypeClassifier] Insufficient training data ({len(X_list)} samples)")
        return None

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"[PrototypeClassifier] Training on {len(X)} labeled samples")

    clf = BaggingClassifier(
        estimator=GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1
        ),
        n_estimators=5,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X, y)

    # Cross-validation
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"[PrototypeClassifier] CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    if save:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(clf, f)
        print(f"[PrototypeClassifier] Model saved to {MODEL_PATH}")

    return clf


# -----------------------------------------------------------------------
if __name__ == "__main__":
    # Train and test
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    print("=" * 70)
    print("  Prototype Classifier — Training & Testing")
    print("=" * 70)

    # Train
    clf = train_classifier(save=True)

    # Test
    test_formulas = [
        "BaTiO3",     # Perovskite
        "SrTiO3",     # Perovskite
        "MgAl2O4",    # Spinel
        "NaCl",       # Rock salt
        "CaF2",       # Fluorite
        "Al2O3",      # Corundum
        "Co2MnSi",    # Heusler
        "TiNiSn",     # Half-Heusler
        "LiCoO2",     # Layered
    ]
    print("\nTest Results:")
    print("-" * 70)
    for f in test_formulas:
        r = classify_prototype(f)
        print(f"  {f:15s}  →  {r['prototype']:15s}  conf={r['confidence']:.2f}  "
              f"method={r['method']}  t={r['features']['tolerance_factor']:.3f}")
