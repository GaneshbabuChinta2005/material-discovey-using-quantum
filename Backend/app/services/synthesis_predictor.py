"""
Synthesis Predictor — Synthesis Feasibility Estimation.

Predicts whether a candidate compound is likely to be experimentally
synthesizable under realistic temperature-pressure conditions.

Approach:
  Uses real Materials Project data where `is_stable=True` compounds
  serve as a proxy for "experimentally realized" materials. Trains
  a GradientBoosting classifier on the 35-feature vector augmented
  with thermodynamic predictions (formation energy, Ehull, band gap).

Output:
  - synthesis_confidence (0.0–1.0)
  - synthesis_label ("Likely", "Possible", "Unlikely", "Improbable")
  - synthesis_factors: which features drove the prediction

Training is performed lazily on first call if no saved model exists.
"""

from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "model")
SYNTH_MODEL_PATH = os.path.join(MODEL_DIR, "synthesis_model.pkl")
SYNTH_SCALER_PATH = os.path.join(MODEL_DIR, "synthesis_scaler.pkl")

_model = None
_scaler = None
_loaded = False

# Augmented feature names (35 base + 3 thermo)
_THERMO_FEATURES = ["pred_formation_energy", "pred_ehull", "pred_bandgap"]


def _load_or_train():
    """Load saved synthesis model, or train one if it doesn't exist."""
    global _model, _scaler, _loaded
    if _loaded:
        return

    if os.path.exists(SYNTH_MODEL_PATH):
        _model = joblib.load(SYNTH_MODEL_PATH)
        if os.path.exists(SYNTH_SCALER_PATH):
            _scaler = joblib.load(SYNTH_SCALER_PATH)
        _loaded = True
        print("[SynthesisPredictor] Loaded existing model.")
        return

    # Train a new model from the MP dataset
    _train_synthesis_model()
    _loaded = True


def _train_synthesis_model():
    """
    Train a synthesis feasibility classifier using MP data.
    Target: is_stable flag as proxy for experimental synthesizability.
    """
    global _model, _scaler

    import sys
    sys.path.insert(0, BASE_DIR)

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    from app.utils.feature_engine import compound_to_features

    print("[SynthesisPredictor] Training synthesis model from MP data...")

    # Load dataset
    dataset_path = None
    for p in [
        os.path.join(BASE_DIR, "Dataset", "real_materials.csv"),
        os.path.join(BASE_DIR, "Dataset", "mp_dataset.csv"),
    ]:
        if os.path.exists(p):
            dataset_path = p
            break

    if dataset_path is None:
        print("[SynthesisPredictor] No dataset found — using rule-based fallback.")
        return

    df = pd.read_csv(dataset_path)

    required_cols = {"formula", "is_stable"}
    if not required_cols.issubset(set(df.columns)):
        print("[SynthesisPredictor] Dataset missing required columns.")
        return

    # Optional thermo columns
    has_fe = "formation_energy_eV_per_atom" in df.columns
    has_ehull = "e_above_hull" in df.columns
    has_bg = "band_gap_eV" in df.columns

    # Build features
    features, labels = [], []
    for _, row in df.iterrows():
        try:
            base_feats = compound_to_features(row["formula"])
            if base_feats is None:
                continue

            # Augment with thermo values
            thermo = [
                float(row["formation_energy_eV_per_atom"]) if has_fe else 0.0,
                float(row["e_above_hull"]) if has_ehull and not pd.isna(row.get("e_above_hull")) else 0.0,
                float(row["band_gap_eV"]) if has_bg else 0.0,
            ]

            features.append(list(base_feats) + thermo)
            labels.append(int(row["is_stable"]))
        except Exception:
            continue

    if len(features) < 100:
        print(f"[SynthesisPredictor] Too few samples ({len(features)}).")
        return

    X = np.array(features)
    y = np.array(labels)

    print(f"[SynthesisPredictor] Dataset: {len(X)} samples, "
          f"{sum(y)} stable ({100*sum(y)/len(y):.1f}%)")

    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # Normalize
    _scaler = StandardScaler()
    X_tr_s = _scaler.fit_transform(X_tr)
    X_te_s = _scaler.transform(X_te)

    # Train
    _model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        min_samples_split=5,
        min_samples_leaf=3,
        subsample=0.85,
        max_features="sqrt",
        random_state=42,
    )
    _model.fit(X_tr_s, y_tr)

    # Evaluate
    y_pred = _model.predict(X_te_s)
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    print(f"[SynthesisPredictor] Test Accuracy: {acc:.4f}  F1: {f1:.4f}")

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(_model, SYNTH_MODEL_PATH)
    joblib.dump(_scaler, SYNTH_SCALER_PATH)
    print(f"[SynthesisPredictor] Model saved to {SYNTH_MODEL_PATH}")


def predict_synthesis_feasibility(
    features: list[float],
    formation_energy: float | None = None,
    ehull: float | None = None,
    bandgap: float | None = None,
) -> dict:
    """
    Predict whether a compound is likely synthesizable.

    Args:
        features: 35-feature vector from feature_engine
        formation_energy: predicted formation energy (eV/atom)
        ehull: predicted Ehull (eV/atom)
        bandgap: predicted band gap (eV)

    Returns:
        {
          "synthesis_confidence": float 0.0–1.0,
          "synthesis_label": str,
          "synthesis_factors": dict,
        }
    """
    _load_or_train()

    # Augment features
    thermo = [
        formation_energy if formation_energy is not None else 0.0,
        ehull if ehull is not None else 0.0,
        bandgap if bandgap is not None else 0.0,
    ]
    X = np.array([list(features) + thermo])

    if _model is not None and _scaler is not None:
        try:
            X_s = _scaler.transform(X)
            proba = _model.predict_proba(X_s)[0]
            # proba[1] = probability of is_stable=True (synthesizable)
            confidence = float(proba[1]) if len(proba) > 1 else float(proba[0])
        except Exception as e:
            print(f"[SynthesisPredictor] Prediction error: {e}")
            confidence = _rule_based_estimate(formation_energy, ehull, bandgap)
    else:
        confidence = _rule_based_estimate(formation_energy, ehull, bandgap)

    # Apply Ehull penalty — high Ehull makes synthesis very unlikely
    if ehull is not None and ehull > 0.5:
        confidence *= max(0.1, 1.0 - (ehull - 0.5) * 0.8)
    elif ehull is not None and ehull > 0.2:
        confidence *= max(0.3, 1.0 - (ehull - 0.2) * 0.5)

    confidence = round(max(0.0, min(1.0, confidence)), 3)

    # Label
    if confidence >= 0.7:
        label = "Likely Synthesizable"
    elif confidence >= 0.45:
        label = "Possibly Synthesizable"
    elif confidence >= 0.2:
        label = "Synthesis Uncertain"
    else:
        label = "Synthesis Improbable"

    # Key factors
    factors = {}
    if formation_energy is not None:
        factors["formation_energy_favorable"] = formation_energy < -0.5
    if ehull is not None:
        factors["thermodynamically_accessible"] = ehull < 0.2
    if bandgap is not None:
        factors["electronic_structure_valid"] = bandgap >= 0.0

    return {
        "synthesis_confidence": confidence,
        "synthesis_label": label,
        "synthesis_factors": factors,
    }


def _rule_based_estimate(
    formation_energy: float | None,
    ehull: float | None,
    bandgap: float | None,
) -> float:
    """
    Rule-based fallback when no trained classifier is available.
    """
    score = 0.5  # neutral baseline

    if formation_energy is not None:
        if formation_energy < -2.0:
            score += 0.25
        elif formation_energy < -1.0:
            score += 0.15
        elif formation_energy < -0.3:
            score += 0.05
        elif formation_energy > 0:
            score -= 0.3

    if ehull is not None:
        if ehull < 0.05:
            score += 0.2
        elif ehull < 0.2:
            score += 0.1
        elif ehull > 0.5:
            score -= 0.25
        elif ehull > 1.0:
            score -= 0.4

    return max(0.0, min(1.0, score))


# -----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, BASE_DIR)
    from app.utils.feature_engine import compound_to_features

    tests = [
        ("BaTiO3", -3.24, 0.01, 3.2),    # Stable perovskite
        ("NaCl", -3.29, 0.0, 8.5),        # Very stable
        ("CrMoN3", -1.50, 0.45, 0.0),     # Uncertain
        ("ZrVSe4", -0.80, 1.20, 0.5),     # Very unlikely
    ]

    print("Synthesis Predictor Test")
    print("=" * 70)
    for formula, ef, ehull, bg in tests:
        feats = compound_to_features(formula)
        if feats:
            r = predict_synthesis_feasibility(feats, ef, ehull, bg)
            print(f"  {formula:12s}  conf={r['synthesis_confidence']:.3f}  "
                  f"{r['synthesis_label']}")
