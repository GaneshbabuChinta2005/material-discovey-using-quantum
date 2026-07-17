"""
Research-Grade Model Training — Dedicated models for thermodynamic prediction.

CRITICAL DATA POLICY:
  - Formation Energy, Ehull, and Band Gap models are trained ONLY on real
    DFT-derived data (real_materials.csv / mp_dataset.csv with source='real').
  - Synthetic data is NEVER used to train thermodynamic models.

Trains THREE models:
  1. Formation Energy (eV/atom)   — primary thermodynamic predictor
  2. Energy Above Hull (eV/atom)  — NEW: direct Ehull regression
  3. Band Gap (eV)                — electronic structure predictor

Uncertainty Estimation:
  - Each model is wrapped in an EnsemblePredictor (BaggingRegressor)
    providing mean + std (epistemic uncertainty) for every prediction.

Splitting:
  - Stratified by energy bin to maintain distribution balance.
  - Normalization scaler fitted ONLY on training split to prevent leakage.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.utils.feature_engine import compound_to_features, FEATURE_NAMES

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
REAL_DATASET_PATH = "Dataset/real_materials.csv"   # REAL data ONLY
MP_DATASET_PATH   = "Dataset/mp_dataset.csv"       # Fallback
MODEL_DIR = "model"

# Output paths
ENERGY_MODEL_PATH        = os.path.join(MODEL_DIR, "energy_model.pkl")
EHULL_MODEL_PATH         = os.path.join(MODEL_DIR, "ehull_model.pkl")
BANDGAP_MODEL_PATH       = os.path.join(MODEL_DIR, "bandgap_model.pkl")
ENERGY_ENSEMBLE_PATH     = os.path.join(MODEL_DIR, "energy_ensemble.pkl")
EHULL_ENSEMBLE_PATH      = os.path.join(MODEL_DIR, "ehull_ensemble.pkl")
BANDGAP_ENSEMBLE_PATH    = os.path.join(MODEL_DIR, "bandgap_ensemble.pkl")
ENERGY_SCALER_PATH       = os.path.join(MODEL_DIR, "energy_scaler.pkl")
FEATURE_NAMES_PATH       = os.path.join(MODEL_DIR, "feature_names.json")
TRAINING_METADATA_PATH   = os.path.join(MODEL_DIR, "training_metadata.json")
LEGACY_MODEL_PATH        = os.path.join(MODEL_DIR, "material_model.pkl")

TEST_SIZE    = 0.15
VAL_SIZE     = 0.10   # of the remaining training set
RANDOM_STATE = 42

N_BINS = 10  # For stratified splitting


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def load_real_data() -> pd.DataFrame:
    """
    Load REAL DFT-derived data for thermodynamic model training.
    Prioritizes real_materials.csv; filters mp_dataset.csv by source='real'.
    NEVER includes synthetic data.
    """
    # Prefer explicit real dataset
    if os.path.exists(REAL_DATASET_PATH):
        df = pd.read_csv(REAL_DATASET_PATH)
        print(f"  Loaded real_materials.csv: {len(df)} compounds")
        df["data_source"] = "real"
        return df

    # Fall back to mp_dataset, filtering by source column
    if os.path.exists(MP_DATASET_PATH):
        df = pd.read_csv(MP_DATASET_PATH)
        if "source" in df.columns:
            real_mask = df["source"].str.lower().str.strip() == "real"
            df_real = df[real_mask].copy()
            df_synth = df[~real_mask]
            print(f"  mp_dataset.csv: {len(df_real)} REAL, {len(df_synth)} synthetic (excluded)")
            df_real["data_source"] = "real"
            return df_real
        else:
            print("  WARNING: No 'source' column found. Treating all data as real.")
            df["data_source"] = "real"
            return df

    raise FileNotFoundError(
        f"No real dataset found. Please place 'real_materials.csv' in {os.path.abspath('Dataset/')}"
    )


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list]:
    """Generate features for all compounds."""
    print(f"  Building feature matrix for {len(df)} compounds...")
    t0 = time.time()
    features, valid_indices, failed = [], [], 0

    for idx, row in df.iterrows():
        try:
            feats = compound_to_features(row["formula"])
            if feats is not None:
                features.append(feats)
                valid_indices.append(idx)
            else:
                failed += 1
        except Exception:
            failed += 1

    elapsed = time.time() - t0
    print(f"  Generated {len(features)} vectors in {elapsed:.1f}s ({failed} failed)")
    return np.array(features), valid_indices


def stratified_split(X, y, test_size=TEST_SIZE, n_bins=N_BINS):
    """
    Stratified train/test split based on binned target values.
    Ensures uniform distribution of target across train and test sets.
    """
    # Create bin labels
    bin_labels = pd.qcut(pd.Series(y), q=n_bins, labels=False, duplicates="drop")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE,
        stratify=bin_labels,
    )
    return X_train, X_test, y_train, y_test


def train_base_model(X_tr, y_tr, target_name: str, **kwargs) -> GradientBoostingRegressor:
    """Train a single GradientBoosting model."""
    params = dict(
        n_estimators=800,
        max_depth=7,
        learning_rate=0.05,
        min_samples_split=4,
        min_samples_leaf=2,
        subsample=0.85,
        max_features="sqrt",
        validation_fraction=0.1,
        n_iter_no_change=30,
        tol=1e-5,
        random_state=RANDOM_STATE,
    )
    params.update(kwargs)
    model = GradientBoostingRegressor(**params)
    model.fit(X_tr, y_tr)
    return model


def train_ensemble(X_tr, y_tr, base_model, n_estimators=10) -> BaggingRegressor:
    """
    Wrap a base GBR in a BaggingRegressor for uncertainty estimation.
    Each bag trains on a different bootstrap sample, enabling std(predictions)
    as an epistemic uncertainty measure.
    """
    ensemble = BaggingRegressor(
        estimator=base_model,
        n_estimators=n_estimators,
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    ensemble.fit(X_tr, y_tr)
    return ensemble


def evaluate_model(model, X_te, y_te, target_name: str) -> dict:
    """Evaluate model performance and print report."""
    y_pred = model.predict(X_te)
    r2   = r2_score(y_te, y_pred)
    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    print(f"\n  [ {target_name} ]")
    print(f"    Test R²:   {r2:.4f}")
    print(f"    Test MAE:  {mae:.4f} eV/atom")
    print(f"    Test RMSE: {rmse:.4f} eV/atom")
    return {"r2": round(r2, 4), "mae": round(mae, 4), "rmse": round(rmse, 4)}


def print_feature_importance(model, top_n=10):
    """Print top feature importances for interpretability."""
    # BaggingRegressor: average over base estimators
    if hasattr(model, "estimators_"):
        importances = np.mean(
            [est.feature_importances_ for est in model.estimators_], axis=0
        )
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return

    order = np.argsort(importances)[::-1][:top_n]
    print(f"    Top {top_n} features:")
    for i in order:
        bar = "█" * int(importances[i] * 60)
        print(f"      {FEATURE_NAMES[i]:24s} {importances[i]:.4f}  {bar}")


# -----------------------------------------------------------------------
# Main Training
# -----------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  RESEARCH-GRADE MODEL TRAINING")
    print("  Data policy: REAL DFT data ONLY for thermodynamic models")
    print("=" * 65)

    # ----- Load real data -----
    print("\n[1/6] Loading real DFT data...")
    df = load_real_data()
    print(f"  Dataset columns: {list(df.columns)}")

    # Validate required columns
    col_map = {}
    for candidate in ["formation_energy_eV_per_atom", "formation_energy", "e_form"]:
        if candidate in df.columns:
            col_map["energy"] = candidate
            break
    for candidate in ["e_above_hull", "ehull", "energy_above_hull"]:
        if candidate in df.columns:
            col_map["ehull"] = candidate
            break
    for candidate in ["band_gap_eV", "band_gap", "bandgap"]:
        if candidate in df.columns:
            col_map["bandgap"] = candidate
            break

    if "energy" not in col_map:
        raise ValueError("Dataset must have a formation energy column.")

    print(f"  Column mapping: {col_map}")

    # ----- Build features -----
    print("\n[2/6] Building feature matrix...")
    X_all, valid_idx = build_feature_matrix(df)
    df_valid = df.loc[valid_idx].reset_index(drop=True)
    print(f"  Feature matrix: {X_all.shape}")

    # ----- Targets -----
    y_energy = df_valid[col_map["energy"]].values.astype(float)

    has_ehull = "ehull" in col_map
    if has_ehull:
        y_ehull = df_valid[col_map["ehull"]].values.astype(float)
        # Remove NaN rows for ehull
        valid_ehull = ~np.isnan(y_ehull)
        X_ehull = X_all[valid_ehull]
        y_ehull  = y_ehull[valid_ehull]
        print(f"  Ehull target: {valid_ehull.sum()} compounds with valid Ehull")
    else:
        print("  WARNING: No Ehull column found — Ehull model will not be trained.")

    has_bandgap = "bandgap" in col_map
    if has_bandgap:
        y_bandgap = df_valid[col_map["bandgap"]].values.astype(float)

    # ----- Stratified split -----
    print("\n[3/6] Stratified data splitting...")
    X_tr_e, X_te_e, y_tr_e, y_te_e = stratified_split(X_all, y_energy)
    print(f"  Energy split  — Train: {len(X_tr_e)}, Test: {len(X_te_e)}")

    if has_ehull and len(X_ehull) > 50:
        X_tr_h, X_te_h, y_tr_h, y_te_h = stratified_split(X_ehull, y_ehull)
        print(f"  Ehull split   — Train: {len(X_tr_h)}, Test: {len(X_te_h)}")

    if has_bandgap:
        X_tr_b, X_te_b, y_tr_b, y_te_b = stratified_split(X_all, y_bandgap)
        print(f"  Bandgap split — Train: {len(X_tr_b)}, Test: {len(X_te_b)}")

    # ----- Normalize features (fit scaler on TRAINING data ONLY) -----
    print("\n[4/6] Fitting feature normalizer on training data...")
    scaler = StandardScaler()
    X_tr_e_s = scaler.fit_transform(X_tr_e)
    X_te_e_s = scaler.transform(X_te_e)

    if has_ehull and len(X_ehull) > 50:
        X_tr_h_s = scaler.transform(X_tr_h)
        X_te_h_s = scaler.transform(X_te_h)

    if has_bandgap:
        X_tr_b_s = scaler.transform(X_tr_b)
        X_te_b_s = scaler.transform(X_te_b)

    joblib.dump(scaler, ENERGY_SCALER_PATH)
    print(f"  Scaler saved to {ENERGY_SCALER_PATH}")

    # ---------------------------------------------------------------
    # TRAIN MODELS
    # ---------------------------------------------------------------
    print("\n[5/6] Training models on REAL DFT data...")
    metrics = {}

    # --- Formation Energy ---
    print("\n  >>> Formation Energy Model")
    t0 = time.time()
    energy_base = train_base_model(X_tr_e_s, y_tr_e, "Formation Energy",
                                   n_estimators=800, max_depth=9, learning_rate=0.04)
    print(f"  Base model trained in {time.time()-t0:.1f}s, building ensemble...")
    energy_ens = train_ensemble(X_tr_e_s, y_tr_e, energy_base, n_estimators=10)
    metrics["formation_energy"] = evaluate_model(energy_ens, X_te_e_s, y_te_e, "Formation Energy")
    print_feature_importance(energy_ens)

    # --- Ehull ---
    if has_ehull and len(X_ehull) > 50:
        print("\n  >>> Energy Above Hull (Ehull) Model")
        t0 = time.time()
        ehull_base = train_base_model(X_tr_h_s, y_tr_h, "Ehull",
                                      n_estimators=600, max_depth=7, learning_rate=0.05)
        print(f"  Base model trained in {time.time()-t0:.1f}s, building ensemble...")
        ehull_ens = train_ensemble(X_tr_h_s, y_tr_h, ehull_base, n_estimators=10)
        metrics["ehull"] = evaluate_model(ehull_ens, X_te_h_s, y_te_h, "Ehull")
        joblib.dump(ehull_base, EHULL_MODEL_PATH)
        joblib.dump(ehull_ens,  EHULL_ENSEMBLE_PATH)
        print(f"  Ehull models saved.")

    # --- Band Gap ---
    if has_bandgap:
        print("\n  >>> Band Gap Model")
        t0 = time.time()
        bg_base = train_base_model(X_tr_b_s, y_tr_b, "Band Gap",
                                   n_estimators=800, max_depth=8, learning_rate=0.05)
        print(f"  Base model trained in {time.time()-t0:.1f}s, building ensemble...")
        bg_ens = train_ensemble(X_tr_b_s, y_tr_b, bg_base, n_estimators=10)
        metrics["band_gap"] = evaluate_model(bg_ens, X_te_b_s, y_te_b, "Band Gap")
        joblib.dump(bg_base, BANDGAP_MODEL_PATH)
        joblib.dump(bg_ens,  BANDGAP_ENSEMBLE_PATH)
        print(f"  Band gap models saved.")

    # --- Save core assets ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(energy_base, ENERGY_MODEL_PATH)
    joblib.dump(energy_ens,  ENERGY_ENSEMBLE_PATH)
    joblib.dump(energy_base, LEGACY_MODEL_PATH)  # backward compat

    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(FEATURE_NAMES, f)

    # --- Training metadata ---
    metadata = {
        "data_policy": "REAL_DFT_ONLY",
        "training_samples": {
            "energy": int(len(X_tr_e)),
            "ehull": int(len(X_tr_h)) if has_ehull and len(X_ehull) > 50 else 0,
            "bandgap": int(len(X_tr_b)) if has_bandgap else 0,
        },
        "test_samples": {
            "energy": int(len(X_te_e)),
        },
        "metrics": metrics,
        "feature_count": len(FEATURE_NAMES),
        "scaler": "StandardScaler (fitted on training split only)",
        "uncertainty_method": "BaggingRegressor (10 estimators)",
        "has_ehull_model": has_ehull and len(X_ehull) > 50 if has_ehull else False,
        "has_bandgap_model": has_bandgap,
    }
    with open(TRAINING_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    # ---------------------------------------------------------------
    print(f"\n[6/6] Summary")
    print(f"{'=' * 65}")
    print(f"  Data policy   : REAL DFT data ONLY (no synthetic contamination)")
    print(f"  Total samples : {len(df_valid)}")
    print(f"  Features      : {len(FEATURE_NAMES)}")
    print(f"  Normalization : StandardScaler (train-only fit)")
    print(f"  Uncertainty   : BaggingRegressor (10 bootstrap ensembles)")
    print(f"\n  Models saved:")
    print(f"    Energy:       {ENERGY_MODEL_PATH}  /  {ENERGY_ENSEMBLE_PATH}")
    if has_ehull and len(X_ehull) > 50:
        print(f"    Ehull:        {EHULL_MODEL_PATH}  /  {EHULL_ENSEMBLE_PATH}")
    if has_bandgap:
        print(f"    Band Gap:     {BANDGAP_MODEL_PATH}  /  {BANDGAP_ENSEMBLE_PATH}")
    print(f"    Scaler:       {ENERGY_SCALER_PATH}")
    print(f"    Metadata:     {TRAINING_METADATA_PATH}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
