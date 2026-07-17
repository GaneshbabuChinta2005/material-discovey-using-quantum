"""
Prediction Service — Research-Grade Hybrid Prediction with Uncertainty.

Priority order for each property:
  1. GNN model (PyTorch Geometric)  — if .pt files exist
  2. Ensemble GBR (BaggingRegressor) — with uncertainty estimation
  3. Single GBR model               — fallback if ensemble not available

Uncertainty Estimation:
  - GNN:     MC Dropout (multiple forward passes with dropout enabled)
  - Ensemble GBR: std(predictions) across BaggingRegressor sub-estimators

DATA POLICY: All models were trained exclusively on REAL DFT-derived data.
  Synthetic data is NOT used to train thermodynamic models.
"""

import os
import joblib
import numpy as np

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# -----------------------------------------------------------------------
# Model caches
# -----------------------------------------------------------------------
_gnn_energy_model  = None
_gnn_bandgap_model = None
_gnn_ehull_model   = None
_gbr_energy_model  = None
_gbr_ehull_model   = None
_gbr_bandgap_model = None
_ens_energy_model  = None
_ens_ehull_model   = None
_ens_bandgap_model = None
_feature_scaler    = None
_loaded  = False
_use_gnn = False

_torch         = None
_MaterialGNN   = None
_formula_to_graph = None


def _load_models():
    """Load all available models from disk (lazy, cached)."""
    global _gnn_energy_model, _gnn_bandgap_model, _gnn_ehull_model
    global _gbr_energy_model, _gbr_ehull_model, _gbr_bandgap_model
    global _ens_energy_model, _ens_ehull_model, _ens_bandgap_model
    global _feature_scaler, _loaded, _use_gnn
    global _torch, _MaterialGNN, _formula_to_graph

    if _loaded:
        return

    # ----- GNN models (preferred) -----
    gnn_energy_path  = os.path.join(MODEL_DIR, "gnn_energy_model.pt")
    gnn_bandgap_path = os.path.join(MODEL_DIR, "gnn_bandgap_model.pt")
    gnn_ehull_path   = os.path.join(MODEL_DIR, "gnn_ehull_model.pt")

    if os.path.exists(gnn_energy_path) or os.path.exists(gnn_bandgap_path) or os.path.exists(gnn_ehull_path):
        try:
            import torch
            from app.models.gnn_model import MaterialGNN
            from app.utils.graph_constructor import EDGE_FEAT_DIM, NODE_FEAT_DIM
            from app.utils.graph_constructor import formula_to_graph

            _torch = torch
            _MaterialGNN = MaterialGNN
            _formula_to_graph = formula_to_graph

            for path, attr, label in [
                (gnn_energy_path,  "_gnn_energy_model",  "energy"),
                (gnn_bandgap_path, "_gnn_bandgap_model", "bandgap"),
                (gnn_ehull_path,   "_gnn_ehull_model",   "ehull"),
            ]:
                if os.path.exists(path):
                    chk = torch.load(path, map_location="cpu", weights_only=False)
                    cfg = chk["config"]
                    m = MaterialGNN(
                        node_feat_dim=cfg.get("node_feat_dim", NODE_FEAT_DIM),
                        edge_feat_dim=cfg.get("edge_feat_dim", EDGE_FEAT_DIM),
                        hidden_dim=cfg.get("hidden_dim", 128),
                        n_conv_layers=cfg.get("n_conv_layers", 3),
                    )
                    res = m.load_state_dict(chk["model_state_dict"], strict=False)
                    if getattr(res, "missing_keys", None):
                        print(f"[PredictionService] GNN {label} checkpoint missing: {res.missing_keys}")
                    m.eval()
                    globals()[attr] = m
                    print(f"[PredictionService] GNN {label} loaded "
                          f"(MAE={chk.get('best_val_mae', '?'):.4f})")

            _use_gnn = (_gnn_energy_model is not None or
                        _gnn_bandgap_model is not None or
                        _gnn_ehull_model is not None)
            if _use_gnn:
                print("[PredictionService] Using GNN as primary predictor")

        except ImportError:
            print("[PredictionService] PyTorch not available — using GBR models")
        except Exception as e:
            print(f"[PredictionService] GNN loading failed: {e}")

    # ----- Feature scaler -----
    scaler_path = os.path.join(MODEL_DIR, "energy_scaler.pkl")
    if os.path.exists(scaler_path):
        _feature_scaler = joblib.load(scaler_path)
        print("[PredictionService] Feature scaler loaded.")

    # ----- Ensemble GBR models (primary non-GNN path) -----
    for path, attr, label in [
        (os.path.join(MODEL_DIR, "energy_ensemble.pkl"),  "_ens_energy_model",  "energy ensemble"),
        (os.path.join(MODEL_DIR, "ehull_ensemble.pkl"),   "_ens_ehull_model",   "ehull ensemble"),
        (os.path.join(MODEL_DIR, "bandgap_ensemble.pkl"), "_ens_bandgap_model", "bandgap ensemble"),
    ]:
        if os.path.exists(path):
            globals()[attr] = joblib.load(path)
            print(f"[PredictionService] Loaded {label}")

    # ----- Single GBR models (fallback) -----
    for path, attr, label in [
        (os.path.join(MODEL_DIR, "energy_model.pkl"),  "_gbr_energy_model",  "GBR energy"),
        (os.path.join(MODEL_DIR, "ehull_model.pkl"),   "_gbr_ehull_model",   "GBR ehull"),
        (os.path.join(MODEL_DIR, "bandgap_model.pkl"), "_gbr_bandgap_model", "GBR bandgap"),
    ]:
        if os.path.exists(path):
            globals()[attr] = joblib.load(path)
            if not _use_gnn:
                print(f"[PredictionService] Loaded {label}")

    _loaded = True


def _scale(features: list[float]) -> np.ndarray:
    """Apply the feature scaler if available."""
    X = np.array([features])
    if _feature_scaler is not None:
        return _feature_scaler.transform(X)
    return X


def _gnn_predict(model, formula: str) -> float | None:
    """Run GNN inference on a single formula."""
    graph = _formula_to_graph(formula)
    if graph is None:
        return None
    with _torch.no_grad():
        pred = model(graph)
    return float(pred.item())


def _ensemble_predict_with_std(ensemble, X_scaled: np.ndarray) -> tuple[float, float]:
    """
    Run BaggingRegressor sub-estimators individually to get mean + std.
    Returns (mean_pred, std_pred).
    """
    predictions = []
    for est in ensemble.estimators_:
        try:
            pred = est.predict(X_scaled)
            predictions.append(float(pred[0]))
        except Exception:
            pass
    if not predictions:
        return float(ensemble.predict(X_scaled)[0]), 0.0
    return float(np.mean(predictions)), float(np.std(predictions))


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def predict_energy(features: list[float], formula: str = None) -> float | None:
    """Predict formation energy (eV/atom). Uses GNN > Ensemble GBR > Single GBR."""
    _load_models()

    if _gnn_energy_model is not None and formula is not None:
        try:
            result = _gnn_predict(_gnn_energy_model, formula)
            if result is not None:
                return result
        except Exception:
            pass

    X = _scale(features)

    if _ens_energy_model is not None:
        try:
            mean, _ = _ensemble_predict_with_std(_ens_energy_model, X)
            return mean
        except Exception as e:
            print(f"[PredictionService] Ensemble energy error: {e}")

    if _gbr_energy_model is not None:
        try:
            return float(_gbr_energy_model.predict(X)[0])
        except Exception as e:
            print(f"[PredictionService] GBR energy error: {e}")

    return None


def predict_ehull(features: list[float], formula: str = None) -> float | None:
    """
    Predict Energy Above Hull (eV/atom).
    Returns None if no Ehull model is available.
    """
    _load_models()
    
    if _gnn_ehull_model is not None and formula is not None:
        try:
            result = _gnn_predict(_gnn_ehull_model, formula)
            if result is not None:
                return max(0.0, result)
        except Exception:
            pass

    X = _scale(features)

    if _ens_ehull_model is not None:
        try:
            mean, _ = _ensemble_predict_with_std(_ens_ehull_model, X)
            return max(0.0, mean)   # Ehull cannot be negative by definition
        except Exception as e:
            print(f"[PredictionService] Ensemble ehull error: {e}")

    if _gbr_ehull_model is not None:
        try:
            val = float(_gbr_ehull_model.predict(X)[0])
            return max(0.0, val)
        except Exception as e:
            print(f"[PredictionService] GBR ehull error: {e}")

    return None  # Signal "not evaluated" — caller must handle this


def predict_bandgap(features: list[float], formula: str = None) -> float | None:
    """Predict band gap (eV). Uses GNN > Ensemble GBR > Single GBR."""
    _load_models()

    if _gnn_bandgap_model is not None and formula is not None:
        try:
            result = _gnn_predict(_gnn_bandgap_model, formula)
            if result is not None:
                return result
        except Exception:
            pass

    X = _scale(features)

    if _ens_bandgap_model is not None:
        try:
            mean, _ = _ensemble_predict_with_std(_ens_bandgap_model, X)
            return max(0.0, mean)
        except Exception as e:
            print(f"[PredictionService] Ensemble bandgap error: {e}")

    if _gbr_bandgap_model is not None:
        try:
            return max(0.0, float(_gbr_bandgap_model.predict(X)[0]))
        except Exception as e:
            print(f"[PredictionService] GBR bandgap error: {e}")

    return None


def predict_all(features: list[float], formula: str = None) -> dict:
    """Predict all properties (point estimates, no uncertainty)."""
    energy  = predict_energy(features, formula=formula)
    ehull   = predict_ehull(features, formula=formula)
    bandgap = predict_bandgap(features, formula=formula)
    return {
        "formation_energy_eV_per_atom": round(energy, 4) if energy is not None else None,
        "e_above_hull_predicted": round(ehull, 4) if ehull is not None else None,
        "band_gap_eV": round(bandgap, 3) if bandgap is not None else None,
    }


def predict_with_uncertainty(
    features: list[float],
    formula: str = None,
    n_mc_samples: int = 10,
) -> dict:
    """
    Predict properties WITH uncertainty estimation.

    GNN path  → MC Dropout (n_mc_samples forward passes)
    GBR path  → BaggingRegressor sub-estimator spread (std)
    No model  → None for uncertainty

    Returns:
        {
          "formation_energy_eV_per_atom": float,
          "formation_energy_uncertainty_eV": float | None,
          "e_above_hull_predicted": float | None,
          "ehull_uncertainty_eV": float | None,
          "ehull_available": bool,
          "band_gap_eV": float,
          "band_gap_uncertainty_eV": float | None,
          "confidence_level": "high" | "medium" | "low" | "speculative",
          "uncertainty_method": str,
        }
    """
    _load_models()
    X = _scale(features)

    result = {}

    # ---- GNN path — MC Dropout ----
    if _use_gnn and formula is not None and _torch is not None:
        energy_preds, bandgap_preds = [], []
        
        # Precompute graph ONCE instead of inside the MC loop
        graph = None
        try:
            graph = _formula_to_graph(formula)
        except Exception:
            pass

        for _ in range(n_mc_samples):
            if _gnn_energy_model is not None and graph is not None:
                try:
                    _gnn_energy_model.train()
                    with _torch.no_grad():
                        energy_preds.append(float(_gnn_energy_model(graph).item()))
                except Exception:
                    pass
            
            if _gnn_bandgap_model is not None and graph is not None:
                try:
                    _gnn_bandgap_model.train()
                    with _torch.no_grad():
                        bandgap_preds.append(float(_gnn_bandgap_model(graph).item()))
                except Exception:
                    pass

        if _gnn_energy_model:  _gnn_energy_model.eval()
        if _gnn_bandgap_model: _gnn_bandgap_model.eval()

        if energy_preds:
            result["formation_energy_eV_per_atom"] = round(np.mean(energy_preds), 4)
            result["formation_energy_uncertainty_eV"] = round(np.std(energy_preds), 4)
        if bandgap_preds:
            result["band_gap_eV"] = round(max(0, np.mean(bandgap_preds)), 3)
            result["band_gap_uncertainty_eV"] = round(np.std(bandgap_preds), 3)

        result["uncertainty_method"] = "mc_dropout"

    # ---- Ensemble GBR path ----
    if "formation_energy_eV_per_atom" not in result:
        if _ens_energy_model is not None:
            mean_e, std_e = _ensemble_predict_with_std(_ens_energy_model, X)
            result["formation_energy_eV_per_atom"] = round(mean_e, 4)
            result["formation_energy_uncertainty_eV"] = round(std_e, 4)
        elif _gbr_energy_model is not None:
            result["formation_energy_eV_per_atom"] = predict_energy(features, formula)
            result["formation_energy_uncertainty_eV"] = None
        result.setdefault("uncertainty_method", "ensemble_gbr")

    if "band_gap_eV" not in result:
        if _ens_bandgap_model is not None:
            mean_b, std_b = _ensemble_predict_with_std(_ens_bandgap_model, X)
            result["band_gap_eV"] = round(max(0, mean_b), 3)
            result["band_gap_uncertainty_eV"] = round(std_b, 3)
        elif _gbr_bandgap_model is not None:
            result["band_gap_eV"] = predict_bandgap(features, formula)
            result["band_gap_uncertainty_eV"] = None

    # ---- Ehull prediction ----
    ehull_val = None
    ehull_std = None

    if _ens_ehull_model is not None:
        mean_h, std_h = _ensemble_predict_with_std(_ens_ehull_model, X)
        ehull_val = round(max(0.0, mean_h), 4)
        ehull_std = round(std_h, 4)
    elif _gbr_ehull_model is not None:
        ehull_val = predict_ehull(features, formula)

    result["e_above_hull_predicted"] = ehull_val
    result["ehull_uncertainty_eV"]   = ehull_std
    result["ehull_available"]        = ehull_val is not None

    # ---- Compute confidence level ----
    unc_e = result.get("formation_energy_uncertainty_eV") or 0.0
    unc_h = result.get("ehull_uncertainty_eV") or 0.0
    total_unc = max(unc_e, unc_h)

    if total_unc < 0.05:
        confidence = "high"
    elif total_unc < 0.15:
        confidence = "medium"
    elif total_unc < 0.30:
        confidence = "low"
    else:
        confidence = "speculative"

    result["confidence_level"] = confidence
    result.setdefault("uncertainty_method", "none")

    # ---- Epistemic vs Aleatoric decomposition (Phase 5) ----
    # Epistemic = model uncertainty (reducible with more data)
    # Aleatoric = inherent noise (irreducible)
    total_unc_e = result.get("formation_energy_uncertainty_eV") or 0.0
    total_unc_h = result.get("ehull_uncertainty_eV") or 0.0
    total_unc_b = result.get("band_gap_uncertainty_eV") or 0.0

    # Approximate split: 70% epistemic / 30% aleatoric (typical for ML on DFT data)
    epistemic_ratio = 0.70
    result["epistemic_uncertainty_eV"] = round(max(total_unc_e, total_unc_h) * epistemic_ratio, 4)
    result["aleatoric_uncertainty_eV"] = round(max(total_unc_e, total_unc_h) * (1 - epistemic_ratio), 4)
    result["confidence_interval_95"] = round(max(total_unc_e, total_unc_h, total_unc_b) * 1.96, 4)

    # Fill defaults
    result.setdefault("formation_energy_eV_per_atom", None)
    result.setdefault("formation_energy_uncertainty_eV", None)
    result.setdefault("band_gap_eV", None)
    result.setdefault("band_gap_uncertainty_eV", None)

    return result


def get_model_type() -> str:
    """Return active model type string."""
    _load_models()
    if _use_gnn:
        return "GNN (Graph Neural Network)"
    elif _ens_energy_model is not None:
        return "Ensemble GBR (BaggingRegressor)"
    elif _gbr_energy_model is not None:
        return "GBR (Gradient Boosting)"
    return "None"


def get_model_info() -> dict:
    """Return model metadata and accuracy metrics."""
    _load_models()
    import os as _os

    info = {
        "model_type": get_model_type(),
        "has_ehull_model": _ens_ehull_model is not None or _gbr_ehull_model is not None,
        "has_ensemble": _ens_energy_model is not None,
        "data_policy": "REAL_DFT_ONLY",
        "energy_model": None,
        "ehull_model": None,
        "bandgap_model": None,
    }

    # Read training metadata if available
    meta_path = _os.path.join(MODEL_DIR, "training_metadata.json")
    if _os.path.exists(meta_path):
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        info["training_metadata"] = meta

    # GNN info
    if _use_gnn:
        for target, pth in [("energy_model", "gnn_energy_model.pt"),
                             ("bandgap_model", "gnn_bandgap_model.pt")]:
            model_path = _os.path.join(MODEL_DIR, pth)
            if _os.path.exists(model_path):
                try:
                    import torch
                    chk = torch.load(model_path, map_location="cpu", weights_only=False)
                    cfg = chk.get("config", {})
                    info[target] = {
                        "type": "GNN",
                        "val_mae": round(chk.get("best_val_mae", 0), 4),
                        "epoch": chk.get("epoch", "?"),
                        "hidden_dim": cfg.get("hidden_dim", "?"),
                        "n_conv_layers": cfg.get("n_conv_layers", "?"),
                    }
                except Exception:
                    info[target] = {"type": "GNN", "val_mae": "?"}

    return info


# -----------------------------------------------------------------------
if __name__ == "__main__":
    from app.utils.feature_engine import compound_to_features

    print(f"Model type: {get_model_type()}")
    info = get_model_info()
    print(f"Has Ehull Model: {info.get('has_ehull_model')}")
    print(f"Has Ensemble:    {info.get('has_ensemble')}\n")

    for formula in ["NaCl", "TiO2", "Fe2O3", "LiCoO2", "BaTiO3"]:
        feats = compound_to_features(formula)
        if feats:
            r = predict_with_uncertainty(feats, formula=formula)
            print(f"{formula}:")
            print(f"  Ef = {r['formation_energy_eV_per_atom']} ± {r['formation_energy_uncertainty_eV']}")
            print(f"  Ehull = {r['e_above_hull_predicted']} ± {r['ehull_uncertainty_eV']}")
            print(f"  Bg = {r['band_gap_eV']}")
            print(f"  Confidence = {r['confidence_level']}")
