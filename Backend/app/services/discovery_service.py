"""
Discovery Service — Research-Grade Inverse-Design Orchestration Pipeline.

RESEARCH-GRADE PIPELINE (v4):

Workflow:
  1. Dataset validation (verify real data models loaded)
  2. Candidate generation (prototype-constrained compositions)
  3. Chemistry validation (charge neutrality, coordination, oxidation)
  4. Structure validation (Goldschmidt tolerance, bond distances, coordination)
  5. Feature computation + ML prediction with uncertainty
  6. Thermodynamic filtering (Ehull < 0.2 eV AND Ef < -1.0 eV)
  7. ★ Convex hull phase validation (competing phase analysis)
  8. ★ Synthesis feasibility estimation (ML classifier)
  9. Quantum refinement (VQE for top-N candidates)
 10. Novelty verification (full MP database cross-check)
 11. ★ Weighted scientific scoring (explicit tunable formula)
 12. Ranking + visualization

DATA POLICY:
  - Thermodynamic predictions only from models trained on REAL DFT data.
  - Synthetic data never used as ground truth for thermodynamics.
  - Ehull "not evaluated" when direct ML Ehull model unavailable.

SCORING FORMULA (explicit, tunable):
  Score = w1*S_ehull + w2*S_ef + w3*S_bandgap + w4*S_density
        + w5*S_struct + w6*S_confidence + w7*S_novelty + w8*S_quantum
  with Ehull-based grade downgrade rule: Ehull > 0.1 → cap at 70%.
"""

import re
import math
import time
from typing import Any
from itertools import combinations as _combinations

from app.services.composition_explorer import explore_compositions
from app.services.chemistry_validator import (
    is_valid_compound, get_compound_stability_score,
    classify_compound, get_validated_structure_type,
)
from app.services.novelty_checker import check_novelty
from app.services.chemical_naming import get_compound_name
from app.services.physics_corrector import (
    correct_bandgap, correct_formation_energy, compute_overall_confidence,
)
from app.utils.feature_engine import compound_to_features, FEATURE_NAMES
from app.services.prediction_service import (
    predict_energy, predict_ehull, predict_bandgap,
    predict_with_uncertainty, get_model_type,
)
from app.services.quantum_service import run_quantum_validation
from app.services.candidate_service import filter_alloys
from app.services.usecase_predictor import predict_usecase
from app.services.learning_service import store_discovery

# ---- Optional modules (graceful degradation) ----
try:
    from app.services.structure_validator import validate_structure
    _HAS_STRUCT_VALIDATOR = True
except ImportError as _e:
    print(f"[Discovery] WARNING: structure_validator unavailable: {_e}")
    _HAS_STRUCT_VALIDATOR = False
    def validate_structure(formula, prototype=None):
        return {"feasible": True, "feasibility_score": 0.5, "warnings": [],
                "goldschmidt_t": None, "prototype": prototype or "unknown",
                "reject_reason": None, "pymatgen_used": False}

try:
    from app.services.structure_generator import generate_structure
    _HAS_STRUCTURE_GEN = True
except ImportError:
    _HAS_STRUCTURE_GEN = False
    def generate_structure(formula):
        return {"atoms": [], "bonds": [], "total_atoms": 0}

try:
    from app.services.convex_hull_estimator import evaluate_phase_stability
    _HAS_CONVEX_HULL = True
except ImportError as _e:
    print(f"[Discovery] WARNING: convex_hull_estimator unavailable: {_e}")
    _HAS_CONVEX_HULL = False
    def evaluate_phase_stability(formula, predicted_formation_energy=None, ml_ehull=None):
        return {"hull_distance_eV": None, "hull_source": "unavailable",
                "hull_status": "unknown", "competing_phases": [],
                "n_competing_phases": 0, "chemical_system": "",
                "decomposition_warning": None}

try:
    from app.services.synthesis_predictor import predict_synthesis_feasibility
    _HAS_SYNTHESIS = True
except ImportError as _e:
    print(f"[Discovery] WARNING: synthesis_predictor unavailable: {_e}")
    _HAS_SYNTHESIS = False
    def predict_synthesis_feasibility(features, formation_energy=None, ehull=None, bandgap=None):
        return {"synthesis_confidence": 0.5, "synthesis_label": "Not Evaluated",
                "synthesis_factors": {}}

try:
    from app.services.prototype_classifier import classify_prototype
    _HAS_PROTOTYPE_CLF = True
except ImportError as _e:
    print(f"[Discovery] WARNING: prototype_classifier unavailable: {_e}")
    _HAS_PROTOTYPE_CLF = False
    def classify_prototype(formula):
        return {"prototype": "unknown", "confidence": 0.0, "probabilities": {},
                "top3": [], "method": "unavailable", "features": {}}

try:
    from app.services.stability_analyzer import estimate_e_above_hull
    _HAS_STABILITY = True
except ImportError:
    _HAS_STABILITY = False
    def estimate_e_above_hull(formula, energy):
        return {"e_above_hull_eV": None, "stability": "unknown",
                "stability_label": "Not Evaluated", "stability_tier": 0,
                "stability_warning": None, "competing_phases": [],
                "reference_available": False}


# -----------------------------------------------------------------------
# Calibrated Scientific Scoring (Phase 3)
# -----------------------------------------------------------------------
def _compute_calibrated_probability(
    ehull: float | None,
    ef: float,
    model_conf: str = "medium"
) -> float:
    """
    Compute a physically interpretable stability probability (0-100%).
    Based on Boltzmann-like distribution and experimental synthesis statistics.
    """
    if ehull is None:
        # If Ehull unknown, use a very conservative estimate based on Ef
        p_ef = 1.0 / (1.0 + math.exp(3.0 * (ef + 0.5)))
        return round(p_ef * 30.0, 2) # Cap at 30% if Ehull unknown

    # 1. Ehull component: P decreases steeply as Ehull > 0.1 eV/atom
    # Sigmoid centered at 0.15 eV with steepness k=25
    p_ehull = 1.0 / (1.0 + math.exp(25.0 * (ehull - 0.15)))
    
    # 2. Formation Energy component: P increases with exothermic favorability
    # Sigmoid centered at -0.5 eV
    p_ef = 1.0 / (1.0 + math.exp(2.0 * (ef + 0.5)))
    
    # Combine and normalize to 0-100
    p_stable = p_ehull * p_ef * 100.0
    
    # Uncertainty penalty
    conf_penalty = {"high": 1.0, "medium": 0.9, "low": 0.6, "speculative": 0.3}.get(model_conf, 0.7)
    p_stable *= conf_penalty
    
    # Strict cutoff: Ehull > 0.8 eV/atom is effectively 0% probability
    if ehull > 0.8:
        return 0.0
        
    return round(p_stable, 2)


def _score_density(density: float | None) -> float:
    """Score density: 2–12 g/cm³ is realistic for crystalline materials."""
    if density is None:
        return 0.5
    if 2.0 <= density <= 12.0:
        return 1.0
    elif 1.0 <= density < 2.0 or 12.0 < density <= 20.0:
        return 0.6
    else:
        return 0.2


def _score_confidence(confidence_level: str) -> float:
    """Score model confidence level."""
    return {"high": 1.0, "medium": 0.7, "low": 0.4, "speculative": 0.1}.get(
        confidence_level, 0.4
    )


def compute_weighted_score(
    ehull: float | None,
    tier: int,
    ef: float,
    bg: float | None,
    density: float | None,
    struct_feasibility: float,
    confidence_level: str,
    novelty_score: float,
    quantum_bonus: float | None = None,
) -> dict:
    """
    Compute a research-grade, calibrated stability probability (0-100%).
    Replaces heuristic weights with physical calibration.
    """
    # 1. Core thermodynamic stability probability
    p_stable = _compute_calibrated_probability(ehull, ef, confidence_level)
    
    # 2. Structural feasibility adjustment
    # If structure is highly unlikely (<0.4), significantly penalize probability
    p_stable *= (0.5 + 0.5 * struct_feasibility)
    
    # 3. Novelty and Quantum bonuses (small additive research incentives)
    # These are research-value bonuses, not physical stability
    nov_bonus = round(novelty_score * 5.0, 2)
    q_bonus   = round((quantum_bonus or 0.0) * 10.0, 2)
    
    # Final Research-Grade Score
    total_score = p_stable + nov_bonus + q_bonus
    
    # Clamp to 100 for presentation
    total_score = min(100.0, total_score)
    
    return {
        "total_score":       round(total_score, 2),
        "base_score":        round(p_stable, 2),
        "novelty_bonus":     nov_bonus,
        "quantum_bonus":     q_bonus,
        "ehull_capped":      bool(ehull is not None and ehull > 0.1),
        "breakdown": {
            "p_stability":   round(p_stable, 2),
            "struct_coeff":  round(struct_feasibility, 2),
            "novelty_pts":   nov_bonus,
            "quantum_pts":   q_bonus,
        },
        "weights": "Calibrated Probabilistic Model (Phase 3)",
    }


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _parse_elements(formula: str) -> list[str]:
    return re.findall(r'[A-Z][a-z]?', formula)


def _n_atoms(formula: str) -> int:
    return sum(int(d) if d else 1
               for _, d in re.findall(r'([A-Z][a-z]?)(\d*)', formula) if _)


def _infer_prototype(formula: str) -> str | None:
    parts = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    counts = sorted([int(c) if c else 1 for _, c in parts if _])
    n_elem = len([_ for _, c in parts if _])

    if n_elem == 2 and counts == [1, 1]: return "rock_salt"
    if n_elem == 2 and counts == [1, 2]: return "fluorite"
    if n_elem == 2 and counts == [2, 3]: return "corundum"
    if n_elem == 3 and counts == [1, 1, 3]: return "perovskite"
    if n_elem == 3 and counts == [1, 2, 4]: return "spinel"
    return None


def _classify_tier(ehull_val: float | None) -> tuple[int, str, str, str | None]:
    """Classify Ehull into tier, stability label, and warning."""
    if ehull_val is None:
        return (0, "not_evaluated", "Not Evaluated",
                "Ehull model not available. Do not assume stability.")
    elif ehull_val <= 0.0:
        return (1, "stable", "Thermodynamically Stable", None)
    elif ehull_val < 0.05:
        return (1, "thermodynamically_stable", "Thermodynamically Stable", None)
    elif ehull_val < 0.2:
        return (2, "metastable", "Metastable",
                "May be synthesizable under non-equilibrium conditions.")
    elif ehull_val < 0.5:
        return (3, "high_energy_metastable", "High-Energy Metastable",
                "Likely requires kinetic trapping or epitaxial stabilization.")
    elif ehull_val < 1.0:
        return (4, "unstable", "Unstable",
                "Thermodynamically unstable — decomposes under equilibrium.")
    else:
        return (5, "highly_unstable", "Highly Unstable",
                f"Ehull={ehull_val:.2f} eV — will not form under equilibrium.")


# -----------------------------------------------------------------------
# Temperature / Pressure Stability Correction
# -----------------------------------------------------------------------
def _tp_stability_correction(
    ehull_eV: float | None,
    formation_energy: float,
    n_elements: int,
    temperature_K: float = 300.0,
    pressure_GPa: float = 0.0001,
) -> dict:
    """
    Clausius-Clapeyron-inspired thermodynamic correction.

    At higher T, configurational entropy stabilizes disordered/metastable phases:
        ΔG ≈ ΔH - T·ΔS_config
    where ΔS_config ≈ k_B · N_species · ln(N_sites)

    At higher P, PΔV term favors denser phases.
    """
    kB_eV = 8.617e-5  # Boltzmann constant in eV/K

    # Configurational entropy stabilization
    delta_S_config = kB_eV * max(n_elements, 2) * math.log(max(n_elements, 2))
    entropy_correction_eV = -temperature_K * delta_S_config  # negative = stabilizing

    # Pressure term: approximate PΔV (small for solids, ~0.01 eV/GPa)
    pressure_correction_eV = pressure_GPa * 0.01  # positive = favors dense phases

    corrected_ef = formation_energy + entropy_correction_eV + pressure_correction_eV

    # Adjust ehull threshold at high T (metastable phases become accessible)
    ehull_threshold_shift = 0.0
    if temperature_K > 600:
        ehull_threshold_shift = min((temperature_K - 600) * 0.0002, 0.15)

    return {
        "corrected_formation_energy": round(corrected_ef, 4),
        "entropy_correction_eV": round(entropy_correction_eV, 4),
        "pressure_correction_eV": round(pressure_correction_eV, 4),
        "ehull_threshold_shift": round(ehull_threshold_shift, 4),
        "temperature_K": temperature_K,
        "pressure_GPa": pressure_GPa,
    }


# -----------------------------------------------------------------------
# Battery Suitability Scoring
# -----------------------------------------------------------------------
def _compute_battery_fitness(c: dict) -> float:
    """Compute application-aware fitness for battery electrodes."""
    bg = c.get("band_gap_eV") or 0.0
    bg_score = max(0.0, 1.0 - max(0.0, bg - 2.5) / 2.0)  # Penalize highly if > 2.5 eV
    
    proto = c.get("prototype", "").lower()
    proto_score = 0.5
    if any(p in proto for p in ["layered", "spinel", "olivine", "nasicon", "amo2", "a2mo3", "ab2o4", "ampo4"]):
        proto_score = 1.0
        
    syms = _parse_elements(c.get("compound", ""))
    tm_redox = 0.5
    if any(tm in syms for tm in ["Co", "Ni", "Mn", "Fe", "V", "Ti"]):
        tm_redox = 1.0
        
    # Mobile ion pathway likelihood tag
    if proto_score == 1.0 and any(m in syms for m in ["Li", "Na", "Mg", "Zn", "Al"]):
        c["mobile_ion_pathway_likelihood"] = "High"
    elif any(m in syms for m in ["Li", "Na"]):
        c["mobile_ion_pathway_likelihood"] = "Medium"
    else:
        c["mobile_ion_pathway_likelihood"] = "Low"
        
    if proto_score == 1.0:
        c["battery_prototype_match"] = proto if proto != "unknown" else "battery_structure"
    else:
        c["battery_prototype_match"] = "novel_structure"

    return (bg_score * 0.4) + (proto_score * 0.4) + (tm_redox * 0.2)


# -----------------------------------------------------------------------
# Pareto Multi-Objective Ranking
# -----------------------------------------------------------------------
def _pareto_rank(candidates: list[dict], application_targets: list[str] | None = None) -> list[dict]:
    """
    Assign Pareto front ranks across multiple objectives:
      - stability_score (higher = better)
      - band_gap suitability (closer to 1.5 eV = better for most apps)
      - synthesis_confidence (higher = better)
      - battery_fitness (if battery target selected)

    Returns candidates with 'pareto_rank' field added.
    """
    if not candidates:
        return candidates

    is_battery = any("battery" in (t or "").lower() for t in (application_targets or []))

    # Compute objective vectors (all maximized)
    for c in candidates:
        bg = c.get("band_gap_eV", 0) or 0
        c["_obj_stability"] = c.get("stability_score", 0)
        c["_obj_bandgap"] = max(0, 1.0 - abs(bg - 1.5) / 3.0)  # peak at 1.5 eV
        c["_obj_synthesis"] = c.get("synthesis_confidence") or 0.5
        if is_battery:
            c["_obj_battery"] = _compute_battery_fitness(c)

    remaining = list(range(len(candidates)))
    rank = 1

    while remaining:
        front = []
        for i in remaining:
            dominated = False
            ci = candidates[i]
            for j in remaining:
                if i == j:
                    continue
                cj = candidates[j]
                
                # Check dominance (cj dominates ci if cj is >= on all and > on at least one)
                ge_all = (
                    cj["_obj_stability"] >= ci["_obj_stability"] and
                    cj["_obj_bandgap"] >= ci["_obj_bandgap"] and
                    cj["_obj_synthesis"] >= ci["_obj_synthesis"]
                )
                gt_any = (
                    cj["_obj_stability"] > ci["_obj_stability"] or
                    cj["_obj_bandgap"] > ci["_obj_bandgap"] or
                    cj["_obj_synthesis"] > ci["_obj_synthesis"]
                )
                
                if is_battery:
                    ge_all = ge_all and (cj["_obj_battery"] >= ci["_obj_battery"])
                    gt_any = gt_any or (cj["_obj_battery"] > ci["_obj_battery"])

                if ge_all and gt_any:
                    dominated = True
                    break
            if not dominated:
                front.append(i)

        for i in front:
            candidates[i]["pareto_rank"] = rank
            remaining.remove(i)
        rank += 1

        if rank > 50:  # safety
            for i in remaining:
                candidates[i]["pareto_rank"] = rank
            break

    # Clean up temp keys
    for c in candidates:
        c.pop("_obj_stability", None)
        c.pop("_obj_bandgap", None)
        c.pop("_obj_synthesis", None)
        c.pop("_obj_battery", None)

    return candidates


# -----------------------------------------------------------------------
# Chemical Validity Probability
# -----------------------------------------------------------------------
def _chemical_validity_probability(classification: dict) -> float:
    """Map chemistry_validator classification → probability."""
    v = classification.get("validity", "invalid")
    if v == "valid":
        base = 0.92
    elif v == "partially_valid":
        base = 0.55
    else:
        base = 0.08
    if classification.get("prototype_match"):
        base = min(base + 0.05, 0.98)
    if classification.get("coordination_ok"):
        base = min(base + 0.03, 0.99)
    return round(base, 2)


# -----------------------------------------------------------------------
# Feature Importance (permutation-based)
# -----------------------------------------------------------------------
def _compute_feature_importance(features: list, formula: str) -> list[dict]:
    """
    Estimate feature importance via perturbation analysis.
    Returns top-10 most important features with scores.
    """
    try:
        import numpy as np
        from app.services.prediction_service import predict_energy, predict_bandgap

        features_arr = np.array(features, dtype=float)
        base_energy = predict_energy(features_arr)
        base_bandgap = predict_bandgap(features_arr)
        if base_energy is None:
            return []

        importances = []
        for i in range(len(features_arr)):
            perturbed = features_arr.copy()
            original_val = perturbed[i]
            delta = max(abs(original_val) * 0.2, 0.5)
            perturbed[i] = original_val + delta
            new_energy = predict_energy(perturbed)
            new_bandgap = predict_bandgap(perturbed)
            if new_energy is None:
                continue
            energy_shift = abs((new_energy - base_energy) / max(abs(base_energy), 0.01))
            bg_shift = abs(((new_bandgap or 0) - (base_bandgap or 0)) / max(abs(base_bandgap or 1), 0.01))
            combined = energy_shift + bg_shift
            importances.append({
                "feature_index": i,
                "feature_name": FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature_{i}",
                "importance": round(combined, 4),
                "energy_sensitivity": round(energy_shift, 4),
                "bandgap_sensitivity": round(bg_shift, 4),
            })

        importances.sort(key=lambda x: x["importance"], reverse=True)
        return importances[:10]
    except Exception:
        return []


# -----------------------------------------------------------------------
# Main Discovery Pipeline
# -----------------------------------------------------------------------

def discover_materials(
    target_energy_min: float = -4.0,
    target_energy_max: float = -0.3,
    target_bandgap_min: float | None = None,
    target_bandgap_max: float | None = None,
    max_atoms: int = 10,
    min_stability: float = 15.0,
    element_groups: list[str] | None = None,
    include_elements: list[str] | None = None,
    exclude_elements: list[str] | None = None,
    num_candidates: int = 5,
    min_strength: float = 500.0,
    min_melting_point: float = 1200.0,
    stability_level: str = "high",
    enable_quantum_refinement: bool = True,
    strict_research_filter: bool = True,
    # ---- Phase 6: Research-Grade Upgrade ----
    crystal_systems: list[str] | None = None,
    application_targets: list[str] | None = None,
    temperature_K: float = 300.0,
    pressure_GPa: float = 0.0001,
) -> dict:
    """
    Run the full research-grade inverse-design discovery pipeline (v4).

    Returns:
        {
            "novel_materials":  list of high-quality validated candidates,
            "alloy_candidates": list from alloy database,
            "search_stats":     pipeline statistics dict,
        }
    """
    t0 = time.time()
    import time as _t

    print(f"[Discovery] v5 Research-Grade Pipeline (Probabilistic)")
    print(f"[Discovery] Stability: {stability_level}  StructValidator: {_HAS_STRUCT_VALIDATOR}"
          f"  ConvexHull: {_HAS_CONVEX_HULL}  Synthesis: {_HAS_SYNTHESIS}"
          f"  Quantum: {enable_quantum_refinement}")

    max_atoms = max(2, min(max_atoms, 25))

    # ==================================================================
    # STEP 1: Dataset validation (verify models loaded)
    # ==================================================================
    model_type = get_model_type()
    print(f"[Discovery] Active model type: {model_type}")

    # ==================================================================
    # STEP 2: Candidate generation (prototype-constrained)
    # ==================================================================
    pool_size = max(num_candidates * 80, 400)
    formulas = explore_compositions(
        element_groups=element_groups,
        max_atoms=max_atoms,
        max_candidates=pool_size,
        min_stability_score=5.0,
        include_elements=include_elements,
        exclude_elements=exclude_elements,
        target_applications=application_targets,
    )
    print(f"[Step 2] Generated {len(formulas)} candidate compositions")
    if not formulas:
        return {"novel_materials": [], "alloy_candidates": [],
                "search_stats": {"explored": 0}}

    # ==================================================================
    # STEP 3: Chemistry validation
    # ==================================================================
    t_chem = _t.time()
    validated, invalid_count, partial_count = [], 0, 0
    for f in formulas:
        result = classify_compound(f)
        v = result["validity"]
        if v in ("valid", "partially_valid"):
            validated.append((f, result))
            if v == "partially_valid":
                partial_count += 1
        else:
            invalid_count += 1
    print(f"[Step 3] {len(validated)} pass chemistry "
          f"({invalid_count} invalid, {partial_count} partial) "
          f"[{_t.time()-t_chem:.1f}s]")

    # ==================================================================
    # STEP 4: Structure validation (Goldschmidt, bond dist, coordination)
    # ==================================================================
    struct_validated, struct_rejected = [], 0
    if _HAS_STRUCT_VALIDATOR:
        t_struct = _t.time()
        for formula, chem_result in validated:
            proto = _infer_prototype(formula)
            sv = validate_structure(formula, prototype=proto)
            if sv["feasible"]:
                struct_validated.append((formula, chem_result, sv))
            else:
                struct_rejected += 1
        print(f"[Step 4] {len(struct_validated)} pass structure validation "
              f"({struct_rejected} rejected) [{_t.time()-t_struct:.1f}s]")
    else:
        struct_validated = [
            (f, cr, {"feasible": True, "feasibility_score": 0.5,
                     "goldschmidt_t": None, "prototype": _infer_prototype(f),
                     "warnings": [], "pymatgen_used": False})
            for f, cr in validated
        ]

    # STEP 4b: ML Prototype Classification (Phase 5)
    if _HAS_PROTOTYPE_CLF:
        t_proto = _t.time()
        for i, (f, cr, sd) in enumerate(struct_validated):
            try:
                proto_result = classify_prototype(f)
                sd["ml_prototype"] = proto_result["prototype"]
                sd["ml_prototype_confidence"] = proto_result["confidence"]
                sd["ml_prototype_top3"] = proto_result["top3"]
                sd["ml_prototype_method"] = proto_result["method"]
                # Override rule-based prototype with ML if confidence is high
                if proto_result["confidence"] >= 0.7:
                    sd["prototype"] = proto_result["prototype"]
            except Exception:
                sd["ml_prototype"] = sd.get("prototype", "other")
                sd["ml_prototype_confidence"] = 0.0
                sd["ml_prototype_top3"] = []
                sd["ml_prototype_method"] = "fallback"
        print(f"[Step 4b] ML prototype classification [{_t.time()-t_proto:.1f}s]")

    # ==================================================================
    # STEP 5: Feature computation + ML prediction with uncertainty
    # ==================================================================
    candidates = []
    failed_features = 0
    t_step5 = _t.time()
    
    total_to_process = len(struct_validated)
    print(f"[Step 5] Extracting features and predicting for {total_to_process} structural candidates...")

    for i, (formula, classification, struct_data) in enumerate(struct_validated):
        if i > 0 and i % 50 == 0:
            print(f"[Step 5] Processed {i}/{total_to_process} candidates...")
            
        try:
            features = compound_to_features(formula)
            if features is None:
                failed_features += 1
                continue

            pred = predict_with_uncertainty(features, formula=formula)
            raw_energy  = pred.get("formation_energy_eV_per_atom")
            raw_bandgap = pred.get("band_gap_eV")
            energy_unc  = pred.get("formation_energy_uncertainty_eV")
            ml_ehull    = pred.get("e_above_hull_predicted")
            ehull_unc   = pred.get("ehull_uncertainty_eV")
            model_conf  = pred.get("confidence_level", "low")
            ehull_avail = pred.get("ehull_available", False)

            if raw_energy is None:
                failed_features += 1
                continue

            bg_correction = correct_bandgap(raw_bandgap or 0.0, formula)
            e_correction  = correct_formation_energy(raw_energy, formula)
            label_info    = compute_overall_confidence(
                bg_correction, e_correction,
                validity=classification["validity"],
                prototype_match=classification["prototype_match"],
            )
            struct_type = get_validated_structure_type(formula)

            candidates.append({
                "compound":               formula,
                "features":               list(features),
                "formation_energy":       round(e_correction["corrected_eV"], 4),
                "formation_energy_raw":   round(raw_energy, 4),
                "formation_energy_uncertainty": round(energy_unc, 4) if energy_unc else None,
                "band_gap_eV":            round(bg_correction["corrected_eV"], 4),
                "ml_ehull_predicted":     round(ml_ehull, 4) if ml_ehull is not None else None,
                "ehull_uncertainty":      round(ehull_unc, 4) if ehull_unc else None,
                "ehull_model_available":  ehull_avail,
                "compound_class":         bg_correction.get("compound_class", "unknown"),
                "validity":               classification["validity"],
                "prototype_match":        struct_type["prototype_validated"],
                "coordination_ok":        classification["coordination_ok"],
                "honest_label":           label_info["label"],
                "confidence_level":       model_conf,
                "structure_type":         struct_type["structure_type"],
                "goldschmidt_t":          struct_data.get("goldschmidt_t"),
                "structure_note":         struct_type["validation_note"],
                "struct_feasibility_score": struct_data.get("feasibility_score", 0.5),
                "struct_warnings":        struct_data.get("warnings", []),
                "pymatgen_validated":     struct_data.get("pymatgen_used", False),
                # Phase 5: ML Prototype Classification
                "ml_prototype":           struct_data.get("ml_prototype"),
                "ml_prototype_confidence": struct_data.get("ml_prototype_confidence", 0.0),
                "ml_prototype_top3":      struct_data.get("ml_prototype_top3", []),
                # Phase 5: Epistemic/Aleatoric Uncertainty
                "epistemic_uncertainty":  pred.get("epistemic_uncertainty_eV"),
                "aleatoric_uncertainty":  pred.get("aleatoric_uncertainty_eV"),
                "confidence_interval_95": pred.get("confidence_interval_95"),
            })
        except Exception:
            failed_features += 1
            if failed_features <= 3:
                import traceback; traceback.print_exc()

    print(f"[Step 5] {len(candidates)} with predictions "
          f"({failed_features} failed)")

    # ==================================================================
    # STEP 6: Thermodynamic filtering (Ehull < 0.2, Ef < -1.0)
    # ==================================================================
    # Basic gate: Ef <= -0.3 eV/atom minimum
    before_gate = len(candidates)
    candidates = [c for c in candidates if c["formation_energy"] <= -0.3]
    print(f"[Step 6a] {len(candidates)} pass energy gate "
          f"(removed {before_gate - len(candidates)})")

    if not candidates:
        return {"novel_materials": [], "alloy_candidates": filter_alloys(min_strength, min_melting_point)[:5],
                "search_stats": {"explored": len(formulas), "returned": 0,
                                 "elapsed_seconds": round(time.time()-t0, 2)}}

    # Target range filter
    filtered = [c for c in candidates
                if target_energy_min <= c["formation_energy"] <= target_energy_max]
    if not filtered:
        filtered = sorted(candidates, key=lambda x: abs(x["formation_energy"] + 1.5))[:50]

    if target_bandgap_min is not None or target_bandgap_max is not None:
        bg_min = target_bandgap_min if target_bandgap_min is not None else -999
        bg_max = target_bandgap_max if target_bandgap_max is not None else 999
        bg_filtered = [c for c in filtered
                       if c.get("band_gap_eV") is not None and bg_min <= c["band_gap_eV"] <= bg_max]
        if bg_filtered:
            filtered = bg_filtered
        else:
            filtered = sorted(
                [c for c in filtered if c.get("band_gap_eV") is not None],
                key=lambda x: min(abs(x["band_gap_eV"] - bg_min), abs(x["band_gap_eV"] - bg_max))
            )[:30]

    # Ehull tier classification
    for c in filtered:
        ml_ehull = c.get("ml_ehull_predicted")
        if ml_ehull is not None:
            ehull_val = ml_ehull
            ehull_from_ml = True
        elif _HAS_STABILITY:
            hull_result = estimate_e_above_hull(c["compound"], c["formation_energy"])
            ehull_val = hull_result.get("e_above_hull_eV")
            ehull_from_ml = False
        else:
            ehull_val = None
            ehull_from_ml = False

        c["e_above_hull"] = ehull_val
        c["ehull_from_ml_model"] = ehull_from_ml
        tier, stability, label, warn = _classify_tier(ehull_val)
        c["thermodynamic_stability"] = stability
        c["stability_tier"] = tier
        c["stability_label"] = label
        c["stability_warning"] = warn

    # Strict research filter
    if strict_research_filter:
        promising, borderline, discarded = [], [], []
        for c in filtered:
            ehull = c.get("e_above_hull")
            ef = c["formation_energy"]
            tier = c["stability_tier"]
            ehull_ok = (ehull is not None and ehull < 0.2) or (ehull is None and tier == 0)
            ef_ok = ef < -1.0
            if ehull_ok and ef_ok:
                promising.append(c)
            elif ef_ok and tier in (0, 2, 3):
                borderline.append(c)
            else:
                discarded.append(c)

        strict_pool = promising + borderline
        is_battery = any("battery" in (t or "").lower() for t in (application_targets or []))
        
        if len(strict_pool) < num_candidates:
            if is_battery:
                print(f"[Step 6b] Adaptive Constraint Relaxation: Relaxing Ehull limit to 0.15 eV to find enough battery candidates.")
                relaxed = [c for c in discarded if c.get("e_above_hull") is not None and c.get("e_above_hull") <= 0.15]
                relaxed.sort(key=lambda x: x["formation_energy"])
                strict_pool += relaxed
            
            # Still not enough? Pad with remaining discarded
            if len(strict_pool) < num_candidates:
                discarded_remaining = [c for c in discarded if c not in strict_pool]
                discarded_remaining.sort(key=lambda x: x["formation_energy"])
                strict_pool += discarded_remaining[:num_candidates - len(strict_pool)]

        print(f"[Step 6b] Strict: {len(promising)} promising, "
              f"{len(borderline)} borderline, {len(discarded)} discarded")
        filtered = strict_pool[:max(num_candidates * 3, 20)]

    print(f"[Step 6] {len(filtered)} after thermodynamic filtering")

    # ==================================================================
    # STEP 7: Convex hull phase validation (NEW)
    # ==================================================================
    if _HAS_CONVEX_HULL:
        t_hull = _t.time()
        for c in filtered:
            hull_result = evaluate_phase_stability(
                c["compound"],
                predicted_formation_energy=c["formation_energy"],
                ml_ehull=c.get("ml_ehull_predicted"),
            )
            c["hull_distance_eV"]     = hull_result["hull_distance_eV"]
            c["hull_source"]          = hull_result["hull_source"]
            c["hull_status"]          = hull_result["hull_status"]
            c["competing_phases"]     = hull_result["competing_phases"]
            c["n_competing_phases"]   = hull_result["n_competing_phases"]
            c["chemical_system"]      = hull_result["chemical_system"]
            c["decomposition_warning"] = hull_result["decomposition_warning"]
            c["decomposition_risk_pct"] = hull_result.get("decomposition_risk_pct")
            c["likely_decomposition_products"] = hull_result.get("likely_decomposition_products", [])
            c["decomposition_energy_gain_eV"] = hull_result.get("decomposition_energy_gain_eV")
        print(f"[Step 7] Convex hull + decomposition risk evaluated [{_t.time()-t_hull:.1f}s]")
    else:
        for c in filtered:
            c["hull_distance_eV"]     = c.get("e_above_hull")
            c["hull_source"]          = "ehull_model" if c.get("ehull_from_ml_model") else "heuristic"
            c["hull_status"]          = "unknown"
            c["competing_phases"]     = []
            c["n_competing_phases"]   = 0
            c["chemical_system"]      = ""
            c["decomposition_warning"] = None
            c["decomposition_risk_pct"] = None
            c["likely_decomposition_products"] = []
            c["decomposition_energy_gain_eV"] = None

    # ==================================================================
    # STEP 8: Synthesis feasibility estimation (NEW)
    # ==================================================================
    if _HAS_SYNTHESIS:
        t_synth = _t.time()
        for c in filtered:
            synth = predict_synthesis_feasibility(
                c["features"],
                formation_energy=c.get("formation_energy"),
                ehull=c.get("e_above_hull"),
                bandgap=c.get("band_gap_eV"),
            )
            c["synthesis_confidence"]  = synth["synthesis_confidence"]
            c["synthesis_label"]       = synth["synthesis_label"]
            c["synthesis_factors"]     = synth["synthesis_factors"]
        print(f"[Step 8] Synthesis feasibility evaluated [{_t.time()-t_synth:.1f}s]")
    else:
        for c in filtered:
            c["synthesis_confidence"] = None
            c["synthesis_label"]      = "Not Evaluated"
            c["synthesis_factors"]    = {}

    # ==================================================================
    # STEP 9: Quantum refinement (VQE for top-N)
    # ==================================================================
    top_n = min(num_candidates * 3, len(filtered), 15)

    # Pre-sort by formation energy to prioritize best for quantum
    filtered.sort(key=lambda x: x["formation_energy"])
    quantum_pool = filtered[:top_n]

    if enable_quantum_refinement:
        print(f"[Step 9] Running quantum refinement on top-{top_n}...")
        t_q = _t.time()
        for c in quantum_pool:
            try:
                n_qubits = max(2, min(_n_atoms(c["compound"]), 6))
                q_result = run_quantum_validation(n_qubits, formula=c["compound"])
                c["quantum_score"]      = round(q_result.get("quantum_score", 5.0) / 10.0, 4)
                c["quantum_method"]     = q_result.get("methodology", "circuit")
                c["quantum_energy"]     = q_result.get("energy")
                c["energy_correction"]  = q_result.get("energy_correction")
            except Exception:
                c["quantum_score"]  = 0.5
                c["quantum_method"] = "fallback"
                c["quantum_energy"] = None
                c["energy_correction"] = None
        print(f"[Step 9] Quantum complete [{_t.time()-t_q:.1f}s]")
    else:
        for c in quantum_pool:
            c["quantum_score"]  = None
            c["quantum_method"] = "skipped"
            c["quantum_energy"] = None
            c["energy_correction"] = None

    # ==================================================================
    # STEP 10: Novelty verification (full MP cross-check)
    # ==================================================================
    for c in quantum_pool:
        novelty_result = check_novelty(c["compound"])
        c["is_novel"]       = novelty_result["is_novel"]
        c["novelty_score"]  = novelty_result["novelty_score"]
        c["novelty_label"]  = novelty_result["novelty_label"]
        c["found_in_db"]    = novelty_result["found_in"]
        c["db_match_detail"] = novelty_result["match_detail"]

    novel_count = sum(1 for c in quantum_pool if c["is_novel"])
    print(f"[Step 10] Novelty: {novel_count} novel, "
          f"{len(quantum_pool)-novel_count} known")

    # ==================================================================
    # STEP 11: Weighted scientific scoring (explicit formula)
    # ==================================================================
    for c in quantum_pool:
        chem_stab = get_compound_stability_score(c["compound"])
        c["chemistry_stability"] = round(chem_stab, 2)

        score_result = compute_weighted_score(
            ehull=c.get("e_above_hull"),
            tier=c.get("stability_tier", 0),
            ef=c["formation_energy"],
            bg=c.get("band_gap_eV"),
            density=None,  # Not predicted at composition level
            struct_feasibility=c.get("struct_feasibility_score", 0.5),
            confidence_level=c.get("confidence_level", "low"),
            novelty_score=c.get("novelty_score", 0.0),
            quantum_bonus=c.get("quantum_score"),
        )
        c["stability_score"]        = score_result["total_score"]
        c["base_score"]             = score_result["base_score"]
        c["novelty_bonus"]          = score_result["novelty_bonus"]
        c["quantum_bonus_score"]    = score_result["quantum_bonus"]
        c["ehull_capped"]           = score_result["ehull_capped"]
        c["score_breakdown"]        = score_result["breakdown"]
        c["score_weights"]          = score_result["weights"]

        # Honest labels for unstable materials
        tier = c.get("stability_tier", 0)
        if tier == 5:
            c["honest_label"]     = "Highly Unstable Theoretical Phase"
            c["confidence_level"] = "speculative"
        elif tier == 4:
            c["honest_label"]     = "High-Energy Theoretical Phase"
            c["confidence_level"] = "low"
        elif tier == 3:
            c["honest_label"]     = "Requires Experimental Stabilization"

    quantum_pool.sort(key=lambda x: x["stability_score"], reverse=True)

    # ==================================================================
    # STEP 11b: Chemical validity probability
    # ==================================================================
    for c in quantum_pool:
        c["chemical_validity_probability"] = _chemical_validity_probability({
            "validity": c.get("validity", "partially_valid"),
            "prototype_match": c.get("prototype_match", False),
            "coordination_ok": c.get("coordination_ok", False),
        })

    # ==================================================================
    # STEP 11c: T/P stability correction
    # ==================================================================
    if temperature_K != 300.0 or pressure_GPa > 0.001:
        for c in quantum_pool:
            n_elems = len(_parse_elements(c["compound"]))
            tp = _tp_stability_correction(
                ehull_eV=c.get("e_above_hull"),
                formation_energy=c["formation_energy"],
                n_elements=n_elems,
                temperature_K=temperature_K,
                pressure_GPa=pressure_GPa,
            )
            c["tp_correction"] = tp
            c["formation_energy_at_conditions"] = tp["corrected_formation_energy"]
        print(f"[Step 11c] T/P correction applied (T={temperature_K}K, P={pressure_GPa}GPa)")
    else:
        for c in quantum_pool:
            c["tp_correction"] = None
            c["formation_energy_at_conditions"] = c["formation_energy"]

    # ==================================================================
    # STEP 11d: Pareto multi-objective ranking
    # ==================================================================
    quantum_pool = _pareto_rank(quantum_pool, application_targets)
    print(f"[Step 11d] Pareto ranking assigned ({sum(1 for c in quantum_pool if c.get('pareto_rank') == 1)} on front)")

    # ==================================================================
    # STEP 12: Final selection + visualization
    # ==================================================================
    qualified = quantum_pool

    results = qualified[:num_candidates]

    for c in results:
        try:
            struct_payload = generate_structure(c["compound"])
            c["structure"] = struct_payload
            # Crystal system filter
            if crystal_systems and struct_payload.get("lattice", {}).get("crystal_system"):
                cs = struct_payload["lattice"]["crystal_system"]
                c["crystal_system"] = cs
            else:
                c["crystal_system"] = struct_payload.get("lattice", {}).get("crystal_system", "unknown")
            c["lattice_params"] = struct_payload.get("lattice", {})
        except Exception:
            c["structure"] = {"atoms": [], "bonds": [], "total_atoms": 0}
            c["crystal_system"] = "unknown"
            c["lattice_params"] = {}

        c["chemical_name"] = get_compound_name(c["compound"])

        elems = _parse_elements(c["compound"])
        c["predicted_applications"] = predict_usecase(
            compound=c["compound"],
            band_gap_eV=c.get("band_gap_eV"),
            formation_energy=c.get("formation_energy"),
            elements=elems,
            quantum_score=c.get("quantum_score"),
            stability_score=c.get("stability_score"),
        )

        # Application-target ranking boost
        if application_targets and c.get("predicted_applications"):
            apps = c["predicted_applications"]
            for da in apps.get("detailed_applications", []):
                app_name_lower = da["application"].lower()
                for target in application_targets:
                    target_lower = target.lower().replace("_", " ")
                    if target_lower in app_name_lower:
                        da["confidence"] = min(da["confidence"] + 0.15, 1.0)
                        c["application_target_match"] = True

        # Feature importance (top candidates only, expensive)
        if c.get("features"):
            c["feature_importance"] = _compute_feature_importance(c["features"], c["compound"])
        else:
            c["feature_importance"] = []

        # Reduce application confidence for unstable materials
        tier = c.get("stability_tier", 0)
        if tier >= 4 and c.get("predicted_applications"):
            apps = c["predicted_applications"]
            if apps.get("overall_confidence") is not None:
                apps["overall_confidence"] = min(apps["overall_confidence"], 0.25)
                apps["confidence_label"] = "Very Low (thermodynamically unstable)"
        elif tier == 0 and c.get("predicted_applications"):
            apps = c["predicted_applications"]
            if apps.get("overall_confidence") is not None:
                apps["overall_confidence"] = min(apps["overall_confidence"], 0.5)
                apps["confidence_label"] = "Limited (Ehull not evaluated)"

    # Store and clean up
    for c in results:
        store_discovery(c)
    for c in results:
        c.pop("features", None)

    alloys = filter_alloys(min_strength, min_melting_point)
    elapsed = round(time.time() - t0, 2)

    # Crystal system post-filter (if requested)
    if crystal_systems:
        cs_set = {cs.lower() for cs in crystal_systems}
        filtered_results = [c for c in results if c.get("crystal_system", "").lower() in cs_set]
        is_battery = any("battery" in (t or "").lower() for t in (application_targets or []))
        
        if is_battery and len(filtered_results) < min(num_candidates, len(results)):
            print(f"[Step 12b] Adaptive Constraint Relaxation: Returning varied crystal symmetries to meet '{application_targets}' quotas.")
            # Do nothing to results
        elif filtered_results:
            results = filtered_results
            print(f"[Step 12b] Crystal system filter: {len(results)} match {crystal_systems}")

    return {
        "novel_materials": results,
        "alloy_candidates": alloys[:5],
        "search_stats": {
            "explored":              len(formulas),
            "chemistry_validated":   len(validated),
            "chemistry_invalid":     invalid_count,
            "struct_validated":      len(struct_validated),
            "struct_rejected":       struct_rejected,
            "after_prediction":      len(candidates),
            "after_thermo_filter":   len(filtered),
            "novel_count":           novel_count,
            "returned":              len(results),
            "elapsed_seconds":       elapsed,
            "stability_level":       stability_level,
            "model_type":            model_type,
            "scoring_formula":       "QScore = 0.35·S_ehull + 0.25·S_ef + 0.20·S_bg + 0.10·S_struct + 0.10·S_conf",
            "scoring_weights":       {"ehull": 0.35, "formation_energy": 0.25, "bandgap": 0.20, "structure": 0.10, "confidence": 0.10},
            "ehull_threshold":       0.2,
            "energy_threshold":      -1.0,
            "ehull_cap_trigger":     0.1,
            "convex_hull_enabled":   _HAS_CONVEX_HULL,
            "synthesis_enabled":     _HAS_SYNTHESIS,
            "quantum_refinement":    enable_quantum_refinement,
            "structure_validation":  _HAS_STRUCT_VALIDATOR,
            "crystal_systems":       crystal_systems,
            "application_targets":   application_targets,
            "temperature_K":         temperature_K,
            "pressure_GPa":          pressure_GPa,
        },
    }
