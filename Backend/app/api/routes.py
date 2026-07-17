"""
Flask API Routes — Material Discovery Endpoints

POST /discover   — Run inverse-design discovery pipeline
POST /simulate   — Run AI screening + VQE quantum simulation for a compound
GET  /health     — Health check
GET  /elements   — List loaded elements and available groups
GET  /model-info — Model metadata and accuracy metrics
GET  /export/<format> — Export structure as CIF / POSCAR / XYZ
GET  /molecule/pubchem-structure — Fetch real 3D structure from PubChem
"""

from flask import Blueprint, request, jsonify, Response
import numpy as np

api = Blueprint("api", __name__)


@api.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "AI + Quantum Materials Discovery Platform",
    })


def _json_safe(obj):
    """Recursively convert numpy scalars/arrays into JSON-safe Python types."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# -----------------------------------------------------------------------
# Model Info Endpoint
# -----------------------------------------------------------------------

@api.route("/model-info", methods=["GET"])
def model_info():
    """Return model metadata and accuracy metrics."""
    try:
        from app.services.prediction_service import get_model_info
        info = get_model_info()
        return jsonify(_json_safe(info))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------------------------------------------------
# Material Discovery
# -----------------------------------------------------------------------

@api.route("/discover", methods=["POST"])
def discover():
    """
    Inverse-design material discovery.

    Accepts target property ranges and returns genuinely novel materials
    with AI predictions, quantum validation, novelty scores, 3D structures,
    and predicted use-cases.

    Request JSON:
    {
        "target_energy_min": -4.0,
        "target_energy_max": -0.3,
        "target_bandgap_min": null,
        "target_bandgap_max": null,
        "max_atoms": 10,
        "min_stability": 5.0,
        "element_groups": ["transition_metals", "nonmetals"],
        "include_elements": ["Ti", "O"],
        "exclude_elements": ["Pb"],
        "num_candidates": 5,
        "min_strength": 500,
        "min_melting_point": 1200,
        "stability_level": "high"
    }
    """
    try:
        data = request.get_json(silent=True) or {}

        target_energy_min = float(data.get("target_energy_min", -4.0))
        target_energy_max = float(data.get("target_energy_max", -0.3))
        target_bandgap_min = data.get("target_bandgap_min")
        target_bandgap_max = data.get("target_bandgap_max")
        if target_bandgap_min is not None:
            target_bandgap_min = float(target_bandgap_min)
        if target_bandgap_max is not None:
            target_bandgap_max = float(target_bandgap_max)

        max_atoms = int(data.get("max_atoms", 10))
        min_stability = float(data.get("min_stability", 5.0))
        element_groups = data.get("element_groups", None)
        include_elements = data.get("include_elements", None)
        exclude_elements = data.get("exclude_elements", None)
        num_candidates = int(data.get("num_candidates", 5))
        min_strength = float(data.get("min_strength", 500))
        min_melting_point = float(data.get("min_melting_point", 1200))
        stability_level = str(data.get("stability_level", "high")).lower()
        if stability_level not in ("high", "medium", "low"):
            stability_level = "high"

        num_candidates = max(1, min(num_candidates, 20))
        max_atoms = max(2, min(max_atoms, 25))

        # Research-grade params (Phase 6)
        crystal_systems = data.get("crystal_systems", None)
        application_targets = data.get("application_targets", None)
        temperature_K = float(data.get("temperature_K", 300.0))
        pressure_GPa = float(data.get("pressure_GPa", 0.0001))

        from app.services.discovery_service import discover_materials
        result = discover_materials(
            target_energy_min=target_energy_min,
            target_energy_max=target_energy_max,
            target_bandgap_min=target_bandgap_min,
            target_bandgap_max=target_bandgap_max,
            max_atoms=max_atoms,
            min_stability=min_stability,
            element_groups=element_groups,
            include_elements=include_elements,
            exclude_elements=exclude_elements,
            num_candidates=num_candidates,
            min_strength=min_strength,
            min_melting_point=min_melting_point,
            stability_level=stability_level,
            crystal_systems=crystal_systems,
            application_targets=application_targets,
            temperature_K=temperature_K,
            pressure_GPa=pressure_GPa,
        )
        return jsonify(_json_safe(result))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# -----------------------------------------------------------------------
# AI + Quantum Simulation
# -----------------------------------------------------------------------

@api.route("/simulate", methods=["POST"])
def simulate():
    """
    Run AI screening + real VQE quantum simulation for a compound.

    Request JSON:
    {
        "formula": "TiO2",
        "n_qubits": 4,
        "max_iterations": 30,
        "simulation_type": "electronic"
    }
    """
    try:
        data = request.get_json(silent=True) or {}
        formula = data.get("formula", "").strip()
        if not formula:
            return jsonify({"error": "formula is required"}), 400

        n_qubits = max(2, min(int(data.get("n_qubits", 3)), 6))
        max_iterations = max(10, min(int(data.get("max_iterations", 30)), 60))

        from app.utils.feature_engine import compound_to_features
        from app.services.prediction_service import predict_with_uncertainty, get_model_info
        from app.services.usecase_predictor import predict_usecase
        from app.services.structure_generator import generate_structure
        from app.services.vqe_service import run_vqe
        from app.services.convex_hull_estimator import evaluate_phase_stability
        import re

        features = compound_to_features(formula)
        if features is None:
            return jsonify({"error": f"Cannot parse formula: {formula}"}), 400

        # Step 1: AI Screening with uncertainty
        predictions = predict_with_uncertainty(features, formula=formula)

        # Step 1.5: Evaluate Phase Stability & Convex Hull
        hull_result = evaluate_phase_stability(
            formula,
            predicted_formation_energy=predictions.get("formation_energy_eV_per_atom"),
            ml_ehull=predictions.get("e_above_hull_predicted"),
        )
        predictions.update(hull_result)

        # Step 2: Generate 3D structure (now with CIF)
        structure = generate_structure(formula)

        # Step 3: Run VQE simulation (now element-aware)
        elements = re.findall(r'[A-Z][a-z]?', formula)
        vqe_result = run_vqe(
            n_qubits=n_qubits,
            max_iterations=max_iterations,
            formula=formula,
        )
        quantum_energy = vqe_result.get("energy", 0.0)

        # Step 4: Predict applications
        applications = predict_usecase(
            compound=formula,
            band_gap_eV=predictions.get("band_gap_eV"),
            formation_energy=predictions.get("formation_energy_eV_per_atom"),
            elements=elements,
            quantum_score=abs(quantum_energy),
        )

        # Step 5: Model info
        model_meta = get_model_info()

        return jsonify(_json_safe({
            "formula": formula,
            "ai_screening": {
                **predictions,
                "is_stable": (predictions.get("formation_energy_eV_per_atom") or 0) < -0.5,
                "feature_count": len(features),
                "model_type": model_meta.get("model_type", "Unknown"),
            },
            "quantum_simulation": vqe_result,
            "predicted_applications": applications,
            "structure": structure,
            "model_info": model_meta,
        }))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# -----------------------------------------------------------------------
# Elements
# -----------------------------------------------------------------------

@api.route("/elements", methods=["GET"])
def elements():
    """Return all loaded elements and available element groups."""
    try:
        from app.utils.element_loader import load_all_elements
        from app.services.composition_explorer import ELEMENT_GROUPS, AVAILABLE_GROUPS

        loaded = load_all_elements()
        return jsonify(_json_safe({
            "count": len(loaded),
            "symbols": sorted(loaded.keys()),
            "groups": {
                name: [s for s in symbols if s in loaded]
                for name, symbols in ELEMENT_GROUPS.items()
            },
            "available_groups": AVAILABLE_GROUPS,
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------------------------------------------------
# Structure Export (CIF / POSCAR / XYZ)
# -----------------------------------------------------------------------

@api.route("/export/<fmt>", methods=["GET"])
def export_structure(fmt):
    """
    Export a structure in CIF, POSCAR, or XYZ format.

    Query params:
        formula: chemical formula (e.g. TiO2)
        format:  cif | poscar | xyz
    """
    from app.services.structure_generator import export_cif, export_poscar, export_xyz

    formula = request.args.get("formula", "").strip()
    if not formula:
        return jsonify({"error": "Missing 'formula' parameter"}), 400

    fmt = fmt.lower()
    if fmt == "cif":
        content = export_cif(formula)
        mime = "chemical/x-cif"
        ext = "cif"
    elif fmt == "poscar":
        content = export_poscar(formula)
        mime = "text/plain"
        ext = "poscar"
    elif fmt == "xyz":
        content = export_xyz(formula)
        mime = "chemical/x-xyz"
        ext = "xyz"
    else:
        return jsonify({"error": f"Unknown format: {fmt}. Use cif, poscar, or xyz"}), 400

    if content is None:
        return jsonify({"error": f"Cannot generate structure for {formula}"}), 400

    return Response(
        content,
        mimetype=mime,
        headers={"Content-Disposition": f"attachment; filename={formula}.{ext}"},
    )


# -----------------------------------------------------------------------
# PubChem 3D Structure Lookup
# -----------------------------------------------------------------------
_pubchem_cache: dict[str, str | None] = {}


@api.route("/molecule/pubchem-structure", methods=["GET"])
def pubchem_structure():
    """
    Fetch a real 3D molecular structure from PubChem.

    Query params:
        formula: molecular formula (e.g. C6H6)
    """
    import requests

    formula = request.args.get("formula", "").strip()
    if not formula:
        return jsonify({"error": "Missing 'formula' parameter"}), 400

    # Check cache first
    if formula in _pubchem_cache:
        sdf = _pubchem_cache[formula]
        return jsonify({
            "sdf": sdf,
            "source": "pubchem" if sdf else "none",
            "formula": formula,
        })

    try:
        # Step 1: Formula → CID (compound ID)
        cid_url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
            f"/fastformula/{formula}/cids/JSON"
        )
        cid_resp = requests.get(cid_url, timeout=10)

        if cid_resp.status_code != 200:
            _pubchem_cache[formula] = None
            return jsonify({"sdf": None, "source": "none", "formula": formula})

        cid_data = cid_resp.json()
        cids = cid_data.get("IdentifierList", {}).get("CID", [])
        if not cids:
            _pubchem_cache[formula] = None
            return jsonify({"sdf": None, "source": "none", "formula": formula})

        cid = cids[0]

        # Step 2: CID → 3D SDF file
        sdf_url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
            f"/cid/{cid}/SDF?record_type=3d"
        )
        sdf_resp = requests.get(sdf_url, timeout=10)

        if sdf_resp.status_code != 200:
            sdf_url_2d = (
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
                f"/cid/{cid}/SDF"
            )
            sdf_resp = requests.get(sdf_url_2d, timeout=10)
            if sdf_resp.status_code != 200:
                _pubchem_cache[formula] = None
                return jsonify({"sdf": None, "source": "none", "formula": formula})

        sdf_text = sdf_resp.text
        _pubchem_cache[formula] = sdf_text

        return jsonify({
            "sdf": sdf_text,
            "source": "pubchem",
            "formula": formula,
            "cid": cid,
        })

    except requests.exceptions.Timeout:
        return jsonify({"sdf": None, "source": "none", "formula": formula,
                        "note": "PubChem timeout"})
    except Exception as e:
        return jsonify({"sdf": None, "source": "none", "formula": formula,
                        "note": str(e)})
