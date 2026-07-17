"""
Model Validation Script
========================
Compares our AI model predictions against experimentally known values
for well-studied compounds.

Reference data sources:
  - Materials Project (mp-xxx)
  - NIST / CRC Handbook
  - Literature values
"""

import requests
import json
import sys
import time

API = "http://localhost:5000"

# -----------------------------------------------------------------------
# Ground-truth reference data for well-known compounds
# Formation energy: eV/atom (from Materials Project)
# Band gap: eV (experimental, from literature)
# Expected applications: what the compound is actually used for
# -----------------------------------------------------------------------
REFERENCE_COMPOUNDS = [
    {
        "formula": "TiO2",
        "name": "Titanium Dioxide (Rutile)",
        "ref_formation_energy": -3.39,   # MP: mp-2657
        "ref_band_gap": 3.03,            # Experimental (rutile)
        "ref_applications": ["Photocatalysis", "Solar Cell", "Pigment", "UV Protection"],
        "ref_stability": "stable",
        "source": "Materials Project mp-2657",
    },
    {
        "formula": "SiO2",
        "name": "Silicon Dioxide (Quartz)",
        "ref_formation_energy": -3.18,   # MP: mp-6930
        "ref_band_gap": 8.9,             # Experimental
        "ref_applications": ["Insulator", "Glass", "Optical Fiber"],
        "ref_stability": "stable",
        "source": "Materials Project mp-6930",
    },
    {
        "formula": "GaN",
        "name": "Gallium Nitride",
        "ref_formation_energy": -1.24,   # MP: mp-804
        "ref_band_gap": 3.4,             # Experimental (direct gap)
        "ref_applications": ["LED", "Power Electronics", "Laser Diode"],
        "ref_stability": "stable",
        "source": "Materials Project mp-804",
    },
    {
        "formula": "NaCl",
        "name": "Sodium Chloride (Rock Salt)",
        "ref_formation_energy": -3.29,   # MP: mp-22862
        "ref_band_gap": 8.5,             # Experimental
        "ref_applications": ["Insulator", "Ionic Conductor"],
        "ref_stability": "stable",
        "source": "Materials Project mp-22862",
    },
    {
        "formula": "ZnO",
        "name": "Zinc Oxide (Wurtzite)",
        "ref_formation_energy": -1.78,   # MP: mp-2133
        "ref_band_gap": 3.37,            # Experimental
        "ref_applications": ["Transparent Conductor", "Varistor", "Piezoelectric", "UV LED"],
        "ref_stability": "stable",
        "source": "Materials Project mp-2133",
    },
    {
        "formula": "Fe2O3",
        "name": "Iron(III) Oxide (Hematite)",
        "ref_formation_energy": -2.67,   # MP: mp-19770
        "ref_band_gap": 2.1,             # Experimental
        "ref_applications": ["Photocatalysis", "Magnetic Material", "Pigment"],
        "ref_stability": "stable",
        "source": "Materials Project mp-19770",
    },
    {
        "formula": "LiFePO4",
        "name": "Lithium Iron Phosphate",
        "ref_formation_energy": -3.07,   # MP: mp-19017
        "ref_band_gap": 3.8,             # Experimental
        "ref_applications": ["Battery Cathode", "Energy Storage"],
        "ref_stability": "stable",
        "source": "Materials Project mp-19017",
    },
    {
        "formula": "BaTiO3",
        "name": "Barium Titanate (Perovskite)",
        "ref_formation_energy": -3.24,   # MP: mp-5020
        "ref_band_gap": 3.2,             # Experimental
        "ref_applications": ["Piezoelectric", "Capacitor", "Ferroelectric"],
        "ref_stability": "stable",
        "source": "Materials Project mp-5020",
    },
    {
        "formula": "GaAs",
        "name": "Gallium Arsenide",
        "ref_formation_energy": -0.47,   # MP: mp-2534
        "ref_band_gap": 1.42,            # Experimental (direct gap)
        "ref_applications": ["Solar Cell", "LED", "Laser Diode", "High-speed Electronics"],
        "ref_stability": "stable",
        "source": "Materials Project mp-2534",
    },
    {
        "formula": "CdTe",
        "name": "Cadmium Telluride",
        "ref_formation_energy": -0.42,   # MP: mp-406
        "ref_band_gap": 1.44,            # Experimental
        "ref_applications": ["Solar Cell", "Infrared Detector", "Radiation Detector"],
        "ref_stability": "stable",
        "source": "Materials Project mp-406",
    },
]


def test_compound(compound: dict) -> dict:
    """Call /simulate for a single compound and compare."""
    formula = compound["formula"]
    print(f"\n{'='*60}")
    print(f"  Testing: {formula} — {compound['name']}")
    print(f"  Source:  {compound['source']}")
    print(f"{'='*60}")

    try:
        resp = requests.post(
            f"{API}/simulate",
            json={
                "formula": formula,
                "n_qubits": 4,
                "max_iterations": 20,
                "simulation_type": "electronic",
            },
            timeout=300,
        )
        if resp.status_code != 200:
            print(f"  ERROR: HTTP {resp.status_code}")
            return {"formula": formula, "status": "HTTP_ERROR", "code": resp.status_code}

        data = resp.json()
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"formula": formula, "status": "CONNECTION_ERROR", "error": str(e)}

    ai = data.get("ai_screening", {})
    apps = data.get("predicted_applications", {})
    quantum = data.get("quantum_simulation", {})

    # ---- Formation Energy Comparison ----
    pred_energy = ai.get("formation_energy_eV_per_atom")
    ref_energy = compound["ref_formation_energy"]
    energy_error = abs(pred_energy - ref_energy) if pred_energy is not None else None
    energy_pct = (energy_error / abs(ref_energy) * 100) if energy_error and ref_energy else None

    print(f"\n  Formation Energy (eV/atom):")
    print(f"    Predicted:  {pred_energy:.4f}" if pred_energy else "    Predicted:  N/A")
    print(f"    Reference:  {ref_energy:.4f}")
    if energy_error is not None:
        print(f"    Error:      {energy_error:.4f} eV ({energy_pct:.1f}%)")
        if energy_error < 0.3:
            print(f"    Verdict:    ✅ GOOD (< 0.3 eV error)")
        elif energy_error < 1.0:
            print(f"    Verdict:    ⚠️  FAIR (< 1.0 eV error)")
        else:
            print(f"    Verdict:    ❌ POOR (> 1.0 eV error)")

    # ---- Uncertainty ----
    energy_unc = ai.get("formation_energy_uncertainty")
    bandgap_unc = ai.get("band_gap_uncertainty")
    if energy_unc is not None:
        print(f"    Uncertainty: ±{energy_unc:.4f} (MC Dropout)")

    # ---- Band Gap Comparison ----
    pred_bg = ai.get("band_gap_eV")
    ref_bg = compound["ref_band_gap"]
    bg_error = abs(pred_bg - ref_bg) if pred_bg is not None else None
    bg_pct = (bg_error / ref_bg * 100) if bg_error and ref_bg else None

    print(f"\n  Band Gap (eV):")
    print(f"    Predicted:  {pred_bg:.3f}" if pred_bg else "    Predicted:  N/A")
    print(f"    Reference:  {ref_bg:.3f}")
    if bg_error is not None:
        print(f"    Error:      {bg_error:.3f} eV ({bg_pct:.1f}%)")
        if bg_error < 0.5:
            print(f"    Verdict:    ✅ GOOD (< 0.5 eV error)")
        elif bg_error < 1.5:
            print(f"    Verdict:    ⚠️  FAIR (< 1.5 eV error)")
        else:
            print(f"    Verdict:    ❌ POOR (> 1.5 eV error)")
    if bandgap_unc is not None:
        print(f"    Uncertainty: ±{bandgap_unc:.3f} (MC Dropout)")

    # ---- Application Comparison ----
    pred_apps = apps.get("primary_applications", [])
    ref_apps = compound["ref_applications"]
    # Check overlap
    pred_set = set(a.lower() for a in pred_apps)
    ref_set = set(a.lower() for a in ref_apps)
    overlap = pred_set & ref_set
    # Fuzzy: check if any reference keyword appears in any predicted app
    fuzzy_matches = 0
    for ra in ref_apps:
        for pa in pred_apps:
            if ra.lower() in pa.lower() or pa.lower() in ra.lower():
                fuzzy_matches += 1
                break

    print(f"\n  Predicted Applications:")
    print(f"    Model:      {pred_apps[:5]}")
    print(f"    Reference:  {ref_apps}")
    print(f"    Exact match: {len(overlap)}/{len(ref_apps)}")
    print(f"    Fuzzy match: {fuzzy_matches}/{len(ref_apps)}")
    if fuzzy_matches >= len(ref_apps) * 0.5:
        print(f"    Verdict:    ✅ GOOD (≥50% match)")
    elif fuzzy_matches > 0:
        print(f"    Verdict:    ⚠️  PARTIAL")
    else:
        print(f"    Verdict:    ❌ NO MATCH")

    # ---- Material Class ----
    mat_class = apps.get("material_class", "N/A")
    confidence = apps.get("overall_confidence", 0)
    conf_label = apps.get("confidence_label", "N/A")
    print(f"\n  Material Class:   {mat_class}")
    print(f"  Confidence:       {conf_label} ({confidence*100:.0f}%)" if confidence else "")

    # ---- Quantum Simulation ----
    if quantum:
        ham_type = quantum.get("hamiltonian_type", "N/A")
        q_energy = quantum.get("energy", "N/A")
        q_unit = quantum.get("energy_unit", "a.u.")
        converged = quantum.get("converged", False)
        print(f"\n  Quantum (VQE):")
        print(f"    Hamiltonian: {ham_type}")
        print(f"    Energy:      {q_energy} {q_unit}")
        print(f"    Converged:   {'✅ Yes' if converged else '❌ No'}")

    return {
        "formula": formula,
        "name": compound["name"],
        "status": "OK",
        "pred_energy": pred_energy,
        "ref_energy": ref_energy,
        "energy_error": energy_error,
        "energy_pct": energy_pct,
        "pred_bandgap": pred_bg,
        "ref_bandgap": ref_bg,
        "bandgap_error": bg_error,
        "bandgap_pct": bg_pct,
        "pred_apps": pred_apps,
        "ref_apps": ref_apps,
        "fuzzy_matches": fuzzy_matches,
        "mat_class": mat_class,
        "confidence": confidence,
    }


def print_summary(results: list):
    """Print a summary table."""
    ok = [r for r in results if r["status"] == "OK"]
    if not ok:
        print("\n❌ No successful results to summarize.")
        return

    print(f"\n\n{'#'*70}")
    print(f"  VALIDATION SUMMARY — {len(ok)} compounds tested")
    print(f"{'#'*70}")

    # Energy summary
    energy_errors = [r["energy_error"] for r in ok if r["energy_error"] is not None]
    if energy_errors:
        mae = sum(energy_errors) / len(energy_errors)
        good = sum(1 for e in energy_errors if e < 0.3)
        fair = sum(1 for e in energy_errors if 0.3 <= e < 1.0)
        poor = sum(1 for e in energy_errors if e >= 1.0)
        print(f"\n  Formation Energy:")
        print(f"    MAE:   {mae:.4f} eV/atom")
        print(f"    Good (< 0.3 eV):  {good}/{len(energy_errors)}")
        print(f"    Fair (< 1.0 eV):  {fair}/{len(energy_errors)}")
        print(f"    Poor (> 1.0 eV):  {poor}/{len(energy_errors)}")

    # Band gap summary
    bg_errors = [r["bandgap_error"] for r in ok if r["bandgap_error"] is not None]
    if bg_errors:
        mae_bg = sum(bg_errors) / len(bg_errors)
        good_bg = sum(1 for e in bg_errors if e < 0.5)
        fair_bg = sum(1 for e in bg_errors if 0.5 <= e < 1.5)
        poor_bg = sum(1 for e in bg_errors if e >= 1.5)
        print(f"\n  Band Gap:")
        print(f"    MAE:   {mae_bg:.3f} eV")
        print(f"    Good (< 0.5 eV):  {good_bg}/{len(bg_errors)}")
        print(f"    Fair (< 1.5 eV):  {fair_bg}/{len(bg_errors)}")
        print(f"    Poor (> 1.5 eV):  {poor_bg}/{len(bg_errors)}")

    # App prediction summary
    total_fuzzy = sum(r["fuzzy_matches"] for r in ok if r.get("fuzzy_matches"))
    total_ref = sum(len(r["ref_apps"]) for r in ok)
    print(f"\n  Application Prediction:")
    print(f"    Fuzzy matches: {total_fuzzy}/{total_ref} ({total_fuzzy/total_ref*100:.0f}%)" if total_ref else "")

    # Table
    print(f"\n  {'Formula':<10} {'Pred E':>8} {'Ref E':>8} {'E Err':>8} │ {'Pred BG':>8} {'Ref BG':>8} {'BG Err':>8} │ {'Apps':>5}")
    print(f"  {'─'*10} {'─'*8} {'─'*8} {'─'*8} │ {'─'*8} {'─'*8} {'─'*8} │ {'─'*5}")
    for r in ok:
        pe = f"{r['pred_energy']:.3f}" if r['pred_energy'] is not None else "N/A"
        re_ = f"{r['ref_energy']:.3f}"
        ee = f"{r['energy_error']:.3f}" if r['energy_error'] is not None else "N/A"
        pb = f"{r['pred_bandgap']:.2f}" if r['pred_bandgap'] is not None else "N/A"
        rb = f"{r['ref_bandgap']:.2f}"
        be = f"{r['bandgap_error']:.2f}" if r['bandgap_error'] is not None else "N/A"
        fm = f"{r.get('fuzzy_matches',0)}/{len(r['ref_apps'])}"
        print(f"  {r['formula']:<10} {pe:>8} {re_:>8} {ee:>8} │ {pb:>8} {rb:>8} {be:>8} │ {fm:>5}")

    print(f"\n{'#'*70}")


if __name__ == "__main__":
    print("=" * 70)
    print("  QuantumMaterialDiscovery — Model Validation Against Real Data")
    print("=" * 70)

    # Check server
    try:
        r = requests.get(f"{API}/health", timeout=5)
        print(f"  Server: {API} — OK")
    except Exception:
        print(f"  ❌ Cannot connect to {API}. Start the backend first.")
        sys.exit(1)

    results = []
    for compound in REFERENCE_COMPOUNDS:
        result = test_compound(compound)
        results.append(result)
        time.sleep(0.5)  # Be gentle with the server

    print_summary(results)
