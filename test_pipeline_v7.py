
import sys
import os
import json

sys.path.insert(0, os.path.join(os.getcwd(), 'Backend'))
os.environ["MATERIALS_PROJECT_API_KEY"] = "test"

from app.services.discovery_service import discover_materials

def test_phase5():
    print("=" * 70)
    print("  Phase 5 Verification — Generative Design & Advanced UQ (v7)")
    print("=" * 70)
    
    results = discover_materials(
        num_candidates=3,
        include_elements=['Ba', 'Ti', 'O'],
        enable_quantum_refinement=True,
        strict_research_filter=False
    )
    
    novel = results.get("novel_materials", [])
    print(f"\nDiscovered: {len(novel)} materials\n")
    
    checks = {"prototype_ok": False, "uncertainty_ok": False, "confidence_ok": False}
    
    for i, m in enumerate(novel):
        print(f"--- Candidate {i+1}: {m['compound']} ---")
        print(f"  Stability Score:        {m.get('stability_score')}%")
        print(f"  Formation Energy:       {m.get('formation_energy')} eV/atom")
        print(f"  Ehull:                  {m.get('e_above_hull')}")
        
        # Phase 4 metrics
        print(f"  Novelty:                {m.get('novelty_label')}")
        print(f"  Databases Checked:      {m.get('databases_checked', 'N/A')}")
        print(f"  Decomposition Risk:     {m.get('decomposition_risk_pct')}%")
        
        # Phase 5: Prototype Classification
        proto = m.get('ml_prototype', 'N/A')
        proto_conf = m.get('ml_prototype_confidence', 0)
        proto_top3 = m.get('ml_prototype_top3', [])
        print(f"  ML Prototype:           {proto} (conf={proto_conf:.2f})")
        if proto_top3:
            top3_str = ", ".join(f"{p['label']}:{p['prob']:.2f}" for p in proto_top3)
            print(f"  Top-3 Prototypes:       {top3_str}")
        if proto and proto != 'N/A':
            checks["prototype_ok"] = True
        
        # Phase 5: Uncertainty Quantification
        epi = m.get('epistemic_uncertainty')
        ale = m.get('aleatoric_uncertainty')
        ci = m.get('confidence_interval_95')
        print(f"  Epistemic Uncertainty:  {epi} eV")
        print(f"  Aleatoric Uncertainty:  {ale} eV")
        print(f"  95% CI:                 ±{ci} eV")
        if epi is not None:
            checks["uncertainty_ok"] = True
        if ci is not None:
            checks["confidence_ok"] = True
        
        # Lattice info
        if m.get('structure', {}).get('lattice'):
            lat = m['structure']['lattice']
            print(f"  Lattice:                a={lat.get('a')}, b={lat.get('b')}, c={lat.get('c')}")
        print()

    # Validation summary
    print("=" * 70)
    print("  PHASE 5 VERIFICATION SUMMARY")
    print("=" * 70)
    for check, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check:30s}  {status}")
    
    all_pass = all(checks.values())
    print(f"\n  Overall: {'✅ ALL CHECKS PASSED' if all_pass else '❌ SOME CHECKS FAILED'}")
    print("=" * 70)
    
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(test_phase5())
