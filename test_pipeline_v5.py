
import sys
import os
import json

# Add backend to path
sys.path.insert(0, os.path.join(os.getcwd(), 'Backend'))

# Dummy environment for testing
os.environ["MATERIALS_PROJECT_API_KEY"] = "test"

from app.services.discovery_service import discover_materials

def test_discovery_v5():
    print("Testing Research-Grade Discovery Pipeline (v5 — Phase 3)")
    print("=" * 60)
    
    # Run discovery for a known system that should trigger tolerance factors
    # e.g., Ba-Ti-O (perovskite territory)
    results = discover_materials(
        num_candidates=2,
        include_elements=['Ba', 'Ti', 'O'],
        enable_quantum_refinement=True,
        strict_research_filter=False
    )
    
    novel = results.get("novel_materials", [])
    print(f"Discovered: {len(novel)} materials")
    
    for i, m in enumerate(novel):
        print(f"\n--- Candidate {i+1}: {m['compound']} ---")
        print(f"Scientific Stability Probability: {m['stability_score']}%")
        print(f"Formation Energy: {m['formation_energy']} eV/atom")
        print(f"Goldschmidt t: {m.get('goldschmidt_t')}")
        print(f"Octahedral factor (mu): {m.get('octahedral_factor')}")
        print(f"Orbital Interaction: {m.get('orbital_interaction')}")
        print(f"DOS Dispersion: {m.get('dos_dispersion')}")
        print(f"Novelty: {m.get('novelty_label')}")
        
        if 'score_breakdown' in m:
            print(f"Score Breakdown: {m['score_breakdown']}")

    print("\nStats:")
    print(json.dumps(results.get("search_stats", {}), indent=2))

if __name__ == "__main__":
    test_discovery_v5()
