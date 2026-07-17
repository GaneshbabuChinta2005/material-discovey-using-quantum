
import sys
import os
import json

# Add backend to path
sys.path.insert(0, os.path.join(os.getcwd(), 'Backend'))

# Dummy environment for testing
os.environ["MATERIALS_PROJECT_API_KEY"] = "test"

from app.services.discovery_service import discover_materials

def test_discovery():
    print("Testing Research-Grade Discovery Pipeline (v4)")
    print("=" * 60)
    
    # Run a small discovery session
    # We use a restricted set of elements to keep it fast
    results = discover_materials(
        num_candidates=2,
        include_elements=['Fe', 'O'],
        enable_quantum_refinement=False,
        strict_research_filter=False
    )
    
    novel = results.get("novel_materials", [])
    print(f"Discovered: {len(novel)} materials")
    
    for i, m in enumerate(novel):
        print(f"\n--- Candidate {i+1}: {m['compound']} ---")
        print(f"Stability Score: {m['stability_score']} / 100")
        print(f"Formation Energy: {m['formation_energy']} eV/atom")
        print(f"Band Gap: {m.get('band_gap_eV')} eV")
        print(f"Synthesis Conf: {m.get('synthesis_confidence')}")
        print(f"Synthesis Label: {m.get('synthesis_label')}")
        print(f"Hull Distance: {m.get('hull_distance_eV')} eV")
        print(f"Hull Status: {m.get('hull_status')}")
        print(f"Novelty: {m.get('novelty_label')}")
        
        if 'score_breakdown' in m:
            print(f"Score Breakdown: {m['score_breakdown']}")

    print("\nStats:")
    print(json.dumps(results.get("search_stats", {}), indent=2))

if __name__ == "__main__":
    test_discovery()
