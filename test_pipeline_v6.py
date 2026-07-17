
import sys
import os
import json

sys.path.insert(0, os.path.join(os.getcwd(), 'Backend'))
os.environ["MATERIALS_PROJECT_API_KEY"] = "test"

from app.services.discovery_service import discover_materials

def test_phase4():
    print("=" * 70)
    print("  Phase 4 Verification — Research-Grade Pipeline (v6)")
    print("=" * 70)
    
    results = discover_materials(
        num_candidates=3,
        include_elements=['Ba', 'Ti', 'O'],
        enable_quantum_refinement=True,
        strict_research_filter=False
    )
    
    novel = results.get("novel_materials", [])
    print(f"\nDiscovered: {len(novel)} materials\n")
    
    for i, m in enumerate(novel):
        print(f"--- Candidate {i+1}: {m['compound']} ---")
        print(f"  Stability Probability:  {m.get('stability_score')}%")
        print(f"  Formation Energy:       {m.get('formation_energy')} eV/atom")
        print(f"  Ehull:                  {m.get('e_above_hull')}")
        print(f"  Novelty:                {m.get('novelty_label')}")
        print(f"  Databases Checked:      {m.get('databases_checked', 'N/A')}")
        print(f"  Found In:               {m.get('found_in', [])}")
        print(f"  Decomposition Risk:     {m.get('decomposition_risk_pct')}%")
        print(f"  Decomp Products:        {m.get('likely_decomposition_products', [])}")
        print(f"  Decomp Energy Gain:     {m.get('decomposition_energy_gain_eV')} eV")
        print(f"  Hull Status:            {m.get('hull_status')}")
        print(f"  Orbital Interaction:    {m.get('orbital_interaction')}")
        print(f"  DOS Dispersion:         {m.get('dos_dispersion')}")
        if m.get('structure', {}).get('lattice'):
            lat = m['structure']['lattice']
            print(f"  Lattice:                a={lat.get('a')}, b={lat.get('b')}, c={lat.get('c')}")
            print(f"  Crystal System:         {lat.get('crystal_system')}")
        print()

    stats = results.get("search_stats", {})
    print(f"Pipeline Stats:")
    print(f"  Explored:          {stats.get('explored')}")
    print(f"  Novel Count:       {stats.get('novel_count')}")
    print(f"  Scoring Model:     {stats.get('scoring_formula')}")
    print(f"  Returned:          {stats.get('returned')}")
    print(f"  Elapsed:           {stats.get('elapsed_seconds')}s")

if __name__ == "__main__":
    test_phase4()
