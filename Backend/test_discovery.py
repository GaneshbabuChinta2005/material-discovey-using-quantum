import sys
import os
sys.path.append(r"c:\QuantumMaterialDiscovery\Backend")
from app.services.discovery_service import discover_materials

res = discover_materials(
    application_targets=["Battery Electrode"], 
    num_candidates=5
)

metals = []
for c in res["novel_materials"]:
    metals.append(c["compound"] + " (E=" + str(round(c["formation_energy"], 2)) + " " + c.get("battery_prototype_match", "") + ")")
    
print("\\nGenerated Battery Candidates:")
for m in metals:
    print(m)
print("\\nStats:")
for k,v in res["search_stats"].items():
    print(f"{k}: {v}")
