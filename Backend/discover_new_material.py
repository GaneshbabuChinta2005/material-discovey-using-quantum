import joblib
import pandas as pd
from compound_generator import generate_valid_compound
from compound_features import compound_to_features
from quantum_simulation import run_quantum_simulation

# Load AI model
model = joblib.load("model/material_model.pkl")

feature_names = [
    "N_atoms",
    "Dipole_debye",
    "Polarizability_bohr3",
    "HOMO_au",
    "LUMO_au",
    "HOMO_LUMO_gap_au",
    "Heatcapacity_Cv_cal_mol_K"
]

print("\nDiscovering new materials...\n")

for _ in range(3):
    # Generate compound
    compound = generate_valid_compound()

    # Convert to features
    features = compound_to_features(compound)
    df = pd.DataFrame([features], columns=feature_names)

    # AI prediction
    ai_energy = model.predict(df)[0]

    # Quantum validation
    quantum_score = run_quantum_simulation()

    # Final stability
    stability = 10 - abs(ai_energy)*0.01 + quantum_score

    print(f"Compound: {compound}")
    print(f"AI Energy: {ai_energy:.2f}")
    print(f"Quantum Score: {quantum_score:.2f}")
    print(f"Stability Score: {stability:.2f}")
    print("-" * 40)
