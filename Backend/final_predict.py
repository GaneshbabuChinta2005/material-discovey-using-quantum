import joblib
import pandas as pd
from quantum_simulation import run_quantum_simulation

# Load AI model
model = joblib.load("model/material_model.pkl")

# Feature names
feature_names = [
    "N_atoms",
    "Dipole_debye",
    "Polarizability_bohr3",
    "HOMO_au",
    "LUMO_au",
    "HOMO_LUMO_gap_au",
    "Heatcapacity_Cv_cal_mol_K"
]

# Example compound features
sample_values = [5, 1.2, 12.5, -0.3, 0.1, 0.4, 20.0]
input_df = pd.DataFrame([sample_values], columns=feature_names)

# AI prediction
ai_energy = model.predict(input_df)[0]

# Quantum simulation
quantum_energy = run_quantum_simulation()

# Combine results
stability_score = 10 - abs(ai_energy) * 0.01 + quantum_energy

print("AI predicted energy:", ai_energy)
print("Quantum energy estimate:", quantum_energy)
print("Final stability score:", round(stability_score, 2))
