import joblib
import pandas as pd

# Load model
model = joblib.load("model/material_model.pkl")

print("Model loaded successfully.")

# Define feature names (must match training data)
feature_names = [
    "N_atoms",
    "Dipole_debye",
    "Polarizability_bohr3",
    "HOMO_au",
    "LUMO_au",
    "HOMO_LUMO_gap_au",
    "Heatcapacity_Cv_cal_mol_K"
]

# Example input
sample_values = [5, 1.2, 12.5, -0.3, 0.1, 0.4, 20.0]

# Convert to DataFrame
input_df = pd.DataFrame([sample_values], columns=feature_names)

# Predict
prediction = model.predict(input_df)

print("Predicted internal energy:", prediction[0])
