import pandas as pd

# Simple atomic property table
ELEMENT_PROPERTIES = {
    "Li": {"mass": 6.9, "en": 1.0},
    "Mg": {"mass": 24.3, "en": 1.2},
    "Al": {"mass": 27.0, "en": 1.5},
    "Fe": {"mass": 55.8, "en": 1.8},
    "C": {"mass": 12.0, "en": 2.5},
    "O": {"mass": 16.0, "en": 3.4},
    "N": {"mass": 14.0, "en": 3.0},
    "Si": {"mass": 28.1, "en": 1.9},
    "Ca": {"mass": 40.1, "en": 1.0}
}


def compound_to_features(formula):
    """
    Convert simple formula like Al2O3 → feature vector
    """

    import re

    parts = re.findall(r'([A-Z][a-z]*)(\d*)', formula)

    total_atoms = 0
    total_mass = 0
    total_en = 0

    for elem, count in parts:
        count = int(count) if count else 1
        props = ELEMENT_PROPERTIES.get(elem, {"mass": 20, "en": 2})

        total_atoms += count
        total_mass += props["mass"] * count
        total_en += props["en"] * count

    avg_mass = total_mass / total_atoms
    avg_en = total_en / total_atoms

    # Create feature vector matching AI model
    features = [
        total_atoms,
        avg_en,         # proxy for dipole
        avg_mass,       # proxy for polarizability
        -0.3,           # dummy HOMO
        0.1,            # dummy LUMO
        0.4,            # dummy gap
        20.0            # dummy heat capacity
    ]

    return features
