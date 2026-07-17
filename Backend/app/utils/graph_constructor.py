"""
Graph Constructor — Build structure-aware PyTorch Geometric graphs.

Instead of fully-connected composition graphs, this module now relies on
pymatgen + ASE to assemble physically plausible neighbor graphs:
    - Pymatgen ``Structure`` objects provide lattice + 3D coordinates
    - ASE ``NeighborList`` delivers sparse, bond-like connectivity
    - Edge features capture inverse distance and a Gaussian radial basis

The output graphs mirror the data fed to the visualization layer, ensuring the
GNN learns from the same structural assumptions that researchers inspect.
"""

import math
import torch
import numpy as np
from torch_geometric.data import Data

from app.utils.structure_builder import (
        DEFAULT_NEIGHBOR_PADDING,
        compute_neighbor_pairs,
        formula_to_structure,
)

# -----------------------------------------------------------------------
# Elemental property databases (same values as feature_engine.py)
# -----------------------------------------------------------------------
ELECTRONEGATIVITY = {
    "H": 2.20, "He": 0.0, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55,
    "N": 3.04, "O": 3.44, "F": 3.98, "Ne": 0.0, "Na": 0.93, "Mg": 1.31,
    "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16, "Ar": 0.0,
    "K": 0.82, "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66,
    "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65,
    "Ga": 1.81, "Ge": 2.01, "As": 2.18, "Se": 2.55, "Br": 2.96, "Kr": 3.00,
    "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33, "Nb": 1.60, "Mo": 2.16,
    "Ru": 2.20, "Rh": 2.28, "Pd": 2.20, "Ag": 1.93, "Cd": 1.69, "In": 1.78,
    "Sn": 1.96, "Sb": 2.05, "Te": 2.10, "I": 2.66, "Cs": 0.79, "Ba": 0.89,
    "La": 1.10, "Ce": 1.12, "Pr": 1.13, "Nd": 1.14, "Sm": 1.17, "Eu": 1.20,
    "Gd": 1.20, "Tb": 1.10, "Dy": 1.22, "Ho": 1.23, "Er": 1.24, "Tm": 1.25,
    "Yb": 1.10, "Lu": 1.27, "Hf": 1.30, "Ta": 1.50, "W": 2.36, "Re": 1.90,
    "Os": 2.20, "Ir": 2.20, "Pt": 2.28, "Au": 2.54, "Pb": 2.33, "Bi": 2.02,
    "Tl": 1.62, "Po": 2.00, "At": 2.20,
}

ATOMIC_RADIUS = {
    "H": 0.53, "He": 0.31, "Li": 1.67, "Be": 1.12, "B": 0.87, "C": 0.67,
    "N": 0.56, "O": 0.48, "F": 0.42, "Ne": 0.38, "Na": 1.90, "Mg": 1.45,
    "Al": 1.18, "Si": 1.11, "P": 0.98, "S": 0.88, "Cl": 0.79, "Ar": 0.71,
    "K": 2.43, "Ca": 1.94, "Sc": 1.84, "Ti": 1.76, "V": 1.71, "Cr": 1.66,
    "Mn": 1.61, "Fe": 1.56, "Co": 1.52, "Ni": 1.49, "Cu": 1.45, "Zn": 1.42,
    "Ga": 1.36, "Ge": 1.25, "As": 1.14, "Se": 1.03, "Br": 0.94, "Kr": 0.88,
    "Rb": 2.65, "Sr": 2.19, "Y": 2.12, "Zr": 2.06, "Nb": 1.98, "Mo": 1.90,
    "Ru": 1.78, "Rh": 1.73, "Pd": 1.69, "Ag": 1.65, "Cd": 1.61, "In": 1.56,
    "Sn": 1.45, "Sb": 1.33, "Te": 1.23, "I": 1.15, "Cs": 2.98, "Ba": 2.53,
    "La": 1.95, "Ce": 1.85, "Hf": 2.08, "Ta": 2.00, "W": 1.93, "Re": 1.88,
    "Os": 1.85, "Ir": 1.80, "Pt": 1.77, "Au": 1.74, "Pb": 1.54, "Bi": 1.43,
    "Pr": 2.03, "Nd": 2.01, "Sm": 1.98, "Eu": 1.98, "Gd": 1.96, "Tb": 1.94,
    "Dy": 1.92, "Ho": 1.92, "Er": 1.89, "Tm": 1.90, "Yb": 1.87, "Lu": 1.87,
    "Tl": 1.56, "Po": 1.40, "At": 1.27,
}

IONIZATION_ENERGY = {
    "H": 13.6, "He": 24.6, "Li": 5.4, "Be": 9.3, "B": 8.3, "C": 11.3,
    "N": 14.5, "O": 13.6, "F": 17.4, "Ne": 21.6, "Na": 5.1, "Mg": 7.6,
    "Al": 6.0, "Si": 8.2, "P": 10.5, "S": 10.4, "Cl": 13.0, "Ar": 15.8,
    "K": 4.3, "Ca": 6.1, "Sc": 6.6, "Ti": 6.8, "V": 6.7, "Cr": 6.8,
    "Mn": 7.4, "Fe": 7.9, "Co": 7.9, "Ni": 7.6, "Cu": 7.7, "Zn": 9.4,
    "Ga": 6.0, "Ge": 7.9, "As": 9.8, "Se": 9.8, "Br": 11.8, "Kr": 14.0,
    "Rb": 4.2, "Sr": 5.7, "Y": 6.2, "Zr": 6.6, "Nb": 6.8, "Mo": 7.1,
    "Ru": 7.4, "Rh": 7.5, "Pd": 8.3, "Ag": 7.6, "Cd": 9.0, "In": 5.8,
    "Sn": 7.3, "Sb": 8.6, "Te": 9.0, "I": 10.5, "Cs": 3.9, "Ba": 5.2,
    "La": 5.6, "Ce": 5.5, "Hf": 6.8, "Ta": 7.5, "W": 7.9, "Re": 7.8,
    "Os": 8.4, "Ir": 8.7, "Pt": 9.0, "Au": 9.2, "Pb": 7.4, "Bi": 7.3,
    "Pr": 5.5, "Nd": 5.5, "Sm": 5.6, "Eu": 5.7, "Gd": 6.2, "Tb": 5.9,
    "Dy": 5.9, "Ho": 6.0, "Er": 6.1, "Tm": 6.2, "Yb": 6.3, "Lu": 5.4,
    "Tl": 6.1, "Po": 8.4, "At": 9.3,
}

ELECTRON_AFFINITY = {
    "H": 0.75, "Li": 0.62, "Be": 0.0, "B": 0.28, "C": 1.26, "N": 0.0,
    "O": 1.46, "F": 3.40, "Na": 0.55, "Mg": 0.0, "Al": 0.43, "Si": 1.39,
    "P": 0.75, "S": 2.08, "Cl": 3.61, "K": 0.50, "Ca": 0.02, "Sc": 0.19,
    "Ti": 0.08, "V": 0.53, "Cr": 0.67, "Mn": 0.0, "Fe": 0.15, "Co": 0.66,
    "Ni": 1.16, "Cu": 1.24, "Zn": 0.0, "Ga": 0.43, "Ge": 1.23, "As": 0.80,
    "Se": 2.02, "Br": 3.36, "Rb": 0.49, "Sr": 0.05, "Y": 0.31, "Zr": 0.43,
    "Nb": 0.92, "Mo": 0.75, "Ru": 1.05, "Rh": 1.14, "Pd": 0.56, "Ag": 1.30,
    "Cd": 0.0, "In": 0.30, "Sn": 1.11, "Sb": 1.05, "Te": 1.97, "I": 3.06,
    "Cs": 0.47, "Ba": 0.14, "La": 0.47, "Ce": 0.55, "Hf": 0.0, "Ta": 0.32,
    "W": 0.82, "Re": 0.15, "Os": 1.10, "Ir": 1.56, "Pt": 2.13, "Au": 2.31,
    "Pb": 0.36, "Bi": 0.94, "Tl": 0.38,
}

VALENCE_ELECTRONS = {
    "H": 1, "He": 2, "Li": 1, "Be": 2, "B": 3, "C": 4,
    "N": 5, "O": 6, "F": 7, "Ne": 8, "Na": 1, "Mg": 2,
    "Al": 3, "Si": 4, "P": 5, "S": 6, "Cl": 7, "Ar": 8,
    "K": 1, "Ca": 2, "Sc": 3, "Ti": 4, "V": 5, "Cr": 6,
    "Mn": 7, "Fe": 8, "Co": 9, "Ni": 10, "Cu": 11, "Zn": 2,
    "Ga": 3, "Ge": 4, "As": 5, "Se": 6, "Br": 7, "Kr": 8,
    "Rb": 1, "Sr": 2, "Y": 3, "Zr": 4, "Nb": 5, "Mo": 6,
    "Ru": 8, "Rh": 9, "Pd": 10, "Ag": 11, "Cd": 2, "In": 3,
    "Sn": 4, "Sb": 5, "Te": 6, "I": 7, "Cs": 1, "Ba": 2,
    "La": 3, "Ce": 4, "Hf": 4, "Ta": 5, "W": 6, "Re": 7,
    "Os": 8, "Ir": 9, "Pt": 10, "Au": 11, "Pb": 4, "Bi": 5,
    "Pr": 3, "Nd": 4, "Sm": 4, "Eu": 3, "Gd": 4, "Tb": 3,
    "Dy": 4, "Ho": 3, "Er": 4, "Tm": 3, "Yb": 2, "Lu": 3,
    "Tl": 3, "Po": 6, "At": 7,
}

ATOMIC_NUMBER = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Ru": 44,
    "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51,
    "Te": 52, "I": 53, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59,
    "Nd": 60, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67,
    "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73, "W": 74,
    "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Pb": 82, "Bi": 83,
    "Tl": 81, "Po": 84, "At": 85,
}

GROUP = {
    "H": 1, "He": 18, "Li": 1, "Be": 2, "B": 13, "C": 14, "N": 15,
    "O": 16, "F": 17, "Ne": 18, "Na": 1, "Mg": 2, "Al": 13, "Si": 14,
    "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 1, "Ca": 2, "Sc": 3,
    "Ti": 4, "V": 5, "Cr": 6, "Mn": 7, "Fe": 8, "Co": 9, "Ni": 10,
    "Cu": 11, "Zn": 12, "Ga": 13, "Ge": 14, "As": 15, "Se": 16,
    "Br": 17, "Kr": 18, "Rb": 1, "Sr": 2, "Y": 3, "Zr": 4, "Nb": 5,
    "Mo": 6, "Ru": 8, "Rh": 9, "Pd": 10, "Ag": 11, "Cd": 12, "In": 13,
    "Sn": 14, "Sb": 15, "Te": 16, "I": 17, "Cs": 1, "Ba": 2, "La": 3,
    "Ce": 3, "Pr": 3, "Nd": 3, "Sm": 3, "Eu": 3, "Gd": 3, "Tb": 3,
    "Dy": 3, "Ho": 3, "Er": 3, "Tm": 3, "Yb": 3, "Lu": 3, "Hf": 4,
    "Ta": 5, "W": 6, "Re": 7, "Os": 8, "Ir": 9, "Pt": 10, "Au": 11,
    "Pb": 14, "Bi": 15, "Tl": 13, "Po": 16, "At": 17,
}

PERIOD = {
    "H": 1, "He": 1, "Li": 2, "Be": 2, "B": 2, "C": 2, "N": 2, "O": 2,
    "F": 2, "Ne": 2, "Na": 3, "Mg": 3, "Al": 3, "Si": 3, "P": 3, "S": 3,
    "Cl": 3, "Ar": 3, "K": 4, "Ca": 4, "Sc": 4, "Ti": 4, "V": 4, "Cr": 4,
    "Mn": 4, "Fe": 4, "Co": 4, "Ni": 4, "Cu": 4, "Zn": 4, "Ga": 4,
    "Ge": 4, "As": 4, "Se": 4, "Br": 4, "Kr": 4, "Rb": 5, "Sr": 5,
    "Y": 5, "Zr": 5, "Nb": 5, "Mo": 5, "Ru": 5, "Rh": 5, "Pd": 5,
    "Ag": 5, "Cd": 5, "In": 5, "Sn": 5, "Sb": 5, "Te": 5, "I": 5,
    "Cs": 6, "Ba": 6, "La": 6, "Ce": 6, "Pr": 6, "Nd": 6, "Sm": 6,
    "Eu": 6, "Gd": 6, "Tb": 6, "Dy": 6, "Ho": 6, "Er": 6, "Tm": 6,
    "Yb": 6, "Lu": 6, "Hf": 6, "Ta": 6, "W": 6, "Re": 6, "Os": 6,
    "Ir": 6, "Pt": 6, "Au": 6, "Pb": 6, "Bi": 6, "Tl": 6, "Po": 6,
    "At": 6,
}

MENDELEEV_NO = {
    "H": 92, "He": 98, "Li": 1, "Be": 67, "B": 72, "C": 77, "N": 82,
    "O": 87, "F": 93, "Ne": 99, "Na": 2, "Mg": 68, "Al": 73, "Si": 78,
    "P": 83, "S": 88, "Cl": 94, "Ar": 100, "K": 3, "Ca": 7, "Sc": 11,
    "Ti": 43, "V": 44, "Cr": 45, "Mn": 46, "Fe": 47, "Co": 48, "Ni": 49,
    "Cu": 50, "Zn": 51, "Ga": 74, "Ge": 79, "As": 84, "Se": 89, "Br": 95,
    "Kr": 101, "Rb": 4, "Sr": 8, "Y": 12, "Zr": 52, "Nb": 53, "Mo": 54,
    "Ru": 56, "Rh": 57, "Pd": 58, "Ag": 59, "Cd": 69, "In": 75, "Sn": 80,
    "Sb": 85, "Te": 90, "I": 96, "Cs": 5, "Ba": 9, "La": 13, "Ce": 14,
    "Pr": 15, "Nd": 16, "Sm": 17, "Eu": 18, "Gd": 19, "Tb": 20, "Dy": 21,
    "Ho": 22, "Er": 23, "Tm": 24, "Yb": 25, "Lu": 26, "Hf": 60, "Ta": 61,
    "W": 62, "Re": 63, "Os": 64, "Ir": 65, "Pt": 66, "Au": 69, "Pb": 81,
    "Bi": 86, "Tl": 76, "Po": 91, "At": 97,
}

# -----------------------------------------------------------------------
# Normalization constants (for scaling features into ~[0,1])
# -----------------------------------------------------------------------
_NORM = {
    "atomic_number": 100.0,
    "group":         18.0,
    "period":         7.0,
    "en":             4.0,
    "radius":         3.0,
    "ie":            25.0,
    "ea":             4.0,
    "valence":       12.0,
    "mendeleev":    103.0,
    # Edge features
    "en_diff":        4.0,
    "radius_sum":     6.0,
    "radius_diff":    3.0,
    "mendeleev_diff": 100.0,
}

# Number of node features and edge features
NODE_FEAT_DIM = 9

# Distance-aware edge encoding parameters
GAUSSIAN_CENTERS = np.linspace(0.75, 4.5, 16)
GAUSSIAN_WIDTH = 0.25
DISTANCE_NORMALIZER = 10.0

# Chemical edge features (added back to preserve ionic information)
CHEM_EDGE_DIM = 3  # en_diff, mendeleev_diff, radius_sum
EDGE_FEAT_DIM = 2 + len(GAUSSIAN_CENTERS) + CHEM_EDGE_DIM


def _get_node_features(symbol: str) -> list[float]:
    """Return 9-dimensional node feature vector for an element."""
    return [
        ATOMIC_NUMBER.get(symbol, 0)     / _NORM["atomic_number"],
        GROUP.get(symbol, 0)             / _NORM["group"],
        PERIOD.get(symbol, 0)            / _NORM["period"],
        ELECTRONEGATIVITY.get(symbol, 0) / _NORM["en"],
        ATOMIC_RADIUS.get(symbol, 1.0)   / _NORM["radius"],
        IONIZATION_ENERGY.get(symbol, 0) / _NORM["ie"],
        ELECTRON_AFFINITY.get(symbol, 0) / _NORM["ea"],
        VALENCE_ELECTRONS.get(symbol, 0) / _NORM["valence"],
        MENDELEEV_NO.get(symbol, 50)     / _NORM["mendeleev"],
    ]


def _distance_features(distance: float) -> list[float]:
    """Encode a bond distance using inverse distance + Gaussian RBF."""
    safe_dist = max(distance, 1e-3)
    inv_d = 1.0 / safe_dist
    normalized_distance = min(distance / DISTANCE_NORMALIZER, 1.0)
    gaussians = [
        math.exp(-0.5 * ((distance - center) / GAUSSIAN_WIDTH) ** 2)
        for center in GAUSSIAN_CENTERS
    ]
    return [inv_d, normalized_distance, *gaussians]


def _chemical_edge_features(sym1: str, sym2: str) -> list[float]:
    en1 = ELECTRONEGATIVITY.get(sym1, 0.0)
    en2 = ELECTRONEGATIVITY.get(sym2, 0.0)
    m1 = MENDELEEV_NO.get(sym1, 50.0)
    m2 = MENDELEEV_NO.get(sym2, 50.0)
    r1 = ATOMIC_RADIUS.get(sym1, 1.0)
    r2 = ATOMIC_RADIUS.get(sym2, 1.0)

    en_diff = abs(en1 - en2) / _NORM["en_diff"]
    mendeleev_diff = abs(m1 - m2) / _NORM["mendeleev_diff"]
    radius_sum = (r1 + r2) / _NORM["radius_sum"]
    return [en_diff, mendeleev_diff, radius_sum]


def formula_to_graph(
    formula: str,
    target_energy: float | None = None,
    target_bandgap: float | None = None,
    target_ehull: float | None = None,
    *,
    density: float | None = None,
    jitter: float = 0.0,
    neighbor_padding: float = DEFAULT_NEIGHBOR_PADDING,
    rng: np.random.Generator | None = None,
) -> Data | None:
    """Convert a chemical formula into a sparse, structure-aware PyG graph."""
    try:
        structure = formula_to_structure(
            formula,
            density=density,
            jitter=jitter,
            rng=rng,
        )
        if structure is None:
            return None

        atoms = [site.specie.symbol for site in structure.sites]
        if not atoms:
            return None

        n_atoms = len(atoms)
        if n_atoms < 1 or n_atoms > 60:
            return None

        node_feats = [_get_node_features(sym) for sym in atoms]
        x = torch.tensor(node_feats, dtype=torch.float32)

        neighbor_pairs = compute_neighbor_pairs(structure, extra_cutoff=neighbor_padding)
        if not neighbor_pairs:
            return None

        src, dst = [], []
        edge_feats: list[list[float]] = []
        for i, j, distance, _ in neighbor_pairs:
            dist_feats = _distance_features(distance)
            chem_feats = _chemical_edge_features(atoms[i], atoms[j])
            feats = dist_feats + chem_feats
            src.append(i)
            dst.append(j)
            edge_feats.append(feats)
            src.append(j)
            dst.append(i)
            edge_feats.append(feats)

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
        pos = torch.tensor(structure.cart_coords, dtype=torch.float32)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
        )
        data.formula = formula
        data.num_atoms = n_atoms

        if target_energy is not None:
            data.y_energy = torch.tensor([target_energy], dtype=torch.float32)
        if target_bandgap is not None:
            data.y_bandgap = torch.tensor([target_bandgap], dtype=torch.float32)
        if target_ehull is not None:
            data.y_ehull = torch.tensor([target_ehull], dtype=torch.float32)

        return data

    except Exception:
        # Gracefully skip formulas that cause pymatgen/ASE errors
        return None


def batch_formulas_to_graphs(
    formulas: list[str],
    energies: list[float] | None = None,
    bandgaps: list[float] | None = None,
    ehulls: list[float] | None = None,
    densities: list[float] | None = None,
    jitter: float = 0.0,
    neighbor_padding: float = DEFAULT_NEIGHBOR_PADDING,
    rng: np.random.Generator | None = None,
) -> list[Data]:
    """
    Convert a batch of formulas into a list of PyG Data objects.

    Skips formulas that can't be parsed (unknown elements, too large, etc.)
    """
    graphs = []
    for i, formula in enumerate(formulas):
        e = energies[i] if energies is not None else None
        b = bandgaps[i] if bandgaps is not None else None
        h = ehulls[i] if ehulls is not None else None
        d = densities[i] if densities is not None else None
        g = formula_to_graph(
            formula,
            target_energy=e,
            target_bandgap=b,
            target_ehull=h,
            density=d,
            jitter=jitter,
            neighbor_padding=neighbor_padding,
            rng=rng,
        )
        if g is not None:
            graphs.append(g)
    return graphs


# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Graph Construction Test")
    print("=" * 60)

    tests = [
        ("NaCl",    -3.29, 5.0),
        ("TiO2",    -3.39, 3.0),
        ("BaTiO3",  -3.24, 2.3),
        ("Fe2O3",   -2.10, 2.2),
        ("LiCoO2",  -2.50, 2.7),
    ]

    for formula, energy, bandgap in tests:
        g = formula_to_graph(formula, energy, bandgap)
        if g is not None:
            print(f"\n  {formula:10s}  nodes={g.x.shape[0]:2d}  "
                  f"edges={g.edge_index.shape[1]:3d}  "
                  f"x_shape={list(g.x.shape)}  "
                  f"edge_attr_shape={list(g.edge_attr.shape)}")
            print(f"              node_feat_dim={g.x.shape[1]}  "
                  f"edge_feat_dim={g.edge_attr.shape[1]}")
        else:
            print(f"  {formula}: FAILED to construct graph")
