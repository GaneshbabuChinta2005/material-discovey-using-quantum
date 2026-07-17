"""
Structure Builder — Convert chemical formulas into approximate 3D structures.

This module centralizes all structure heuristics so both the GNN graph
constructor and the visualization stack share the exact same physical
scaffolding:
  * Pymatgen ``Structure`` objects are used to store lattice + coordinates
  * ASE ``NeighborList`` provides realistic bonding connectivity
  * Lightweight chemistry heuristics (coordination, VSEPR, radii) provide
    initial atomic placement when no CIF is available

The resulting structures are not ab initio relaxed, but the combination of
chemistry-informed placement + density-constrained lattice scaling produces
reasonable inter-atomic distances that can be refined via jitter-based data
augmentation during training.
"""

from __future__ import annotations

import math
import re

import numpy as np
from ase.data import covalent_radii as ASE_COVALENT_RADII
from ase.neighborlist import NeighborList
from pymatgen.core import Composition, Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from app.services.chemistry_validator import (
    PREFERRED_COORDINATION,
    OXIDATION_STATES,
    d_electron_count,
)

AVOGADRO = 6.02214076e23
CM3_TO_ANGSTROM3 = 1e24
MIN_LATTICE_A = 5.0
LATTICE_BUFFER = 2.0
DEFAULT_NEIGHBOR_PADDING = 0.45
DEFAULT_RADIUS = 1.2
DEFAULT_JITTER = 0.05


# -----------------------------------------------------------------------
# Chemistry data (shared with visualization pipeline)
# -----------------------------------------------------------------------
CPK_COLORS = {
    "H":  "#FFFFFF", "He": "#D9FFFF",
    "Li": "#CC80FF", "Be": "#C2FF00", "B":  "#FFB5B5", "C":  "#909090",
    "N":  "#3050F8", "O":  "#FF0D0D", "F":  "#90E050", "Ne": "#B3E3F5",
    "Na": "#AB5CF2", "Mg": "#8AFF00", "Al": "#BFA6A6", "Si": "#F0C8A0",
    "P":  "#FF8000", "S":  "#FFFF30", "Cl": "#1FF01F", "Ar": "#80D1E3",
    "K":  "#8F40D4", "Ca": "#3DFF00", "Sc": "#E6E6E6", "Ti": "#BFC2C7",
    "V":  "#A6A6AB", "Cr": "#8A99C7", "Mn": "#9C7AC7", "Fe": "#E06633",
    "Co": "#F090A0", "Ni": "#50D050", "Cu": "#C88033", "Zn": "#7D80B0",
    "Ga": "#C28F8F", "Ge": "#668F8F", "As": "#BD80E3", "Se": "#FFA100",
    "Br": "#A62929", "Kr": "#5CB8D1", "Rb": "#702EB0", "Sr": "#00FF00",
    "Y":  "#94FFFF", "Zr": "#94E0E0", "Nb": "#73C2C9", "Mo": "#54B5B5",
    "Ru": "#248F8F", "Rh": "#0A7D8C", "Pd": "#006985", "Ag": "#C0C0C0",
    "Cd": "#FFD98F", "In": "#A67573", "Sn": "#668080", "Sb": "#9E63B5",
    "Te": "#D47A00", "I":  "#940094", "Cs": "#57178F", "Ba": "#00C900",
    "La": "#70D4FF", "Ce": "#FFFFC7", "W":  "#2194D6", "Pt": "#D0D0E0",
    "Au": "#FFD123", "Pb": "#575961", "Bi": "#9E4FB5",
}

# Approximate covalent radii (Angstroms)
COVALENT_RADII = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84,
    "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07,
    "S": 1.05, "Cl": 1.02, "Ar": 1.06, "K": 2.03, "Ca": 1.76,
    "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39,
    "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20,
    "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64,
    "Mo": 1.54, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45,
    "Cd": 1.44, "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38,
    "I": 1.39, "Cs": 2.44, "Ba": 2.15, "La": 2.07, "Ce": 2.04,
    "W": 1.62, "Pt": 1.36, "Au": 1.36, "Pb": 1.46, "Bi": 1.48,
}

METALS = {
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Cs",
    "Ba", "La", "Ce", "W", "Pt", "Au", "Pb", "Bi",
}

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
    "La": 1.10, "Ce": 1.12, "W": 2.36, "Pt": 2.28, "Au": 2.54, "Pb": 2.33,
    "Bi": 2.02,
}

__all__ = [
    "formula_to_structure",
    "compute_neighbor_pairs",
    "COVALENT_RADII",
    "CPK_COLORS",
    "DEFAULT_NEIGHBOR_PADDING",
]


# -----------------------------------------------------------------------
# Formula parsing and atom placement heuristics
# -----------------------------------------------------------------------
def _parse_formula(formula: str) -> list[tuple[str, int]]:
    parts = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    result = []
    for sym, count_str in parts:
        if not sym:
            continue
        count = int(count_str) if count_str else 1
        result.append((sym, count))
    return result


def _expand_atoms(parsed: list[tuple[str, int]]) -> list[str]:
    atoms: list[str] = []
    for sym, count in parsed:
        atoms.extend([sym] * count)
    return atoms


def _bond_length(sym1: str, sym2: str) -> float:
    r1 = COVALENT_RADII.get(sym1, DEFAULT_RADIUS)
    r2 = COVALENT_RADII.get(sym2, DEFAULT_RADIUS)
    return r1 + r2


def _classify_atoms(atom_list: list[str]) -> tuple[list[int], list[int]]:
    central, ligand = [], []
    for idx, sym in enumerate(atom_list):
        en = ELECTRONEGATIVITY.get(sym, 2.0)
        if sym in METALS or en < 2.0:
            central.append(idx)
        else:
            ligand.append(idx)

    if not central and ligand:
        ligand.sort(key=lambda i: ELECTRONEGATIVITY.get(atom_list[i], 2.0))
        n_central = max(1, len(ligand) // 3)
        central = ligand[:n_central]
        ligand = ligand[n_central:]

    if not ligand and len(central) > 1:
        mid = len(central) // 2
        ligand = central[mid:]
        central = central[:mid]

    return central, ligand


def _get_preferred_cn(sym: str) -> int:
    states = [s for s in OXIDATION_STATES.get(sym, []) if s > 0]
    for ox in states:
        cn_list = PREFERRED_COORDINATION.get((sym, ox))
        if cn_list:
            return cn_list[0]
    return 6


def _is_square_planar(sym: str) -> bool:
    states = [s for s in OXIDATION_STATES.get(sym, []) if s > 0]
    for ox in states:
        d_count = d_electron_count(sym, ox) or -1
        cn_pref = PREFERRED_COORDINATION.get((sym, ox), [])
        if d_count == 8 and cn_pref and cn_pref[0] == 4:
            return True
    return False


def _ligand_positions_around(
    center: tuple[float, float, float],
    n_ligands: int,
    bond_len: float,
    square_planar: bool = False,
) -> list[tuple[float, float, float]]:
    cx, cy, cz = center
    d = bond_len

    if n_ligands == 0:
        return []
    if n_ligands == 1:
        return [(cx + d, cy, cz)]
    if n_ligands == 2:
        return [(cx + d, cy, cz), (cx - d, cy, cz)]
    if n_ligands == 3:
        return [
            (cx + d * math.cos(2 * math.pi * k / 3),
             cy + d * math.sin(2 * math.pi * k / 3), cz)
            for k in range(3)
        ]
    if n_ligands == 4:
        if square_planar:
            return [
                (cx + d, cy, cz), (cx - d, cy, cz),
                (cx, cy + d, cz), (cx, cy - d, cz),
            ]
        return [
            (cx + d * 0.0,    cy + d * 0.0,    cz + d * 1.0),
            (cx + d * 0.9428, cy + d * 0.0,    cz - d * 0.3333),
            (cx - d * 0.4714, cy + d * 0.8165, cz - d * 0.3333),
            (cx - d * 0.4714, cy - d * 0.8165, cz - d * 0.3333),
        ]
    if n_ligands == 5:
        base = [
            (cx, cy, cz + d),
            (cx, cy, cz - d),
        ]
        for k in range(3):
            angle = 2 * math.pi * k / 3
            base.append((cx + d * math.cos(angle), cy + d * math.sin(angle), cz))
        return base
    if n_ligands == 6:
        return [
            (cx + d, cy, cz), (cx - d, cy, cz),
            (cx, cy + d, cz), (cx, cy - d, cz),
            (cx, cy, cz + d), (cx, cy, cz - d),
        ]
    if n_ligands == 8:
        s = d * 0.577
        return [
            (cx + s, cy + s, cz + s), (cx + s, cy + s, cz - s),
            (cx + s, cy - s, cz + s), (cx + s, cy - s, cz - s),
            (cx - s, cy + s, cz + s), (cx - s, cy + s, cz - s),
            (cx - s, cy - s, cz + s), (cx - s, cy - s, cz - s),
        ]

    positions = []
    golden_ratio = (1 + math.sqrt(5)) / 2
    for i in range(n_ligands):
        theta = math.acos(1 - 2 * (i + 0.5) / n_ligands)
        phi = 2 * math.pi * i / golden_ratio
        positions.append((
            cx + d * math.sin(theta) * math.cos(phi),
            cy + d * math.sin(theta) * math.sin(phi),
            cz + d * math.cos(theta),
        ))
    return positions


def _place_central_atoms(
    n_central: int,
    central_symbols: list[str],
    base_spacing: float = 2.5,
) -> list[tuple[float, float, float]]:
    if n_central == 0:
        return []
    if n_central == 1:
        return [(0.0, 0.0, 0.0)]

    d = base_spacing
    if n_central == 2:
        return [(-d / 2, 0, 0), (d / 2, 0, 0)]
    if n_central == 3:
        return [
            (0, d * 0.577, 0),
            (-d / 2, -d * 0.289, 0),
            (d / 2, -d * 0.289, 0),
        ]
    if n_central == 4:
        return [
            (-d / 2, -d / 2, 0), (d / 2, -d / 2, 0),
            (-d / 2, d / 2, 0), (d / 2, d / 2, 0),
        ]

    positions: list[tuple[float, float, float]] = []
    side = math.ceil(n_central ** (1 / 3))
    idx = 0
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                if idx >= n_central:
                    break
                x = (ix - (side - 1) / 2) * d
                y = (iy - (side - 1) / 2) * d
                z = (iz - (side - 1) / 2) * d
                positions.append((x, y, z))
                idx += 1
    return positions[:n_central]


def _generate_positions(atom_list: list[str]) -> list[tuple[float, float, float]]:
    n = len(atom_list)
    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0, 0.0)]
    if n == 2:
        d = _bond_length(atom_list[0], atom_list[1])
        return [(-d / 2, 0, 0), (d / 2, 0, 0)]

    central_idx, ligand_idx = _classify_atoms(atom_list)
    if central_idx and ligand_idx:
        pairings = [
            _bond_length(atom_list[ci], atom_list[li])
            for ci in central_idx[:3] for li in ligand_idx[:3]
        ]
        avg_bond = sum(pairings) / len(pairings)
    else:
        avg_bond = 1.8

    central_spacing = avg_bond * 2.2
    central_symbols = [atom_list[i] for i in central_idx]
    central_positions = _place_central_atoms(len(central_idx), central_symbols, central_spacing)

    allocations: dict[int, list[int]] = {ci: [] for ci in range(len(central_idx))}
    for i, li in enumerate(ligand_idx):
        ci = i % max(len(central_idx), 1)
        allocations[ci].append(li)

    positions: list[tuple[float, float, float] | None] = [None] * n
    for ci_local, ci_global in enumerate(central_idx):
        positions[ci_global] = central_positions[ci_local]

    for ci_local, ci_global in enumerate(central_idx):
        my_ligands = allocations.get(ci_local, [])
        if not my_ligands:
            continue
        center = central_positions[ci_local]
        sq_planar = _is_square_planar(atom_list[ci_global])
        pref_cn = _get_preferred_cn(atom_list[ci_global])
        avg_bl = sum(_bond_length(atom_list[ci_global], atom_list[li]) for li in my_ligands) / len(my_ligands)
        n_target = max(len(my_ligands), pref_cn)
        lig_positions = _ligand_positions_around(center, n_target, avg_bl, square_planar=sq_planar)

        for k, li in enumerate(my_ligands):
            if k < len(lig_positions):
                positions[li] = lig_positions[k]
            else:
                angle = 2 * math.pi * k / len(my_ligands)
                positions[li] = (
                    center[0] + avg_bl * math.cos(angle),
                    center[1] + avg_bl * math.sin(angle),
                    center[2],
                )

    for i in range(n):
        if positions[i] is None:
            positions[i] = (0.0, 0.0, 0.0)

    return positions  # type: ignore[return-value]


def _estimate_target_volume(formula: str, density: float | None) -> float | None:
    if density is None or density <= 0:
        return None
    comp = Composition(formula)
    mass_per_formula = comp.weight / AVOGADRO
    volume_cm3 = mass_per_formula / density
    return volume_cm3 * CM3_TO_ANGSTROM3


def _center_positions(coords: np.ndarray) -> np.ndarray:
    centered = coords - coords.mean(axis=0)
    return centered


def _build_lattice_and_coords(
    positions: list[tuple[float, float, float]],
    formula: str,
    density: float | None,
) -> tuple[Lattice, np.ndarray]:
    coords = np.array(positions, dtype=float)
    centered = _center_positions(coords)
    extent = np.ptp(centered, axis=0)
    max_extent = float(np.max(extent)) + LATTICE_BUFFER
    base_a = max(MIN_LATTICE_A, max_extent)

    target_volume = _estimate_target_volume(formula, density)
    if target_volume is not None:
        a_from_density = target_volume ** (1 / 3)
        a = max(base_a, a_from_density)
    else:
        a = base_a

    lattice = Lattice.cubic(a)
    shift = a / 2.0
    cart_coords = centered + shift
    return lattice, cart_coords


def _apply_jitter(structure: Structure, magnitude: float, rng: np.random.Generator) -> Structure:
    perturbed = structure.copy()
    displacements = rng.normal(loc=0.0, scale=magnitude, size=(len(perturbed), 3))
    perturbed.translate_sites(range(len(perturbed)), displacements, frac_coords=False, to_unit_cell=True)
    return perturbed


# -----------------------------------------------------------------------
# Prototype Structural Mapping (Crystallography-Aware)
# -----------------------------------------------------------------------
def _build_from_prototype(parsed: list[tuple[str, int]], formula: str, density: float | None) -> Structure | None:
    """Generate periodic crystal structures using exact spacegroups and Wyckoff positions."""
    n_elements = len(parsed)
    if n_elements < 2:
        return None
        
    # Classify into cations/anions for mapping
    metals = [(s, c) for s, c in parsed if s in METALS or ELECTRONEGATIVITY.get(s, 2.0) < 2.0]
    nonmetals = [(s, c) for s, c in parsed if s not in [m[0] for m in metals]]
    
    if not metals or not nonmetals:
        return None

    # Sort by amount
    metals.sort(key=lambda x: x[1])
    
    # Estimate lattice parameter (a)
    vol = _estimate_target_volume(formula, density)
    if not vol:
        # fallback based on radii sum
        avg_r = sum(COVALENT_RADII.get(s, 1.2) for s, c in parsed) / n_elements
        vol = (avg_r * 2.5) ** 3 * sum(c for s, c in parsed)
    
    a = vol ** (1/3)

    try:
        # Rock Salt (AX) - Fm-3m (225)
        if n_elements == 2 and metals[0][1] == 1 and nonmetals[0][1] == 1:
            lattice = Lattice.cubic(max(a * 0.8, 4.0))
            return Structure.from_spacegroup("Fm-3m", lattice, [metals[0][0], nonmetals[0][0]], [[0,0,0], [0.5,0.5,0.5]])
            
        # Fluorite (AX2) - Fm-3m (225)
        if n_elements == 2 and metals[0][1] == 1 and nonmetals[0][1] == 2:
            lattice = Lattice.cubic(max(a * 0.9, 4.5))
            return Structure.from_spacegroup("Fm-3m", lattice, [metals[0][0], nonmetals[0][0]], [[0,0,0], [0.25,0.25,0.25]])
            
        # Rutile (AX2) - P4_2/mnm (136)
        if n_elements == 2 and metals[0][1] == 1 and nonmetals[0][1] == 2:
            # Alternately, could be rutile. Let's make an arbitrary choice based on radius? 
            # We'll just stick to fluorite for AX2 for simplicity, unless we do both.
            pass

        # Perovskite (ABX3) - Pm-3m (221)
        if n_elements == 3 and len(metals) == 2 and metals[0][1] == 1 and metals[1][1] == 1 and nonmetals[0][1] == 3:
            lattice = Lattice.cubic(max(a * 0.7, 3.8))
            # A at corners (0,0,0), B at body-center (0.5,0.5,0.5), X at face-centers (0.5,0.5,0)
            return Structure.from_spacegroup("Pm-3m", lattice, [metals[0][0], metals[1][0], nonmetals[0][0]], [[0,0,0], [0.5,0.5,0.5], [0.5,0.5,0]])

        # Spinel (AB2X4) - Fd-3m (227)
        if n_elements == 3 and len(metals) == 2 and metals[0][1] == 1 and metals[1][1] == 2 and nonmetals[0][1] == 4:
            lattice = Lattice.cubic(max(a * 1.2, 8.0))
            return Structure.from_spacegroup("Fd-3m", lattice, [metals[0][0], metals[1][0], nonmetals[0][0]], [[0.125,0.125,0.125], [0.5,0.5,0.5], [0.26,0.26,0.26]])
            
        # Layered Oxide (AMO2) - R-3m (166)
        if n_elements == 3 and len(metals) == 2 and metals[0][1] == 1 and metals[1][1] == 1 and nonmetals[0][1] == 2:
            # Hexagonal lattice
            lattice = Lattice.hexagonal(max(a*0.5, 2.8), max(a*1.5, 14.0))
            return Structure.from_spacegroup("R-3m", lattice, [metals[0][0], metals[1][0], nonmetals[0][0]], [[0,0,0], [0,0,0.5], [0,0,0.24]])

    except Exception as e:
        print(f"Prototype generation failed: {e}")
        pass

    return None


# -----------------------------------------------------------------------
# Public helpers
# -----------------------------------------------------------------------
def formula_to_structure(
    formula: str,
    density: float | None = None,
    jitter: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Structure | None:
    parsed = _parse_formula(formula)
    if not parsed:
        return None

    # Phase 7: Try exact crystallographic prototypes first
    structure = _build_from_prototype(parsed, formula, density)
    
    if structure is None:
        # Fallback to heuristic Cartesian graph generation
        atoms = _expand_atoms(parsed)
        if not atoms:
            return None
        positions = _generate_positions(atoms)
        lattice, cart_coords = _build_lattice_and_coords(positions, formula, density)
        structure = Structure(lattice, atoms, cart_coords, coords_are_cartesian=True)

    structure = structure.get_sorted_structure()

    if jitter > 0:
        generator = rng or np.random.default_rng()
        structure = _apply_jitter(structure, jitter, generator)

    return structure


def compute_neighbor_pairs(
    structure: Structure,
    extra_cutoff: float = DEFAULT_NEIGHBOR_PADDING,
) -> list[tuple[int, int, float, np.ndarray]]:
    if structure is None or structure.num_sites < 2:
        return []

    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(structure)
    numbers = atoms.get_atomic_numbers()
    cutoffs = []
    for num in numbers:
        radius = ASE_COVALENT_RADII[num]
        if radius <= 0:
            radius = 1.0
        cutoffs.append(radius + extra_cutoff)

    nl = NeighborList(cutoffs=cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    cell = atoms.get_cell()

    pair_map: dict[tuple[int, int], tuple[float, np.ndarray]] = {}
    
    # Use periodic boundary distances directly from Pymatgen Structure where available
    try:
        # Use get_all_neighbors from pymatgen, this respects minimum image convention properly
        # and returns all neighbors within a specified radius (max cutoff)
        max_cutoff = max(cutoffs) * 2.0
        all_neighbors = structure.get_all_neighbors(r=max_cutoff)
        
        for i, site_neighbors in enumerate(all_neighbors):
            rad_i = cutoffs[i]
            for neighbor in site_neighbors:
                j = neighbor.index
                if i >= j: continue
                
                rad_j = cutoffs[j]
                dist = neighbor.nn_distance
                
                # Bond distance criteria
                if dist <= (rad_i + rad_j):
                    key = (i, j)
                    # For visualization, we need the vector pointing from i to j in Cartesian space
                    vec = neighbor.coords - structure.sites[i].coords
                    
                    stored = pair_map.get(key)
                    if stored is None or dist < stored[0]:
                        pair_map[key] = (dist, vec)
                        
    except Exception as e:
        # Fallback to pure ASE NeighborList if pymatgen approach fails
        print(f"Pymatgen neighbor fallback: {e}")
        for i in range(len(atoms)):
            indices, offsets = nl.get_neighbors(i)
            for j, offset in zip(indices, offsets):
                if i == j:
                    continue
                vec = atoms.positions[j] + np.dot(offset, cell) - atoms.positions[i]
                dist = float(np.linalg.norm(vec))
                if dist <= 1e-6:
                    continue
                key = (min(i, j), max(i, j))
                stored = pair_map.get(key)
                if stored is None or dist < stored[0]:
                    vec_store = vec if i < j else -vec
                    pair_map[key] = (dist, vec_store)

    result = []
    for (i, j), (dist, vec) in sorted(pair_map.items()):
        result.append((i, j, dist, vec))
    return result
