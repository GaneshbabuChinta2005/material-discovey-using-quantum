"""
Quantum Service — Research-Grade Element-Aware Quantum Validation.

Provides a lightweight quantum-circuit-based screening score for candidate
materials. Acts as the FAST validation path during batch discovery. The
full VQE service (vqe_service.py) is reserved for detailed analysis of
top-ranked individual candidates.

Research-grade v3 improvements:
- Element-aware Hamiltonian-inspired circuit angles (EN & IE based)
- Quantum-inspired energy correction term appended to output
- Ionization energy mapped to circuit depth contribution
- Honest output with energy_correction label for UI display
- Clear fallback chain: Qiskit → classical approximation
"""

from __future__ import annotations

import re
import math


_ELECTRONEGATIVITY = {
    "H": 2.20, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55,
    "N": 3.04, "O": 3.44, "F": 3.98, "Na": 0.93, "Mg": 1.31,
    "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16,
    "K": 0.82, "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66,
    "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65,
    "Ga": 1.81, "Ge": 2.01, "As": 2.18, "Se": 2.55, "Br": 2.96,
    "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33, "Nb": 1.60,
    "Mo": 2.16, "Pd": 2.20, "Ag": 1.93, "Cd": 1.69, "Sn": 1.96,
    "Sb": 2.05, "Te": 2.10, "I": 2.66, "Cs": 0.79, "Ba": 0.89,
    "La": 1.10, "Ce": 1.12, "Hf": 1.30, "Ta": 1.50, "W": 2.36,
    "Pt": 2.28, "Au": 2.54, "Pb": 2.33, "Bi": 2.02,
}

_IONIZATION_ENERGY = {
    "H": 13.6, "Li": 5.4, "Be": 9.3, "B": 8.3, "C": 11.3, "N": 14.5,
    "O": 13.6, "F": 17.4, "Na": 5.1, "Mg": 7.6, "Al": 6.0, "Si": 8.2,
    "P": 10.5, "S": 10.4, "Cl": 13.0, "K": 4.3, "Ca": 6.1,
    "Ti": 6.8, "V": 6.7, "Cr": 6.8, "Mn": 7.4, "Fe": 7.9, "Co": 7.9,
    "Ni": 7.6, "Cu": 7.7, "Zn": 9.4, "Ga": 6.0, "Ge": 7.9,
    "Zr": 6.6, "Nb": 6.8, "Mo": 7.1, "Ru": 7.4, "Pd": 8.3, "Ag": 7.6,
    "Sn": 7.3, "Ba": 5.2, "La": 5.6, "W": 7.9, "Pt": 9.0,
    "Au": 9.2, "Pb": 7.4, "Bi": 7.3, "Sr": 5.7, "Y": 6.2,
}


def _compute_surrogate_electronic_structure(elements: list[str]) -> dict:
    """
    Surrogate model for electronic structure insights.
    Estimates orbital interaction strength and DOS dispersion character.
    """
    if not elements:
        return {"orbital_interaction": 0.5, "dos_dispersion": 0.5}

    en_vals = [_ELECTRONEGATIVITY.get(e, 2.0) for e in elements]
    ie_vals = [_IONIZATION_ENERGY.get(e, 8.0) for e in elements]
    
    # Orbital Interaction Strength (~ reciprocal of IE spread)
    # High spread = localized levels; Low spread = strong hybridization potential
    ie_spread = max(ie_vals) - min(ie_vals) if len(ie_vals) > 1 else 1.0
    orbital_interaction = 1.0 / (1.0 + math.exp(0.5 * (ie_spread - 5.0)))
    
    # DOS Dispersion Proxy (Electronegativity mean-range balance)
    en_mean = sum(en_vals) / len(en_vals)
    en_diff = max(en_vals) - min(en_vals)
    # Covalent systems (low EN diff, mid EN) have higher dispersion
    dos_dispersion = 1.0 / (1.0 + math.exp(2.0 * (en_diff - 1.5)))
    
    return {
        "orbital_interaction": round(orbital_interaction, 3),
        "dos_dispersion": round(dos_dispersion, 3),
        "en_mean": round(en_mean, 2)
    }


def _compute_quantum_energy_correction(elements: list[str]) -> float:
    """
    Compute a quantum-inspired energy correction (VQE surrogate).
    Includes orbital interaction and DOS dispersion terms.
    """
    surr = _compute_surrogate_electronic_structure(elements)
    oi = surr["orbital_interaction"]
    dd = surr["dos_dispersion"]
    
    # Effective correction: favor systems with high orbital interaction and moderate dispersion
    correction = -(0.2 * oi) - (0.1 * dd)
    return round(correction, 4)


def run_quantum_validation(
    n_qubits: int = 3,
    formula: str | None = None,
) -> dict:
    """
    Run a fast quantum-circuit validation for a candidate material.

    Circuit design:
    - Layer 1: Ry(θ) with element-aware θ from electronegativity
    - Layer 2: Linear CNOT entanglement
    - Layer 3: Ry(θ × IE_factor) — ionization energy modulation
    - Measurement + weighted expectation value

    The result is used as a REFINEMENT stage bonus, not a primary predictor.

    Args:
        n_qubits: circuit width (capped 2–6)
        formula:  chemical formula for element-aware angles

    Returns:
        dict with quantum_score (0–10), energy_correction (a.u.),
        methodology, and metadata
    """
    n_qubits = max(2, min(n_qubits, 6))
    elements = re.findall(r'[A-Z][a-z]?', formula) if formula else ["H"]
    n_elem   = len(elements)

    # Build element-aware rotation angles
    thetas_l1, thetas_l3 = [], []
    for q in range(n_qubits):
        elem = elements[q % n_elem]
        en = _ELECTRONEGATIVITY.get(elem, 2.0)
        ie = _IONIZATION_ENERGY.get(elem, 8.0)
        thetas_l1.append((en / 4.0) * math.pi)
        thetas_l3.append(((ie - 4.0) / 16.0) * math.pi)

    # Quantum-inspired energy correction (always computed, even without Qiskit)
    energy_correction = _compute_quantum_energy_correction(elements)

    try:
        from qiskit import QuantumCircuit, transpile
        try:
            from qiskit_aer import AerSimulator
            simulator = AerSimulator()
        except ImportError:
            # qiskit_aer not installed — fall through to classical fallback below
            raise ImportError("qiskit_aer not available")

        qc = QuantumCircuit(n_qubits)

        # Layer 1: EN-based Ry rotations
        for i in range(n_qubits):
            qc.ry(thetas_l1[i], i)

        # Entanglement
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)

        # Layer 3: IE-based second rotation
        for i in range(n_qubits):
            qc.ry(thetas_l3[i], i)

        qc.measure_all()

        transpiled = transpile(qc, simulator)
        result = simulator.run(transpiled, shots=512).result()
        counts = result.get_counts()

        total = sum(counts.values())
        score = 0.0
        for bitstring, count in counts.items():
            bits = [int(b) for b in bitstring]
            weight = sum(bits) / len(bits)
            score += weight * count
        score /= total

        quantum_score = round(score * 10.0, 3)
        methodology = "element-aware-circuit-qiskit"

    except ImportError:
        # Classical fallback — EN-weighted approximation
        avg_en = sum(_ELECTRONEGATIVITY.get(e, 2.0) for e in elements) / n_elem
        avg_ie = sum(_IONIZATION_ENERGY.get(e, 8.0) for e in elements) / n_elem
        score_raw = 5.0 + (avg_en - 2.0) * 1.2 - (avg_ie - 8.0) * 0.1
        quantum_score = round(max(0.5, min(9.5, score_raw)), 3)
        methodology = "classical-fallback"

    except Exception as ex:
        quantum_score = 5.0
        methodology = f"error-fallback: {ex}"

    # Surrogate electronic structure insights
    surr = _compute_surrogate_electronic_structure(elements)

    return {
        "quantum_score":             quantum_score,
        "energy_correction":         energy_correction,
        "orbital_interaction":       surr["orbital_interaction"],
        "dos_dispersion":            surr["dos_dispersion"],
        "energy_correction_label":   "Quantum-inspired correction (a.u., not DFT)",
        "n_qubits":                  n_qubits,
        "methodology":               methodology,
        "formula":                   formula,
        "elements_used":             list(dict.fromkeys(elements))[:n_qubits],
        "role":                      "Refinement ranking bonus (not primary predictor)",
    }


# -----------------------------------------------------------------------
if __name__ == "__main__":
    tests = ["TiO2", "NaCl", "Fe2O3", "LiCoO2", "BaTiO3", "CsPbBr3"]
    for f in tests:
        r = run_quantum_validation(n_qubits=3, formula=f)
        print(f"{f:12s}  score={r['quantum_score']:.3f}  "
              f"correction={r['energy_correction']:+.4f} a.u.  "
              f"method={r['methodology']}")
