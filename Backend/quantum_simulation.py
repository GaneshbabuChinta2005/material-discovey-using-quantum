from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
import numpy as np


def run_quantum_simulation(num_qubits=2):
    """
    Simple quantum-inspired energy estimation.
    Uses a parameterized quantum circuit.
    """

    # Create a quantum circuit
    qc = QuantumCircuit(num_qubits)

    # Apply parameterized rotations
    theta = np.pi / 4
    for q in range(num_qubits):
        qc.ry(theta, q)

    # Add entanglement
    if num_qubits > 1:
        qc.cx(0, 1)

    # Measure
    qc.measure_all()

    # Run simulation
    sampler = StatevectorSampler()
    job = sampler.run([qc])
    result = job.result()

    # Extract probabilities
    probs = result[0].data.meas.get_counts()

    # Convert to a simple “energy-like” score
    energy_score = sum(int(state, 2) * count for state, count in probs.items())
    energy_score /= sum(probs.values())

    return energy_score


if __name__ == "__main__":
    energy = run_quantum_simulation()
    print("Quantum energy estimate:", energy)
