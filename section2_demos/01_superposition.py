"""Demo 1 - Superposition (report Sections 2.2 and 2.3).

1. A Hadamard gate puts one qubit into |+> = (|0> + |1>)/sqrt(2).
2. Hadamards on 3 qubits give a uniform superposition of all 8 basis states.
Outputs: circuit diagrams, Bloch spheres, measurement histograms.
"""
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector, plot_bloch_vector, plot_histogram

from _common import SHOTS, draw_circuit, run_counts, save_fig, write_csv

print("Demo 1: superposition")

# --- Bloch-sphere reference states for Section 2.2 ---------------------------
fig = plt.figure(figsize=(14, 3.8))
states = {"|0>": [0, 0, 1], "|1>": [0, 0, -1], "|+>": [1, 0, 0], "|+i>": [0, 1, 0]}
for i, (label, vec) in enumerate(states.items(), start=1):
    ax = fig.add_subplot(1, 4, i, projection="3d")
    plot_bloch_vector(vec, title=label, ax=ax)
save_fig(fig, "s2_bloch_basis_states")

# --- 1 qubit: H|0> = |+> ------------------------------------------------------
qc1 = QuantumCircuit(1, 1)
qc1.h(0)
state = Statevector(qc1)                      # state before measurement
print("  H|0> amplitudes:", state.data.round(4))
save_fig(plot_bloch_multivector(state), "s2_superposition_bloch")

qc1.measure(0, 0)
draw_circuit(qc1, "s2_superposition_1q_circuit")
counts1 = run_counts(qc1)
save_fig(plot_histogram(counts1, title=f"H|0>, {SHOTS} shots"), "s2_superposition_1q_hist")

# --- 3 qubits: H on every qubit ---------------------------------------------
qc3 = QuantumCircuit(3, 3)
qc3.h(range(3))
qc3.measure(range(3), range(3))
draw_circuit(qc3, "s2_superposition_3q_circuit")
counts3 = run_counts(qc3)
save_fig(plot_histogram(counts3, title=f"H on 3 qubits, {SHOTS} shots"),
         "s2_superposition_3q_hist")

rows = [{"circuit": "1 qubit H", "state": k, "counts": v, "probability": v / SHOTS,
         "ideal_probability": 0.5} for k, v in sorted(counts1.items())]
rows += [{"circuit": "3 qubit H", "state": k, "counts": v, "probability": v / SHOTS,
          "ideal_probability": 0.125} for k, v in sorted(counts3.items())]
write_csv("s2_superposition", rows)
for r in rows:
    print(f"  {r['circuit']:10s} {r['state']}: {r['probability']:.4f}")
