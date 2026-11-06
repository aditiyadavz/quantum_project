"""Demo 2 - Entanglement with Bell states (report Section 2.4).

H on qubit 0 followed by CNOT(0 -> 1) turns |00> into
|Phi+> = (|00> + |11>)/sqrt(2). Measuring gives only 00 or 11: the two
qubits are perfectly correlated. The other three Bell states are also built.
"""
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram

from _common import SHOTS, draw_circuit, run_counts, save_fig, write_csv

print("Demo 2: Bell states")


def bell_circuit(kind):
    """Return the circuit that prepares one of the four Bell states."""
    qc = QuantumCircuit(2, 2)
    if kind in ("Psi+", "Psi-"):
        qc.x(1)            # start from |01> instead of |00>
    if kind in ("Phi-", "Psi-"):
        qc.x(0)            # start from |1> on qubit 0 -> minus sign after H
    qc.h(0)
    qc.cx(0, 1)
    return qc


rows = []
for kind in ["Phi+", "Phi-", "Psi+", "Psi-"]:
    qc = bell_circuit(kind)
    amps = Statevector(qc).data.round(4)   # amplitudes of |00>,|01>,|10>,|11>
    qc.measure([0, 1], [0, 1])
    counts = run_counts(qc)
    rows.append({"bell_state": kind,
                 "amplitudes_00_01_10_11": " ".join(str(a) for a in amps),
                 **{f"P({b})": counts.get(b, 0) / SHOTS for b in ["00", "01", "10", "11"]}})
    print(f"  {kind}: amplitudes {amps}, counts {dict(sorted(counts.items()))}")
    if kind == "Phi+":
        draw_circuit(qc, "s2_bell_circuit")
        save_fig(plot_histogram(counts, title=f"Bell state |Phi+>, {SHOTS} shots"),
                 "s2_bell_hist")

write_csv("s2_bell_states", rows)
