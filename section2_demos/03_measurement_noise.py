"""Demo 3 - Measurement and quantum noise (report Sections 2.6 and 2.7).

1. Born rule / shots: the estimated P(1) of H|0> converges to 0.5 as shots grow.
2. The Bell state on an ideal simulator vs a noise model copied from a
   5-qubit IBM device (FakeManilaV2): noise adds the 'wrong' outcomes 01 and 10.
3. Noise grows with depth: adding identity pairs CX.CX lowers the Bell fidelity.
The device's T1, T2, gate and readout errors are saved for the report's table.
"""
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit.visualization import plot_histogram

from _common import SEED, draw_circuit, run_counts, save_fig, write_csv

print("Demo 3: measurement and noise")
SHOTS = 4096

# --- 1. Shot noise: estimate P(1) for H|0> with more and more shots ----------
qc = QuantumCircuit(1, 1)
qc.h(0)
qc.measure(0, 0)
ideal = AerSimulator()
shot_list = [10, 50, 100, 500, 1000, 5000, 10000, 50000]
rows = []
for shots in shot_list:
    p1 = run_counts(qc, ideal, shots=shots).get("1", 0) / shots
    rows.append({"shots": shots, "estimated_P1": p1, "abs_error": abs(p1 - 0.5),
                 "expected_std_error": np.sqrt(0.25 / shots)})
write_csv("s2_shots_convergence", rows)
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.semilogx(shot_list, [r["estimated_P1"] for r in rows], "o-", label="estimated P(1)")
se = np.array([r["expected_std_error"] for r in rows])
ax.fill_between(shot_list, 0.5 - 2 * se, 0.5 + 2 * se, alpha=0.2, label="0.5 ± 2 std. error")
ax.axhline(0.5, color="k", ls="--", lw=1, label="Born rule P(1) = 0.5")
ax.set_xlabel("number of shots")
ax.set_ylabel("P(1)")
ax.legend()
save_fig(fig, "s2_shots_convergence")

# --- 2. Bell state: ideal vs noisy ---------------------------------------------
fake = FakeManilaV2()
noisy = AerSimulator.from_backend(fake)

bell = QuantumCircuit(2, 2)
bell.h(0)
bell.cx(0, 1)
bell.measure([0, 1], [0, 1])
counts_ideal = run_counts(bell, ideal, shots=SHOTS)
counts_noisy = run_counts(bell, noisy, shots=SHOTS)
save_fig(plot_histogram([counts_ideal, counts_noisy],
                        legend=["ideal simulator", f"noisy ({fake.name})"],
                        title=f"Bell state |Phi+>, {SHOTS} shots"),
         "s2_noise_bell_ideal_vs_noisy")


def bell_fidelity(counts, shots=SHOTS):
    """Fraction of correct outcomes (00 or 11) - a simple success measure."""
    return (counts.get("00", 0) + counts.get("11", 0)) / shots


rows = [{"simulator": name, **{b: c.get(b, 0) for b in ["00", "01", "10", "11"]},
         "P(00)+P(11)": bell_fidelity(c)}
        for name, c in [("ideal", counts_ideal), (f"noisy {fake.name}", counts_noisy)]]
write_csv("s2_noise_bell", rows)
for r in rows:
    print(f"  {r['simulator']:20s} P(00)+P(11) = {r['P(00)+P(11)']:.4f}")

# --- 3. Deeper circuits collect more noise --------------------------------------
rows = []
for pairs in [0, 2, 4, 8, 16, 32]:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    for _ in range(pairs):        # CX.CX = identity, so the ideal state is unchanged
        qc.barrier()
        qc.cx(0, 1)
        qc.barrier()
        qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    if pairs == 2:
        draw_circuit(qc, "s2_noise_depth_circuit_example")
    counts = run_counts(qc, noisy, shots=SHOTS, optimization_level=0)
    rows.append({"extra_cx_pairs": pairs, "total_cx": 1 + 2 * pairs,
                 "P(00)+P(11)": bell_fidelity(counts)})
write_csv("s2_noise_vs_depth", rows)
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot([r["total_cx"] for r in rows], [r["P(00)+P(11)"] for r in rows], "o-")
ax.axhline(1.0, color="k", ls="--", lw=1, label="ideal")
ax.set_xlabel("number of CNOT gates")
ax.set_ylabel("P(00) + P(11)")
ax.set_title(f"Bell-state success vs circuit depth ({fake.name} noise)")
ax.legend()
save_fig(fig, "s2_noise_vs_depth")
for r in rows:
    print(f"  {r['total_cx']:3d} CNOTs: P(00)+P(11) = {r['P(00)+P(11)']:.4f}")

# --- 4. Device noise parameters for the report table ---------------------------
target = fake.target
rows = []
for q in range(fake.num_qubits):
    props = fake.qubit_properties(q)
    sx_err = target["sx"][(q,)].error
    ro_err = target["measure"][(q,)].error
    rows.append({"qubit": q, "T1_us": round(props.t1 * 1e6, 2), "T2_us": round(props.t2 * 1e6, 2),
                 "gate_error (sx or 2q)": sx_err, "readout_error": ro_err})
cx_name = "cx" if "cx" in target.operation_names else "ecr"
for qargs, inst in target[cx_name].items():
    rows.append({"qubit": f"{cx_name} {qargs}", "T1_us": "", "T2_us": "",
                 "gate_error (sx or 2q)": inst.error, "readout_error": ""})
write_csv("s2_fake_backend_properties", rows)
print(f"  saved {fake.name} properties ({fake.num_qubits} qubits), seed {SEED}")
