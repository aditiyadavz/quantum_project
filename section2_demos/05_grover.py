"""Demo 5 - Grover's search algorithm (report Section 2.10).

Searches N = 2^n items for one marked item. Each Grover iteration =
oracle (flip the sign of the marked state) + diffusion (reflect about the mean).
The optimal number of iterations is about (pi/4) sqrt(N), versus ~N/2
classical guesses on average.
"""
import math

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram

from _common import SHOTS, draw_circuit, run_counts, save_fig, write_csv

print("Demo 5: Grover search")


def multi_controlled_z(qc, n):
    """Apply a Z on the last qubit controlled by all others (H-MCX-H)."""
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)


def oracle(n, marked):
    """Phase oracle: |marked> -> -|marked>, all other states unchanged."""
    qc = QuantumCircuit(n, name="Oracle")
    zeros = [q for q in range(n) if marked[::-1][q] == "0"]  # Qiskit is little-endian
    if zeros:
        qc.x(zeros)
    multi_controlled_z(qc, n)
    if zeros:
        qc.x(zeros)
    return qc


def diffuser(n):
    """Diffusion operator 2|s><s| - I (inversion about the mean)."""
    qc = QuantumCircuit(n, name="Diffuser")
    qc.h(range(n))
    qc.x(range(n))
    multi_controlled_z(qc, n)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def grover(n, marked, iterations, measure=True):
    qc = QuantumCircuit(n, n) if measure else QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(iterations):
        qc.barrier()
        qc.compose(oracle(n, marked), inplace=True)
        qc.compose(diffuser(n), inplace=True)
    if measure:
        qc.measure(range(n), range(n))
    return qc


def optimal_iterations(n):
    return math.floor(math.pi / 4 * math.sqrt(2 ** n))


# --- 2-qubit and 3-qubit searches ------------------------------------------------
rows = []
for n, marked in [(2, "11"), (3, "101")]:
    k = optimal_iterations(n)
    if n == 3:
        draw_circuit(oracle(n, marked), "s2_grover_oracle_3q")
        draw_circuit(diffuser(n), "s2_grover_diffuser_3q")
    qc = grover(n, marked, k)
    draw_circuit(qc, f"s2_grover_circuit_{n}q", fold=-1)
    counts = run_counts(qc)
    save_fig(plot_histogram(counts, title=f"Grover, n = {n}, marked |{marked}>, "
                                          f"{k} iteration(s), {SHOTS} shots"),
             f"s2_grover_hist_{n}q")
    theta = math.asin(1 / math.sqrt(2 ** n))
    rows.append({"n_qubits": n, "N": 2 ** n, "marked": marked, "iterations": k,
                 "measured_success": counts.get(marked, 0) / SHOTS,
                 "theoretical_success": math.sin((2 * k + 1) * theta) ** 2})
    print(f"  n={n} marked={marked} k={k}: success {rows[-1]['measured_success']:.4f} "
          f"(theory {rows[-1]['theoretical_success']:.4f})")
write_csv("s2_grover_runs", rows)

# --- success probability vs number of iterations ---------------------------------
rows = []
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for n, marked, style in [(3, "101", "o"), (4, "1011", "s")]:
    theta = math.asin(1 / math.sqrt(2 ** n))
    ks = list(range(0, 9))
    sim = [Statevector(grover(n, marked, k, measure=False)).probabilities_dict()
           .get(marked, 0.0) for k in ks]
    for k, p in zip(ks, sim):
        rows.append({"n_qubits": n, "iterations": k, "success_probability": p,
                     "theory": math.sin((2 * k + 1) * theta) ** 2})
    fine = np.linspace(0, 8, 200)
    line, = ax.plot(fine, np.sin((2 * fine + 1) * theta) ** 2, lw=1, alpha=0.6)
    ax.plot(ks, sim, style, color=line.get_color(),
            label=f"n = {n} (N = {2 ** n}), optimum k = {optimal_iterations(n)}")
ax.set_xlabel("number of Grover iterations k")
ax.set_ylabel("P(marked state)")
ax.set_title("Grover success probability (points: simulation, lines: theory)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
save_fig(fig, "s2_grover_success_vs_iterations")
write_csv("s2_grover_success_vs_iterations", rows)

# --- query scaling: classical N/2 vs quantum (pi/4) sqrt(N) -----------------------
rows = [{"n": n, "N": 2 ** n, "classical_average_queries": 2 ** n / 2,
         "grover_queries": optimal_iterations(n)} for n in range(2, 21, 2)]
write_csv("s2_grover_query_scaling", rows)
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.loglog([r["N"] for r in rows], [r["classical_average_queries"] for r in rows], "o-",
          label="classical: N/2  (O(N))")
ax.loglog([r["N"] for r in rows], [r["grover_queries"] for r in rows], "s-",
          label=r"Grover: $\lfloor\frac{\pi}{4}\sqrt{N}\rfloor$  (O($\sqrt{N}$))")
ax.set_xlabel("search space size N")
ax.set_ylabel("oracle queries")
ax.legend()
save_fig(fig, "s2_grover_query_scaling")
