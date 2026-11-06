"""Demo 4 - Deutsch-Jozsa algorithm, n = 3 (report Section 2.9).

Promise: f:{0,1}^3 -> {0,1} is either constant or balanced.
One query to the oracle decides which: measuring 000 means constant,
anything else means balanced. A classical algorithm needs up to
2^(n-1) + 1 = 5 queries in the worst case.
"""
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram

from _common import SHOTS, draw_circuit, run_counts, save_fig, write_csv

print("Demo 4: Deutsch-Jozsa")
N = 3  # number of input qubits


def constant_oracle(n, value=1):
    """f(x) = value for every x. For value 1 the output qubit is flipped."""
    oracle = QuantumCircuit(n + 1, name="U_f const")
    if value == 1:
        oracle.x(n)
    return oracle


def balanced_oracle(n, mask="101"):
    """f(x) = parity of x XOR mask -> exactly half the inputs give 1."""
    oracle = QuantumCircuit(n + 1, name="U_f bal")
    flips = [q for q in range(n) if mask[::-1][q] == "1"]
    oracle.x(flips)
    for q in range(n):
        oracle.cx(q, n)            # output ^= x_q  (parity)
    oracle.x(flips)
    return oracle


def deutsch_jozsa(oracle, n):
    """Build the full DJ circuit around an oracle."""
    qc = QuantumCircuit(n + 1, n)
    qc.x(n)                        # output qubit in |1>
    qc.h(range(n + 1))             # inputs -> |+>^n, output -> |->
    qc.barrier()
    qc.compose(oracle, inplace=True)   # phase kickback: (-1)^f(x)
    qc.barrier()
    qc.h(range(n))                 # interfere
    qc.measure(range(n), range(n))
    return qc


results = {}
rows = []
for label, oracle in [("constant", constant_oracle(N)), ("balanced", balanced_oracle(N))]:
    draw_circuit(oracle, f"s2_dj_oracle_{label}")
    qc = deutsch_jozsa(oracle, N)
    draw_circuit(qc, f"s2_dj_circuit_{label}")
    counts = run_counts(qc)
    results[label] = counts
    p_zero = counts.get("0" * N, 0) / SHOTS
    verdict = "constant" if p_zero > 0.5 else "balanced"
    rows.append({"oracle": label, "counts": dict(counts), "P(000)": p_zero,
                 "decision": verdict, "correct": verdict == label})
    print(f"  {label:8s} oracle: counts {dict(counts)} -> decided '{verdict}'")
write_csv("s2_deutsch_jozsa", rows)

save_fig(plot_histogram([results["constant"], results["balanced"]],
                        legend=["constant oracle", "balanced oracle"],
                        title=f"Deutsch-Jozsa, n = {N}, {SHOTS} shots"),
         "s2_dj_hist")

# --- query complexity comparison --------------------------------------------
rows = [{"n": n, "classical_worst_case_queries": 2 ** (n - 1) + 1, "quantum_queries": 1}
        for n in range(1, 11)]
write_csv("s2_dj_query_complexity", rows)
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.semilogy([r["n"] for r in rows], [r["classical_worst_case_queries"] for r in rows],
            "o-", label="classical deterministic: $2^{n-1}+1$")
ax.semilogy([r["n"] for r in rows], [1] * len(rows), "s-", label="Deutsch-Jozsa: 1")
ax.set_xlabel("number of input bits n")
ax.set_ylabel("oracle queries (log scale)")
ax.legend()
save_fig(fig, "s2_dj_query_complexity")
