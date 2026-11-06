"""Demo 6 - Shor's algorithm overview: factoring N = 15 with a = 7 (Section 2.11).

Factoring reduces to finding the period r of f(x) = a^x mod N.
Quantum phase estimation on U|y> = |a*y mod N> gives phases s/r; continued
fractions recover r; then gcd(a^(r/2) +- 1, N) gives the factors.
Uses the textbook circuit: 8 counting qubits + 4 work qubits.
"""
import math
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Operator
from qiskit.visualization import plot_histogram

from _common import SHOTS, draw_circuit, run_counts, save_fig, write_csv

print("Demo 6: Shor overview (N = 15, a = 7)")
N, A, N_COUNT = 15, 7, 8


def qft(n, barriers=False):
    """Quantum Fourier Transform built from H and controlled-phase gates."""
    qc = QuantumCircuit(n, name="QFT")
    for j in reversed(range(n)):
        qc.h(j)
        for k in reversed(range(j)):
            qc.cp(math.pi / 2 ** (j - k), k, j)
        if barriers:
            qc.barrier()
    for q in range(n // 2):
        qc.swap(q, n - q - 1)
    return qc


# sanity check: our QFT equals Qiskit's QFTGate
assert Operator(qft(4)).equiv(Operator(QFTGate(4))), "QFT construction is wrong"
draw_circuit(qft(4, barriers=True), "s2_qft_4q_circuit")


def c_amod15(a, power):
    """Controlled U^power with U|y> = |a*y mod 15> (Qiskit textbook construction)."""
    U = QuantumCircuit(4)
    for _ in range(power):
        if a in [2, 13]:
            U.swap(2, 3); U.swap(1, 2); U.swap(0, 1)
        if a in [7, 8]:
            U.swap(0, 1); U.swap(1, 2); U.swap(2, 3)
        if a in [4, 11]:
            U.swap(1, 3); U.swap(0, 2)
        if a in [7, 11, 13]:
            U.x(range(4))
    gate = U.to_gate()
    gate.name = f"{a}^{power} mod 15"
    return gate.control()


# --- 1. classical view of the period -------------------------------------------
xs = np.arange(16)
fx = [pow(A, int(x), N) for x in xs]
fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.plot(xs, fx, "o-")
ax.set_xlabel("x")
ax.set_ylabel(f"{A}^x mod {N}")
ax.set_title(f"f(x) = {A}^x mod {N} repeats with period r = 4")
ax.set_xticks(xs)
save_fig(fig, "s2_shor_modexp_period")

# --- 2. quantum period finding ---------------------------------------------------
qc = QuantumCircuit(N_COUNT + 4, N_COUNT)
qc.h(range(N_COUNT))                 # counting register in superposition
qc.x(N_COUNT)                        # work register = |1>
for q in range(N_COUNT):
    qc.append(c_amod15(A, 2 ** q), [q] + list(range(N_COUNT, N_COUNT + 4)))
qc.append(qft(N_COUNT).inverse().to_gate(label="QFT†"), range(N_COUNT))
qc.measure(range(N_COUNT), range(N_COUNT))
draw_circuit(qc, "s2_shor_circuit", fold=-1)

counts = run_counts(qc)
phase_counts = {}
rows = []
for bits, c in sorted(counts.items(), key=lambda kv: int(kv[0], 2)):
    dec = int(bits, 2)
    phase = dec / 2 ** N_COUNT
    frac = Fraction(phase).limit_denominator(N)
    r = frac.denominator
    is_period = pow(A, r, N) == 1          # check the guess classically
    factors = ""
    if is_period and r % 2 == 0 and pow(A, r // 2, N) != N - 1:
        factors = f"{math.gcd(pow(A, r // 2) - 1, N)} x {math.gcd(pow(A, r // 2) + 1, N)}"
    rows.append({"register": bits, "decimal": dec, "phase": phase, "counts": c,
                 "fraction": f"{frac.numerator}/{frac.denominator}", "guess_r": r,
                 "a^r mod N == 1": is_period,
                 "factors_from_r": factors or "none (retry)"})
    phase_counts[f"{phase:.3f}"] = c
    print(f"  {bits} -> phase {phase:.3f} = {frac} -> r = {r} -> {factors or 'retry'}")
write_csv("s2_shor_results", rows)
save_fig(plot_histogram(phase_counts, title=f"Measured phases s/r (N={N}, a={A}, {SHOTS} shots)"),
         "s2_shor_phase_hist")

r = 4
print(f"  worked example: r = {r}, gcd({A}^{r // 2} - 1, {N}) = {math.gcd(A ** 2 - 1, N)}, "
      f"gcd({A}^{r // 2} + 1, {N}) = {math.gcd(A ** 2 + 1, N)}")
