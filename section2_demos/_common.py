"""Shared helpers for the Section 2 Qiskit demos.

Every demo saves its figures to ../figures (300 dpi PNG) and its numbers to
../results (CSV), so the report only uses outputs that were actually produced.
"""
from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")  # draw to files, no window
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit_aer import AerSimulator

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

SEED = 42      # fixed seed for reproducible results
SHOTS = 1024   # default number of measurement shots


def save_fig(fig, name):
    """Save a matplotlib figure as figures/<name>.png at 300 dpi."""
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure  -> {path.relative_to(ROOT)}")


def draw_circuit(circuit, name, **kwargs):
    """Draw a circuit with Qiskit's matplotlib drawer and save it."""
    save_fig(circuit.draw("mpl", **kwargs), name)


def run_counts(circuit, backend=None, shots=SHOTS, optimization_level=1):
    """Transpile and run a circuit on a simulator; return measurement counts."""
    backend = backend or AerSimulator()
    compiled = transpile(circuit, backend, seed_transpiler=SEED,
                         optimization_level=optimization_level)
    job = backend.run(compiled, shots=shots, seed_simulator=SEED)
    return job.result().get_counts()


def write_csv(name, rows):
    """Write a list of dicts to results/<name>.csv."""
    path = RES_DIR / f"{name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  results -> {path.relative_to(ROOT)}")
