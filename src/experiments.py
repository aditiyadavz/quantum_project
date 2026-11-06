"""Experiment runner. Every table and figure in Sections 4-5 comes from here.

Usage (from the project folder):
    python -m src.experiments eda        # dataset figures (Section 4.8)
    python -m src.experiments e1         # main comparison, n = 4 qubits
    python -m src.experiments circuits   # circuit diagrams (Section 4.14)
    python -m src.experiments kernelcheck  # circuit kernel == statevector kernel
    python -m src.experiments e2 e3 ...  # experiment grid, see src/grid.py
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel

from .classical_models import build_classical_models, tuned_rbf_svm
from .evaluate import (fit_and_evaluate, plot_confusion_grid, plot_kernel_pair,
                       plot_loss_curve, plot_metric_bars, plot_roc, plot_train_time)
from .grid import TASKS as GRID_TASKS
from .preprocess import RES_DIR, ROOT, SEED, eda_figures, prepare_data, save_fig
from .quantum_models import (TunedQSVC, VQCModel, kernel_circuit, make_ansatz,
                             make_feature_map, circuit_stats)

N_QUBITS = 4
VQC_MAXITER = 100   # spec default: COBYLA(maxiter=100)
TIMING_REPEATS = 5  # fast models are refitted 5x and the median time is reported


def _save(df, name):
    path = RES_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"  results -> {path.relative_to(ROOT)}")


# ----------------------------------------------------------------------------- E1
def run_e1(n_qubits=N_QUBITS, seed=SEED):
    """Main hold-out comparison of all 8 models on the same 80/20 split."""
    print(f"E1: main comparison, {n_qubits} PCA features / qubits, seed {seed}")
    d = prepare_data(n_components=n_qubits, seed=seed)
    print(f"  train {len(d.y_train)} / test {len(d.y_test)}, "
          f"PCA variance kept {d.explained_variance.sum():.3f}")

    models = {"QSVC (quantum kernel)": TunedQSVC(n_qubits=n_qubits, seed=seed),
              "VQC": VQCModel(n_qubits=n_qubits, maxiter=VQC_MAXITER, seed=seed)}
    models.update(build_classical_models(seed))
    models["SVM (RBF, 30 features)"] = tuned_rbf_svm(seed)

    rows, cms, curves, preds, extra = [], {}, {}, {"y_true": d.y_test}, []
    for name, model in models.items():
        full = name.endswith("30 features)")
        Xtr, Xte = (d.X_train_full, d.X_test_full) if full else (d.X_train, d.X_test)
        quantum = name.startswith(("QSVC", "VQC"))
        m, y_pred, y_score = fit_and_evaluate(model, Xtr, d.y_train, Xte, d.y_test,
                                              repeats=1 if quantum else TIMING_REPEATS)
        m.update({"model": name, "quantum": quantum,
                  "features": Xtr.shape[1]})
        if hasattr(model, "best_params_"):
            m["hyperparameters"] = json.dumps(model.best_params_)
        if isinstance(model, TunedQSVC):
            m["hyperparameters"] = json.dumps({"C": model.best_C_})
            extra.append({"model": name, **model.stats(len(d.y_train)),
                          "cv_accuracy_per_C": json.dumps(model.cv_scores_)})
        if isinstance(model, VQCModel):
            extra.append({"model": name, **model.stats(len(d.y_train))})
            pd.DataFrame({"evaluation": np.arange(1, len(model.loss_history_) + 1),
                          "loss": model.loss_history_}).to_csv(
                RES_DIR / "e1_vqc_loss_history.csv", index=False)
            plot_loss_curve({"VQC": model.loss_history_}, "s5_vqc_loss_curve",
                            title=f"VQC training (COBYLA, maxiter {VQC_MAXITER}, {n_qubits} qubits, "
                                  f"{len(d.y_train)} samples)")
        rows.append(m)
        cms[name] = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        preds[name] = y_pred
        if y_score is not None:
            curves[name] = y_score
        print(f"  {name:24s} acc {m['accuracy']:.4f}  rec {m['recall']:.4f}  "
              f"f1 {m['f1']:.4f}  auc {m['roc_auc']:.4f}  train {m['train_time_s']:.2f}s  "
              f"FN {m['fn']}")

    df = pd.DataFrame(rows)
    cols = ["model", "quantum", "features", "accuracy", "precision", "recall", "specificity",
            "f1", "roc_auc", "tn", "fp", "fn", "tp", "train_time_s", "inference_time_s",
            "timing_repeats", "n_train", "n_test", "hyperparameters"]
    df = df[cols]
    _save(df, "e1_main_holdout")
    _save(pd.DataFrame(extra), "e1_quantum_circuit_stats")
    _save(pd.DataFrame(preds), "e1_test_predictions")

    plot_confusion_grid(cms, "s5_confusion_matrices")
    plot_metric_bars(df, "s5_metric_bars")
    plot_roc(curves, d.y_test, "s5_roc_curves")
    plot_train_time(df, "s5_training_time")

    # quantum vs RBF kernel matrices on the training data
    qsvc = models["QSVC (quantum kernel)"]
    svm = models["SVM (RBF)"]
    gamma = svm.best_params_["gamma"]
    if gamma == "scale":
        gamma = 1.0 / (d.X_train.shape[1] * d.X_train.var())
    plot_kernel_pair(qsvc.train_kernel_, rbf_kernel(d.X_train, gamma=gamma), d.y_train,
                     "s5_kernel_matrices")
    off = ~np.eye(len(d.y_train), dtype=bool)
    print(f"  quantum kernel off-diagonal mean {qsvc.train_kernel_[off].mean():.4f}, "
          f"std {qsvc.train_kernel_[off].std():.4f}")
    return df


# ----------------------------------------------------------------------------- circuits
def draw_circuits(n_qubits=N_QUBITS):
    print("Circuit diagrams")
    fm = make_feature_map(n_qubits, reps=2, entanglement="linear")
    ans = make_ansatz(n_qubits, reps=3)
    save_fig(fm.draw("mpl", fold=-1), "s4_zz_feature_map")
    save_fig(ans.draw("mpl", fold=-1), "s4_real_amplitudes_ansatz")
    full = fm.compose(ans)
    save_fig(full.decompose().draw("mpl", fold=-1), "s4_vqc_full_circuit")
    d = prepare_data(n_components=n_qubits)
    kc = kernel_circuit(fm, d.X_train[0], d.X_train[1])
    save_fig(kc.draw("mpl", fold=-1), "s4_kernel_circuit")
    rows = [{"circuit": "ZZFeatureMap (reps=2, linear)", **circuit_stats(fm)},
            {"circuit": "RealAmplitudes ansatz (reps=3)", **circuit_stats(ans)},
            {"circuit": "VQC full circuit", **circuit_stats(full)},
            {"circuit": "Kernel compute-uncompute circuit",
             **circuit_stats(kc.remove_final_measurements(inplace=False))}]
    _save(pd.DataFrame(rows), "s4_circuit_stats")
    print(pd.DataFrame(rows).to_string(index=False))


# ----------------------------------------------------------------------------- kernel check
def kernel_check(n_samples=60, n_qubits=N_QUBITS):
    """Show the fast statevector kernel equals the compute-uncompute circuit kernel,
    and that our tuned model gives the same predictions as Qiskit's QSVC class."""
    print(f"Kernel check on {n_samples} training samples")
    d = prepare_data(n_components=n_qubits)
    X = d.X_train[:n_samples]
    times, mats = {}, {}
    for backend in ["statevector", "circuits"]:
        model = TunedQSVC(n_qubits=n_qubits, backend=backend, C=1.0)
        t0 = time.perf_counter()
        mats[backend] = model.fit(X, d.y_train[:n_samples]).train_kernel_
        times[backend] = time.perf_counter() - t0
    diff = np.abs(mats["statevector"] - mats["circuits"]).max()

    tuned = TunedQSVC(n_qubits=n_qubits, C=1.0).fit(d.X_train, d.y_train)
    qsvc = tuned.as_qiskit_qsvc().fit(d.X_train, d.y_train)
    same = bool(np.all(qsvc.predict(d.X_test) == tuned.predict(d.X_test)))
    row = {"samples": n_samples, "kernel_entries": n_samples * (n_samples - 1) // 2,
           "statevector_time_s": times["statevector"], "circuit_time_s": times["circuits"],
           "max_abs_difference": diff, "qiskit_QSVC_same_test_predictions": same}
    _save(pd.DataFrame([row]), "kernel_backend_check")
    print(f"  {row}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    tasks = {"eda": eda_figures, "e1": run_e1, "circuits": draw_circuits,
             "kernelcheck": kernel_check, **GRID_TASKS}
    p.add_argument("tasks", nargs="+", choices=list(tasks))
    args = p.parse_args()
    for t in args.tasks:
        t0 = time.perf_counter()
        tasks[t]()
        print(f"  [{t} finished in {time.perf_counter() - t0:.1f} s]\n")
