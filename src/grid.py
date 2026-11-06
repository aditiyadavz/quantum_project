"""Experiment grid E1-CV, E2-E9 (Sections 5.3-5.5). Run through src/experiments.py.

Repeated experiments use seeds 42, 43, 44 (each seed gives a different
stratified train/test split and a different VQC starting point) and report
mean +- standard deviation.

Run-time choices (stated in the report):
  * QSVC uses the exact statevector kernel (proved identical to the circuit
    kernel by `kernelcheck`) on the full 455-sample training set.
  * VQC sweeps (E2, E5, E6) use a stratified 200-sample training subset,
    because each full VQC training run takes several minutes on the laptop.
  * The noisy simulation (E8) uses a 100-sample QSVC training subset, because
    every kernel entry is a separate noisy circuit run (~22 ms each).
"""
import json
import time

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit_machine_learning.neural_networks import SamplerQNN
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

from .classical_models import build_classical_models, tuned_rbf_svm
from .evaluate import compute_metrics, fit_and_evaluate, plot_loss_curve
from .preprocess import RES_DIR, ROOT, SEED, load_dataset, prepare_data, save_fig, transform_split
from .quantum_models import TunedQSVC, VQCModel

SEEDS = [42, 43, 44]
VQC_TRAIN = 200
NOISE_TRAIN = 100
REPEATS = 5   # timing repeats for fast classical models


def _save(df, name):
    path = RES_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"  results -> {path.relative_to(ROOT)}")
    return df


def _summary(df, keys, name):
    """mean and std over seeds for the main metrics."""
    cols = [c for c in ["accuracy", "recall", "f1", "roc_auc", "train_time_s",
                        "inference_time_s", "kernel_offdiag_mean", "final_loss"] if c in df]
    s = df.groupby(keys, sort=False)[cols].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s = s.reset_index()
    s.insert(len(keys), "runs", df.groupby(keys, sort=False).size().values)
    return _save(s, name)


def _kernel_offdiag(K):
    off = K[~np.eye(len(K), dtype=bool)]
    return float(off.mean()), float(off.std())


# ----------------------------------------------------------------------------- single runs
def qsvc_once(seed, n=4, reps=2, entanglement="linear", kind="zz", train_size=None, data=None):
    d = data or prepare_data(n, train_size=train_size, seed=seed)
    model = TunedQSVC(n_qubits=n, reps=reps, entanglement=entanglement, kind=kind, seed=seed)
    m, _, _ = fit_and_evaluate(model, d.X_train, d.y_train, d.X_test, d.y_test)
    mean, std = _kernel_offdiag(model.train_kernel_)
    st = model.stats(len(d.y_train))
    return {**m, "model": "QSVC", "seed": seed, "C": model.best_C_,
            "kernel_offdiag_mean": mean, "kernel_offdiag_std": std,
            "circuit_depth": st["depth"], "cx_gates": st["cx_gates"],
            "kernel_circuits_train": st["circuit_evaluations_train"],
            "pca_variance": float(d.explained_variance.sum())}


def svm_once(seed, n=4, train_size=None, data=None):
    d = data or prepare_data(n, train_size=train_size, seed=seed)
    model = tuned_rbf_svm(seed)
    m, _, _ = fit_and_evaluate(model, d.X_train, d.y_train, d.X_test, d.y_test, repeats=REPEATS)
    return {**m, "model": "SVM (RBF)", "seed": seed, "C": model.best_params_["C"],
            "pca_variance": float(d.explained_variance.sum())}


def vqc_once(seed, n=4, ansatz_reps=3, optimizer="cobyla", maxiter=100, train_size=VQC_TRAIN,
             data=None, return_model=False):
    d = data or prepare_data(n, train_size=train_size, seed=seed)
    model = VQCModel(n_qubits=n, ansatz_reps=ansatz_reps, optimizer=optimizer,
                     maxiter=maxiter, seed=seed)
    m, _, _ = fit_and_evaluate(model, d.X_train, d.y_train, d.X_test, d.y_test)
    st = model.stats(len(d.y_train))
    row = {**m, "model": "VQC", "seed": seed, "optimizer": optimizer,
           "ansatz_reps": ansatz_reps, "parameters": st["parameters"],
           "circuit_depth": st["depth"], "cx_gates": st["cx_gates"],
           "loss_history_points": len(model.loss_history_),
           "objective_evaluations": st["objective_evaluations"],
           "circuit_evaluations_train": st["circuit_evaluations_train"],
           "initial_loss": model.loss_history_[0], "final_loss": model.loss_history_[-1],
           "min_loss": min(model.loss_history_)}
    return (row, model, d) if return_model else row


def _log(row, label):
    print(f"  {label:38s} acc {row['accuracy']:.4f} rec {row['recall']:.4f} "
          f"train {row['train_time_s']:.2f}s")


# ----------------------------------------------------------------------------- plotting
def plot_sweep(df, x, name, xlabel, models=None, xticks=None, logx=False):
    """Two panels: accuracy (mean +- std over seeds) and training time."""
    models = models or list(dict.fromkeys(df["model"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for mdl in models:
        g = df[df["model"] == mdl].groupby(x, sort=False)
        xs = list(g.groups.keys())
        for ax, col in [(axes[0], "accuracy"), (axes[1], "train_time_s")]:
            mean, std = g[col].mean().values, g[col].std().fillna(0).values
            ax.errorbar(xs, mean, yerr=std, marker="o", capsize=3, label=mdl)
    axes[0].set_ylabel("test accuracy (mean ± std)")
    axes[1].set_ylabel("training time (s, log scale)")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
        if logx:
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        if xticks is not None:
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(t) for t in xticks])
    axes[0].legend()
    fig.tight_layout()
    save_fig(fig, name)


# ----------------------------------------------------------------------------- E1 with CV
def run_e1_cv(n=4, folds=5, seed=SEED):
    """5-fold stratified CV of all 8 models on the full dataset (preprocessing per fold)."""
    print(f"E1-CV: {folds}-fold stratified cross-validation, all models")
    X, y = load_dataset()
    X, y = X.values, y.values
    # resume support: folds already saved are kept and not trained again
    path = RES_DIR / "e1_cv_folds.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {r["fold"] for r in rows}
    if done:
        print(f"  resuming: folds {sorted(done)} already saved")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(cv.split(X, y), start=1):
        if fold in done:
            continue
        d = transform_split(X[tr], X[te], y[tr], y[te], n, seed)
        models = {"QSVC (quantum kernel)": TunedQSVC(n_qubits=n, seed=seed),
                  "VQC": VQCModel(n_qubits=n, seed=seed)}
        models.update(build_classical_models(seed))
        models["SVM (RBF, 30 features)"] = tuned_rbf_svm(seed)
        for name, model in models.items():
            full = name.endswith("30 features)")
            quantum = name.startswith(("QSVC", "VQC"))
            Xtr, Xte = (d.X_train_full, d.X_test_full) if full else (d.X_train, d.X_test)
            m, _, _ = fit_and_evaluate(model, Xtr, d.y_train, Xte, d.y_test,
                                       repeats=1 if quantum else REPEATS)
            rows.append({"fold": fold, "model": name, "quantum": quantum, **m})
            _log(rows[-1], f"fold {fold} {name}")
        _save(pd.DataFrame(rows), "e1_cv_folds")      # save after every fold
    df = pd.DataFrame(rows)
    cols = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc",
            "train_time_s", "inference_time_s", "fn", "fp"]
    s = df.groupby("model", sort=False)[cols].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    _save(s.reset_index(), "e1_cv_summary")


# ----------------------------------------------------------------------------- E2
def run_e2():
    print("E2: number of qubits (PCA components)")
    rows = []
    for n in [2, 4, 6, 8]:
        for seed in SEEDS:
            for fn in (qsvc_once, svm_once):
                r = fn(seed, n=n)
                rows.append({"n_qubits": n, **r})
                _log(r, f"n={n} seed={seed} {r['model']}")
        r = vqc_once(SEED, n=n)                    # one seed: VQC is slow
        rows.append({"n_qubits": n, **r, "model": f"VQC ({VQC_TRAIN} train, 1 seed)"})
        _log(r, f"n={n} VQC")
        _save(pd.DataFrame(rows), "e2_qubits")
    df = pd.DataFrame(rows)
    _summary(df, ["model", "n_qubits"], "e2_qubits_summary")
    plot_sweep(df, "n_qubits", "s5_e2_qubits", "number of qubits = PCA components",
               xticks=[2, 4, 6, 8])


# ----------------------------------------------------------------------------- E3
def run_e3():
    print("E3: feature-map depth (ZZFeatureMap reps)")
    rows = []
    for reps in [1, 2, 3, 4]:
        for seed in SEEDS:
            r = qsvc_once(seed, reps=reps)
            rows.append({"fm_reps": reps, **r})
            _log(r, f"reps={reps} seed={seed}")
    df = _save(pd.DataFrame(rows), "e3_fm_depth")
    _summary(df, ["model", "fm_reps"], "e3_fm_depth_summary")
    plot_sweep(df, "fm_reps", "s5_e3_fm_depth", "ZZFeatureMap repetitions (reps)",
               xticks=[1, 2, 3, 4])
    _plot_concentration(df, "fm_reps", "s5_e3_kernel_concentration",
                        "ZZFeatureMap repetitions (reps)")


def _plot_concentration(df, x, name, xlabel):
    g = df.groupby(x, sort=False)
    fig, ax1 = plt.subplots(figsize=(6.5, 3.8))
    xs = [str(k) for k in g.groups.keys()]
    ax1.bar(xs, g["kernel_offdiag_mean"].mean(), yerr=g["kernel_offdiag_mean"].std(),
            color="C2", alpha=0.7, capsize=3, label="mean off-diagonal kernel value")
    ax1.set_ylabel("mean off-diagonal k(x, x')")
    ax1.set_xlabel(xlabel)
    ax2 = ax1.twinx()
    ax2.plot(xs, g["accuracy"].mean(), "o-", color="C3", label="QSVC test accuracy")
    ax2.set_ylabel("test accuracy")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    ax1.tick_params(axis="x", rotation=15)
    save_fig(fig, name)


# ----------------------------------------------------------------------------- E4
def run_e4():
    print("E4: entanglement and feature-map type")
    configs = [("ZZ linear", "zz", "linear"), ("ZZ circular", "zz", "circular"),
               ("ZZ full", "zz", "full"), ("Z (no entanglement)", "z", "linear"),
               ("Pauli Z,Y,ZZ linear", "pauli", "linear")]
    rows = []
    for label, kind, ent in configs:
        for seed in SEEDS:
            r = qsvc_once(seed, kind=kind, entanglement=ent)
            rows.append({"feature_map": label, **r})
            _log(r, f"{label} seed={seed}")
    df = _save(pd.DataFrame(rows), "e4_entanglement")
    _summary(df, ["model", "feature_map"], "e4_entanglement_summary")
    g = df.groupby("feature_map", sort=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    labels = list(g.groups.keys())
    axes[0].bar(labels, g["accuracy"].mean(), yerr=g["accuracy"].std(), capsize=3, color="C0")
    axes[0].set_ylabel("QSVC test accuracy (mean ± std)")
    axes[0].set_ylim(0.5, 1.0)
    axes[1].bar(labels, g["kernel_offdiag_mean"].mean(), yerr=g["kernel_offdiag_mean"].std(),
                capsize=3, color="C2")
    axes[1].set_ylabel("mean off-diagonal kernel value")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "s5_e4_entanglement")


# ----------------------------------------------------------------------------- E5
def run_e5():
    print(f"E5: VQC ansatz depth (RealAmplitudes reps), {VQC_TRAIN} training samples")
    rows, curves = [], {}
    for reps in [1, 2, 3, 4]:
        for seed in SEEDS:
            r, model, _ = vqc_once(seed, ansatz_reps=reps, return_model=True)
            rows.append({"ansatz_reps": reps, **r})
            if seed == SEED:
                curves[f"reps = {reps} ({r['parameters']} parameters)"] = model.loss_history_
            _log(r, f"ansatz reps={reps} seed={seed}")
        _save(pd.DataFrame(rows), "e5_ansatz_depth")
    df = pd.DataFrame(rows)
    _summary(df, ["model", "ansatz_reps"], "e5_ansatz_depth_summary")
    plot_sweep(df, "ansatz_reps", "s5_e5_ansatz_depth", "RealAmplitudes repetitions (reps)",
               xticks=[1, 2, 3, 4])
    plot_loss_curve(curves, "s5_e5_loss_curves", title="VQC loss vs ansatz depth (seed 42)")


# ----------------------------------------------------------------------------- E6
def run_e6():
    print(f"E6: optimiser COBYLA vs SPSA (maxiter 100), {VQC_TRAIN} training samples")
    rows, curves = [], {}
    for opt in ["cobyla", "spsa"]:
        for seed in SEEDS:
            r, model, _ = vqc_once(seed, optimizer=opt, return_model=True)
            rows.append(r)
            if seed == SEED:
                curves[opt.upper()] = model.loss_history_
            _log(r, f"{opt} seed={seed}")
        _save(pd.DataFrame(rows), "e6_optimizer")
    df = pd.DataFrame(rows)
    _summary(df, ["model", "optimizer"], "e6_optimizer_summary")
    plot_loss_curve(curves, "s5_e6_optimizer_loss",
                    title="VQC loss: COBYLA vs SPSA, maxiter 100 (seed 42)")


# ----------------------------------------------------------------------------- E7
def run_e7():
    print("E7: training-set size")
    rows = []
    for size in [50, 100, 200, 455]:
        for seed in SEEDS:
            for fn in (qsvc_once, svm_once):
                r = fn(seed, train_size=size)
                rows.append({"train_size": size, **r})
                _log(r, f"size={size} seed={seed} {r['model']}")
    df = _save(pd.DataFrame(rows), "e7_train_size")
    _summary(df, ["model", "train_size"], "e7_train_size_summary")
    plot_sweep(df, "train_size", "s5_e7_train_size", "number of training samples",
               xticks=[50, 100, 200, 455], logx=True)

    # real compute-uncompute circuits: kernel cost grows as M(M-1)/2
    rows = []
    for size in [50, 100, 200]:
        d = prepare_data(4, train_size=size, seed=SEED)
        for backend in ["circuits", "statevector"]:
            t0 = time.perf_counter()
            TunedQSVC(backend=backend, C=1.0).fit(d.X_train, d.y_train)
            rows.append({"train_size": size, "kernel_backend": backend,
                         "kernel_circuits": size * (size - 1) // 2,
                         "train_kernel_time_s": time.perf_counter() - t0})
            print(f"  size={size} {backend}: {rows[-1]['train_kernel_time_s']:.2f}s")
    df = _save(pd.DataFrame(rows), "e7_kernel_cost")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for be, lab in [("circuits", "compute-uncompute circuits (one per pair)"),
                    ("statevector", "statevector shortcut (one state per sample)")]:
        g = df[df["kernel_backend"] == be]
        ax.loglog(g["train_size"], g["train_kernel_time_s"], "o-", label=lab)
    c = df[df["kernel_backend"] == "circuits"]
    ref = c["train_kernel_time_s"].iloc[0] * (c["train_size"] / c["train_size"].iloc[0]) ** 2
    ax.loglog(c["train_size"], ref, "k--", lw=1, label="$\\propto M^2$ reference")
    ax.set_xlabel("training samples M")
    ax.set_ylabel("time to build training kernel (s)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    save_fig(fig, "s5_e7_kernel_cost")


# ----------------------------------------------------------------------------- E8
def run_e8(seed=SEED):
    print(f"E8: noise (QSVC {NOISE_TRAIN} training samples, VQC {VQC_TRAIN})")
    fake = FakeManilaV2()
    pm = generate_preset_pass_manager(backend=fake, optimization_level=1, seed_transpiler=seed)
    noise = NoiseModel.from_backend(fake)
    settings = {
        "exact statevector": None,
        "ideal sampler, 1024 shots": (AerSampler(default_shots=1024, seed=seed), None),
        f"noisy {fake.name}, 1024 shots": (AerSampler(default_shots=1024, seed=seed,
                                                      options={"backend_options":
                                                               {"noise_model": noise}}), pm),
    }
    rows = []
    # --- QSVC: the kernel itself is estimated on the (noisy) simulator
    d = prepare_data(4, train_size=NOISE_TRAIN, seed=seed)
    K_exact = None
    for label, s in settings.items():
        backend = "statevector" if s is None else {"sampler": s[0], "pm": s[1]}
        model = TunedQSVC(backend=backend, seed=seed)
        m, _, _ = fit_and_evaluate(model, d.X_train, d.y_train, d.X_test, d.y_test)
        if K_exact is None:
            K_exact = model.train_kernel_
        dev = np.abs(model.train_kernel_ - K_exact)[~np.eye(len(K_exact), dtype=bool)]
        rows.append({"model": "QSVC", "setting": label, **m, "C": model.best_C_,
                     "kernel_mean_abs_error_vs_exact": float(dev.mean())})
        _log(rows[-1], f"QSVC {label}")
        _save(pd.DataFrame(rows), "e8_noise")
    # --- VQC: trained on the ideal simulator, then run on each setting
    r, vqc, dv = vqc_once(seed, return_model=True)
    nn = vqc.vqc_.neural_network
    for label, s in settings.items():
        if s is None:
            y_pred, t = vqc.predict(dv.X_test), r["inference_time_s"]
            prob = vqc.predict_proba(dv.X_test)
        else:
            qnn = SamplerQNN(circuit=nn.circuit, input_params=nn.input_params,
                             weight_params=nn.weight_params, interpret=nn._interpret,
                             output_shape=nn.output_shape, sampler=s[0], pass_manager=s[1])
            t0 = time.perf_counter()
            prob = qnn.forward(dv.X_test, vqc.vqc_.weights)
            t = time.perf_counter() - t0
            y_pred = prob.argmax(axis=1)
        m = compute_metrics(dv.y_test, y_pred, prob[:, 1])
        rows.append({"model": "VQC", "setting": label, **m, "train_time_s": r["train_time_s"],
                     "inference_time_s": t, "n_train": len(dv.y_train), "n_test": len(dv.y_test)})
        _log(rows[-1], f"VQC {label}")
    plot_e8(_save(pd.DataFrame(rows), "e8_noise"))


def plot_e8(df):
    """Ideal vs noisy accuracy bar chart, drawn from the saved E8 table."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    labels = list(dict.fromkeys(df["setting"]))
    w = 0.38
    for i, mdl in enumerate(["QSVC", "VQC"]):
        vals = [df[(df.model == mdl) & (df.setting == lab)]["accuracy"].iloc[0] for lab in labels]
        bars = ax.bar(np.arange(3) + (i - 0.5) * w, vals, w, label=mdl)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in vals], fontsize=8)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([lab.replace(", ", "\n") for lab in labels])
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.4, 1.0)
    ax.legend()
    save_fig(fig, "s5_e8_noise")


def run_e8_check(seed=SEED):
    """Compare exact and noisy QSVC test scores at the same C, sample by sample."""
    print(f"E8 check: exact vs noisy QSVC scores, {NOISE_TRAIN} training samples, C = 1")
    fake = FakeManilaV2()
    pm = generate_preset_pass_manager(backend=fake, optimization_level=1, seed_transpiler=seed)
    noisy = AerSampler(default_shots=1024, seed=seed, options={"backend_options": {
        "noise_model": NoiseModel.from_backend(fake)}})
    d = prepare_data(4, train_size=NOISE_TRAIN, seed=seed)
    exact = TunedQSVC(C=1.0, seed=seed).fit(d.X_train, d.y_train)
    noise = TunedQSVC(backend={"sampler": noisy, "pm": pm}, C=1.0, seed=seed).fit(d.X_train, d.y_train)
    s_e, s_n = exact.decision_function(d.X_test), noise.decision_function(d.X_test)
    p_e, p_n = (s_e > 0).astype(int), (s_n > 0).astype(int)
    _save(pd.DataFrame({"y_true": d.y_test, "exact_score": s_e, "noisy_score": s_n,
                        "exact_pred": p_e, "noisy_pred": p_n}), "e8_qsvc_scores")
    m_e, m_n = compute_metrics(d.y_test, p_e, s_e), compute_metrics(d.y_test, p_n, s_n)
    print(f"  predictions that differ: {int((p_e != p_n).sum())} of {len(p_e)}; "
          f"max score difference {np.abs(s_e - s_n).max():.4f}")
    print(f"  ROC-AUC exact {m_e['roc_auc']:.10f}, noisy {m_n['roc_auc']:.10f}")
    print(f"  (tn, fp, fn, tp) exact {m_e['tn'], m_e['fp'], m_e['fn'], m_e['tp']}, "
          f"noisy {m_n['tn'], m_n['fp'], m_n['fn'], m_n['tp']}")


# ----------------------------------------------------------------------------- E9
def run_e9():
    print("E9: Iris multiclass (4 features = 4 qubits, no PCA)")
    data = load_iris()
    rows = []
    for seed in SEEDS:
        X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.3,
                                                  stratify=data.target, random_state=seed)
        sc = MinMaxScaler((0, np.pi)).fit(X_tr)
        X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)
        for name, model in [("QSVC", TunedQSVC(n_qubits=4, seed=seed)),
                            ("SVM (RBF)", tuned_rbf_svm(seed))]:
            t0 = time.perf_counter()
            model.fit(X_tr, y_tr)
            tt = time.perf_counter() - t0
            y_pred = model.predict(X_te)
            rows.append({"model": name, "seed": seed, "accuracy": accuracy_score(y_te, y_pred),
                         "macro_f1": f1_score(y_te, y_pred, average="macro"),
                         "train_time_s": tt, "n_train": len(y_tr), "n_test": len(y_te)})
            print(f"  seed={seed} {name}: acc {rows[-1]['accuracy']:.4f}")
    df = _save(pd.DataFrame(rows), "e9_iris")
    s = df.groupby("model")[["accuracy", "macro_f1", "train_time_s"]].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    _save(s.reset_index(), "e9_iris_summary")


TASKS = {"e1cv": run_e1_cv, "e2": run_e2, "e3": run_e3, "e4": run_e4, "e5": run_e5,
         "e6": run_e6, "e7": run_e7, "e8": run_e8, "e8check": run_e8_check, "e9": run_e9}
