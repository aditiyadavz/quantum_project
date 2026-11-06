"""Every result number used in the report text, read from results/*.csv."""
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results"
MODELS = {"QSVC (quantum kernel)": "QSVC", "VQC": "VQC", "SVM (RBF)": "SVM", "Logistic Regression": "LR",
          "Random Forest": "RF", "KNN": "KNN", "MLP": "MLP", "SVM (RBF, 30 features)": "SVM30"}


def csv(name):
    return pd.read_csv(RES / f"{name}.csv")


def f3(x): return f"{float(x):.3f}"
def f4(x): return f"{float(x):.4f}"
def seconds(x):
    x = float(x)
    if x < 0.01:
        return "<0.01"
    return f"{x:.2f}" if x < 100 else f"{x:.0f}"


def load():
    V = {}
    # dataset and PCA
    ds = csv("dataset_summary").iloc[0]
    V.update(n_samples=int(ds.samples), n_features=int(ds.features), n_mal=int(ds.malignant), n_ben=int(ds.benign),
             pct_mal=f"{ds.malignant_fraction:.1%}", missing=int(ds.missing_values))
    pca = csv("pca_explained_variance")
    for n in (2, 4, 6, 8):
        V[f"pca_{n}"] = f"{pca.cumulative.iloc[n - 1]:.1%}"
    V["pc1"] = f"{pca.explained_variance_ratio.iloc[0]:.1%}"
    V["pc2"] = f"{pca.explained_variance_ratio.iloc[1]:.1%}"

    # hold-out (E1)
    h = csv("e1_main_holdout")
    h["code"] = h.model.map(MODELS)
    for _, r in h.iterrows():
        for m, short in [("accuracy", "acc"), ("precision", "prec"), ("recall", "rec"), ("specificity", "spec"), ("f1", "f1"), ("roc_auc", "auc")]:
            V[f"h_{r.code}_{short}"] = f3(r[m])
        V[f"h_{r.code}_fn"] = int(r.fn)
        V[f"h_{r.code}_fp"] = int(r.fp)
        V[f"h_{r.code}_time"] = seconds(r.train_time_s)
        V[f"h_{r.code}_inf"] = seconds(r.inference_time_s)
    V["n_train"], V["n_test"] = int(h.n_train.iloc[0]), int(h.n_test.iloc[0])
    V["h_test_mal"] = int(h.tp.iloc[0] + h.fn.iloc[0])
    V["h_test_ben"] = int(h.tn.iloc[0] + h.fp.iloc[0])
    V["majority"] = f3(ds.benign / ds.samples)

    # 5-fold CV
    cv = csv("e1_cv_summary")
    cv["code"] = cv.model.map(MODELS)
    folds = csv("e1_cv_folds")
    folds["code"] = folds.model.map(MODELS)
    for _, r in cv.iterrows():
        for m, short in [("accuracy", "acc"), ("precision", "prec"), ("recall", "rec"), ("specificity", "spec"), ("f1", "f1"), ("roc_auc", "auc")]:
            V[f"cv_{r.code}_{short}"] = f3(r[f"{m}_mean"])
            V[f"cv_{r.code}_{short}_sd"] = f3(r[f"{m}_std"])
        V[f"cv_{r.code}_time"] = seconds(r.train_time_s_mean)
        V[f"cv_{r.code}_inf"] = seconds(r.inference_time_s_mean)
        V[f"cv_{r.code}_fn"] = int(folds[folds.code == r.code].fn.sum())
        V[f"cv_{r.code}_fp"] = int(folds[folds.code == r.code].fp.sum())
    V["cv_fn_ratio"] = f"{V['cv_QSVC_fn'] / V['cv_SVM_fn']:.1f}"
    V["cv_vqc_ratio"] = f"{V['cv_VQC_fn'] / V['cv_SVM_fn']:.1f}"

    # circuits
    cs = csv("s4_circuit_stats").set_index("circuit")
    for label, key in [("ZZFeatureMap (reps=2, linear)", "zz"), ("RealAmplitudes ansatz (reps=3)", "ra"),
                       ("VQC full circuit", "vqcc"), ("Kernel compute-uncompute circuit", "kc")]:
        for col in ("depth", "cx_gates", "total_gates", "parameters"):
            V[f"{key}_{col}"] = int(cs.loc[label, col])
    q = csv("e1_quantum_circuit_stats").set_index("model")
    V["qsvc_circuits"] = f"{int(q.loc['QSVC (quantum kernel)', 'circuit_evaluations_train']):,}"
    V["vqc_circuits"] = f"{int(q.loc['VQC', 'circuit_evaluations_train']):,}"
    V["qsvc_test_circuits"] = f"{V['n_train'] * V['n_test']:,}"
    vl = csv("e1_vqc_loss_history").loss
    V.update(vqc_loss_first=f3(vl.iloc[0]), vqc_loss_last=f3(vl.iloc[-1]), vqc_loss_min=f3(vl.min()))

    # kernel equivalence check
    kc = csv("kernel_backend_check").iloc[0]
    V.update(kc_samples=int(kc.samples), kc_entries=f"{int(kc.kernel_entries):,}", kc_sv=f"{kc.statevector_time_s:.2f}",
             kc_circ=f"{kc.circuit_time_s:.1f}", kc_diff=f"{kc.max_abs_difference:.1e}",
             kc_ratio=f"{kc.circuit_time_s / kc.statevector_time_s:.0f}")

    # E2
    e2 = csv("e2_qubits_summary")
    for _, r in e2.iterrows():
        code = "VQC" if r.model.startswith("VQC") else ("QSVC" if r.model == "QSVC" else "SVM")
        n = int(r.n_qubits)
        V[f"e2_{code}_{n}_acc"] = f3(r.accuracy_mean)
        V[f"e2_{code}_{n}_rec"] = f3(r.recall_mean)
        V[f"e2_{code}_{n}_time"] = seconds(r.train_time_s_mean)
        if code == "QSVC":
            V[f"e2_k_{n}"] = f3(r.kernel_offdiag_mean_mean)
    # E3
    for _, r in csv("e3_fm_depth_summary").iterrows():
        V[f"e3_{int(r.fm_reps)}_acc"] = f3(r.accuracy_mean)
        V[f"e3_{int(r.fm_reps)}_rec"] = f3(r.recall_mean)
        V[f"e3_{int(r.fm_reps)}_k"] = f3(r.kernel_offdiag_mean_mean)
    # E4
    labels = {"ZZ linear": "zzl", "ZZ circular": "zzc", "ZZ full": "zzf", "Z (no entanglement)": "z", "Pauli Z,Y,ZZ linear": "pauli"}
    for _, r in csv("e4_entanglement_summary").iterrows():
        code = labels[r.feature_map]
        V[f"e4_{code}_acc"] = f3(r.accuracy_mean)
        V[f"e4_{code}_sd"] = f3(r.accuracy_std)
        V[f"e4_{code}_rec"] = f3(r.recall_mean)
        V[f"e4_{code}_k"] = f3(r.kernel_offdiag_mean_mean)
    # E5, E6
    for _, r in csv("e5_ansatz_depth_summary").iterrows():
        n = int(r.ansatz_reps)
        V[f"e5_{n}_acc"] = f3(r.accuracy_mean)
        V[f"e5_{n}_sd"] = f3(r.accuracy_std)
        V[f"e5_{n}_time"] = seconds(r.train_time_s_mean)
        V[f"e5_{n}_loss"] = f3(r.final_loss_mean)
    for _, r in csv("e6_optimizer_summary").iterrows():
        V[f"e6_{r.optimizer}_acc"] = f3(r.accuracy_mean)
        V[f"e6_{r.optimizer}_sd"] = f3(r.accuracy_std)
        V[f"e6_{r.optimizer}_rec"] = f3(r.recall_mean)
        V[f"e6_{r.optimizer}_time"] = seconds(r.train_time_s_mean)
        V[f"e6_{r.optimizer}_loss"] = f3(r.final_loss_mean)
    e6 = csv("e6_optimizer")
    V["e6_spsa_evals"] = int(e6[e6.optimizer == "spsa"].objective_evaluations.iloc[0])
    V["e6_cobyla_evals"] = int(e6[e6.optimizer == "cobyla"].objective_evaluations.iloc[0])
    # E7
    for _, r in csv("e7_train_size_summary").iterrows():
        code = "QSVC" if r.model == "QSVC" else "SVM"
        n = int(r.train_size)
        V[f"e7_{code}_{n}_acc"] = f3(r.accuracy_mean)
        V[f"e7_{code}_{n}_rec"] = f3(r.recall_mean)
    for _, r in csv("e7_kernel_cost").iterrows():
        tag = "c" if r.kernel_backend == "circuits" else "s"
        V[f"e7{tag}_{int(r.train_size)}"] = f"{r.train_kernel_time_s:.2f}" if tag == "s" else f"{r.train_kernel_time_s:.1f}"
        V[f"e7n_{int(r.train_size)}"] = f"{int(r.kernel_circuits):,}"
    # E8
    e8 = csv("e8_noise")
    tags = {"exact statevector": "exact", "ideal sampler, 1024 shots": "shots", "noisy fake_manila, 1024 shots": "noisy"}
    for _, r in e8.iterrows():
        m, t = r.model, tags[r.setting]
        for col, short in [("accuracy", "acc"), ("recall", "rec"), ("f1", "f1"), ("roc_auc", "auc")]:
            V[f"e8_{m}_{t}_{short}"] = f3(r[col])
        V[f"e8_{m}_{t}_fn"] = int(r.fn)
        V[f"e8_{m}_{t}_time"] = seconds(r.train_time_s)
        V[f"e8_{m}_{t}_inf"] = seconds(r.inference_time_s)
        if m == "QSVC":
            V[f"e8_{t}_C"] = f"{r.C:g}"
            V[f"e8_{t}_err"] = f"{r.kernel_mean_abs_error_vs_exact:.3f}"
    s = csv("e8_qsvc_scores")
    V["e8_changed"] = int((s.exact_pred != s.noisy_pred).sum())
    V["e8_total"] = len(s)
    import numpy as np
    y = s.y_true.values
    a = np.sign(s.exact_score.values[y == 1][:, None] - s.exact_score.values[y == 0][None, :])
    b = np.sign(s.noisy_score.values[y == 1][:, None] - s.noisy_score.values[y == 0][None, :])
    V["e8_flip"], V["e8_pairs"] = int((a != b).sum()), f"{a.size:,}"
    # E9
    e9 = csv("e9_iris_summary").set_index("model")
    V["e9_q"], V["e9_q_sd"] = f3(e9.loc["QSVC", "accuracy_mean"]), f3(e9.loc["QSVC", "accuracy_std"])
    V["e9_s"], V["e9_s_sd"] = f3(e9.loc["SVM (RBF)", "accuracy_mean"]), f3(e9.loc["SVM (RBF)", "accuracy_std"])

    # Section 2 demos
    sup = pd.read_csv(RES / "s2_superposition.csv", dtype={"state": str})
    one = sup[sup.circuit == "1 qubit H"].set_index("state").probability
    V["sup_p0"], V["sup_p1"] = f3(one["0"]), f3(one["1"])
    three = sup[sup.circuit == "3 qubit H"].probability
    V["sup3_min"], V["sup3_max"] = f3(three.min()), f3(three.max())
    bell = csv("s2_bell_states").iloc[0]
    V["bell_00"], V["bell_11"] = int(round(bell["P(00)"] * 1024)), int(round(bell["P(11)"] * 1024))
    nb = csv("s2_noise_bell")
    V["noise_ideal"], V["noise_noisy"] = f3(nb["P(00)+P(11)"].iloc[0]), f3(nb["P(00)+P(11)"].iloc[1])
    V["noise_01"], V["noise_10"] = int(nb["01"].iloc[1]), int(nb["10"].iloc[1])
    nd = csv("s2_noise_vs_depth")
    V["noise_d_first"], V["noise_d_last"], V["noise_cx_last"] = f3(nd["P(00)+P(11)"].iloc[0]), f3(nd["P(00)+P(11)"].iloc[-1]), int(nd.total_cx.iloc[-1])
    g = csv("s2_grover_runs").set_index("n_qubits")
    V["gr2_meas"], V["gr3_meas"], V["gr3_theory"] = f3(g.loc[2, "measured_success"]), f3(g.loc[3, "measured_success"]), f3(g.loc[3, "theoretical_success"])
    sh = pd.read_csv(RES / "s2_shor_results.csv", dtype={"register": str})
    V["shor_total"] = int(sh.counts.sum())
    for _, r in sh.iterrows():
        V[f"shor_{int(r.decimal)}"] = int(r.counts)
    fb = csv("s2_fake_backend_properties")
    q = fb[~fb.qubit.astype(str).str.startswith("cx")]
    cx = fb[fb.qubit.astype(str).str.startswith("cx")]
    t1, t2 = q.T1_us.astype(float), q.T2_us.astype(float)
    ro = q.readout_error.astype(float)
    g1 = q["gate_error (sx or 2q)"].astype(float)
    g2 = cx["gate_error (sx or 2q)"].astype(float)
    V.update(t1_min=f"{t1.min():.1f}", t1_max=f"{t1.max():.1f}", t2_min=f"{t2.min():.1f}", t2_max=f"{t2.max():.1f}",
             ro_min=f"{ro.min():.1%}", ro_max=f"{ro.max():.1%}", g1_min=f"{g1.min():.2%}", g1_max=f"{g1.max():.2%}",
             g2_min=f"{g2.min():.2%}", g2_max=f"{g2.max():.2%}")
    return V, {"holdout": h, "cv": cv, "shor": sh}
