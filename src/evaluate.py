"""Metrics, timing and result plots shared by all experiments.

Positive class = malignant (label 1). Recall (sensitivity) on this class is the
clinically most important number: a false negative is a missed cancer.
"""
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from .preprocess import CLASS_NAMES, save_fig

METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]


def scores_of(model, X):
    """Continuous score for ROC-AUC (decision function or P(malignant))."""
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X))
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    return None


def compute_metrics(y_true, y_pred, y_score=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score) if y_score is not None else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def fit_and_evaluate(model, X_train, y_train, X_test, y_test, repeats=1):
    """Train, time and score one model. Returns (metrics dict, y_pred, y_score).

    With repeats > 1 the model is refitted that many times (same seed, same
    result) and the median train / inference time is reported, which removes
    one-off slow-downs of the laptop from fast classical models.
    """
    train_times, infer_times = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        y_pred = np.asarray(model.predict(X_test)).astype(int)
        infer_times.append(time.perf_counter() - t0)
    train_time, inference_time = float(np.median(train_times)), float(np.median(infer_times))

    y_score = scores_of(model, X_test)
    m = compute_metrics(y_test, y_pred, y_score)
    m.update({"train_time_s": train_time, "inference_time_s": inference_time,
              "timing_repeats": repeats, "n_train": len(y_train), "n_test": len(y_test)})
    return m, y_pred, y_score


# ----------------------------------------------------------------------------- plots
def plot_confusion_grid(cms, name, ncols=4):
    """cms: {model name: 2x2 confusion matrix}."""
    n = len(cms)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows))
    for ax, (model, cm) in zip(np.ravel(axes), cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cbar=False, cmap="Blues", ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, annot_kws={"size": 12})
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.tight_layout()
    save_fig(fig, name)


def plot_metric_bars(df, name, metrics=("accuracy", "precision", "recall", "f1")):
    """Grouped bar chart: one group per model, one bar per metric."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(df))
    w = 0.8 / len(metrics)
    for i, met in enumerate(metrics):
        ax.bar(x + (i - (len(metrics) - 1) / 2) * w, df[met], w, label=met)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=20, ha="right")
    ax.set_ylim(max(0.0, df[list(metrics)].min().min() - 0.1), 1.0)
    ax.set_ylabel("score")
    ax.legend(ncol=len(metrics), loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, name)


def plot_roc(curves, y_true, name):
    """curves: {model name: score array}."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for model, s in curves.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        ax.plot(fpr, tpr, label=f"{model} (AUC = {roc_auc_score(y_true, s):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("false positive rate (1 - specificity)")
    ax.set_ylabel("true positive rate (recall)")
    ax.legend(fontsize=8, loc="lower right")
    save_fig(fig, name)


def plot_train_time(df, name):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    order = df.sort_values("train_time_s")
    bars = ax.barh(order["model"], order["train_time_s"],
                   color=["C3" if q else "C0" for q in order["quantum"]])
    ax.bar_label(bars, labels=[f"{t:.3g} s" for t in order["train_time_s"]], padding=3, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("training time (s, log scale)  -  red = quantum (simulated)")
    save_fig(fig, name)


def plot_kernel_pair(K_quantum, K_rbf, y, name):
    """Quantum vs RBF kernel matrices, samples sorted by class."""
    order = np.argsort(y, kind="stable")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, K, title in [(axes[0], K_quantum, "quantum kernel (ZZFeatureMap)"),
                         (axes[1], K_rbf, "classical RBF kernel")]:
        im = ax.imshow(K[np.ix_(order, order)], cmap="viridis", vmin=0, vmax=1)
        b = int((y == 0).sum())
        ax.axhline(b - 0.5, color="w", lw=0.8)
        ax.axvline(b - 0.5, color="w", lw=0.8)
        ax.set_title(f"{title}\n(benign first, then malignant)", fontsize=10)
        ax.set_xlabel("training sample")
        ax.set_ylabel("training sample")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    save_fig(fig, name)


def plot_loss_curve(histories, name, title=None):
    """histories: {label: list of objective values}."""
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for label, h in histories.items():
        ax.plot(np.arange(1, len(h) + 1), h, label=label)
    ax.set_xlabel("optimiser step (COBYLA: per evaluation, SPSA: per iteration)")
    ax.set_ylabel("cross-entropy loss")
    if title:
        ax.set_title(title)
    if len(histories) > 1:
        ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, name)
