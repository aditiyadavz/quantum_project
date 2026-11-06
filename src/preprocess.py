"""Data loading, preprocessing and exploratory figures.

Pipeline (every scaler / PCA is fitted on the TRAINING data only):
    load -> re-map labels (malignant = 1) -> stratified 80/20 split
    -> StandardScaler -> PCA(n components) -> MinMaxScaler to [0, pi]
The [0, pi] range is used because each feature becomes a rotation angle
in the quantum feature map.
"""
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)
CLASS_NAMES = ["benign", "malignant"]   # label 0, label 1


def save_fig(fig, name):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure  -> {path.relative_to(ROOT)}")


def load_dataset():
    """Return features X (DataFrame, 30 columns) and labels y (1 = malignant)."""
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = (1 - data.target).rename("malignant")   # scikit-learn uses 0 = malignant
    assert not X.isna().any().any(), "unexpected missing values"
    return X, y


@dataclass
class DataSplit:
    X_train: np.ndarray       # PCA features scaled to [0, pi]  (quantum + classical models)
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    X_train_full: np.ndarray  # all 30 standardised features (upper-reference SVM)
    X_test_full: np.ndarray
    explained_variance: np.ndarray


def prepare_data(n_components=4, train_size=None, seed=SEED, test_size=0.2):
    """Build one train/test split.

    n_components : number of PCA components = number of qubits.
    train_size   : optional smaller stratified training subset (e.g. 200).
    seed         : controls the split, so repeated runs with different seeds
                   give different (but reproducible) splits.
    """
    X, y = load_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X.values, y.values, test_size=test_size, stratify=y.values, random_state=seed)
    if train_size is not None and train_size < len(y_tr):
        X_tr, _, y_tr, _ = train_test_split(
            X_tr, y_tr, train_size=train_size, stratify=y_tr, random_state=seed)
    return transform_split(X_tr, X_te, y_tr, y_te, n_components, seed)


def transform_split(X_tr, X_te, y_tr, y_te, n_components=4, seed=SEED):
    """Fit StandardScaler -> PCA -> [0, pi] scaling on the training part only."""
    std = StandardScaler().fit(X_tr)
    Z_tr, Z_te = std.transform(X_tr), std.transform(X_te)

    pca = PCA(n_components=n_components, random_state=seed).fit(Z_tr)
    P_tr, P_te = pca.transform(Z_tr), pca.transform(Z_te)

    angle = MinMaxScaler(feature_range=(0, np.pi)).fit(P_tr)
    # test values can fall slightly outside [0, pi]; they are still valid angles
    A_tr, A_te = angle.transform(P_tr), angle.transform(P_te)

    return DataSplit(A_tr, A_te, y_tr, y_te, Z_tr, Z_te, pca.explained_variance_ratio_)


def eda_figures():
    """Exploratory analysis figures and summary tables for Section 4.8."""
    X, y = load_dataset()
    labels = y.map({0: "benign", 1: "malignant"})

    # dataset summary
    pd.DataFrame([{"samples": len(y), "features": X.shape[1], "malignant": int(y.sum()),
                   "benign": int((y == 0).sum()), "malignant_fraction": y.mean(),
                   "missing_values": int(X.isna().sum().sum())}]
                 ).to_csv(RES_DIR / "dataset_summary.csv", index=False)
    X.describe().T.to_csv(RES_DIR / "dataset_feature_statistics.csv")

    # 1. class distribution
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    counts = labels.value_counts().reindex(CLASS_NAMES)
    bars = ax.bar(counts.index, counts.values, color=["#4C72B0", "#C44E52"])
    ax.bar_label(bars, labels=[f"{v} ({v / len(y):.1%})" for v in counts.values])
    ax.set_ylabel("number of samples")
    ax.set_ylim(0, counts.max() * 1.15)
    save_fig(fig, "s4_class_distribution")

    # 2. correlation heatmap of the 30 features
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(X.corr(), cmap="coolwarm", center=0, ax=ax, square=True,
                cbar_kws={"shrink": 0.7, "label": "Pearson correlation"})
    save_fig(fig, "s4_correlation_heatmap")

    # 3. box plots of the 6 features most correlated with the label
    corr_y = X.apply(lambda c: np.corrcoef(c, y)[0, 1]).abs().sort_values(ascending=False)
    corr_y.rename("abs_corr_with_malignant").to_csv(RES_DIR / "feature_label_correlation.csv")
    top6 = corr_y.index[:6]
    Z = (X[top6] - X[top6].mean()) / X[top6].std()
    long = Z.assign(diagnosis=labels).melt(id_vars="diagnosis", var_name="feature",
                                           value_name="standardised value")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    sns.boxplot(data=long, x="feature", y="standardised value", hue="diagnosis",
                hue_order=CLASS_NAMES, palette=["#4C72B0", "#C44E52"], ax=ax, fliersize=2)
    ax.tick_params(axis="x", rotation=20)
    save_fig(fig, "s4_top6_feature_boxplots")

    # 4. PCA explained variance (fitted on the training split, all 30 components)
    X_tr, _, y_tr, _ = train_test_split(X.values, y.values, test_size=0.2,
                                        stratify=y.values, random_state=SEED)
    Z_tr = StandardScaler().fit_transform(X_tr)
    pca = PCA(random_state=SEED).fit(Z_tr)
    ev = pca.explained_variance_ratio_
    pd.DataFrame({"component": np.arange(1, 31), "explained_variance_ratio": ev,
                  "cumulative": np.cumsum(ev)}).to_csv(RES_DIR / "pca_explained_variance.csv",
                                                      index=False)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    k = np.arange(1, 13)
    ax.bar(k, ev[:12], label="per component")
    ax.plot(k, np.cumsum(ev)[:12], "o-", color="C3", label="cumulative")
    for n in (2, 4, 6, 8):
        ax.annotate(f"{np.cumsum(ev)[n - 1]:.1%}", (n, np.cumsum(ev)[n - 1]),
                    textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xlabel("principal component")
    ax.set_ylabel("explained variance ratio")
    ax.set_xticks(k)
    ax.set_ylim(0, 1.05)
    ax.legend()
    save_fig(fig, "s4_pca_explained_variance")

    # 5. 2D scatter of the first two components
    P = pca.transform(Z_tr)[:, :2]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for lab, col in [(0, "#4C72B0"), (1, "#C44E52")]:
        m = y_tr == lab
        ax.scatter(P[m, 0], P[m, 1], s=12, alpha=0.7, color=col, label=CLASS_NAMES[lab])
    ax.set_xlabel(f"PC1 ({ev[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%} variance)")
    ax.legend()
    save_fig(fig, "s4_pca_scatter")
    print(f"  PCA cumulative variance: 2 -> {np.cumsum(ev)[1]:.3f}, 4 -> {np.cumsum(ev)[3]:.3f}, "
          f"6 -> {np.cumsum(ev)[5]:.3f}, 8 -> {np.cumsum(ev)[7]:.3f}")
