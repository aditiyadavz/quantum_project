"""Block diagrams and flowcharts for Sections 2 and 4, drawn with matplotlib.

Run from the project folder:  python -m src.diagrams
Numbers shown in the diagrams (split sizes, PCA variance) are read from results/.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc, Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle

from .preprocess import RES_DIR, save_fig

STYLES = {
    "data": ("#EFEFEF", "#666666"),
    "classical": ("#DCE9F7", "#2F5D8C"),
    "quantum": ("#E9DFF5", "#6A3FA0"),
    "eval": ("#DFF0E0", "#2E7D32"),
    "decision": ("#FFF4D1", "#B28704"),
    "terminal": ("#333333", "#333333"),
}
LEGEND = {"data": "data", "classical": "classical (CPU)", "quantum": "quantum circuit (simulator)",
          "eval": "evaluation", "decision": "decision"}


def canvas(width, height):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def box(ax, cx, cy, w, h, text, style="classical", size=9, bold=False):
    face, edge = STYLES[style]
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                facecolor=face, edgecolor=edge, linewidth=1.2, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=size, zorder=3,
            color="white" if style == "terminal" else "black", fontweight="bold" if bold else "normal")
    return {"l": (cx - w / 2, cy), "r": (cx + w / 2, cy), "t": (cx, cy + h / 2), "b": (cx, cy - h / 2)}


def diamond(ax, cx, cy, w, h, text, size=8.5):
    face, edge = STYLES["decision"]
    ax.add_patch(Polygon([(cx - w / 2, cy), (cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2)],
                         closed=True, facecolor=face, edgecolor=edge, linewidth=1.2, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=size, zorder=3)
    return {"l": (cx - w / 2, cy), "r": (cx + w / 2, cy), "t": (cx, cy + h / 2), "b": (cx, cy - h / 2)}


def arrow(ax, start, end, label=None, rad=0.0, dashed=False, color="#333333", label_offset=(0.06, 0.0)):
    ax.annotate("", xy=end, xytext=start, zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, linestyle="--" if dashed else "-",
                                connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if label:
        ax.text((start[0] + end[0]) / 2 + label_offset[0], (start[1] + end[1]) / 2 + label_offset[1],
                label, fontsize=8, ha="left", va="center", color=color, zorder=4)


def elbow(ax, points, label=None, dashed=False, color="#333333", label_at=None):
    """Polyline through points, with an arrow head on the last segment."""
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, color=color, lw=1.2, ls="--" if dashed else "-", zorder=1)
    arrow(ax, points[-2], points[-1], dashed=dashed, color=color)
    if label and label_at:
        ax.text(*label_at, label, fontsize=8, color=color, ha="center", va="center", zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=1))


def legend(ax, keys, x, y, step=1.75):
    for i, key in enumerate(keys):
        face, edge = STYLES[key]
        ax.add_patch(Rectangle((x + i * step, y), 0.22, 0.16, facecolor=face, edgecolor=edge, lw=1))
        ax.text(x + i * step + 0.28, y + 0.08, LEGEND[key], fontsize=8, va="center")


# ----------------------------------------------------------------------------- Section 2
def stacks():
    fig, ax = canvas(8.4, 5.6)
    classical = ["Application\n(spreadsheet, ML model)", "High-level language\n(Python, C++)",
                 "Compiler and operating system", "Instruction set\n(x86, ARM)",
                 "Logic gates and registers\n(deterministic bits)",
                 "Transistors at room temperature"]
    quantum = ["Application and algorithm\n(Grover, VQC, QSVC)", "Quantum SDK\n(Qiskit, Cirq)",
               "Transpiler\n(gate decomposition, qubit routing)", "Native gate set\n(for example CX, RZ, SX, X)",
               "Classical control electronics\n(pulses and readout)",
               "Noisy qubits in a cryostat\n(superconducting: about 15 mK)"]
    ax.text(2.1, 5.3, "Classical computing stack", ha="center", fontsize=10.5, fontweight="bold")
    ax.text(6.3, 5.3, "Quantum computing stack", ha="center", fontsize=10.5, fontweight="bold")
    for x, items, style in [(2.1, classical, "classical"), (6.3, quantum, "quantum")]:
        previous = None
        for i, text in enumerate(items):
            b = box(ax, x, 4.72 - i * 0.84, 3.5, 0.62, text, style if i < 5 else ("data" if style == "classical" else style))
            if previous:
                arrow(ax, previous["b"], b["t"])
            previous = b
    save_fig(fig, "s2_classical_vs_quantum_stack")


def vertical_flow(steps, name, width=5.4, box_w=4.2, step_h=0.92, loop=None):
    """steps: list of (text, style or 'decision'). loop: (decision index, return index, yes, no)."""
    height = step_h * len(steps) + 0.35
    fig, ax = canvas(width, height)
    cx = width / 2
    nodes = []
    for i, (text, style) in enumerate(steps):
        cy = height - 0.45 - i * step_h
        if style == "decision":
            nodes.append(diamond(ax, cx, cy, box_w * 0.82, step_h * 0.82, text))
        elif style == "terminal":
            nodes.append(box(ax, cx, cy, 1.6, step_h * 0.5, text, "terminal", bold=True))
        else:
            nodes.append(box(ax, cx, cy, box_w, step_h * 0.66, text, style))
    for i in range(len(nodes) - 1):
        label = loop[3] if loop and i == loop[0] else None
        arrow(ax, nodes[i]["b"], nodes[i + 1]["t"], label=label)
    if loop:
        d, r, yes, _ = loop
        x = 0.14
        elbow(ax, [nodes[d]["l"], (x, nodes[d]["l"][1]), (x, nodes[r]["l"][1]), nodes[r]["l"]],
              label=yes, label_at=(x + 0.17, (nodes[d]["l"][1] + nodes[r]["l"][1]) / 2))
    save_fig(fig, name)


def flowcharts():
    vertical_flow([
        ("Start", "terminal"),
        ("Oracle $U_f$ promised constant or balanced;\nprepare $|0\\rangle^{\\otimes n}|1\\rangle$", "data"),
        ("Apply $H^{\\otimes (n+1)}$: uniform superposition,\noutput qubit in $|-\\rangle$", "quantum"),
        ("Query $U_f$ once: phase kickback $(-1)^{f(x)}$", "quantum"),
        ("Apply $H^{\\otimes n}$ to the input register", "quantum"),
        ("Measure the $n$ input qubits", "quantum"),
        ("Result is $0^n$?", "decision"),
        ("Yes: constant.  No: balanced.", "eval"),
        ("End", "terminal"),
    ], "s2_dj_flowchart")
    vertical_flow([
        ("Start", "terminal"),
        ("Prepare $|0\\rangle^{\\otimes n}$; set $k_{opt}\\approx\\frac{\\pi}{4}\\sqrt{N}$ and $k=0$", "data"),
        ("Apply $H^{\\otimes n}$: uniform superposition $|s\\rangle$", "quantum"),
        ("Oracle: flip the sign of the marked state", "quantum"),
        ("Diffusion: reflect about $|s\\rangle$; $k \\leftarrow k+1$", "quantum"),
        ("$k < k_{opt}$?", "decision"),
        ("Measure all qubits", "quantum"),
        ("Check result with the oracle; repeat if wrong", "classical"),
        ("End", "terminal"),
    ], "s2_grover_flowchart", loop=(5, 3, "yes", "no"))
    vertical_flow([
        ("Start", "terminal"),
        ("Composite $N$: pick random $a < N$", "classical"),
        ("$\\gcd(a, N) = 1$?\n(if not, it is a factor: done)", "decision"),
        ("Quantum period finding: phase estimation on\n$U|y\\rangle=|ay\\ \\mathrm{mod}\\ N\\rangle$ with inverse QFT", "quantum"),
        ("Measure phase $s/r$; continued fractions give $r$", "classical"),
        ("$r$ even and $a^{r/2}\\neq -1\\ (\\mathrm{mod}\\ N)$?", "decision"),
        ("Factors $=\\gcd(a^{r/2}\\pm 1,\\ N)$", "eval"),
        ("End", "terminal"),
    ], "s2_shor_flowchart", width=6.6, box_w=4.8, loop=(5, 1, "no: retry", "yes"))


def grover_geometry():
    theta = np.arcsin(1 / np.sqrt(16))        # one marked item in N = 16
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Arc((0, 0), 2, 2, theta1=-25, theta2=100, ec="#BBBBBB", lw=1))
    states =[(theta, "$|s\\rangle$ (start, angle $\\theta$)", "#2F5D8C"),
              (-theta, "after oracle (angle $-\\theta$)", "#B28704"),
              (3 * theta, "after diffusion (angle $3\\theta$)", "#6A3FA0"),
              (5 * theta, "after second iteration ($5\\theta$)", "#2E7D32")]
    ax.annotate("", (1.2, 0), (-0.1, 0), arrowprops=dict(arrowstyle="-|>", color="#333333"))
    ax.annotate("", (0, 1.2), (0, -0.45), arrowprops=dict(arrowstyle="-|>", color="#333333"))
    ax.text(1.22, 0, "$|s'\\rangle$ (unmarked)", va="center", fontsize=9)
    ax.text(0, 1.23, "$|w\\rangle$ (marked)", ha="center", fontsize=9)
    for angle, label, colour in states:
        x, y = np.cos(angle), np.sin(angle)
        ax.annotate("", (x, y), (0, 0), arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.8))
        ax.text(x * 1.03 + 0.02, y * 1.03, label, fontsize=8.5, color=colour, va="center")
    ax.text(0.72, -0.62, f"Each iteration rotates the state by $2\\theta$, with $\\sin\\theta = 1/\\sqrt{{N}}$ (drawn for N = 16)",
            ha="center", fontsize=8.5)
    ax.set_xlim(-0.6, 2.05)
    ax.set_ylim(-0.72, 1.35)
    save_fig(fig, "s2_grover_geometry")


def complexity_classes():
    fig, ax = canvas(7.4, 5.3)
    ax.set_ylim(-0.4, 4.9)
    def region(x, y, w, h, face, edge, alpha=0.45):
        ax.add_patch(Ellipse((x, y), w, h, facecolor=face, edgecolor=edge, lw=1.3, alpha=alpha))
    region(3.7, 2.42, 7.2, 4.7, "#F4F4F4", "#777777", 1)
    region(2.7, 2.3, 3.9, 3.0, "#DCE9F7", "#2F5D8C")
    region(4.55, 2.3, 3.9, 3.0, "#E9DFF5", "#6A3FA0")
    region(3.62, 2.3, 1.45, 1.35, "#DFF0E0", "#2E7D32", 0.9)
    ax.text(3.7, 4.42, "PSPACE", ha="center", fontsize=10.5, fontweight="bold", color="#555555")
    ax.text(1.75, 3.15, "NP", fontsize=11, fontweight="bold", color="#2F5D8C")
    ax.text(5.5, 3.15, "BQP", fontsize=11, fontweight="bold", color="#6A3FA0")
    ax.text(3.62, 2.56, "P", ha="center", fontsize=11, fontweight="bold", color="#2E7D32")
    ax.text(3.62, 2.1, "sorting,\nshortest path", ha="center", va="center", fontsize=7.5)
    ax.text(3.62, 3.25, "factoring,\ndiscrete log", ha="center", va="center", fontsize=8)
    ax.text(1.8, 2.05, "NP-complete:\nSAT, travelling\nsalesman (decision)", ha="center", va="center", fontsize=8)
    ax.text(5.5, 2.05, "simulating\nquantum systems", ha="center", va="center", fontsize=8)
    ax.text(3.7, -0.22, "Widely believed picture. None of these separations has been proven.",
            ha="center", fontsize=8.5, style="italic")
    save_fig(fig, "s2_complexity_classes")


# ----------------------------------------------------------------------------- Section 4
def system_architecture():
    e1 = pd.read_csv(RES_DIR / "e1_main_holdout.csv").iloc[0]
    variance = pd.read_csv(RES_DIR / "pca_explained_variance.csv")["cumulative"].iloc[3]
    fig, ax = canvas(10.2, 6.5)
    y = 5.75
    a = box(ax, 1.15, y, 1.95, 1.0, "Raw data\nWDBC: 569 samples,\n30 features", "data")
    b = box(ax, 3.75, y, 2.5, 1.0, f"Preprocessing\nmalignant = 1; stratified split\n{e1.n_train} train / {e1.n_test} test;\nStandardScaler", "classical", size=8.5)
    c = box(ax, 6.45, y, 2.0, 1.0, f"PCA\nn = 4 components\n({variance:.1%} variance)", "classical")
    d = box(ax, 8.95, y, 2.0, 1.0, "Angle scaling\nMinMax to $[0, \\pi]$\n(rotation angles)", "classical")
    arrow(ax, a["r"], b["l"]); arrow(ax, b["r"], c["l"]); arrow(ax, c["r"], d["l"])

    fm = box(ax, 2.1, 3.75, 2.25, 0.95, "Quantum feature map\nZZFeatureMap $U_\\Phi(x)$\n4 qubits, reps = 2", "quantum")
    kern = box(ax, 5.1, 4.45, 2.3, 0.82, "Quantum kernel path\n$k(x,x')=|\\langle\\phi(x)|\\phi(x')\\rangle|^2$", "quantum", size=8.5)
    svm = box(ax, 8.3, 4.45, 2.3, 0.82, "Classical SVM\nC tuned by 5-fold CV", "classical")
    ans = box(ax, 5.1, 3.05, 2.3, 0.82, "Variational path (VQC)\nRealAmplitudes $W(\\theta)$\nreps = 3", "quantum", size=8.5)
    opt = box(ax, 8.3, 3.05, 2.3, 0.82, "Classical optimiser\nCOBYLA, 100 iterations", "classical")
    elbow(ax, [d["b"], (8.95, 5.0), (2.1, 5.0), fm["t"]])
    arrow(ax, fm["r"], kern["l"]); arrow(ax, fm["r"], ans["l"])
    arrow(ax, kern["r"], svm["l"], label="kernel\nmatrix", label_offset=(-0.27, 0.24))
    arrow(ax, (6.25, 3.17), (7.15, 3.17), label="loss", label_offset=(-0.12, 0.13))
    arrow(ax, (7.15, 2.93), (6.25, 2.93), label=r"new $\theta$", label_offset=(-0.2, -0.15))

    cl = box(ax, 4.15, 2.0, 6.3, 0.72, "Classical baselines on the same features: RBF SVM, Logistic Regression,\nRandom Forest, KNN, MLP.  Also an RBF SVM on all 30 scaled features (reference).", "classical", size=8.5)
    elbow(ax, [(2.1, 5.0), (0.55, 5.0), (0.55, 2.0), cl["l"]])
    pred = box(ax, 3.6, 0.55, 2.3, 0.72, "Prediction\nmalignant or benign", "eval")
    ev = box(ax, 7.35, 0.55, 4.3, 0.72, "Evaluation: accuracy, precision, recall, specificity,\nF1, ROC-AUC, missed cancers, time, circuit cost", "eval", size=8.5)
    arrow(ax, cl["b"], (cl["b"][0], 0.91))
    elbow(ax, [svm["r"], (9.75, 4.45), (9.75, 1.3), (3.2, 1.3), (3.2, 0.91)],
          label="trained quantum models", label_at=(7.0, 1.3))
    ax.plot([opt["r"][0], 9.75], [3.05, 3.05], color="#333333", lw=1.2, zorder=1)
    arrow(ax, pred["r"], ev["l"])
    save_fig(fig, "s4_system_architecture")


def hybrid_model_architecture():
    fig, ax = canvas(10.2, 7.3)
    def band(y, h, text, face):
        ax.add_patch(Rectangle((0.1, y), 10.0, h, facecolor=face, edgecolor="#BBBBBB", lw=0.8, zorder=0))
        ax.text(0.22, y + h - 0.17, text, fontsize=9, fontweight="bold", va="center")

    # (a) QSVC
    band(3.85, 3.35, "(a) QSVC: the circuit computes similarities, the SVM finds the boundary", "#FBFBFB")
    x = box(ax, 1.15, 5.65, 1.6, 0.8, "Sample pair\n$(x_i, x_j)$", "data")
    u = box(ax, 3.2, 5.65, 1.35, 0.8, "$U_\\Phi(x_i)$", "quantum", size=10)
    ud = box(ax, 4.85, 5.65, 1.35, 0.8, "$U_\\Phi^\\dagger(x_j)$", "quantum", size=10)
    m = box(ax, 6.65, 5.65, 1.65, 0.8, "Measure\n$P(0^n)=k(x_i,x_j)$", "quantum", size=8.5)
    km = box(ax, 8.85, 5.65, 1.9, 0.8, "Kernel matrix $K$\n$M(M-1)/2$ entries", "classical", size=8.5)
    for s, t in [(x, u), (u, ud), (ud, m), (m, km)]:
        arrow(ax, s["r"], t["l"])
    ax.text(5.0, 6.28, "on the simulator", fontsize=8, color=STYLES["quantum"][1], ha="center")
    svm = box(ax, 8.85, 4.4, 1.9, 0.85, "SVM dual solver\nfinds $\\alpha_i, b$", "classical", size=8.5)
    f = box(ax, 5.45, 4.4, 3.5, 0.85, "$f(x)=\\mathrm{sign}\\,(\\Sigma_i\\,\\alpha_i y_i k(x_i,x)+b)$\nnew $x$: kernel with training set", "eval", size=8.5)
    arrow(ax, km["b"], svm["t"]); arrow(ax, svm["l"], f["r"])

    # (b) VQC
    band(0.1, 3.6, "(b) VQC: the circuit predicts, the classical optimiser trains $\\theta$", "#FBFBFB")
    xv = box(ax, 1.15, 2.45, 1.6, 0.8, "Sample $x$", "data")
    uv = box(ax, 3.2, 2.45, 1.35, 0.8, "$U_\\Phi(x)$", "quantum", size=10)
    w = box(ax, 4.85, 2.45, 1.35, 0.8, "$W(\\theta)$", "quantum", size=10)
    mv = box(ax, 6.65, 2.45, 1.65, 0.8, "Measure:\nclass = $b\\ \\mathrm{mod}\\ 2$", "quantum", size=8.5)
    p = box(ax, 8.85, 2.45, 1.9, 0.8, "Probabilities\n$p_c(x;\\theta)$", "classical", size=8.5)
    for s, t in [(xv, uv), (uv, w), (w, mv), (mv, p)]:
        arrow(ax, s["r"], t["l"])
    ax.text(5.0, 3.08, "on the simulator", fontsize=8, color=STYLES["quantum"][1], ha="center")
    loss = box(ax, 8.85, 0.85, 1.9, 0.8, "Cross-entropy\nloss $L(\\theta)$", "classical", size=8.5)
    opt = box(ax, 4.85, 0.85, 2.7, 0.8, "COBYLA updates $\\theta$\n(no gradients; 100 iterations)", "classical", size=8.5)
    arrow(ax, p["b"], loss["t"]); arrow(ax, loss["l"], opt["r"]); arrow(ax, opt["t"], w["b"])
    ax.text(1.45, 1.2, "$b$ = measured bitstring as an\ninteger, so class = value of\nqubit 0 (Qiskit default)",
            fontsize=8, ha="center", va="center")
    save_fig(fig, "s4_hybrid_model_architecture")


def workflow():
    fig, ax = canvas(9.6, 12.6)
    cx = 4.8
    start = box(ax, cx, 12.2, 1.4, 0.42, "Start", "terminal", bold=True)
    prev = start
    for i, (text, style) in enumerate([
            ("Load WDBC: 569 samples, 30 features", "data"),
            ("Re-map labels (malignant = 1); check missing values; EDA", "classical"),
            ("Stratified 80/20 split with seed", "classical"),
            ("Fit StandardScaler, PCA(n) and MinMax $[0,\\pi]$ on training data only", "classical")]):
        node = box(ax, cx, 11.42 - i * 0.82, 5.6, 0.56, text, style)
        arrow(ax, prev["b"], node["t"])
        prev = node

    tops = {"q": 1.75, "v": 4.8, "c": 7.85}
    y = 7.72
    q1 = box(ax, tops["q"], y, 2.7, 0.62, "QSVC: build ZZFeatureMap\nand kernel", "quantum")
    v1 = box(ax, tops["v"], y, 2.7, 0.62, "VQC: ZZFeatureMap +\nRealAmplitudes", "quantum")
    c1 = box(ax, tops["c"], y, 2.7, 0.62, "Classical: fit 5 models on\nPCA features; SVM on all 30", "classical", size=8.5)
    elbow(ax, [prev["b"], (cx, 8.3), (tops["q"], 8.3), q1["t"]])
    arrow(ax, prev["b"], v1["t"])
    elbow(ax, [prev["b"], (cx, 8.3), (tops["c"], 8.3), c1["t"]])

    q2 = box(ax, tops["q"], 6.82, 2.7, 0.62, "Compute training kernel\nmatrix $K$", "quantum")
    q3 = box(ax, tops["q"], 5.92, 2.7, 0.62, "Tune C in {0.1, 1, 10}\nby 5-fold CV", "classical")
    q4 = box(ax, tops["q"], 5.02, 2.7, 0.62, "Fit SVM on $K$", "classical")
    arrow(ax, q1["b"], q2["t"]); arrow(ax, q2["b"], q3["t"]); arrow(ax, q3["b"], q4["t"])

    v2 = box(ax, tops["v"], 6.82, 2.7, 0.62, "Run circuits; compute\ncross-entropy loss", "quantum")
    v3 = box(ax, tops["v"], 5.92, 2.7, 0.62, "COBYLA updates $\\theta$", "classical")
    v4 = diamond(ax, tops["v"], 5.02, 2.5, 0.78, "iterations\n< 100?")
    arrow(ax, v1["b"], v2["t"]); arrow(ax, v2["b"], v3["t"]); arrow(ax, v3["b"], v4["t"])
    elbow(ax, [v4["r"], (6.33, 5.02), (6.33, 6.82), v2["r"]], label="yes", label_at=(6.33, 5.9))

    pred = box(ax, cx, 3.95, 6.6, 0.56, "Predict the test set; record time and circuit statistics", "classical")
    elbow(ax, [q4["b"], (tops["q"], 4.45), (3.2, 4.45), (3.2, 4.23)])
    arrow(ax, v4["b"], (tops["v"], 4.23), label="no", label_offset=(0.07, 0.03))
    elbow(ax, [c1["b"], (tops["c"], 4.45), (6.4, 4.45), (6.4, 4.23)])

    met = box(ax, cx, 3.15, 6.6, 0.56, "Compute accuracy, precision, recall, specificity, F1, ROC-AUC, confusion matrix", "eval", size=8.5)
    dec = diamond(ax, cx, 2.15, 3.6, 0.9, "More seeds, folds or\ngrid settings (E2-E9)?")
    out = box(ax, cx, 1.05, 6.6, 0.56, "Save CSV tables and figures; compare quantum and classical models", "eval")
    end = box(ax, cx, 0.35, 1.4, 0.42, "End", "terminal", bold=True)
    arrow(ax, pred["b"], met["t"]); arrow(ax, met["b"], dec["t"]); arrow(ax, dec["b"], out["t"], label="no")
    arrow(ax, out["b"], end["t"])
    elbow(ax, [dec["r"], (9.42, 2.15), (9.42, 9.78), (cx + 2.8, 9.78)], label="yes", label_at=(9.42, 5.6))
    save_fig(fig, "s4_workflow_flowchart")


def stick_figure(ax, x, y, label):
    ax.add_patch(Circle((x, y + 0.42), 0.14, fill=False, lw=1.4, ec="#333333"))
    ax.plot([x, x], [y + 0.28, y - 0.15], color="#333333", lw=1.4)
    ax.plot([x - 0.25, x + 0.25], [y + 0.12, y + 0.12], color="#333333", lw=1.4)
    ax.plot([x, x - 0.2], [y - 0.15, y - 0.5], color="#333333", lw=1.4)
    ax.plot([x, x + 0.2], [y - 0.15, y - 0.5], color="#333333", lw=1.4)
    ax.text(x, y - 0.68, label, ha="center", va="center", fontsize=9)


def use_case():
    fig, ax = canvas(9.4, 6.0)
    ax.add_patch(Rectangle((2.0, 0.25), 5.4, 5.5, fill=False, ec="#555555", lw=1.2))
    ax.text(4.7, 5.53, "Breast cancer diagnosis system (hybrid QML)", ha="center", fontsize=9.5, fontweight="bold")

    def use(cx, cy, text, style="classical", w=2.1, h=0.62):
        face, edge = STYLES[style]
        ax.add_patch(Ellipse((cx, cy), w, h, facecolor=face, edgecolor=edge, lw=1.2, zorder=2))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=8.5, zorder=3)
        return (cx, cy, w / 2, h / 2)

    def edge(node, side):
        cx, cy, a, _ = node
        return (cx + a, cy) if side == "r" else (cx - a, cy)

    enter = use(3.3, 4.75, "Enter tumour\nfeatures")
    predict = use(3.3, 3.75, "Predict malignant\nor benign")
    view = use(3.3, 2.75, "View prediction", "eval")
    ax.text(3.3, 1.25, "Clinical use is shown as a concept\nonly: no clinical tool was built\nor validated in this work.",
            ha="center", va="center", fontsize=8, style="italic", color="#555555")
    prep = use(6.1, 4.75, "Preprocess data")
    qsvc = use(6.1, 3.75, "Train QSVC", "quantum")
    vqc = use(6.1, 2.75, "Train VQC", "quantum")
    evalu = use(6.1, 1.75, "Evaluate models", "eval")
    comp = use(6.1, 0.8, "Compare with\nclassical models", "eval")

    stick_figure(ax, 0.9, 3.5, "Clinician / user")
    stick_figure(ax, 8.5, 3.0, "Researcher")
    for node in (enter, view):
        ax.plot([1.2, edge(node, "l")[0]], [3.6, node[1]], color="#333333", lw=1, zorder=1)
    for node in (prep, qsvc, vqc, evalu, comp):
        ax.plot([8.2, edge(node, "r")[0]], [3.1, node[1]], color="#333333", lw=1, zorder=1)

    def relation(a, b, text, offset=(0.05, 0.0)):
        arrow(ax, a, b, dashed=True, color="#555555")
        ax.text((a[0] + b[0]) / 2 + offset[0], (a[1] + b[1]) / 2 + offset[1], text, fontsize=7.5,
                color="#555555", ha="left", va="center")
    relation((3.3, 4.44), (3.3, 4.06), "«include»", (0.05, 0))
    relation((3.3, 3.06), (3.3, 3.44), "«include»", (0.05, 0))
    relation((6.1, 4.06), (6.1, 4.44), "«include»", (0.05, 0))
    elbow(ax, [edge(vqc, "l"), (4.72, vqc[1]), (4.72, prep[1]), edge(prep, "l")], dashed=True, color="#555555",
          label="«include»", label_at=(4.72, 3.25))
    relation((6.1, 1.44), (6.1, 1.11), "«include»", (0.05, 0))
    save_fig(fig, "s4_use_case")


if __name__ == "__main__":
    stacks()
    flowcharts()
    grover_geometry()
    complexity_classes()
    system_architecture()
    hybrid_model_architecture()
    workflow()
    use_case()
