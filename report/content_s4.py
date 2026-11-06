"""Section 4: the proposed mini project."""


def section4(d):
    d.h1("Proposed Mini Project: Design and Development of a Hybrid Quantum Machine Learning Model")

    d.h2("Problem Statement")
    d.p("The World Health Organization (WHO) estimates that 2.4 million women were diagnosed with breast cancer in 2024 and that the disease "
        "caused 694,000 deaths; it was the most common cancer in women in 164 of 186 countries [[c:who2026]]. GLOBOCAN estimates for 2022 "
        "place female breast cancer at 11.6% of all new cancers worldwide [[c:bray2024]]. Accurate and early diagnosis supports timely treatment, "
        "and in a diagnostic test a false negative, which is a missed malignant tumour, is far more harmful than a false alarm. This project "
        "classifies tumours as malignant or benign from fine-needle aspirate (FNA) measurements, with malignant-class recall as the main clinical "
        "measure, and asks whether small quantum circuits can do this as well as classical models.")

    d.h2("Background and Motivation")
    d.p("FNA features are a well-studied benchmark on which classical ML is already strong, so any quantum result can be judged fairly. QML papers "
        "report mixed results on this and similar data. Some report strong quantum performance [[c:r30]], while others find that classical models "
        "remain ahead [[c:r32,mpofu2025]]. Quantum kernels and VQCs are the two leading near-term approaches [[c:r7]], but each has known problems: "
        "kernel concentration [[c:r8]] and barren plateaus [[c:r20,r21]]. Testing them together, on identical features, shows how these issues appear "
        "on a real diagnostic task. QML is included here as the subject of a fair experiment, not because it is expected to win. A careful negative result "
        "is still useful, as argued in [[c:r2,r6]].")

    d.h2("Objectives")
    d.numbered([
        "**O1.** Build QSVC and VQC pipelines in Qiskit and evaluate both on the Wisconsin Diagnostic Breast Cancer (WDBC) data with the same preprocessing.",
        "**O2.** Compare them with classical models on identical reduced features using accuracy, precision, recall, specificity, F1-score, ROC-AUC and missed malignant cases, with 5-fold cross-validation.",
        "**O3.** Measure how qubit count (2 to 8), feature-map depth, entanglement, ansatz depth, optimiser and training size affect accuracy and cost.",
        "**O4.** Measure the effect of shot noise and a fake IBM device noise model on accuracy and recall.",
        "**O5.** Report the computational cost of each model and explain where quantum models stand today.",
    ])

    d.h2("Research Gaps Addressed")
    d.bullets([
        "A direct VQC and quantum kernel comparison on WDBC, which only [[c:r32]] offers among the reviewed papers.",
        "A fair baseline on identical features, in contrast to weaker or unclear comparisons in [[c:r24,r25,r30]].",
        "A joint sweep of circuit settings on one medical dataset, extending the partial studies in [[c:r31,r32]] and connecting them to theory [[c:r8,r22]].",
        "Malignant recall under device noise, which goes beyond kernel-level and gradient-level noise studies [[c:r10,r21]].",
        "Missed-cancer counts and cost figures for every model, which the reviewed abstracts rarely report [[c:r29,r33]].",
    ])

    d.h2("Proposed Solution and Contributions")
    d.p("The solution is a single pipeline in which the data is scaled, reduced by Principal Component Analysis (PCA) and scaled again into rotation angles. "
        "The same features then go to a QSVC, a VQC and classical baselines, and every model is scored in the same way. The expected contributions are: "
        "(1) a working hybrid QML pipeline for medical diagnosis in Qiskit; (2) a fair comparison on identical reduced features with a focus on malignant "
        "recall; (3) an analysis of how qubit count, depth, entanglement, training size and noise affect accuracy and cost; and (4) an honest discussion of "
        "where quantum models stand in the NISQ era.")

    d.h2("Techniques and Justification")
    d.p("[[T:techniques]] gives the reason for each choice. The default settings follow common practice in the reviewed work and keep the circuits small enough "
        "for a laptop simulator.")
    d.table("techniques", "Selected techniques and why they were chosen.", ["Technique", "Role", "Justification"], [
        ["PCA [[c:bishop2006]]", "Reduces 30 features to $n$", "One qubit per feature limits the input size; PCA keeps the most variance with no use of labels"],
        ["Angle encoding", "Maps features to rotation angles", "Simple, uses few qubits, and repeated encoding enriches the model [[c:r16]]"],
        ["ZZFeatureMap [[c:r7]]", "Encodes data with entanglement", "Standard map, believed hard to simulate at large scale"],
        ["Fidelity quantum kernel", "Computes sample similarity", "Lets a classical SVM use the quantum feature space, with convex training"],
        ["SVM [[c:cortes1995]]", "Finds the decision boundary", "Same algorithm as the RBF baseline, so only the kernel differs"],
        ["RealAmplitudes", "Trainable circuit $W(\\theta)$", "Compact: only RY rotations and CNOT gates; gives a real-valued ansatz"],
        ["COBYLA [[c:powell1994]]", "Trains the VQC", "Gradient-free, so it avoids extra circuit runs for gradients"],
        ["SPSA [[c:spall1992]]", "Alternative optimiser", "Estimates gradients with two circuits and is designed to tolerate noise"],
        ["Classical baselines [[c:pedregosa2011]]", "Comparison models", "RBF SVM, logistic regression, random forest, KNN and MLP cover the main model families"],
    ], widths=[1.6, 1.8, 3.1])

    d.h2("Mathematical Formulation")
    d.p("**Preprocessing.** PCA and scaling are given in [[E:pca]], where $\\mathbf{x}$ is a standardised sample, $\\boldsymbol{\\mu}$ the training mean, $W_k$ the first $k$ "
        "principal directions, and $\\min_j$ and $\\max_j$ the training extremes of component $j$.")
    d.eq("pca", r"\mathbf{z} = W_k^\top (\mathbf{x} - \boldsymbol{\mu}), \qquad \tilde{z}_j = \pi \cdot \frac{z_j - \min_j}{\max_j - \min_j}")
    d.p("**Feature map.** One layer of the ZZFeatureMap applies [[E:fmap]], where $Z_j$ is the Pauli-Z operator on qubit $j$ and $H$ is the Hadamard gate. In this work $x_j$ denotes "
        "the scaled feature $\\tilde{z}_j$. The layer is repeated reps times.")
    d.eq("fmap", r"U_{\Phi}(\mathbf{x}) = \exp\left(i\sum_{j} x_j Z_j + i\sum_{j<k} (\pi - x_j)(\pi - x_k) Z_j Z_k\right) H^{\otimes n}")
    d.p("**Quantum kernel.** The kernel in [[E:kernel]] measures the overlap of two encoded states. On a device it is estimated as the probability of measuring all zeros.")
    d.eq("kernel", r"k(\mathbf{x}, \mathbf{x}') = |\langle \phi(\mathbf{x}) | \phi(\mathbf{x}') \rangle|^2 = |\langle 0^n | U_{\Phi}^{\dagger}(\mathbf{x}') U_{\Phi}(\mathbf{x}) | 0^n \rangle|^2")
    d.p("**SVM.** Training solves the dual problem in [[E:svm]] with labels $y_i\\in\\{-1,+1\\}$ and regularisation $C$, subject to $0\\le\\alpha_i\\le C$ and $\\sum_i\\alpha_i y_i=0$. Prediction uses [[E:decision]], where $b$ is the bias.")
    d.eq("svm", r"\max_{\boldsymbol{\alpha}} \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j k(\mathbf{x}_i, \mathbf{x}_j)")
    d.eq("decision", r"f(\mathbf{x}) = \text{sign}\left(\sum_i \alpha_i y_i k(\mathbf{x}_i, \mathbf{x}) + b\right)")
    d.p("**VQC.** The circuit output gives class probabilities, and training minimises the cross-entropy in [[E:loss]], where $M$ is the number of samples and $y_{i,c}$ is 1 for the true class and 0 otherwise. "
        "In the Qiskit version used, the default mapping assigns the class as the measured bitstring value modulo 2, as in [[E:vqcprob]]. Although that function is named parity, it depends only on qubit 0.")
    d.eq("loss", r"L(\boldsymbol{\theta}) = -\frac{1}{M}\sum_{i=1}^{M} \sum_{c} y_{i,c} \log p_c(\mathbf{x}_i; \boldsymbol{\theta}), \qquad \boldsymbol{\theta}^* = \mathrm{arg\,min}_{\boldsymbol{\theta}} L(\boldsymbol{\theta})")
    d.eq("vqcprob", r"p_c(\mathbf{x};\boldsymbol{\theta}) = \sum_{b:\; b \,\text{mod}\, 2 = c} |\langle b | W(\boldsymbol{\theta}) U_{\Phi}(\mathbf{x}) | 0^n \rangle|^2")

    d.h2("Dataset and Preprocessing")
    d.p("The Breast Cancer Wisconsin (Diagnostic) dataset [[c:uci1995]] was loaded from scikit-learn [[c:pedregosa2011]]. [[T:dataset]] summarises it. Labels were re-mapped so that malignant is the positive class. "
        "There were <<missing>> missing values.")
    d.table("dataset", "Dataset summary.", ["Item", "Detail"], [
        ["Samples", "<<n_samples>>: <<n_mal>> malignant (<<pct_mal>>) and <<n_ben>> benign"],
        ["Features", "<<n_features>> real values: mean, standard error and worst value of 10 measurements"],
        ["Measurements", "Radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension"],
        ["Split", "Stratified 80/20: <<n_train>> training and <<n_test>> test samples (<<h_test_mal>> malignant, <<h_test_ben>> benign in the test set)"],
        ["Labels", "Malignant = 1 (re-mapped from scikit-learn, where 0 = malignant)"],
    ], widths=[1.3, 5.1])
    d.p("[[F:eda1]] shows the class balance and the variance kept by PCA. Four components keep <<pca_4>> of the variance, and two, six and eight keep <<pca_2>>, <<pca_6>> and <<pca_8>>. "
        "Always predicting benign would score <<majority>>, which is the minimum that a useful model must beat. [[F:eda2]] shows that the features are strongly correlated, which is why PCA "
        "compresses them well. [[F:eda3]] shows that the six features most correlated with the label separate the classes clearly, and [[F:eda4]] shows the same separation in the first two components.")
    d.fig("eda1", ["s4_class_distribution", "s4_pca_explained_variance"], "Class distribution (left) and PCA explained variance (right).", 2.95)
    d.fig("eda2", "s4_correlation_heatmap", "Correlation heatmap of the 30 features.", 4.4)
    d.fig("eda3", "s4_top6_feature_boxplots", "Standardised box plots of the six features most correlated with the label.", 5.2)
    d.fig("eda4", "s4_pca_scatter", "Training samples in the first two principal components.", 3.4)
    d.p("The preprocessing steps were: stratified split; StandardScaler; PCA to $n=4$ components (2 to 8 in E2); and MinMaxScaler to $[0,\\pi]$. Every scaler and the PCA were fitted on training data only, "
        "which prevents information from the test set leaking into the models. Test values can fall slightly outside $[0,\\pi]$, and they were not clipped because they remain valid angles.")

    d.h2("System Architecture")
    d.p("[[F:sysarch]] shows the complete system. Data flows through preprocessing to the quantum and classical paths, and all models are scored in the same evaluation step.")
    d.fig("sysarch", "s4_system_architecture", "Overall architecture of the proposed system.", 5.8)

    d.h2("Hybrid Model Architecture")
    d.p("[[F:hybrid]] separates the work done on the simulator from the work done on the CPU. In the QSVC, circuits only produce the kernel matrix, and all training is classical. In the VQC, the circuit "
        "makes the prediction, and a classical optimiser repeatedly updates its angles.")
    d.fig("hybrid", "s4_hybrid_model_architecture", "Architecture of the hybrid QSVC path (a) and VQC loop (b).", 5.6)

    d.h2("Algorithms and Pseudocode")
    d.p("Algorithms 1 and 2 give the two quantum models in pseudocode.")
    d.code("""
Algorithm 1: QSVC training and prediction
Input: training set (X, y), test set T, feature map U, grid C = {0.1, 1, 10}
 1  for each pair (x_i, x_j) with i < j do
 2      K[i][j] <- |<0| U†(x_j) U(x_i) |0>|^2            # one kernel value
 3  K[i][i] <- 1;  K[j][i] <- K[i][j]
 4  for each C value do
 5      score[C] <- 5-fold CV accuracy of SVM(C) on K
 6  C* <- value with the highest score
 7  train SVM(C*) on K to obtain alpha and b
 8  for each t in T do
 9      K_t[i] <- k(t, x_i) for all training samples
10      predict sign(sum_i alpha_i y_i K_t[i] + b)
""")
    d.code("""
Algorithm 2: VQC training
Input: training set (X, y), feature map U, ansatz W, maxiter = 100
 1  theta <- random values in [-pi, pi] (fixed seed)
 2  repeat
 3      for each sample x_i do
 4          run W(theta) U(x_i) |0> and obtain p_c(x_i; theta)
 5      L <- mean cross-entropy over the samples
 6      record L; COBYLA proposes a new theta
 7  until maxiter evaluations are used
 8  return theta
Prediction: class with the larger p_c(x; theta)
""")

    d.h2("Workflow Flowchart")
    d.p("[[F:workflow]] gives the full workflow from raw data to the final comparison, including the loop that repeats it for each seed, fold and experiment setting.")
    d.fig("workflow", "s4_workflow_flowchart", "Complete workflow of the project.", 5.0)

    d.h2("Use Case Diagram")
    d.p("[[F:usecase]] shows the two actors. A researcher preprocesses data, trains the quantum models, and evaluates and compares them. A clinician or other user would enter tumour features and view a prediction. "
        "That clinical use is only a concept here: no clinical tool was built or validated.")
    d.fig("usecase", "s4_use_case", "Use case diagram of the system.", 4.8)

    d.h2("Quantum Circuit Design")
    d.p("[[F:circuits1]] and [[F:circuits2]] show the circuits for four qubits, and [[T:circuitstats]] gives their size after decomposition into the basis gates CX, RZ, SX and X. The kernel circuit is the feature map followed by "
        "its inverse with the second sample, so its depth is about twice that of the map.")
    d.fig("circuits1", ["s4_zz_feature_map", "s4_real_amplitudes_ansatz"], "ZZFeatureMap with reps = 2 and linear entanglement (top); RealAmplitudes ansatz with reps = 3 (bottom).", 6)
    d.fig("circuits2", ["s4_vqc_full_circuit", "s4_kernel_circuit"], "Full VQC circuit, decomposed (top); compute-uncompute kernel circuit for one pair of samples (bottom).", 6)
    d.table("circuitstats", "Circuit size for four qubits after decomposition.", ["Circuit", "Qubits", "Depth", "CX gates", "Total gates", "Parameters"], [
        ["ZZFeatureMap (reps = 2, linear)", 4, "<<zz_depth>>", "<<zz_cx_gates>>", "<<zz_total_gates>>", "<<zz_parameters>> (data)"],
        ["RealAmplitudes (reps = 3)", 4, "<<ra_depth>>", "<<ra_cx_gates>>", "<<ra_total_gates>>", "<<ra_parameters>> (trainable)"],
        ["Full VQC", 4, "<<vqcc_depth>>", "<<vqcc_cx_gates>>", "<<vqcc_total_gates>>", "<<vqcc_parameters>> (4 data + 16 trainable)"],
        ["Kernel circuit", 4, "<<kc_depth>>", "<<kc_cx_gates>>", "<<kc_total_gates>>", "None once samples are bound"],
    ], align="lrrrrl", widths=[2.3, 0.6, 0.6, 0.7, 0.7, 1.6])

    d.h2("Implementation in Qiskit")
    d.p("The code uses Qiskit [[c:qiskit2024,qiskitdocs]] and Qiskit Machine Learning [[c:qmlrepo]]. The excerpts below come from *src/quantum_models.py* and *src/grid.py*. The kernel is first built on the feature map:")
    d.code("""
feature_map = zz_feature_map(n_qubits, reps=reps, entanglement=entanglement)
kernel = FidelityStatevectorKernel(feature_map=feature_map)  # exact and fast
# kernel = FidelityQuantumKernel(feature_map=feature_map)    # one circuit per pair
K = kernel.evaluate(X_train)
search = GridSearchCV(SVC(kernel="precomputed"), {"C": [0.1, 1, 10]}, cv=cv)
search.fit(K, y_train)
""")
    d.p("The VQC uses the same feature map with an ansatz, an optimiser and a callback that records the loss:")
    d.code("""
ansatz = real_amplitudes(n_qubits, reps=3)
init = np.random.default_rng(seed).uniform(-np.pi, np.pi, ansatz.num_parameters)
vqc = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=COBYLA(maxiter=100),
          callback=callback, initial_point=init)
vqc.fit(X_train, y_train)
""")
    d.p("Noisy execution uses an Aer sampler with a noise model copied from the fake backend, with circuits transpiled for that device:")
    d.code("""
fake = FakeManilaV2()
pm = generate_preset_pass_manager(backend=fake, optimization_level=1,
                                  seed_transpiler=seed)
noise = NoiseModel.from_backend(fake)
sampler = AerSampler(default_shots=1024, seed=seed,
                     options={"backend_options": {"noise_model": noise}})
fidelity = ComputeUncompute(sampler=sampler, pass_manager=pm)
kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
""")
    d.p("**Simulation and execution.** Three modes were used. (1) The *exact statevector kernel* computes one state per sample and takes inner products. (2) The *compute-uncompute circuit kernel* simulates one circuit for each pair. "
        "A check on <<kc_samples>> samples (<<kc_entries>> kernel entries) showed that the two agree to within <<kc_diff>>, while the circuit form took <<kc_circ>> s against <<kc_sv>> s, about <<kc_ratio>> times longer. The same check showed that Qiskit's QSVC class "
        "gives the same test predictions as an SVM on the precomputed kernel. Mode 1 was therefore used for the main runs and sweeps. (3) *Sampled simulation*, with or without the device noise model, was used in E8. The VQC used Qiskit's default exact sampler. No real quantum hardware was used.")

    d.h2("Implementation Environment")
    d.p("[[T:env]] lists the configuration. All randomness uses seed 42, or seeds 42, 43 and 44 in repeated experiments. The exact package versions are in *requirements.txt*.")
    d.table("env", "Implementation environment.", ["Item", "Detail"], [
        ["Operating system", "Windows 11 Pro"], ["CPU and RAM", "Intel Core i3-1115G4 at 3.00 GHz; 7.7 GB RAM; no GPU"],
        ["Language", "Python 3.12.10"],
        ["Quantum libraries", "Qiskit 2.6.1; Qiskit Machine Learning 0.9.1; Qiskit Aer 0.18.2; Qiskit IBM Runtime 0.51.0"],
        ["Classical libraries", "scikit-learn 1.9.1; NumPy 2.5.4; SciPy 1.18.1; pandas 3.1.0; Matplotlib 3.11.2; seaborn 0.13.2"],
        ["Simulators", "Statevector simulation; Aer SamplerV2; FakeManilaV2 noise model (5 qubits)"],
        ["Dataset", "Breast Cancer Wisconsin (Diagnostic) from scikit-learn; Iris for E9"],
        ["Development tools", "Python scripts run from a terminal; report built with python-docx and WPS Office"],
    ], widths=[1.6, 4.8])
