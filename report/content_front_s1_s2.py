"""Front matter, Section 1 and Section 2."""

TITLE = ("Hybrid Quantum-Classical Machine Learning for Breast Cancer Diagnosis: "
         "A Comparative Study of Quantum Kernel SVM and Variational Quantum Classifier")
ABBREVIATIONS = [
    ("AI", "Artificial Intelligence"), ("AUC", "Area Under the Curve"), ("BQP", "Bounded-error Quantum Polynomial time"),
    ("CNN", "Convolutional Neural Network"), ("CNOT / CX", "Controlled-NOT gate"),
    ("COBYLA", "Constrained Optimization BY Linear Approximation"), ("CV", "Cross-Validation"),
    ("EDA", "Exploratory Data Analysis"), ("FN / FP", "False Negative / False Positive"), ("FNA", "Fine-Needle Aspirate"),
    ("KNN", "K-Nearest Neighbours"), ("MLP", "Multilayer Perceptron"), ("ML", "Machine Learning"),
    ("NISQ", "Noisy Intermediate-Scale Quantum"), ("PCA", "Principal Component Analysis"),
    ("QAOA", "Quantum Approximate Optimization Algorithm"), ("QCNN", "Quantum Convolutional Neural Network"),
    ("QFT", "Quantum Fourier Transform"), ("QML", "Quantum Machine Learning"), ("QNN", "Quantum Neural Network"),
    ("QSVC", "Quantum Support Vector Classifier"), ("QSVM", "Quantum Support Vector Machine"), ("RBF", "Radial Basis Function"),
    ("ROC", "Receiver Operating Characteristic"), ("SPSA", "Simultaneous Perturbation Stochastic Approximation"),
    ("SVM", "Support Vector Machine"), ("VQA", "Variational Quantum Algorithm"), ("VQC", "Variational Quantum Classifier"),
    ("WDBC", "Wisconsin Diagnostic Breast Cancer dataset"), ("WHO", "World Health Organization"),
]


def front(d):
    d._add("cover", school="SHARDA SCHOOL OF COMPUTING SCIENCE AND ENGINEERING (SSCSE)",
           department="COMPUTER SCIENCE & ENGINEERING", title=TITLE,
           course="Quantum Computing with Artificial Intelligence (Course Code: NV62010)",
           lines=[("Final Assignment Technical Report", 8), ("Semester- 7th", 8), ("Submission date: [fill in]", 60)],
           by=["Student Name: Aditi Yadav", "System ID: 2023215091", "Roll No.: 2301010045", "Section/Group: BTech CSB (G1)"],
           to=["Dr. Dharam Raj", "Sharda University, Greater Noida"])

    d.h1("Declaration", numbered=False)
    d.p("I declare that this report is my own submission for course NV62010, that the experiments were run on the "
        "configuration stated in Section 4, and that every source used is listed in Section 7.")
    d.p("**Use of AI assistance [confirm against the course policy; edit or remove as required]:** an AI assistant (Claude, "
        "by Anthropic) was used to help write and run the code, find and check literature records, and draft and format "
        "this report. All result numbers come from code that was actually executed. The literature reviews are based on "
        "abstracts as stated in Section 3.")
    d.p("Signature: [fill in]          Date: [fill in]")

    d.h1("Abstract", numbered=False)
    d.p("Breast cancer is a leading cause of cancer death in women, and missing a malignant tumour is the most serious "
        "diagnostic error. This work tests whether small quantum circuits can classify tumours from fine-needle aspirate "
        "measurements. Two hybrid quantum-classical models were built in Qiskit on the Breast Cancer Wisconsin (Diagnostic) "
        "dataset of <<n_samples>> samples: a quantum kernel support vector classifier (QSVC) and a variational quantum "
        "classifier (VQC). The <<n_features>> features were reduced to four by principal component analysis and encoded "
        "into four qubits. Both models were compared with five classical models on identical features, and with a support "
        "vector machine (SVM) on all <<n_features>> features. In 5-fold cross-validation, QSVC reached <<cv_QSVC_acc>> "
        "accuracy and <<cv_QSVC_rec>> malignant recall, and VQC reached <<cv_VQC_acc>> and <<cv_VQC_rec>>. The radial basis "
        "function SVM on the same features reached <<cv_SVM_acc>> and <<cv_SVM_rec>>. The quantum models missed "
        "<<cv_QSVC_fn>> and <<cv_VQC_fn>> of <<n_mal>> malignant cases, against <<cv_SVM_fn>> for that SVM. Experiments on "
        "qubit count, circuit depth, entanglement, training size, optimiser and noise showed that a feature map without "
        "entanglement performed best (<<e4_z_acc>> accuracy), and that kernel values were very small for entangling maps, "
        "which is consistent with kernel concentration. All results came from simulators, and VQC training had not converged "
        "after 100 iterations. No quantum advantage was found on this task. Classical models remain the better choice for "
        "this dataset, while the pipeline provides a fair, reproducible basis for further study.")

    d.h1("Table of Contents", numbered=False)
    d.toc('TOC \\o "1-3" \\h \\z \\u')
    d.h1("List of Figures", numbered=False)
    d.toc('TOC \\h \\z \\t "Fig Caption,1"')
    d.h1("List of Tables", numbered=False)
    d.toc('TOC \\h \\z \\t "Table Caption,1"')
    d.h1("List of Abbreviations", numbered=False)
    d._add("abbrev", rows=[list(pair) for pair in ABBREVIATIONS])


def section1(d):
    d.h1("Introduction and Synopsis")
    d.p("Quantum Machine Learning (QML) studies how quantum computers can support learning from data [[c:biamonte2017]]. "
        "Most current work is hybrid: a parametrised quantum circuit processes data, while a classical computer stores the "
        "data and trains the model. Within Artificial Intelligence (AI) [[c:russell2020]], QML is therefore best seen as an "
        "experimental extension of machine learning (ML) rather than a replacement for it. Today's hardware belongs to the "
        "noisy intermediate-scale quantum (NISQ) era, so most studies use small circuits or simulators [[c:r1,r15]].")
    d.p("**Major applications.** In image recognition, hybrid networks with parallel quantum circuits have been tested on "
        "benchmark image sets [[c:r36]]. In healthcare, quantum classifiers and kernels have been applied to clinical tables "
        "and electronic health records [[c:r28,r33]]. In financial modelling, quantum methods target Monte Carlo estimation, "
        "optimisation and learning [[c:r43]]. In logistics, vehicle routing has been reformulated for variational solvers "
        "[[c:r44]]. In optimisation, the Quantum Approximate Optimization Algorithm (QAOA) and analogue devices are studied "
        "on hard combinatorial problems [[c:r38,r39]].")
    d.p("**Potential advantages.** A quantum feature map places data in a Hilbert space that grows exponentially with qubit "
        "number [[c:r7]], and speed-ups have been proven for specially constructed learning problems [[c:r4]]. Some bounds "
        "also suggest good generalisation from few training points [[c:r5]]. These remain potential benefits: with enough "
        "data, classical models can compete even on problems designed for quantum models [[c:r3]].")
    d.p("**Current trends.** Active topics include variational algorithms [[c:r14]], quantum kernels [[c:r7,r8]], barren "
        "plateaus (vanishing training gradients) [[c:r20]], error mitigation [[c:kim2023]], and dequantization, in which "
        "classical surrogates reproduce some quantum models [[c:schreiber2023]]. Hardware roadmaps are also moving towards "
        "error correction: surface-code memories have operated below threshold [[c:google2025]], and IBM has published a plan "
        "for a fault-tolerant machine by 2029 [[c:ibm2025]]. Industry reports describe surging investment [[c:mckinsey2025]], "
        "which is a sign of interest and not evidence of practical advantage.")
    d.p("**Challenges and limitations.** Noise, small qubit counts, the cost of encoding classical data, poor trainability "
        "and exponential concentration of kernel values all limit current models [[c:r8,r21]]. Large benchmarks have also "
        "found that standard classical models usually beat quantum classifiers on small tasks [[c:r6]], and it has been argued "
        "that beating classical ML may not be the most useful research goal at this stage [[c:r2]].")
    d.p("**Future scope.** Useful directions include better encodings, trainable kernels, error mitigation and larger "
        "hardware. In healthcare, QML could one day support decision making, but it must first show reliable performance, "
        "strong recall of disease, and privacy-safe handling of data.")
    d.p("**This report.** The mini project in this report classifies breast tumours as malignant or benign using two hybrid "
        "models, a quantum kernel support vector classifier and a variational quantum classifier, and compares them honestly "
        "with classical models. Section 2 covers quantum computing fundamentals and algorithms. Section 3 reviews 45 research "
        "papers. Section 4 describes the proposed project, Section 5 presents the results, Section 6 concludes, and Section 7 "
        "lists the references.")


def section2(d):
    d.h1("Fundamentals of Quantum Computing and Quantum Algorithms")
    d.p("This section explains the concepts used later in the mini project. Wherever possible, the working of a concept is "
        "shown with a circuit that was run in Qiskit [[c:qiskit2024,qiskitdocs]]. The same circuits could also be written in "
        "Cirq [[c:cirq]]. All demonstrations use a fixed random seed of 42 and the scripts in the folder *section2_demos*.")

    d.h2("Introduction and Principles")
    d.p("Feynman argued that simulating quantum physics efficiently calls for a computer that itself follows quantum rules "
        "[[c:feynman1982]]. Deutsch then described a universal quantum computer [[c:deutsch1985]]. Later results by Deutsch and "
        "Jozsa [[c:deutsch1992]], Shor [[c:shor1997]] and Grover [[c:grover1996]] showed that quantum algorithms can beat known "
        "classical methods on particular problems. The standard textbook treatment is given in [[c:nielsen2010]], and an applied "
        "introduction in [[c:hidary2021]]. In simple terms, four postulates describe the model: a system's state is a unit vector "
        "in a complex vector space; it evolves by unitary operations; measurement gives random outcomes with fixed "
        "probabilities; and joint systems are described by the tensor product of their parts.")

    d.h2("Classical Bits versus Qubits")
    d.p("A classical bit is either 0 or 1. A qubit can be in a superposition of both basis states, as written in [[E:qubit]].")
    d.eq("qubit", r"|\psi\rangle = \alpha|0\rangle + \beta|1\rangle, \quad |\alpha|^2 + |\beta|^2 = 1")
    d.p("Here $\\alpha$ and $\\beta$ are complex amplitudes, and $|\\alpha|^2$ and $|\\beta|^2$ are the probabilities of measuring 0 and "
        "1. Up to a global phase, any single-qubit state is a point on the Bloch sphere, as in [[F:bloch]]. [[T:bitqubit]] compares "
        "the two units of information.")
    d.fig("bloch", "s2_bloch_basis_states", "Bloch sphere positions of the states $|0\\rangle$, $|1\\rangle$, $|+\\rangle$ and $|+i\\rangle$ (Qiskit output).", 6)
    d.table("bitqubit", "Comparison of a classical bit and a qubit.", ["Property", "Classical bit", "Qubit"], [
        ["State", "0 or 1", "$\\alpha|0\\rangle+\\beta|1\\rangle$"],
        ["State of $n$ units", "One of $2^n$ values", "$2^n$ complex amplitudes"],
        ["Reading", "Does not change the value", "Gives a random outcome and collapses the state"],
        ["Copying", "Free", "Forbidden for unknown states (no-cloning)"],
        ["Errors", "Rare in practice", "Common: decoherence and gate errors"],
    ], widths=[1.6, 2.2, 2.7])

    d.h2("Quantum Superposition")
    d.p("The Hadamard gate $H$ in [[E:hadamard]] creates an equal superposition. For example, $H|0\\rangle=(|0\\rangle+|1\\rangle)/\\sqrt{2}=|+\\rangle$.")
    d.eq("hadamard", r"H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}")
    d.p("Measuring $H|0\\rangle$ over 1024 shots gave 0 with frequency <<sup_p0>> and 1 with frequency <<sup_p1>>. Applying $H$ to three "
        "qubits gave all eight outcomes, as shown in [[F:superposition]], with frequencies between <<sup3_min>> and <<sup3_max>> "
        "around the ideal value 0.125. A superposition is not the same as trying every answer: a measurement returns only one "
        "outcome, so a useful algorithm must use interference to raise the probability of the right one.")
    d.fig("superposition", ["s2_superposition_3q_circuit", "s2_superposition_3q_hist"], "Three-qubit uniform superposition: circuit (left) and measured histogram (right).", 2.9)

    d.h2("Quantum Entanglement")
    d.p("An entangled state of two qubits cannot be written as a product of two separate states. The Bell state in [[E:bell]] is "
        "made by a Hadamard gate followed by a controlled-NOT (CNOT) gate.")
    d.eq("bell", r"|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}")
    d.p("Each qubit alone looks random, yet the two results always agree. In the run shown in [[F:bell]], the outcomes were 00 "
        "(<<bell_00>> counts) and 11 (<<bell_11>> counts), and never 01 or 10. Entanglement is a key resource for many algorithms. "
        "It does not allow faster-than-light communication, and, as Section 5 shows, more of it does not always improve a model.")
    d.fig("bell", ["s2_bell_circuit", "s2_bell_hist"], "Bell state $|\\Phi^+\\rangle$: circuit (left) and histogram showing only 00 and 11 (right).", 2.9)

    d.h2("Quantum Gates and Circuits")
    d.p("Quantum gates are unitary matrices, so they are reversible. A circuit is a sequence of gates applied to qubits, read from "
        "left to right and usually ending in measurement. [[T:gates]] lists the common gates. A small set, such as $H$, $T$ and CNOT, "
        "is universal: it can approximate any unitary operation.")
    d.table("gates", "Common quantum gates.", ["Gate", "Matrix or action", "Effect"], [
        ["X", "$\\begin{pmatrix}0&1\\\\1&0\\end{pmatrix}$", "Bit flip"],
        ["Y", "$\\begin{pmatrix}0&-i\\\\i&0\\end{pmatrix}$", "Bit and phase flip"],
        ["Z", "$\\begin{pmatrix}1&0\\\\0&-1\\end{pmatrix}$", "Phase flip"],
        ["H", "$\\frac{1}{\\sqrt{2}}\\begin{pmatrix}1&1\\\\1&-1\\end{pmatrix}$", "Creates superposition"],
        ["S", "$\\begin{pmatrix}1&0\\\\0&i\\end{pmatrix}$", "Phase of $\\pi/2$"],
        ["T", "$\\begin{pmatrix}1&0\\\\0&e^{i\\pi/4}\\end{pmatrix}$", "Phase of $\\pi/4$"],
        ["RX($\\theta$)", "$\\begin{pmatrix}\\cos\\frac{\\theta}{2}&-i\\sin\\frac{\\theta}{2}\\\\-i\\sin\\frac{\\theta}{2}&\\cos\\frac{\\theta}{2}\\end{pmatrix}$", "Rotation about x"],
        ["RY($\\theta$)", "$\\begin{pmatrix}\\cos\\frac{\\theta}{2}&-\\sin\\frac{\\theta}{2}\\\\\\sin\\frac{\\theta}{2}&\\cos\\frac{\\theta}{2}\\end{pmatrix}$", "Rotation about y"],
        ["RZ($\\theta$)", "$\\begin{pmatrix}e^{-i\\theta/2}&0\\\\0&e^{i\\theta/2}\\end{pmatrix}$", "Rotation about z"],
        ["CNOT", "$|c,t\\rangle\\rightarrow|c,t\\oplus c\\rangle$", "Flips target if control is 1"],
        ["CZ", "$|c,t\\rangle\\rightarrow(-1)^{ct}|c,t\\rangle$", "Phase flip if both are 1"],
        ["SWAP", "$|a,b\\rangle\\rightarrow|b,a\\rangle$", "Exchanges two qubits"],
        ["Toffoli", "$|a,b,t\\rangle\\rightarrow|a,b,t\\oplus ab\\rangle$", "Flips target if both controls are 1"],
    ], align="ccl", widths=[0.9, 2.9, 2.5])

    d.h2("Quantum Measurement")
    d.p("Measurement follows the Born rule: for a state $|\\psi\\rangle$, the probability of basis outcome $x$ is $P(x)=|\\langle x|\\psi\\rangle|^2$, "
        "and the state then collapses to $|x\\rangle$. Probabilities are estimated by repeating the circuit many times, and each repeat is "
        "called a shot. The estimate has a standard error of $\\sqrt{p(1-p)/s}$ for $s$ shots, which is the shot noise. [[F:shots]] shows "
        "the estimated probability approaching 0.5 as shots increase.")
    d.fig("shots", "s2_shots_convergence", "Estimated $P(1)$ of $H|0\\rangle$ against the number of shots.", 3.9)

    d.h2("Quantum Noise")
    d.p("Real qubits lose information to their surroundings. Energy relaxation has a time constant $T_1$, and loss of phase has a time "
        "constant $T_2$. Gates and readout also introduce errors. A simple model is the depolarizing channel in [[E:depol]].")
    d.eq("depol", r"\mathcal{E}(\rho) = (1-p)\rho + p\frac{I}{2^n}")
    d.p("Here $\\rho$ is the density matrix of the state, $p$ is the error probability, $I$ is the identity matrix and $n$ is the number of "
        "qubits. Devices of this kind are the NISQ machines described by Preskill [[c:preskill2018]]. [[T:noiseprops]] gives the properties "
        "of the noise model used in this work, which copies the 5-qubit IBM Manila device through a fake backend.")
    d.table("noiseprops", "Calibration ranges of the fake IBM Manila noise model.", ["Property", "Range across the device"], [
        ["$T_1$", "<<t1_min>> to <<t1_max>> $\\mu$s"], ["$T_2$", "<<t2_min>> to <<t2_max>> $\\mu$s"],
        ["Single-qubit (SX) gate error", "<<g1_min>> to <<g1_max>>"], ["Two-qubit (CX) gate error", "<<g2_min>> to <<g2_max>>"],
        ["Readout error", "<<ro_min>> to <<ro_max>>"]], widths=[2.8, 2.6])
    d.p("As [[F:noise]] shows, the Bell state on this noise model gave wrong outcomes (<<noise_01>> counts of 01 and <<noise_10>> of 10 "
        "in 4096 shots), and the correct-outcome probability $P(00)+P(11)$ fell from <<noise_ideal>> to <<noise_noisy>>. Adding pairs of "
        "CNOT gates, which cancel in theory, lowered it further to <<noise_d_last>> at <<noise_cx_last>> CNOTs. Deep circuits therefore "
        "suffer more.")
    d.fig("noise", ["s2_noise_bell_ideal_vs_noisy", "s2_noise_vs_depth"], "Bell state on ideal and noisy simulators (left) and success against circuit depth under noise (right).", 2.95)

    d.h2("Classical versus Quantum Computing Architectures")
    d.p("Both kinds of machine use layered designs, as [[F:stacks]] shows, but the lowest layers differ greatly. A quantum processor is "
        "not a stand-alone computer, since classical electronics control it and a classical computer processes its results. "
        "[[T:arch]] summarises the differences.")
    d.fig("stacks", "s2_classical_vs_quantum_stack", "Block diagram of the classical and quantum computing stacks.", 5.0)
    d.table("arch", "Classical and quantum architectures compared.", ["Feature", "Classical", "Quantum"], [
        ["Data unit", "Bit", "Qubit"], ["Operations", "Logic gates, mostly irreversible", "Unitary gates, reversible until measured"],
        ["Parallelism", "Many cores and threads", "Superposition with interference; not simple parallel search"],
        ["Errors", "Very low, handled in hardware", "High today; error correction is still emerging"],
        ["Hardware", "Silicon transistors", "Superconducting circuits, trapped ions, neutral atoms and others"],
        ["Temperature", "Room temperature", "About 15 mK for superconducting qubits"],
    ], widths=[1.2, 2.4, 2.9])

    d.h2("Deutsch-Jozsa Algorithm")
    d.p("**Problem.** A function $f:\\{0,1\\}^n\\rightarrow\\{0,1\\}$ is promised to be either constant or balanced (0 on exactly half of the "
        "inputs). A deterministic classical algorithm may need $2^{n-1}+1$ queries in the worst case, while Deutsch-Jozsa needs one "
        "[[c:deutsch1992]]. **Method.** With the output qubit in $|-\\rangle$, the oracle adds a phase through phase kickback, as in [[E:dj1]].")
    d.eq("dj1", r"\frac{1}{\sqrt{2^n}}\sum_{x}|x\rangle \;\xrightarrow{U_f}\; \frac{1}{\sqrt{2^n}}\sum_{x}(-1)^{f(x)}|x\rangle")
    d.p("Applying $H^{\\otimes n}$ again gives the amplitude of $|0^n\\rangle$ in [[E:dj2]].")
    d.eq("dj2", r"a_{0} = \frac{1}{2^n}\sum_{x}(-1)^{f(x)}")
    d.p("If $f$ is constant, $|a_0|=1$, so the result is always $0^n$. If $f$ is balanced, the terms cancel and $a_0=0$, so the result is never "
        "$0^n$. [[F:djcircuit]] shows the 3-qubit circuit with a balanced oracle, and [[F:djresult]] shows the flowchart and results. The "
        "constant oracle gave 000 in all 1024 shots and the balanced oracle gave 111 in all 1024 shots. Although it solves an artificial "
        "problem, this algorithm clearly shows the value of interference.")
    d.fig("djcircuit", "s2_dj_circuit_balanced", "Deutsch-Jozsa circuit for $n=3$ with a balanced oracle.", 5.2)
    d.fig("djresult", ["s2_dj_flowchart", "s2_dj_hist"], "Deutsch-Jozsa flowchart (left) and measured results for both oracles (right).", 2.9)

    d.h2("Grover's Search Algorithm")
    d.p("**Problem.** Find one marked item among $N=2^n$ unsorted items. A classical search needs about $N/2$ checks on average, whereas "
        "Grover's algorithm needs about $\\sqrt{N}$ [[c:grover1996]]. **Method.** Each iteration applies an oracle, which flips the sign of "
        "the marked state, followed by a diffusion operator $2|s\\rangle\\langle s|-I$ that reflects about the uniform state $|s\\rangle$. "
        "Geometrically, as in [[F:grovergeo]], the two reflections rotate the state towards the marked state by $2\\theta$, with "
        "$\\sin\\theta=1/\\sqrt{N}$. After $k$ iterations the success probability and optimum are given in [[E:grover]].")
    d.eq("grover", r"P_k = \sin^2\left((2k+1)\theta\right), \qquad k_{opt} \approx \frac{\pi}{4}\sqrt{N}")
    d.p("Going beyond the optimum lowers the success probability, as the curves in [[F:grovergeo]] show. [[F:grovercircuit]] shows the "
        "3-qubit circuit. In the runs, the 2-qubit search found the marked state in every shot (success <<gr2_meas>>), and the 3-qubit search "
        "with two iterations reached <<gr3_meas>> over 1024 shots, against a theoretical <<gr3_theory>>; the exact statevector value matches "
        "theory, so the difference is shot noise. [[F:groverflow]] gives the flowchart and the query scaling.")
    d.fig("grovergeo", ["s2_grover_geometry", "s2_grover_success_vs_iterations"], "Geometric picture of Grover's algorithm (left) and success probability against iterations (right).", 2.95)
    d.fig("grovercircuit", "s2_grover_circuit_3q", "Grover circuit for three qubits, marked state $|101\\rangle$, two iterations.", 6)
    d.fig("groverflow", ["s2_grover_flowchart", "s2_grover_query_scaling"], "Grover flowchart (left) and oracle queries against search space size (right).", 2.9)

    d.h2("Shor's Algorithm (Overview)")
    d.p("Shor's algorithm factors an integer $N$ in polynomial time [[c:shor1997]]. Factoring reduces to finding the period $r$ of "
        "$f(x)=a^x\\bmod N$, for a random $a$ that shares no factor with $N$. When $r$ is even and $a^{r/2}\\neq -1\\pmod N$, the values "
        "$\\gcd(a^{r/2}\\pm 1,N)$ give factors. The quantum step is phase estimation, which uses the inverse of the Quantum Fourier Transform "
        "(QFT) in [[E:qft]].")
    d.eq("qft", r"\text{QFT}|x\rangle = \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1} e^{2\pi i xk/N}|k\rangle")
    d.p("**Worked example, $N=15$, $a=7$.** The powers of 7 modulo 15 repeat as 1, 7, 4, 13, so $r=4$. Then $7^{2}=49$, $\\gcd(48,15)=3$ and "
        "$\\gcd(50,15)=5$, which gives $15=3\\times5$. A textbook circuit with eight counting qubits and four work qubits was simulated. "
        "As [[F:shor]] and [[T:shor]] show, the four measured phases were 0, 1/4, 1/2 and 3/4, as expected for $r=4$. Only 1/4 and 3/4 reveal $r$ "
        "directly. The result 1/2 suggests $r=2$, but $7^2\\bmod 15\\neq1$, so a classical check rejects it and the run is repeated. "
        "**Impact.** A large fault-tolerant machine could break RSA encryption, whose security rests on the difficulty of factoring. "
        "This is far beyond current devices, which is why this simulation of $N=15$ is only an illustration.")
    d.fig("shor", ["s2_qft_4q_circuit", "s2_shor_phase_hist"], "Four-qubit QFT circuit (left) and measured phases for $N=15$, $a=7$ (right).", [3.8, 2.5])
    d.table("shor", "Measured phases and the period guess from each outcome (<<shor_total>> shots).",
            ["Register", "Phase", "Fraction", "Guess $r$", "$7^r$ mod 15 = 1?", "Factors", "Counts"], d.shor_rows,
            align="lrcccll", widths=[1.0, 0.6, 0.7, 0.6, 1.0, 1.1, 0.6])
    d.p("[[F:shorflow]] shows how the classical and quantum steps fit together.")
    d.fig("shorflow", "s2_shor_flowchart", "Flowchart of Shor's algorithm.", 2.5)

    d.h2("Computational Complexity and Quantum Advantage")
    d.p("P contains problems that a classical computer solves efficiently, NP contains problems whose solutions can be checked efficiently, "
        "and BQP contains problems that a quantum computer solves efficiently with bounded error. [[F:complexity]] shows the widely believed "
        "relationships. Factoring is in BQP but is not known to be in P, while NP-complete problems are not expected to be in BQP. So quantum "
        "computers are not expected to solve every hard problem quickly. [[T:complexity]] compares the algorithms discussed. A proven gain on a "
        "particular problem also does not imply a gain for practical ML, which is the question tested in Section 5.")
    d.fig("complexity", "s2_complexity_classes", "Believed relationship between the classes P, NP, BQP and PSPACE.", 3.3)
    d.table("complexity", "Classical and quantum complexity of selected problems.", ["Problem", "Best classical", "Quantum", "Gain"], [
        ["Deutsch-Jozsa", "$2^{n-1}+1$ queries (deterministic)", "1 query", "Exponential, but only against deterministic algorithms"],
        ["Unstructured search", "$O(N)$ queries", "$O(\\sqrt{N})$ queries", "Quadratic"],
        ["Factoring an $n$-bit number", "Sub-exponential (number field sieve)", "Polynomial in $n$", "Super-polynomial"],
    ], widths=[1.6, 1.9, 1.2, 1.8])
