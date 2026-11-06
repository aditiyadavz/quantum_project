"""Sections 5, 6 and 7."""
from values import f4, seconds

ORDER = [("QSVC", "QSVC"), ("VQC", "VQC"), ("SVM", "SVM (RBF)"), ("LR", "Logistic Regression"), ("RF", "Random Forest"), ("KNN", "KNN"),
         ("MLP", "MLP"), ("SVM30", "SVM (RBF, 30 features)")]


def section5(d, tables):
    h = tables["holdout"].set_index("code")
    cv = tables["cv"].set_index("code")
    d.h1("Results, Performance Evaluation and Discussion")

    d.h2("Implementation Outcomes")
    d.p("All planned code ran on the laptop in [[T:env]]. Eight models were trained on the same data: QSVC, VQC, five classical models on the same four PCA features, and an RBF SVM on all 30 features as an upper reference. "
        "Experiments E1 to E9 were completed. Optional experiment E10, on real hardware, was not run. Reduced training sets were used where run time required it, and these are stated with each result. Malignant is the positive class throughout. "
        "**Timing caveat:** the laptop's speed changed with its power state, and it slept during some runs. Runs affected by sleep were repeated, classical times are medians of five fits, and times should still be read as rough sizes rather than precise values. "
        "The cross-validation run was interrupted and resumed from its saved folds.")

    d.h2("Main Comparison (E1)")
    d.p("[[T:holdout]] gives the results on the single 80/20 split, and [[T:cv]] gives the 5-fold cross-validation results. Cross-validation is the more reliable of the two because every sample is tested once. On the hold-out split, a single test sample changes accuracy by nearly 0.009, so small differences between the classical models should not be over-read.")
    d.table("holdout", "Performance of all models on the hold-out split (455 training and 114 test samples, four qubits).",
            ["Model", "Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC", "Train (s)", "Infer (s)"],
            [[name] + [f4(h.loc[c, m]) for m in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc")] +
             [seconds(h.loc[c, "train_time_s"]), seconds(h.loc[c, "inference_time_s"])] for c, name in ORDER],
            align="lrrrrrrrr", widths=[1.48, 0.7, 0.7, 0.58, 0.8, 0.52, 0.62, 0.5, 0.5], size=9)
    d.table("cv", "Five-fold cross-validation results: mean over folds, with standard deviation for accuracy and recall.",
            ["Model", "Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC", "Train (s)"],
            [[name, f"{cv.loc[c, 'accuracy_mean']:.3f} ± {cv.loc[c, 'accuracy_std']:.3f}", f"{cv.loc[c, 'precision_mean']:.3f}",
              f"{cv.loc[c, 'recall_mean']:.3f} ± {cv.loc[c, 'recall_std']:.3f}", f"{cv.loc[c, 'specificity_mean']:.3f}", f"{cv.loc[c, 'f1_mean']:.3f}",
              f"{cv.loc[c, 'roc_auc_mean']:.3f}", seconds(cv.loc[c, "train_time_s_mean"])] for c, name in ORDER],
            align="lrrrrrrr", widths=[1.45, 0.98, 0.7, 0.98, 0.8, 0.46, 0.62, 0.47], size=9)
    d.p("The classical models clearly outperformed both quantum models. In cross-validation, the RBF SVM reached <<cv_SVM_acc>> on the same features, against <<cv_QSVC_acc>> for QSVC and <<cv_VQC_acc>> for VQC. VQC was only slightly above the <<majority>> achieved by always predicting benign. "
        "The all-feature SVM reached <<cv_SVM30_acc>>, so reducing the data to four components cost about <<pca_loss>> in accuracy for the SVM. That loss is much smaller than the gap between the classical and quantum models, so PCA alone does not explain their weaker results. [[F:bars]] compares the metrics, [[F:confusion]] the confusion matrices, and [[F:rocloss]] the ROC curves and VQC training.")
    d.fig("bars", "s5_metric_bars", "Accuracy, precision, recall and F1-score of all models on the hold-out split.", 5.4)
    d.fig("confusion", "s5_confusion_matrices", "Confusion matrices of all models on the hold-out split.", 5.8)
    d.fig("rocloss", ["s5_roc_curves", "s5_vqc_loss_curve"], "ROC curves (left) and VQC training loss with COBYLA (right), hold-out split.", 2.95)
    d.p("The VQC loss fell only from <<vqc_loss_first>> to <<vqc_loss_last>>. That is still above $\\ln 2\\approx0.693$, the loss of a model that always outputs equal probabilities, so training had clearly not converged within the 100 iterations set for this project. "
        "[[F:kernels]] compares the two kernels. The RBF kernel shows clear benign and malignant blocks, while most off-diagonal values of the quantum kernel are close to zero, with a mean of about <<e2_k_4>> across seeds. Most pairs of samples therefore look almost unrelated, giving the SVM little to use.")
    d.fig("kernels", "s5_kernel_matrices", "Training kernel matrices: quantum (left) and RBF (right), with samples sorted by class.", 5.6)

    d.h2("Experiments E2 to E9")
    d.p("Unless stated otherwise, these runs use three seeds (a different split and starting point for each), and values are means over seeds.")
    d.h3("E2: Number of Qubits")
    d.p("[[F:e2]] varies the qubit count and PCA components together. QSVC fell from <<e2_QSVC_2_acc>> at 2 qubits to <<e2_QSVC_8_acc>> at 8, even though more qubits carried more variance (<<pca_2>> to <<pca_8>>). The RBF SVM rose from <<e2_SVM_2_acc>> to <<e2_SVM_8_acc>>. "
        "The mean kernel value fell from <<e2_k_2>> to <<e2_k_8>>, which matches the trend expected from concentration [[c:r8]]. VQC, run with one seed and 200 training samples, scored <<e2_VQC_2_acc>>, <<e2_VQC_4_acc>>, <<e2_VQC_6_acc>> and <<e2_VQC_8_acc>>, and its time rose from <<e2_VQC_2_time>> s to <<e2_VQC_8_time>> s.")
    d.fig("e2", "s5_e2_qubits", "Accuracy and training time against the number of qubits (E2).", 5.6)
    d.h3("E3: Feature Map Depth")
    d.p("In [[F:e3]], deeper feature maps were worse. QSVC accuracy for reps = 1, 2, 3 and 4 was <<e3_1_acc>>, <<e3_2_acc>>, <<e3_3_acc>> and <<e3_4_acc>>. The mean kernel value stayed near 0.09 throughout (<<e3_1_k>>, <<e3_2_k>>, <<e3_3_k>>, <<e3_4_k>>), so a low mean alone does not explain this decline. Extra depth also makes the map vary more quickly with its input, which may reduce useful structure.")
    d.fig("e3", "s5_e3_fm_depth", "Accuracy and training time against feature map repetitions (E3).", 5.4)
    d.h3("E4: Entanglement")
    d.p("[[F:e4]] gives the clearest result of the study. The Z feature map, which has no entanglement, reached <<e4_z_acc>> with recall <<e4_z_rec>>, and its mean kernel value was <<e4_z_k>>. The ZZ maps with linear, circular and full entanglement reached <<e4_zzl_acc>>, <<e4_zzc_acc>> and <<e4_zzf_acc>>, and the Pauli map reached <<e4_pauli_acc>>, all with mean kernel values near 0.08. "
        "On this task more entanglement did not help, which agrees with the large benchmark in [[c:r6]]. The Z map is simple to simulate classically, so its better result is not evidence of a quantum benefit.")
    d.fig("e4", "s5_e4_entanglement", "QSVC accuracy and mean kernel value for each feature map (E4).", 5.4)
    d.h3("E5: Ansatz Depth")
    d.p("With 200 training samples, VQC accuracy for reps = 1, 2, 3 and 4 was <<e5_1_acc>>, <<e5_2_acc>>, <<e5_3_acc>> and <<e5_4_acc>>, with standard deviations of <<e5_1_sd>> to <<e5_3_sd>>. As [[F:e5]] shows, the differences are within the spread between seeds, so there is no clear trend. Final loss improved only slightly with depth (<<e5_1_loss>> to <<e5_4_loss>>), and [[F:e5loss]] shows that all runs stayed above 0.693.")
    d.fig("e5", "s5_e5_ansatz_depth", "VQC accuracy and time against ansatz depth (E5).", 5.0)
    d.fig("e5loss", "s5_e5_loss_curves", "VQC loss curves for each ansatz depth (E5).", 3.4)
    d.h3("E6: Optimiser")
    d.p("COBYLA reached <<e6_cobyla_acc>> ± <<e6_cobyla_sd>> using <<e6_cobyla_evals>> evaluations and about <<e6_cobyla_time>> s. SPSA reached <<e6_spsa_acc>> ± <<e6_spsa_sd>> using <<e6_spsa_evals>> evaluations and about <<e6_spsa_time>> s. As [[F:e6]] shows, SPSA's loss curve was smoother but ended higher (<<e6_spsa_loss>> against <<e6_cobyla_loss>> on average). With these settings, COBYLA was both cheaper and better. These two sets of runs were completed before the laptop's power problem was fixed, so their times are less reliable than those in E5.")
    d.fig("e6", "s5_e6_optimizer_loss", "VQC loss for COBYLA and SPSA (E6).", 3.8)
    d.h3("E7: Training Size")
    d.p("As [[F:e7]] shows, QSVC needed much more data than the RBF SVM. With 50, 100, 200 and 455 samples, QSVC reached <<e7_QSVC_50_acc>>, <<e7_QSVC_100_acc>>, <<e7_QSVC_200_acc>> and <<e7_QSVC_455_acc>>. At 50 samples its recall was <<e7_QSVC_50_rec>>, meaning it predicted every test sample as benign. The RBF SVM already reached <<e7_SVM_50_acc>> at 50 samples. "
        "[[F:e7cost]] shows the cost of the circuit kernel: <<e7c_50>> s, <<e7c_100>> s and <<e7c_200>> s for 50, 100 and 200 samples, following the expected $M^2$ growth.")
    d.fig("e7", "s5_e7_train_size", "Accuracy and training time against training size (E7).", 5.4)
    d.fig("e7cost", "s5_e7_kernel_cost", "Time to build the training kernel against training size (E7).", 3.6)
    d.h3("E8: Noise")
    d.p("[[T:e8]] and [[F:e8]] compare exact simulation, an ideal sampler with 1024 shots, and the fake Manila noise model, using one seed. QSVC used 100 training samples because every kernel entry is a separate noisy circuit. The VQC was trained on the exact simulator with 200 samples, then evaluated on each setting.")
    d.table("e8", "Effect of shots and device noise (one seed).", ["Model", "Setting", "Accuracy", "Recall", "Missed", "ROC-AUC"], [
        ["QSVC", "Exact statevector", "<<e8_QSVC_exact_acc>>", "<<e8_QSVC_exact_rec>>", "<<e8_QSVC_exact_fn>>", "<<e8_QSVC_exact_auc>>"],
        ["QSVC", "Ideal sampler, 1024 shots", "<<e8_QSVC_shots_acc>>", "<<e8_QSVC_shots_rec>>", "<<e8_QSVC_shots_fn>>", "<<e8_QSVC_shots_auc>>"],
        ["QSVC", "Noisy fake Manila, 1024 shots", "<<e8_QSVC_noisy_acc>>", "<<e8_QSVC_noisy_rec>>", "<<e8_QSVC_noisy_fn>>", "<<e8_QSVC_noisy_auc>>"],
        ["VQC", "Exact statevector", "<<e8_VQC_exact_acc>>", "<<e8_VQC_exact_rec>>", "<<e8_VQC_exact_fn>>", "<<e8_VQC_exact_auc>>"],
        ["VQC", "Ideal sampler, 1024 shots", "<<e8_VQC_shots_acc>>", "<<e8_VQC_shots_rec>>", "<<e8_VQC_shots_fn>>", "<<e8_VQC_shots_auc>>"],
        ["VQC", "Noisy fake Manila, 1024 shots", "<<e8_VQC_noisy_acc>>", "<<e8_VQC_noisy_rec>>", "<<e8_VQC_noisy_fn>>", "<<e8_VQC_noisy_auc>>"],
    ], align="llrrrr", widths=[0.7, 2.5, 0.75, 0.7, 0.7, 0.8])
    d.p("At this small scale, noise changed the summary scores very little. This does not mean that noise has no effect. QSVC's exact and noisy rows are identical in their totals, yet a direct check at the same C showed that <<e8_changed>> of <<e8_total>> individual predictions differed and that <<e8_flip>> of <<e8_pairs>> ranked pairs changed order. These changes happened to cancel exactly. "
        "The noise also changed the kernel values, with a mean absolute error of <<e8_noisy_err>> against the exact kernel, compared with <<e8_shots_err>> for shots alone. The higher score with shots should not be read as shot noise helping, because cross-validation picked C = <<e8_shots_C>> there and C = <<e8_exact_C>> in the other two cases, and this is only one split.")
    d.fig("e8", "s5_e8_noise", "Accuracy under exact simulation, shots and device noise (E8).", 4.0)
    d.h3("E9: Iris Multiclass")
    d.p("To check multiclass support, QSVC and an RBF SVM were run on the Iris data (150 samples, 3 classes) with four qubits and no PCA. QSVC reached <<e9_q>> ± <<e9_q_sd>> and the RBF SVM reached <<e9_s>> ± <<e9_s_sd>>. The quantum model handled three classes without changes, and the classical model again scored higher.")

    d.h2("Computational Cost Analysis")
    d.p("[[T:cost]] compares the costs. QSVC needs about $M^2/2$ circuits for $M$ training samples, plus $M\\times T$ for $T$ test samples. VQC needs about one circuit per sample for every optimiser evaluation. Simulating $n$ qubits requires storing $2^n$ amplitudes. On real hardware each circuit must also be repeated for many shots, and that would multiply these counts. [[F:timecost]] shows measured training times on a log scale.")
    d.table("cost", "Circuit counts and measured time on the main task (455 training and 114 test samples).", ["Model", "Training circuits", "Test circuits", "Mean train time in CV"], [
        ["QSVC", "$M(M-1)/2$ = <<qsvc_circuits>>", "$M\\times T$ = <<qsvc_test_circuits>>", "<<cv_QSVC_time>> s (exact statevector kernel)"],
        ["VQC", "Iterations $\\times M$ = <<vqc_circuits>>", "$T$ = <<n_test>>", "<<cv_VQC_time>> s"],
        ["RBF SVM", "None", "None", "<<cv_SVM_time>> s (includes tuning)"],
        ["Logistic regression", "None", "None", "<<cv_LR_time>> s"],
    ], widths=[1.3, 2.0, 1.4, 1.8])
    d.fig("timecost", "s5_training_time", "Training time of all models on the hold-out split (log scale).", 4.8)
    d.p("The short QSVC time depends on the exact statevector shortcut, which a real quantum computer cannot use. With real circuits, simulated cost grows quickly: the 200-sample kernel alone took <<e7c_200>> s. So no computational advantage was observed; every quantum model cost more than the classical models it was compared with. This is a result of classical simulation and small problem size, and it does not rule out advantages for other problems on future hardware [[c:r3,r4]].")

    d.h2("Clinical View: Missed Cancers")
    d.p("[[T:clinical]] totals errors across the five folds, so each of the <<n_mal>> malignant samples was tested exactly once. All models used their default decision threshold. QSVC missed about <<cv_fn_ratio>> times as many cancers as the RBF SVM, and VQC about <<cv_vqc_ratio>> times as many. Missing <<cv_VQC_fn>> of <<n_mal>> cancers would be unacceptable in a diagnostic setting. "
        "It should also be noted that even the classical models missed some cancers, so they too would need clinical validation. Changing the threshold could trade false alarms for fewer misses, as a VQC study with high recall on the same dataset suggests [[c:mpofu2025]], but this was not tested here.")
    d.table("clinical", "Missed cancers and false alarms over 5-fold cross-validation (212 malignant and 357 benign samples).", ["Model", "Missed cancers (FN)", "False alarms (FP)", "Mean recall"],
            [[name, f"<<cv_{c}_fn>>", f"<<cv_{c}_fp>>", f"<<cv_{c}_rec>>"] for c, name in ORDER], align="lrrr", widths=[2.3, 1.3, 1.2, 1.0])

    d.h2("Discussion")
    d.p("**Effectiveness.** Both quantum pipelines worked correctly from end to end, but neither matched the classical models. QSVC's best setting (Z feature map, <<e4_z_acc>>) was still below the RBF SVM (<<e2_SVM_4_acc>>, with the same three seeds at four qubits). These findings align with [[c:r6]], which reported that classical models generally win on small tasks, and with [[c:r32]], which reported that a ZZ quantum kernel under-performed on WDBC.")
    d.p("**Why the quantum models fell short.** The following are likely causes. Each is supported by the results above or by the literature, but this study does not separate them with controlled tests.")
    d.bullets([
        "**Unsuitable kernel scale.** Angles in $[0,\\pi]$ give large ZZ phases, and off-diagonal kernel values were small. Tuning kernel bandwidth has been shown to help [[c:r11]], but it was not tested here.",
        "**Kernel concentration.** Mean kernel values fell as qubits were added, as predicted in [[c:r8]], although at this size the values are not yet exponentially small.",
        "**Too little VQC training.** The loss stayed above 0.693 after 100 iterations with 16 parameters. A well-tuned eight-qubit VQC reached an external AUC of 0.983 in [[c:r32]], so the VQC result here should not be read as typical of the model family.",
        "**Single-qubit readout.** The default class mapping uses only qubit 0, which may limit the VQC. Its effect was not tested.",
        "**Information lost in PCA.** Four components keep <<pca_4>> of the variance. This has a real but smaller effect, as the SVM comparison shows.",
        "**Uneven tuning.** The RBF SVM tuned both C and gamma over 16 combinations, QSVC tuned only C over three values, and the VQC had no tuning. Classical models therefore had an advantage in this design.",
    ])
    d.p("**Benefits.** The work gives a reproducible pipeline, a fair baseline, and direct evidence on how design choices affect results. QSVC training is convex, so it did not show the training failures seen in the VQC. "
        "**Implementation challenges.** Qiskit's APIs change quickly, simulation time limited the experiment grid, noisy kernels were slow, and long runs were affected by the laptop's power state. "
        "**Limitations.** All runs used simulators; E2's VQC and all of E8 used one seed; VQC sweeps and noisy QSVC used reduced training sets; seeds and folds are few, so small differences are not significant; WDBC is a curated benchmark, so nothing here shows clinical readiness; and the literature review rests on abstracts. "
        "**Claims about advantage.** No claim of quantum advantage is made. The observed results support the opposite conclusion for this task on these simulated small circuits.")


def section6(d):
    d.h1("Conclusion and Future Scope")
    d.h2("Conclusion")
    d.p("This work developed an understanding of quantum computing from qubits, gates, measurement and noise through to the Deutsch-Jozsa, Grover and Shor algorithms, each demonstrated in Qiskit. It showed that quantum algorithms offer proven gains on particular problems, and that these gains do not automatically carry over to machine learning. "
        "The review of 45 papers showed a field with strong theory and real hardware progress, but also serious issues: barren plateaus, kernel concentration, noise and a lack of proven practical advantage on classical data.")
    d.p("The mini project built and compared two hybrid models on breast cancer data. In 5-fold cross-validation, QSVC reached <<cv_QSVC_acc>> accuracy with <<cv_QSVC_rec>> recall, and VQC reached <<cv_VQC_acc>> with <<cv_VQC_rec>>. On identical features, the RBF SVM reached <<cv_SVM_acc>> with <<cv_SVM_rec>>, and the all-feature SVM reached <<cv_SVM30_acc>>. The quantum models missed <<cv_QSVC_fn>> and <<cv_VQC_fn>> of <<n_mal>> cancers, compared with <<cv_SVM_fn>> for the RBF SVM. "
        "Removing entanglement improved QSVC to <<e4_z_acc>>, deeper feature maps reduced its accuracy, and kernel values shrank as qubits were added. Small-scale noise changed summary scores little in one seed, although individual predictions changed.")
    d.p("The contributions are a working Qiskit pipeline, a fair comparison focused on missed cancers, a joint analysis of circuit settings and cost, and an honest statement of where these models stand. The main limitations are that all results are simulated, the VQC did not converge in its fixed training budget, and tuning was uneven. On this dataset, classical models remain the better choice, and that is a valid finding.")
    d.h2("Future Scope")
    d.bullets([
        "**Advanced Quantum Neural Networks.** Test deeper or problem-specific QNNs, and use a readout based on more than one qubit.",
        "**Improved Variational Quantum Circuits.** Train for longer, tune hyperparameters, try better ansatz designs and data re-uploading [[c:r17]], and use layerwise training [[c:r18]].",
        "**Better kernels.** Tune kernel bandwidth [[c:r11]], use trainable kernels with alignment [[c:r10]], and test projected kernels [[c:r3]].",
        "**Quantum optimisation.** Use QAOA [[c:r38]] for feature selection before encoding.",
        "**Noise and hardware.** Apply error mitigation [[c:kim2023]], and repeat the tests on real IBM devices and on larger, error-corrected machines as they appear [[c:ibm2025,google2025]].",
        "**Scalable hybrid models.** Build hybrid quantum CNNs for mammograms and histopathology images [[c:r23,r29]], and draw on the hybrid-model background in [[c:schuld2021book]].",
        "**Real-world use.** Test on real clinical cohorts with external validation, calibration and threshold tuning, and consider clinical decision support only with attention to privacy, regulation and adversarial robustness [[c:r27,r32]].",
    ])


def section7(d):
    d.h1("References and Bibliography")
    d.p("References are numbered in order of first citation and follow the IEEE style. They include the 45 reviewed papers, books, organisational reports, official framework documentation and the dataset citation. Records of journal papers were checked against Crossref or arXiv, and web pages were opened on the access date shown.")
    d.references_here()
