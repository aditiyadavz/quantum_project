"""Hybrid quantum-classical models built with Qiskit Machine Learning.

QSVC : a quantum circuit computes the kernel k(x, x') = |<phi(x)|phi(x')>|^2,
       a classical SVM finds the decision boundary.
VQC  : a parameterised circuit (feature map + ansatz) predicts the class,
       a classical optimiser (COBYLA / SPSA) trains the ansatz angles.

Kernel back-ends
  "statevector": FidelityStatevectorKernel - computes one statevector per
                 sample and takes exact inner products. Mathematically the same
                 kernel as the compute-uncompute circuit, much faster on a laptop.
  "circuits"   : FidelityQuantumKernel + ComputeUncompute - builds and simulates
                 one U(x')^dagger U(x) circuit per pair of samples, as a real
                 quantum computer would.
  a sampler    : FidelityQuantumKernel on any SamplerV2 (e.g. a noisy Aer sampler).
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import (pauli_feature_map, real_amplitudes, z_feature_map,
                                    zz_feature_map)
from qiskit_machine_learning.algorithms import QSVC, VQC
from qiskit_machine_learning.kernels import FidelityQuantumKernel, FidelityStatevectorKernel
from qiskit_machine_learning.optimizers import COBYLA, SPSA
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.utils import algorithm_globals
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from .preprocess import SEED

BASIS = ["cx", "rz", "sx", "x"]  # IBM-style basis used to report circuit depth


# ----------------------------------------------------------------------------- circuits
def make_feature_map(n_qubits, reps=2, entanglement="linear", kind="zz"):
    """Data-encoding circuit U_Phi(x)."""
    if kind == "zz":
        return zz_feature_map(n_qubits, reps=reps, entanglement=entanglement)
    if kind == "z":         # no entangling gates
        return z_feature_map(n_qubits, reps=reps)
    if kind == "pauli":     # Z, Y single-qubit terms + ZZ two-qubit terms
        return pauli_feature_map(n_qubits, reps=reps, entanglement=entanglement,
                                 paulis=["Z", "Y", "ZZ"])
    raise ValueError(kind)


def make_ansatz(n_qubits, reps=3):
    """Trainable circuit W(theta): RY rotations + CNOT ladders."""
    return real_amplitudes(n_qubits, reps=reps)


def circuit_stats(circuit):
    """Qubits, depth, CNOT count and parameter count after decomposing to BASIS."""
    flat = transpile(circuit, basis_gates=BASIS, optimization_level=0)
    ops = flat.count_ops()
    return {"qubits": circuit.num_qubits, "depth": flat.depth(),
            "cx_gates": int(ops.get("cx", 0)), "total_gates": int(sum(ops.values())),
            "parameters": circuit.num_parameters}


def kernel_circuit(feature_map, x1, x2):
    """Compute-uncompute circuit whose P(0...0) equals k(x1, x2)."""
    qc = feature_map.assign_parameters(x1)
    qc = qc.compose(feature_map.assign_parameters(x2).inverse())
    qc.measure_all()
    return qc


# ----------------------------------------------------------------------------- kernels
def make_kernel(feature_map, backend="statevector"):
    if backend == "statevector":
        return FidelityStatevectorKernel(feature_map=feature_map)
    if backend == "circuits":
        return FidelityQuantumKernel(feature_map=feature_map)  # exact compute-uncompute
    # otherwise: a SamplerV2 (e.g. noisy simulator); circuits are transpiled for it
    return FidelityQuantumKernel(feature_map=feature_map,
                                 fidelity=ComputeUncompute(sampler=backend["sampler"],
                                                           pass_manager=backend["pm"]))


class TunedQSVC(BaseEstimator, ClassifierMixin):
    """QSVC whose regularisation C is chosen by 5-fold CV on the quantum kernel.

    The training kernel matrix is computed once; C in `Cs` is tuned with
    SVC(kernel='precomputed') (identical maths to QSVC), then the final
    Qiskit QSVC is trained with the best C.
    """

    def __init__(self, n_qubits=4, reps=2, entanglement="linear", kind="zz",
                 backend="statevector", Cs=(0.1, 1, 10), C=None, seed=SEED):
        self.n_qubits, self.reps, self.entanglement, self.kind = n_qubits, reps, entanglement, kind
        self.backend, self.Cs, self.C, self.seed = backend, Cs, C, seed

    def fit(self, X, y):
        self.feature_map_ = make_feature_map(self.n_qubits, self.reps, self.entanglement, self.kind)
        self.kernel_ = make_kernel(self.feature_map_, self.backend)
        self.train_kernel_ = self.kernel_.evaluate(X)       # M x M quantum kernel matrix
        if self.C is None:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
            search = GridSearchCV(SVC(kernel="precomputed"), {"C": list(self.Cs)}, cv=cv)
            search.fit(self.train_kernel_, y)
            self.best_C_ = search.best_params_["C"]
            self.cv_scores_ = dict(zip(self.Cs, search.cv_results_["mean_test_score"]))
        else:
            self.best_C_ = self.C
        # final model: SVM on the precomputed quantum kernel (what QSVC does internally)
        self.svc_ = SVC(kernel="precomputed", C=self.best_C_).fit(self.train_kernel_, y)
        self.X_train_ = np.asarray(X)
        self.classes_ = self.svc_.classes_
        return self

    def _test_kernel(self, X):
        return self.kernel_.evaluate(np.asarray(X), self.X_train_)

    def predict(self, X):
        return self.svc_.predict(self._test_kernel(X))

    def decision_function(self, X):
        return self.svc_.decision_function(self._test_kernel(X))

    def as_qiskit_qsvc(self):
        """The same model as a Qiskit QSVC object (used to check equivalence)."""
        return QSVC(quantum_kernel=self.kernel_, C=self.best_C_)

    def stats(self, n_train):
        s = circuit_stats(kernel_circuit(self.feature_map_, np.zeros(self.n_qubits),
                                         np.zeros(self.n_qubits)).remove_final_measurements(
                                             inplace=False))
        s.update({"parameters": 0, "feature_map_depth": circuit_stats(self.feature_map_)["depth"],
                  "circuit_evaluations_train": n_train * (n_train - 1) // 2})
        return s


# ----------------------------------------------------------------------------- VQC
class VQCModel(BaseEstimator, ClassifierMixin):
    """Qiskit VQC with a loss-history callback and evaluation counting."""

    def __init__(self, n_qubits=4, fm_reps=2, entanglement="linear", ansatz_reps=3,
                 optimizer="cobyla", maxiter=100, sampler=None, seed=SEED):
        self.n_qubits, self.fm_reps, self.entanglement = n_qubits, fm_reps, entanglement
        self.ansatz_reps, self.optimizer, self.maxiter = ansatz_reps, optimizer, maxiter
        self.sampler, self.seed = sampler, seed

    def fit(self, X, y):
        algorithm_globals.random_seed = self.seed      # SPSA uses this generator
        self.feature_map_ = make_feature_map(self.n_qubits, self.fm_reps, self.entanglement)
        self.ansatz_ = make_ansatz(self.n_qubits, self.ansatz_reps)
        self.loss_history_ = []

        def callback(*args):
            # COBYLA (SciPy) calls callback(weights, loss) once per evaluation;
            # SPSA calls callback(nfev, weights, loss, step, accepted) once per iteration.
            self.loss_history_.append(float(args[1] if len(args) == 2 else args[2]))

        opt = (COBYLA(maxiter=self.maxiter) if self.optimizer == "cobyla"
               else SPSA(maxiter=self.maxiter))
        rng = np.random.default_rng(self.seed)
        init = rng.uniform(-np.pi, np.pi, self.ansatz_.num_parameters)
        self.vqc_ = VQC(feature_map=self.feature_map_, ansatz=self.ansatz_, optimizer=opt,
                        callback=callback, initial_point=init, sampler=self.sampler)
        self.vqc_.fit(np.asarray(X), np.asarray(y))
        self.classes_ = np.unique(y)
        self.n_train_ = len(y)
        return self

    def predict(self, X):
        return np.asarray(self.vqc_.predict(np.asarray(X))).ravel().astype(int)

    def predict_proba(self, X):
        return self.vqc_.predict_proba(np.asarray(X))

    def full_circuit(self):
        qc = QuantumCircuit(self.n_qubits)
        qc.compose(self.feature_map_, inplace=True)
        qc.compose(self.ansatz_, inplace=True)
        return qc

    def stats(self, n_train):
        s = circuit_stats(self.full_circuit())
        s.update({"parameters": self.ansatz_.num_parameters,
                  "feature_map_depth": circuit_stats(self.feature_map_)["depth"],
                  "optimizer_iterations": len(self.loss_history_),
                  "objective_evaluations": int(self.vqc_.fit_result.nfev),
                  "circuit_evaluations_train": int(self.vqc_.fit_result.nfev) * n_train})
        return s
