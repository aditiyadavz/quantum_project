"""Phase 0 check: confirm every package imports, list versions, and load the dataset."""
import importlib
import warnings

warnings.simplefilter("default")

PACKAGES = [
    "qiskit", "qiskit_machine_learning", "qiskit_aer", "qiskit_ibm_runtime",
    "sklearn", "pandas", "numpy", "matplotlib", "seaborn", "pylatexenc", "docx",
]
for name in PACKAGES:
    mod = importlib.import_module(name)
    print(f"{name:26s} {getattr(mod, '__version__', 'ok')}")

# Qiskit 2.x API names the project will use
from qiskit.circuit.library import zz_feature_map, real_amplitudes, z_feature_map, pauli_feature_map  # noqa
from qiskit.primitives import StatevectorSampler  # noqa
from qiskit_machine_learning.kernels import FidelityQuantumKernel  # noqa
from qiskit_machine_learning.algorithms import QSVC, VQC  # noqa
from qiskit_machine_learning.optimizers import COBYLA, SPSA  # noqa
from qiskit_ibm_runtime.fake_provider import FakeManilaV2  # noqa
print("Qiskit ML API imports: ok")

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
y = 1 - data.target  # re-map: malignant = 1
print(f"Dataset: {data.data.shape[0]} samples, {data.data.shape[1]} features, "
      f"malignant={int(y.sum())}, benign={int((y == 0).sum())}")
