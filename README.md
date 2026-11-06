# Hybrid Quantum-Classical ML for Breast Cancer Diagnosis

Course NV62010 – Quantum Computing with Artificial Intelligence (final assignment).

Compares a Quantum Kernel SVM (QSVC) and a Variational Quantum Classifier (VQC), built with Qiskit,
against classical ML models on the Breast Cancer Wisconsin (Diagnostic) dataset. All results come from
simulators. Random seeds are fixed (42, or 42, 43 and 44 for repeated experiments).

**Final report:** `report/NV62010_QML_Breast_Cancer_Report.docx` (PDF copy alongside it).

## Setup (Windows, Python 3.12)

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python check_env.py
```

## Running the code

Run everything from the project folder.

| Command | What it does | Approx. time |
|---|---|---|
| `python section2_demos\01_superposition.py` (also 02 to 06) | Section 2 Qiskit demos | under 1 min each |
| `python -m src.experiments eda circuits kernelcheck` | Dataset figures, circuit diagrams, kernel check | 1 min |
| `python -m src.experiments e1` | Main comparison, all 8 models | 5 min |
| `python -m src.experiments e3 e4 e7 e9` | QSVC and classical sweeps | 10 min |
| `python -m src.experiments e2 e5 e6` | Sweeps that include the VQC | 45 min |
| `python -m src.experiments e8` | Noise experiment | 40 min |
| `python -m src.experiments e1cv` | 5-fold cross-validation (resumes from saved folds) | 15 min |
| `python -m src.diagrams` | Architecture diagrams and flowcharts | under 1 min |

Times are for the laptop used (Intel i3, 7.7 GB RAM) on mains power. Keep the laptop plugged in and awake
for the long runs: on battery they were about four times slower.

## Rebuilding the report

```
python report\build_report.py
powershell -ExecutionPolicy Bypass -File report\finalize_wps.ps1
python report\quality_check.py
```

- `build_report.py` writes the `.docx`. Every result number is read from `results/*.csv`, not typed by hand.
- It converts equations using `MML2OMML.XSL`, which comes with Microsoft Office. If your path differs, change `XSL` in `report/framework.py`.
- `finalize_wps.ps1` uses WPS Office to fill in the contents lists and export the PDF.
- `quality_check.py` compares the finished report with the result files and reference records.

## Layout

| Folder | Contents |
|---|---|
| `section2_demos/` | Qiskit demos: superposition, Bell states, noise, Deutsch-Jozsa, Grover, Shor |
| `src/` | Preprocessing, classical and quantum models, evaluation, experiment grid, diagrams |
| `results/` | CSV result tables |
| `figures/` | Every figure, 300 dpi |
| `literature/` | Verification records, tracking sheet (`literature.csv`), reviews, comparative matrix |
| `report/` | Report source, final `.docx` and PDF |
| `logs/` | Run logs |

`literature/abstracts/` holds copies of abstracts for reference only. Do not submit that folder.

## Known limits

- QSVC normally uses the exact statevector kernel. `kernelcheck` confirms that it matches the circuit-based kernel.
- VQC sweeps use 200 training samples, and the noisy QSVC uses 100.
- The VQC uses COBYLA with 100 iterations as specified, and its loss had not converged.
- Qiskit's default VQC class mapping is the measured value modulo 2, so it reads qubit 0 only.
- E10 (real hardware) and the optional Streamlit demo were not done.
