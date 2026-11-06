"""The 45 reviewed papers in review order, grouped by the spec's theme targets.

Numbers here are review numbers only. Final IEEE numbers are assigned in the
report phase, in order of first citation.
"""
SELECTED = [
    # QML surveys and foundations (6)
    ("Foundations", "10.1038/s43588-022-00311-3"), ("Foundations", "10.1103/PRXQuantum.3.030101"),
    ("Foundations", "10.1038/s41467-021-22539-9"), ("Foundations", "10.1038/s41567-021-01287-z"),
    ("Foundations", "10.1038/s41467-022-32550-3"), ("Foundations", "arXiv:2403.07059"),
    # Quantum kernel methods and QSVM (7)
    ("Kernels", "10.1038/s41586-019-0980-2"), ("Kernels", "10.1038/s41467-024-49287-w"),
    ("Kernels", "10.1038/s41467-023-36159-y"), ("Kernels", "10.1103/PhysRevA.106.042431"),
    ("Kernels", "10.1103/PhysRevA.106.042407"), ("Kernels", "10.1038/s41534-021-00498-9"),
    ("Kernels", "10.22331/q-2021-08-30-531"),
    # Variational quantum algorithms and VQC (5)
    ("VQA", "10.1038/s42254-021-00348-9"), ("VQA", "10.1103/RevModPhys.94.015004"),
    ("VQA", "10.1103/PhysRevA.103.032430"), ("VQA", "10.22331/q-2020-02-06-226"),
    ("VQA", "10.1007/s42484-020-00036-4"),
    # Quantum neural networks and trainability (4)
    ("QNN", "10.1038/s43588-021-00084-1"), ("QNN", "10.1038/s41467-021-21728-w"),
    ("QNN", "10.1038/s41467-021-27045-6"), ("QNN", "10.1103/PRXQuantum.3.010313"),
    # Hybrid quantum-classical models (5)
    ("Hybrid", "10.22331/q-2020-10-09-340"), ("Hybrid", "10.1007/s11433-021-1734-3"),
    ("Hybrid", "10.1007/s42484-021-00061-x"), ("Hybrid", "10.1038/s41467-023-36144-5"),
    ("Hybrid", "10.1038/s43588-022-00351-9"),
    # QML in healthcare and medical diagnosis (7)
    ("Health", "10.1038/s41598-022-05971-9"), ("Health", "10.1007/s42484-022-00062-4"),
    ("Health", "10.1049/qtc2.12113"), ("Health", "arXiv:2407.09930"),
    ("Health", "10.3390/diagnostics16131996"), ("Health", "10.1109/TQE.2022.3176806"),
    ("Health", "10.22331/q-2022-12-22-881"),
    # Quantum-enhanced image recognition (3)
    ("Image", "10.1007/s42484-020-00012-y"), ("Image", "10.1088/2632-2153/ad2aef"),
    ("Image", "10.1007/s43673-021-00030-3"),
    # Quantum optimization (3)
    ("Optimization", "10.1016/j.physrep.2024.03.002"), ("Optimization", "10.1126/science.abo6587"),
    ("Optimization", "10.1038/s42254-024-00770-9"),
    # AI/ML for quantum computing (2)
    ("AIforQC", "10.1038/s41586-024-08148-8"), ("AIforQC", "10.1038/s41467-025-65836-3"),
    # Applications: finance, logistics, cybersecurity (3)
    ("Applications", "10.1038/s42254-023-00603-1"), ("Applications", "10.1109/TQE.2021.3049230"),
    ("Applications", "10.1007/s11416-022-00435-0"),
]
assert len(SELECTED) == 45 and len({i for _, i in SELECTED}) == 45
