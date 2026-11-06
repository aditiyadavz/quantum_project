"""Section 3: built from the verified reviews and matrix in literature/."""
import csv
import json
from pathlib import Path

LIT = Path(__file__).resolve().parent.parent / "literature"
THEMES = [("Foundations", "QML Surveys and Foundations"), ("Kernels", "Quantum Kernel Methods and QSVM"),
          ("VQA", "Variational Quantum Algorithms and VQC"), ("QNN", "Quantum Neural Networks and Trainability"),
          ("Hybrid", "Hybrid Quantum-Classical Models"), ("Health", "QML in Healthcare and Medical Diagnosis"),
          ("Image", "Quantum-Enhanced Image Recognition"), ("Optimization", "Quantum Optimization"),
          ("AIforQC", "AI and ML for Quantum Computing"), ("Applications", "Applications in Finance, Logistics and Cybersecurity")]
PARTS = [("problem", "Problem and objective"), ("gap", "Research gap"), ("method", "Methodology"), ("results", "Results"), ("limits", "Limitations / scope")]


def load_reviews():
    sheet = list(csv.DictReader(open(LIT / "literature.csv", encoding="utf-8")))
    matrix = list(csv.DictReader(open(LIT / "comparative_matrix.csv", encoding="utf-8")))
    for m in matrix:      # this author is registered with a one-letter family name
        m["Authors (Year)"] = m["Authors (Year)"].replace("P et al. (2024)", "Jose P et al. (2024)")
    texts = {}
    for path in sorted((LIT / "reviews").glob("batch_*.json")):
        for item in json.load(open(path, encoding="utf-8")):
            texts[item["id"]] = item
    return sheet, matrix, texts


def section3(d):
    sheet, matrix, texts = load_reviews()
    d.h1("Critical Review of Advanced Machine Learning and Quantum Machine Learning Research")
    d.h2("Review Method")
    d.p("Candidate papers came from the assignment's seed list, from well-known work in each theme, and from web searches using keywords "
        "such as \"quantum support vector machine breast cancer\", \"variational quantum classifier healthcare\" and \"quantum machine learning "
        "intrusion detection\". Each candidate was then checked against its registered record: Crossref for journal papers, with a check that the "
        "DOI is registered, and the arXiv service for preprints. OpenAlex supplied retraction flags. Abstracts came from Crossref or OpenAlex, "
        "Semantic Scholar, an arXiv version with the same title and first author, or, in one case, the publisher's page. Google Scholar and IEEE "
        "Xplore were not searched directly. From 68 verified candidates, 45 papers were selected to meet the theme targets: 41 from 2021 onwards and "
        "4 foundational papers from 2019 and 2020. Two are labelled preprints. Papers were included when they addressed a listed theme and an "
        "abstract could be read, and relevance to breast cancer or tabular medical data was favoured. No selected journal paper carries a retraction "
        "flag in OpenAlex; preprints were not checked for this. **All reviews are based on abstracts rather than full texts, so their claims are kept "
        "general.** The bracketed number at the start of each review is its IEEE reference number.")
    for theme, title in THEMES:
        d.h2(title)
        for row in sheet:
            if row["theme"] != theme:
                continue
            match = next(m for m in matrix if m["S. No."] == f"[{row['number']}]")
            pid = row["doi_or_url"].replace("https://doi.org/", "")
            if pid.startswith("https://arxiv.org/abs/"):
                pid = "arXiv:" + pid.rsplit("/", 1)[-1]
            review = texts[pid]
            title_text = row["title"].replace("‐", "-").replace("*", "")
            body = " ".join(f"*{label}:* {review[key]}" for key, label in PARTS)
            d.review(f"**[[c:r{row['number']}]] {match['Authors (Year)']}, \"{title_text}.\"** {body}")
    d.begin_landscape()
    d.h2("Consolidated Comparative Research Matrix")
    d.p("[[T:matrix]] summarises all reviewed papers, in the same theme order as the reviews above.")
    order = {theme: i for i, (theme, _) in enumerate(THEMES)}
    theme_of = {f"[{r['number']}]": r["theme"] for r in sheet}
    rows = []
    for m in sorted(matrix, key=lambda m: (order[theme_of[m["S. No."]]], int(m["S. No."][1:-1]))):
        number = m["S. No."][1:-1]
        rows.append([f"[[c:r{number}]]", m["Authors (Year)"], m["Technique"], m["Methodology / Dataset"], m["Key findings"], m["Limitations"]])
    d.table("matrix", "Comparative research matrix of the reviewed papers.",
            ["S. No.", "Authors (Year)", "Technique", "Methodology / Dataset", "Key findings", "Limitations"], rows,
            widths=[0.5, 1.4, 1.85, 2.2, 2.2, 1.7], size=8.5)
    d.end_landscape()
    d.h2("Identified Research Gaps")
    d.p("The following gaps describe the 45 reviewed papers as seen through their abstracts. They are not a claim about the whole field. Each gap links "
        "to an objective of the mini project in Section 4.3.")
    d.bullets([
        "**G1. Few direct comparisons of quantum kernels and variational classifiers on the same medical dataset.** Medical studies mostly evaluate one "
        "quantum family: kernels in [[c:r28,r30,r31,r33]], and quantum or hybrid networks in [[c:r29,r34]]. Only [[c:r32]] compares a VQC and a ZZ quantum "
        "kernel on WDBC. This project adds an independent comparison (objective O1).",
        "**G2. Limited fair comparison on identical reduced features.** Some positive reports use a same-structure baseline [[c:r24]], train under similar "
        "conditions [[c:r25]], use the authors' own data [[c:r45]], or give no accuracy figure in the abstract [[c:r30]], while a large benchmark found that "
        "classical models generally win [[c:r6]]. Here every reduced-feature model receives the same PCA features (O2).",
        "**G3. Little joint analysis of how circuit design changes medical results.** Theory explains the effects of expressibility, entanglement and depth "
        "[[c:r8,r21,r22]], and [[c:r11]] tunes kernel bandwidth. Medical studies examine parts of the design, such as feature maps [[c:r31]] or embeddings, "
        "depth and noise [[c:r32]]. No reviewed abstract varies qubit count, feature-map depth, entanglement, ansatz depth and training size together on one "
        "medical dataset (O3).",
        "**G4. Little study of noise effects on diagnostic recall.** Noise is studied for kernels [[c:r10,r13]], gradients [[c:r21]] and hardware [[c:r12]], and "
        "[[c:r32]] tests depolarising noise on a VQC. No reviewed abstract reports malignant-class recall under a device noise model (O4).",
        "**G5. Diagnostic errors are rarely the headline result.** Medical abstracts mostly report accuracy or AUC [[c:r29,r32,r33]], and the reviewed abstracts "
        "do not report false-negative counts. This project reports missed cancers for every model (O2).",
        "**G6. Costs are rarely reported next to accuracy.** The quadratic kernel cost is noted but left open in [[c:r10]], and speed claims can be hard to "
        "interpret [[c:r30,r45]]. This project records time, circuit counts and depth (O5).",
    ])
