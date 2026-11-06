"""Phase 7 quality checklist. Reads the FINISHED report (not the build data) and compares it
with results/*.csv and the reference records, so that it can catch build errors.

Run from the project folder:  python report/quality_check.py
"""
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
import pypdfium2
from docx import Document

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "report" / "NV62010_QML_Breast_Cancer_Report.pdf"
DOCX = ROOT / "report" / "NV62010_QML_Breast_Cancer_Report.docx"
RES = ROOT / "results"
UA = {"User-Agent": "qml-literature-check/1.0"}
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + str(detail) if detail else ''}")


pdf = pypdfium2.PdfDocument(str(PDF))
pages = [pdf[i].get_textpage().get_text_range().replace("\r", "") for i in range(len(pdf))]
text = "\n".join(pages)
flat = re.sub(r"\s+", " ", text)
doc = Document(str(DOCX))

# ---------- structure
print("\n== Structure")
check("page count within 50-65", 50 <= len(pages) <= 65, f"{len(pages)} pages")
titles = ["Introduction and Synopsis", "Fundamentals of Quantum Computing", "Critical Review of Advanced",
          "Proposed Mini Project", "Results, Performance Evaluation", "Conclusion and Future Scope", "References and Bibliography"]
headings = [p.text for p in doc.paragraphs if p.style.name == "Heading 1"]
order = [next((i for i, h in enumerate(headings) if t in h), -1) for t in titles]
check("all 7 sections present, in order", all(x >= 0 for x in order) and order == sorted(order), headings[-7:] and "")
for needed in ["Table of Contents", "List of Figures", "List of Tables", "List of Abbreviations", "Abstract", "Declaration"]:
    check(f"front matter: {needed}", any(needed in h for h in headings))
subs = [p.text for p in doc.paragraphs if p.style.name == "Heading 2"]
for label, n in [("2.", 12), ("4.", 16)]:
    count = sum(1 for s in subs if re.match(label.replace(".", r"\.") + r"\d+ ", s))
    check(f"Section {label[0]} has {n} required subsections", count == n, count)
check("contents lists filled in", "Update this field" not in text and "Error!" not in text)
check("no leftover markup", not re.search(r"\[\[|\]\]|<<|>>|\*\*|\$", text))
left = sorted(set(re.findall(r"\[(?:fill in|confirm[^\]]*)\]", text)))
print(f"INFO  placeholders left for the student: {left} on pages "
      f"{[i + 1 for i, t in enumerate(pages) if '[fill in]' in t or '[confirm' in t]}")

# ---------- abstract and front
abstract = next(p.text for p in doc.paragraphs if p.text.startswith("Breast cancer is a leading"))
check("abstract is 200-250 words", 200 <= len(abstract.split()) <= 250, len(abstract.split()))
for value in ["Aditi Yadav", "2301010045", "2023215091", "Dr. Dharam Raj", "NV62010"]:
    check(f"cover has {value}", value in pages[0])

# ---------- figures and tables
print("\n== Figures, tables, equations")
body = "\n".join(pages[8:])   # after the front-matter lists
figs = sorted(set(re.findall(r"Fig\. (\d+\.\d+)\. ", body)), key=lambda s: [int(x) for x in s.split(".")])
tabs = sorted(set(re.findall(r"TABLE (\d+\.\d+)\. ", body)), key=lambda s: [int(x) for x in s.split(".")])
missing_ref = []
for f in figs:
    first_ref = re.search(rf"Fig\. {re.escape(f)}(?!\d)", body)
    if first_ref and body[first_ref.start():first_ref.end() + 2].endswith(". ") and body.count(f"Fig. {f}") < 2:
        missing_ref.append(f)
check("every figure has a caption and is referred to in text", not missing_ref, f"{len(figs)} figures {missing_ref or ''}")
flat_body = re.sub(r"\s+", " ", body)    # references can break across two lines
unreferenced = [t for t in tabs if f"Table {t}" not in flat_body]
check("every table is referred to in text", not unreferenced, f"{len(tabs)} tables {unreferenced or ''}")
for s in "245":
    nums = [int(f.split(".")[1]) for f in figs if f.startswith(s + ".")]
    check(f"Section {s} figures numbered without gaps", nums == list(range(1, len(nums) + 1)), len(nums))
check("figure images embedded", len(doc.inline_shapes) >= len(figs), len(doc.inline_shapes))

# ---------- results agree with CSV
print("\n== Result numbers against results/*.csv")
names = {"QSVC (quantum kernel)": "QSVC", "VQC": "VQC", "SVM (RBF)": "SVM (RBF)", "Logistic Regression": "Logistic Regression",
         "Random Forest": "Random Forest", "KNN": "KNN", "MLP": "MLP", "SVM (RBF, 30 features)": "SVM (RBF, 30 features)"}
tables = {t.rows[0].cells[0].text: t for t in doc.tables}
holdout = pd.read_csv(RES / "e1_main_holdout.csv")
cv = pd.read_csv(RES / "e1_cv_summary.csv")
folds = pd.read_csv(RES / "e1_cv_folds.csv")
bad = []
checked = 0
for t in doc.tables:
    header = [c.text for c in t.rows[0].cells]
    if header[:2] == ["Model", "Accuracy"] and "Infer (s)" in header:
        for row in t.rows[1:]:
            cells = [c.text for c in row.cells]
            source = holdout[holdout.model.map(names) == cells[0]].iloc[0]
            for col, name in zip(cells[1:7], ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]):
                checked += 1
                if col != f"{source[name]:.4f}":
                    bad.append((cells[0], name, col))
    if header[:2] == ["Model", "Accuracy"] and "Infer (s)" not in header:
        for row in t.rows[1:]:
            cells = [c.text for c in row.cells]
            source = cv[cv.model.map(names) == cells[0]].iloc[0]
            for col, name in zip(cells[1:7], ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]):
                checked += 1
                if not col.startswith(f"{source[name + '_mean']:.3f}"):
                    bad.append((cells[0], name, col))
    if header[0] == "Model" and "Missed cancers (FN)" in header:
        for row in t.rows[1:]:
            cells = [c.text for c in row.cells]
            m = folds[folds.model.map(names) == cells[0]]
            checked += 2
            if int(cells[1]) != int(m.fn.sum()) or int(cells[2]) != int(m.fp.sum()):
                bad.append((cells[0], "FN/FP", cells[1:3]))
check("Tables 5.1, 5.2 and 5.5 match the CSV files", not bad and checked > 90, f"{checked} values {bad or ''}")

e = pd.read_csv(RES / "e4_entanglement_summary.csv").set_index("feature_map")
e3 = pd.read_csv(RES / "e3_fm_depth_summary.csv").set_index("fm_reps")
e7 = pd.read_csv(RES / "e7_train_size_summary.csv")
e9 = pd.read_csv(RES / "e9_iris_summary.csv").set_index("model")
e8 = pd.read_csv(RES / "e8_noise.csv")
expected = {
    "E4 Z map accuracy": f"{e.loc['Z (no entanglement)', 'accuracy_mean']:.3f}",
    "E3 reps=4 accuracy": f"{e3.loc[4, 'accuracy_mean']:.3f}",
    "E7 QSVC at 50": f"{e7[(e7.model == 'QSVC') & (e7.train_size == 50)].accuracy_mean.iloc[0]:.3f}",
    "E9 QSVC": f"{e9.loc['QSVC', 'accuracy_mean']:.3f}",
    "E8 QSVC noisy": f"{e8[(e8.model == 'QSVC') & e8.setting.str.startswith('noisy')].accuracy.iloc[0]:.3f}",
}
for label, value in expected.items():
    check(f"{label} = {value} appears in report", value in flat)

# ---------- citations and references
print("\n== Citations and references")
ref_start = next(i for i, t in enumerate(pages) if "References and Bibliography" in t and i > 10)
main = "\n".join(pages[8:ref_start])
refs_text = re.sub(r"\s+", " ", "\n".join(pages[ref_start:]))
numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]\s", refs_text)})
check("reference list numbered without gaps", numbers == list(range(1, len(numbers) + 1)), f"{len(numbers)} references")
cited = set()
for a, b in re.findall(r"\[(\d+)\]–\[(\d+)\]", main):
    cited.update(range(int(a), int(b) + 1))
cited.update(int(n) for n in re.findall(r"\[(\d+)\]", main))
check("every citation has a reference", cited <= set(numbers), sorted(cited - set(numbers)))
check("every reference is cited", set(numbers) <= cited, sorted(set(numbers) - cited))
first = [int(re.search(r"\d+", m).group()) for m in re.findall(r"\[\d+\]", main)]
order_ok, top = True, 0
for n in first:
    if n > top + 1 and n not in range(top + 1):
        order_ok = order_ok and n <= top + 8     # ranges and sorted groups can jump slightly
    top = max(top, n)
check("numbered in order of first citation", order_ok)

sheet = pd.read_csv(ROOT / "literature" / "literature.csv")
check("45 reviewed papers, 40+ from 2021 on", len(sheet) == 45 and (sheet.year >= 2021).sum() >= 40, f"{(sheet.year >= 2021).sum()} from 2021+")
check("reviews are 100-150 words", sheet.review_words.between(100, 150).all(), f"{sheet.review_words.min()}-{sheet.review_words.max()}")
matrix = next(t for t in doc.tables if t.rows[0].cells[0].text == "S. No.")
check("comparative matrix has 45 rows", len(matrix.rows) == 46, len(matrix.rows) - 1)
landscape = [s for s in doc.sections if s.page_width > s.page_height]
check("matrix is on landscape pages", len(landscape) >= 1)

extra = pd.read_csv(ROOT / "literature" / "extra_verification.csv")
dois = [i for i in extra.id if not str(i).startswith("arXiv")] + ["10.24432/C5DW2B"]
failed = []
for doi in dois:
    try:
        req = urllib.request.Request("https://doi.org/api/handles/" + urllib.parse.quote(doi), headers=UA)
        if json.loads(urllib.request.urlopen(req, timeout=30).read()).get("responseCode") != 1:
            failed.append(doi)
    except Exception as exc:
        failed.append(f"{doi}: {exc}")
check("DOIs of the additional references are registered", not failed, f"{len(dois)} checked {failed or ''}")

print("\n== Formatting (student's ML file)")
s = doc.sections[0]
check("A4 page", abs(s.page_width.inches - 8.27) < 0.02)
check("0.9 inch margins", abs(s.left_margin.inches - 0.9) < 0.01)
check("page border on all sections", all(x._sectPr.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pgBorders") is not None for x in doc.sections), len(doc.sections))

print(f"\nRESULT: {sum(results)} of {len(results)} checks passed")
