"""Build report/NV62010_QML_Breast_Cancer_Report.docx from code, results and literature.

Run from the project folder:  python report/build_report.py
Then run report/finalize_word.ps1 to update the table of contents and count pages.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import content_front_s1_s2 as c12
import content_s3 as c3
import content_s4 as c4
import content_s5_s6_s7 as c5
import refs
import values
from framework import Report

OUTPUT = HERE / "NV62010_QML_Breast_Cancer_Report.docx"


def main():
    V, tables = values.load()
    h = tables["holdout"]
    cv = tables["cv"].set_index("code")
    V["pca_loss"] = f"{cv.loc['SVM30', 'accuracy_mean'] - cv.loc['SVM', 'accuracy_mean']:.3f}"
    references, _ = refs.load()
    d = Report(V, references)
    d.shor_rows = [[r.register, f"{r.phase:.3f}", r.fraction, int(r.guess_r), "yes" if r["a^r mod N == 1"] else "no", r.factors_from_r, int(r.counts)]
                   for _, r in tables["shor"].iterrows()]
    c12.front(d)
    d.begin_body()
    c12.section1(d)
    c12.section2(d)
    c3.section3(d)
    c4.section4(d)
    c5.section5(d, tables)
    c5.section6(d)
    c5.section7(d)
    problems = d.render(OUTPUT, "QUANTUM COMPUTING WITH AI: TECHNICAL REPORT", "NV62010")

    abstract = next(b["text"] for i, b in enumerate(d.blocks) if b["kind"] == "p" and b["text"].startswith("Breast cancer is a leading"))
    words = len(d._fill(abstract).split())
    print(f"abstract words: {words}")
    if not 200 <= words <= 250:
        problems.append(f"abstract has {words} words")
    print(f"references: {len(d.cite_no)}; figures: {len(d.fig_no)}; tables: {len(d.tab_no)}; equations: {len(d.eq_no)}")
    body = sum(len(d._fill(b["text"]).split()) for b in d.blocks if b["kind"] == "p")
    print(f"paragraph words: {body}")
    if problems:
        print("PROBLEMS:")
        print("\n".join(sorted(set(problems))))
        sys.exit(1)
    print("built", OUTPUT.name)


if __name__ == "__main__":
    main()
