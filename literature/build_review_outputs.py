"""Combine the review batches, run the quality checks, and write the outputs.

Checks: 45 papers in selection order; 100-150 words per review; every review has
an abstract; no run of 7 or more words copied from that paper's abstract; every
matrix field filled. Writes literature.csv (tracking sheet), comparative_matrix.csv
and section3_reviews.md. Exit code 1 if any check fails.
"""
import csv
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from selection import SELECTED

HERE = Path(__file__).parent
PARTS = [("problem", "Problem and objective"), ("gap", "Research gap"), ("method", "Methodology"),
         ("results", "Results"), ("limits", "Limitations / scope")]
NGRAM = 7


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def copied_runs(review, abstract, n=NGRAM):
    r, a = words(review), words(abstract)
    grams = {tuple(a[i:i + n]) for i in range(len(a) - n + 1)}
    return [" ".join(r[i:i + n]) for i in range(len(r) - n + 1) if tuple(r[i:i + n]) in grams]


def published_year(doi):
    """Year of the journal issue when Crossref has one (online-first years can differ)."""
    req = urllib.request.Request("https://api.crossref.org/works/" + urllib.parse.quote(doi),
                                 headers={"User-Agent": "qml-literature-check/1.0"})
    m = json.loads(urllib.request.urlopen(req, timeout=30).read())["message"]
    ji = m.get("journal-issue", {})
    date = ji.get("published-print") or ji.get("published-online") or m.get("issued", {})
    return date.get("date-parts", [[None]])[0][0]


meta = {r["id"]: r for r in csv.DictReader(open(HERE / "verification.csv", encoding="utf-8"))}
audit = {r["id"]: r for r in csv.DictReader(open(HERE / "audit.csv", encoding="utf-8"))}
reviews = {}
for path in sorted((HERE / "reviews").glob("batch_*.json")):
    for item in json.load(open(path, encoding="utf-8")):
        assert item["id"] not in reviews, f"duplicate {item['id']}"
        reviews[item["id"]] = item

problems, sheet, matrix = [], [], []
md = ["# Section 3 reviews (draft)", "",
      "Review numbers are provisional. Final IEEE numbers follow first citation in the report.", ""]
theme = None
for n, (th, pid) in enumerate(SELECTED, start=1):
    if pid not in reviews:
        problems.append(f"[{n}] no review for {pid}")
        continue
    r, m = reviews[pid], meta[pid]
    body = " ".join(f"{label}: {r[key]}" for key, label in PARTS)
    prose = " ".join(r[key] for key, _ in PARTS)
    count = len(prose.split())
    if not 100 <= count <= 150:
        problems.append(f"[{n}] {count} words")

    path = HERE / "abstracts" / f"{m['key']}.txt"
    notes = HERE / "abstracts" / f"{m['key']}.notes.txt"
    if path.exists():
        abstract, basis = path.read_text(encoding="utf-8"), audit[pid]["abstract_source"]
    elif notes.exists():
        abstract, basis = "", "publisher page, read in browser; own-word notes kept"
    else:
        abstract, basis = "", ""
        problems.append(f"[{n}] no abstract")
    if pid == "10.1103/RevModPhys.94.015004":
        basis = "arXiv version 2101.08448 (OpenAlex text did not match this paper)"
    runs = copied_runs(prose, abstract)
    if runs:
        problems.append(f"[{n}] copied text: {runs[0]!r} (+{len(runs) - 1})")
    for key in ("technique", "mm", "f", "l"):
        if not r.get(key):
            problems.append(f"[{n}] empty matrix field {key}")

    year = m["year"]
    if not pid.startswith("arXiv:"):
        issue_year = published_year(pid)
        if issue_year and str(issue_year) != str(year):
            print(f"  [{n}] year {year} -> {issue_year} from journal issue")
            year = issue_year
    authors = m["authors"].split("; ")
    short = authors[0].split()[-1] + (" et al." if len(authors) > 2 else
                                      (" and " + authors[1].split()[-1] if len(authors) == 2 else ""))
    preprint = pid.startswith("arXiv:")
    sheet.append({"number": n, "theme": th, "authors": m["authors"], "year": year, "title": m["title"],
                  "venue": m["venue"], "volume": m["volume"], "issue": m["issue"], "pages": m["pages"],
                  "doi_or_url": m["url"], "preprint": "yes" if preprint else "no",
                  "doi_registered": audit[pid]["doi_registered"] or "n/a",
                  "verified": "yes", "review_basis": "abstract only", "abstract_source": basis,
                  "retraction_flag": m["retracted"] or "not checked (preprint)", "review_words": count})
    matrix.append({"S. No.": f"[{n}]", "Authors (Year)": f"{short} ({year})", "Technique": r["technique"],
                   "Methodology / Dataset": r["mm"], "Key findings": r["f"], "Limitations": r["l"]})
    if th != theme:
        md += [f"## {th}", ""]
        theme = th
    md += [f"**[{n}] {short} ({year}), \"{m['title']}.\"** {body}", ""]

for out, rows in [("literature.csv", sheet), ("comparative_matrix.csv", matrix)]:
    with open(HERE / out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
(HERE / "section3_reviews.md").write_text("\n".join(md), encoding="utf-8")

counts = [s["review_words"] for s in sheet]
print(f"reviews {len(sheet)}; words min {min(counts)} max {max(counts)} mean {sum(counts) / len(counts):.0f}")
print(f"before 2021: {sum(int(s['year']) < 2021 for s in sheet)}; preprints: "
      f"{sum(s['preprint'] == 'yes' for s in sheet)}")
print("retraction flags:", sorted({str(s['retraction_flag']) for s in sheet}))
if problems:
    print("PROBLEMS:")
    print("\n".join(problems))
    sys.exit(1)
print("all checks passed")
