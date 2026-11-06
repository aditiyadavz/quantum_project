"""Second check on the candidates.

1. Every abstract taken from an arXiv title search must share its first author's
   surname with the published record; otherwise the abstract is deleted.
2. Abstracts still missing are retried on Semantic Scholar, slowly.
3. Every DOI is checked at the doi.org handle service.
Writes audit.csv.
"""
import csv
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).parent
UA = {"User-Agent": "qml-literature-check/1.0"}
NS = {"a": "http://www.w3.org/2005/Atom"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return r.read()


def surname(name):
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return plain.split()[-1].lower()


sources = {r["key"]: r["abstract_source"] for r in
           csv.DictReader(open(HERE / "abstract_sources.csv", encoding="utf-8"))}
rows = []
for row in csv.DictReader(open(HERE / "verification.csv", encoding="utf-8")):
    key, source = row["key"], sources.get(row["key"], "")
    path = HERE / "abstracts" / f"{key}.txt"
    arxiv_author_ok, doi_ok = "", ""

    if source.startswith("arXiv version"):
        aid = source.split("/abs/")[1].split("v")[0]
        root = ET.fromstring(get("https://export.arxiv.org/api/query?id_list=" + aid))
        first = root.find("a:entry", NS).find("a:author", NS).find("a:name", NS).text
        arxiv_author_ok = surname(first) == surname(row["authors"].split(";")[0])
        if not arxiv_author_ok:
            path.unlink(missing_ok=True)
            source = f"REJECTED arXiv {aid} (first author {first})"
        time.sleep(1)

    if not path.exists() or len(path.read_text(encoding="utf-8").split()) < 40:
        if not row["id"].startswith("arXiv:"):
            for attempt in range(3):
                time.sleep(12)
                try:
                    url = ("https://api.semanticscholar.org/graph/v1/paper/DOI:"
                           + urllib.parse.quote(row["id"]) + "?fields=abstract")
                    text = re.sub(r"\s+", " ", json.loads(get(url)).get("abstract") or "").strip()
                    if text:
                        path.write_text(text, encoding="utf-8")
                        source = "semantic scholar"
                    break
                except Exception as e:
                    print(f"  retry {attempt + 1}: {e}")

    if not row["id"].startswith("arXiv:"):
        try:
            handle = json.loads(get("https://doi.org/api/handles/" + urllib.parse.quote(row["id"])))
            doi_ok = handle.get("responseCode") == 1
        except Exception as e:
            doi_ok = f"failed: {e}"
        time.sleep(0.3)

    words = len(path.read_text(encoding="utf-8").split()) if path.exists() else 0
    rows.append({"key": key, "id": row["id"], "abstract_source": source, "abstract_words": words,
                 "arxiv_first_author_ok": arxiv_author_ok, "doi_registered": doi_ok})
    if arxiv_author_ok is False or words < 40 or doi_ok not in ("", True):
        print(f"CHECK {row['id']}: {source} words={words} doi={doi_ok}", flush=True)

with open(HERE / "audit.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print("arXiv-sourced abstracts with matching first author:",
      sum(r["arxiv_first_author_ok"] is True for r in rows))
print("DOIs registered:", sum(r["doi_registered"] is True for r in rows))
