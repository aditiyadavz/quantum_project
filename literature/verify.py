"""Check each candidate paper against Crossref (DOIs) or arXiv (preprints).

Writes verification.csv with the metadata as recorded by the publisher, a title
match score against the expected title, a retraction flag from OpenAlex, and
saves any abstract to abstracts/<key>.txt. Abstracts are data for reading only.
"""
import csv
import difflib
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).parent
UA = {"User-Agent": "qml-literature-check/1.0"}
FIELDS = ["theme", "id", "key", "status", "title_match", "expected_title", "title", "authors",
          "year", "venue", "volume", "issue", "pages", "type", "url", "source", "retracted",
          "cited_by", "abstract_words", "journal_ref"]


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return r.read()


def clean(text):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()


def crossref(doi):
    m = json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
    date = (m.get("published-print") or m.get("published-online") or m.get("issued"))
    authors = [f"{a.get('given', '')} {a.get('family', a.get('name', ''))}".strip()
               for a in m.get("author", [])]
    return {"title": clean(" ".join(m.get("title", []))), "authors": "; ".join(authors),
            "year": date["date-parts"][0][0], "venue": clean(" ".join(m.get("container-title", []))),
            "volume": m.get("volume", ""), "issue": m.get("issue", ""),
            "pages": m.get("page", "") or m.get("article-number", ""), "type": m.get("type", ""),
            "abstract": clean(m.get("abstract", "")), "url": f"https://doi.org/{doi}",
            "source": "crossref"}


def arxiv(aid):
    root = ET.fromstring(get("https://export.arxiv.org/api/query?id_list=" + aid))
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    e = root.find("a:entry", ns)
    jr = e.find("x:journal_ref", ns)
    return {"title": clean(e.find("a:title", ns).text),
            "authors": "; ".join(a.find("a:name", ns).text for a in e.findall("a:author", ns)),
            "year": e.find("a:published", ns).text[:4], "venue": f"arXiv:{aid}", "type": "preprint",
            "abstract": clean(e.find("a:summary", ns).text),
            "journal_ref": jr.text if jr is not None else "",
            "url": f"https://arxiv.org/abs/{aid}", "source": "arxiv"}


def openalex(doi):
    url = ("https://api.openalex.org/works/doi:" + urllib.parse.quote(doi)
           + "?select=is_retracted,abstract_inverted_index,cited_by_count")
    m = json.loads(get(url))
    words = sorted((p, w) for w, ps in (m.get("abstract_inverted_index") or {}).items() for p in ps)
    return m.get("is_retracted"), " ".join(w for _, w in words), m.get("cited_by_count")


rows = []
for c in csv.DictReader(open(HERE / "candidates.csv", encoding="utf-8")):
    key = re.sub(r"[^A-Za-z0-9]+", "_", c["id"]).strip("_")
    row = {"theme": c["theme"], "id": c["id"], "key": key, "expected_title": c["expected_title"]}
    try:
        info = arxiv(c["id"][6:]) if c["id"].startswith("arXiv:") else crossref(c["id"])
        row.update(info)
        row["title_match"] = round(difflib.SequenceMatcher(
            None, c["expected_title"].lower(), info["title"].lower()).ratio(), 2)
        if info["source"] == "crossref":
            try:
                row["retracted"], abstract, row["cited_by"] = openalex(c["id"])
                if len(abstract) > len(info["abstract"]):
                    row["abstract"] = abstract
            except Exception as e:
                row["retracted"] = f"openalex failed: {e}"
        abstract = row.pop("abstract", "")
        row["abstract_words"] = len(abstract.split())
        if abstract:
            (HERE / "abstracts" / f"{key}.txt").write_text(abstract, encoding="utf-8")
        row["status"] = "found"
    except Exception as e:
        row["status"] = f"FAILED: {e}"
    rows.append(row)
    print(f"{row['status'][:32]:32s} match={row.get('title_match', '')!s:5s} {row.get('year', '')!s:5s} "
          f"abs={row.get('abstract_words', '')!s:5s} {c['id']}", flush=True)
    time.sleep(0.4)

with open(HERE / "verification.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
