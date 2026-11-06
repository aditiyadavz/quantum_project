"""Find abstracts missing from Crossref/OpenAlex.

Tries Semantic Scholar by DOI, then an arXiv title search. An arXiv abstract is
accepted only if its title matches the published title closely (>= 0.95). The
source of each abstract is recorded in abstract_sources.csv.
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
NS = {"a": "http://www.w3.org/2005/Atom"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return r.read()


def clean(text):
    return re.sub(r"\s+", " ", text or "").strip()


def semantic_scholar(doi):
    url = ("https://api.semanticscholar.org/graph/v1/paper/DOI:" + urllib.parse.quote(doi)
           + "?fields=abstract")
    return clean(json.loads(get(url)).get("abstract"))


def arxiv_by_title(title):
    words = re.sub(r"[^A-Za-z0-9 ]+", " ", title)
    query = urllib.parse.quote(f'ti:"{clean(words)}"')
    root = ET.fromstring(get(f"https://export.arxiv.org/api/query?search_query={query}&max_results=5"))
    best = (0, "", "")
    for e in root.findall("a:entry", NS):
        found = clean(e.find("a:title", NS).text)
        score = difflib.SequenceMatcher(None, title.lower(), found.lower()).ratio()
        if score > best[0]:
            best = (score, clean(e.find("a:summary", NS).text), e.find("a:id", NS).text)
    return best


sources = []
for row in csv.DictReader(open(HERE / "verification.csv", encoding="utf-8")):
    path = HERE / "abstracts" / f"{row['key']}.txt"
    if path.exists() and len(path.read_text(encoding="utf-8").split()) >= 40:
        sources.append({"key": row["key"], "abstract_source": row["source"] + "/openalex"})
        continue
    found, origin = "", "NOT FOUND"
    try:
        found = semantic_scholar(row["id"])
        origin = "semantic scholar" if found else origin
    except Exception as e:
        print(f"  semantic scholar: {e}")
    time.sleep(3.2)                      # public rate limit
    if not found:
        try:
            score, text, link = arxiv_by_title(row["title"])
            if score >= 0.95 and text:
                found, origin = text, f"arXiv version {link}"
        except Exception as e:
            print(f"  arxiv: {e}")
        time.sleep(3.2)
    if found:
        path.write_text(found, encoding="utf-8")
    sources.append({"key": row["key"], "abstract_source": origin})
    print(f"{origin[:44]:44s} {len(found.split()):4d}w  {row['title'][:60]}", flush=True)

with open(HERE / "abstract_sources.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["key", "abstract_source"])
    writer.writeheader()
    writer.writerows(sources)
