"""IEEE reference text, built from the verified records in literature/.

Journal details come from Crossref/arXiv records (see literature/). Books, web
pages and the dataset are written by hand and were checked on 2026-11-06.
"""
import csv
import re
from pathlib import Path

LIT = Path(__file__).resolve().parent.parent / "literature"
ACCESSED = "(accessed Nov. 6, 2026)"

JOURNALS = {
    "Nature Computational Science": "Nat. Comput. Sci.", "PRX Quantum": "PRX Quantum",
    "Nature Communications": "Nat. Commun.", "Nature Physics": "Nat. Phys.", "Nature": "Nature",
    "Physical Review A": "Phys. Rev. A", "npj Quantum Information": "npj Quantum Inf.",
    "Quantum": "Quantum", "Nature Reviews Physics": "Nat. Rev. Phys.",
    "Reviews of Modern Physics": "Rev. Mod. Phys.", "Quantum Machine Intelligence": "Quantum Mach. Intell.",
    "Physical Review Letters": "Phys. Rev. Lett.", "Scientific Reports": "Sci. Rep.",
    "Science China Physics, Mechanics & Astronomy": "Sci. China Phys. Mech. Astron.",
    "Science China Physics, Mechanics &amp; Astronomy": "Sci. China Phys. Mech. Astron.",
    "IET Quantum Communication": "IET Quantum Commun.", "Diagnostics": "Diagnostics",
    "IEEE Transactions on Quantum Engineering": "IEEE Trans. Quantum Eng.",
    "Machine Learning: Science and Technology": "Mach. Learn.: Sci. Technol.",
    "AAPPS Bulletin": "AAPPS Bull.", "Physics Reports": "Phys. Rep.", "Science": "Science",
    "Journal of Computer Virology and Hacking Techniques": "J. Comput. Virol. Hacking Techn.",
    "CA: A Cancer Journal for Clinicians": "CA Cancer J. Clin.",
    "International Journal of Theoretical Physics": "Int. J. Theor. Phys.",
    "SIAM Journal on Computing": "SIAM J. Comput.", "Machine Learning": "Mach. Learn.",
    "IEEE Transactions on Automatic Control": "IEEE Trans. Autom. Control",
    "MATEC Web of Conferences": "MATEC Web Conf.",
    "Proceedings of the Royal Society of London. A. Mathematical and Physical Sciences": "Proc. R. Soc. Lond. A",
    "Proceedings of the Royal Society of London. Series A: Mathematical and Physical Sciences": "Proc. R. Soc. Lond. A",
    "Proceedings of the Royal Society of London. Series A: Mathematical, Physical and Engineering Sciences": "Proc. R. Soc. Lond. A",
}
NAME_FIXES = {"Shehab Omar": "Omar Shehab", "Zoran Krunic": "Zoran Krunic", "Frederik Flother": "Frederik F. Flöther",
              "Nate Earnest-Noble": "Nathan Earnest-Noble"}


def initials(name):
    """'Hsin-Yuan Huang' -> 'H.-Y. Huang'. Names whose family part is one letter stay as registered."""
    name = NAME_FIXES.get(name.strip(), name.strip())
    parts = name.split()
    if len(parts) == 1 or len(parts[-1].rstrip(".")) == 1:
        return name
    given = []
    for part in parts[:-1]:
        given.append("-".join(p[0].upper() + "." for p in part.split("-") if p))
    return " ".join(given) + " " + parts[-1]


def authors(text):
    names = [n for n in text.split("; ") if n]
    if len(names) > 6:
        return initials(names[0]) + " *et al.*"
    formatted = [initials(n) for n in names]
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    return ", ".join(formatted[:-1]) + ", and " + formatted[-1]


def clean_number(value):
    return re.sub(r"\.0$", "", str(value or "")).strip()


def journal_reference(r, doi):
    title = re.sub(r"\s+", " ", r["title"]).replace("‐", "-").rstrip(".")
    venue = JOURNALS.get(r["venue"])
    if venue is None:
        raise KeyError(f"no journal abbreviation for {r['venue']!r}")
    vol, issue, pages = clean_number(r["volume"]), clean_number(r["issue"]), clean_number(r["pages"])
    bits = [f"*{venue}*"]
    if vol:
        bits.append(f"vol. {vol}")
    if issue:
        bits.append(f"no. {issue}")
    if pages:
        bits.append(f"pp. {pages.replace('-', chr(0x2013))}" if "-" in pages else f"Art. no. {pages}")
    bits.append(str(r["year"]))
    return f"{authors(r['authors'])}, \"{title},\" {', '.join(bits)}, doi: {doi}."


def preprint_reference(r):
    title = re.sub(r"\s+", " ", r["title"]).rstrip(".")
    aid = r["venue"] if r.get("venue", "").startswith("arXiv:") else r["doi_or_url"].rsplit("/", 1)[-1]
    if not aid.startswith("arXiv:"):
        aid = f"arXiv:{aid}"
    return f"{authors(r['authors'])}, \"{title},\" {aid}, {r['year']}."


def load():
    refs, reviewed = {}, {}
    for r in csv.DictReader(open(LIT / "literature.csv", encoding="utf-8")):
        key = f"r{r['number']}"
        reviewed[key] = r
        if r["preprint"] == "yes":
            refs[key] = preprint_reference(r)
        else:
            refs[key] = journal_reference(r, r["doi_or_url"].replace("https://doi.org/", ""))

    extra = {r["key"]: r for r in csv.DictReader(open(LIT / "extra_verification.csv", encoding="utf-8"))}
    for key in ("bray2024", "feynman1982", "deutsch1985", "deutsch1992", "shor1997", "preskill2018", "biamonte2017",
                "schreiber2023", "kim2023", "google2025", "cortes1995", "spall1992", "mpofu2025"):
        refs[key] = journal_reference(extra[key], extra[key]["id"])
    # Crossref's first "author" is the group name, which must not be shortened to initials
    refs["google2025"] = re.sub(r"^.*?\*et al\.\*", "Google Quantum AI and Collaborators", refs["google2025"])
    refs["qiskit2024"] = preprint_reference({**extra["arXiv_qiskit"], "doi_or_url": "arXiv:2405.08810"})
    refs["grover1996"] = (f"{authors(extra['grover1996']['authors'])}, \"A fast quantum mechanical algorithm for database "
                          "search,\" in *Proc. 28th Annu. ACM Symp. Theory Comput.*, Philadelphia, PA, USA, 1996, "
                          "pp. 212–219, doi: 10.1145/237814.237866.")
    refs["powell1994"] = ("M. J. D. Powell, \"A direct search optimization method that models the objective and constraint "
                          "functions by linear interpolation,\" in *Advances in Optimization and Numerical Analysis*, "
                          "S. Gomez and J.-P. Hennart, Eds. Dordrecht, Netherlands: Springer, 1994, pp. 51–67, "
                          "doi: 10.1007/978-94-015-8330-5_4.")
    refs.update({
        "nielsen2010": "M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information*, 10th Anniversary ed. "
                       "Cambridge, U.K.: Cambridge Univ. Press, 2010, doi: 10.1017/CBO9780511976667.",
        "schuld2021book": "M. Schuld and F. Petruccione, *Machine Learning with Quantum Computers*, 2nd ed. Cham, Switzerland: "
                          "Springer, 2021, doi: 10.1007/978-3-030-83098-4.",
        "hidary2021": "J. D. Hidary, *Quantum Computing: An Applied Approach*, 2nd ed. Cham, Switzerland: Springer, 2021, "
                      "doi: 10.1007/978-3-030-83274-2.",
        "russell2020": "S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed. Hoboken, NJ, USA: "
                       "Pearson, 2020.",
        "bishop2006": "C. M. Bishop, *Pattern Recognition and Machine Learning*. New York, NY, USA: Springer, 2006.",
        "pedregosa2011": "F. Pedregosa *et al.*, \"Scikit-learn: Machine learning in Python,\" *J. Mach. Learn. Res.*, vol. 12, "
                         "pp. 2825–2830, 2011. [Online]. Available: https://jmlr.org/papers/v12/pedregosa11a.html",
        "uci1995": "W. Wolberg, O. Mangasarian, N. Street, and W. Street, \"Breast Cancer Wisconsin (Diagnostic),\" UCI Machine "
                   "Learning Repository, 1995, doi: 10.24432/C5DW2B.",
        "who2026": "World Health Organization, \"Breast cancer,\" Fact Sheet, Jul. 3, 2026. [Online]. Available: "
                   f"https://www.who.int/news-room/fact-sheets/detail/breast-cancer {ACCESSED}.",
        "mckinsey2025": "H. Soller, M. Gschwendtner, S. Shabani, and W. Svejstrup, \"The year of quantum: From concept to reality "
                        "in 2025,\" McKinsey & Company, Rep., Jun. 23, 2025. [Online]. Available: https://www.mckinsey.com/"
                        f"capabilities/tech-and-ai/our-insights/the-year-of-quantum-from-concept-to-reality-in-2025 {ACCESSED}.",
        "ibm2025": "R. Mandelbaum *et al.*, \"How IBM will build the world's first large-scale, fault-tolerant quantum computer,\" "
                   "IBM Quantum Computing Blog, Jun. 10, 2025. [Online]. Available: https://www.ibm.com/quantum/blog/large-scale-ftqc "
                   f"{ACCESSED}.",
        "qiskitdocs": "IBM Quantum, \"Introduction to Qiskit and IBM Quantum,\" Qiskit documentation. [Online]. Available: "
                      f"https://quantum.cloud.ibm.com/docs/en/guides {ACCESSED}.",
        "qmlrepo": "Qiskit Machine Learning contributors, \"Qiskit Machine Learning,\" software repository and documentation "
                   f"links. [Online]. Available: https://github.com/stfc/qiskit-machine-learning {ACCESSED}.",
        "cirq": f"Google Quantum AI, \"Cirq,\" documentation. [Online]. Available: https://quantumai.google/cirq {ACCESSED}.",
    })
    return refs, reviewed
