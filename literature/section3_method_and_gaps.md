# Section 3: method note and research gaps (draft)

Numbers in brackets are the provisional review numbers in `literature.csv`. Final IEEE numbers are assigned in the report phase, in order of first citation.

## Method note (opening paragraph of Section 3)

Candidate papers were identified from the assignment's seed list, from knowledge of well-known work in each theme, and from web searches using keywords such as "quantum support vector machine breast cancer", "variational quantum classifier healthcare" and "quantum machine learning intrusion detection". Each candidate was then checked against its registered record: Crossref for journal papers, with a doi.org check that the DOI is registered, and the arXiv API for preprints. OpenAlex supplied retraction flags. Title, authors, year, venue and volume were taken from these records. Abstracts came from Crossref or OpenAlex, from Semantic Scholar, from an arXiv version of the same paper with the same title and first author, or, in one case, from the publisher's page. From 68 verified candidates, 45 were selected to meet the theme targets: 41 from 2021 onwards and 4 foundational papers from 2019 to 2020. Two are labelled preprints. Papers were included when they addressed a listed theme and an abstract could be read; relevance to breast cancer or tabular medical data was favoured. No selected journal paper carries a retraction flag in OpenAlex; preprints were not checked for this. All reviews are based on abstracts, not full texts, so their claims are kept general.

## Identified research gaps

These gaps describe the 45 reviewed papers as seen through their abstracts. They are not a claim about the whole field.

**G1. Few direct comparisons of quantum kernels and variational classifiers on the same medical dataset.** Medical studies mostly evaluate one quantum family: quantum kernels in [28], [30], [31] and [33], quantum or hybrid networks in [29] and [34]. Only [32] compares a VQC and a ZZ quantum kernel on WDBC. This project adds an independent comparison of the two on WDBC. *Links to objective O1.*

**G2. Limited fair comparison on identical reduced features.** Some positive reports use a same-structure baseline [24], train under similar conditions [25], use the authors' own data [45], or give no accuracy figure in the abstract [30]. The large benchmark in [6] found that classical models generally outperform quantum ones. This project gives every reduced-feature model the same PCA features. *Links to O2.*

**G3. Little joint analysis of how circuit design changes medical results.** Theory explains the effects of expressibility, entanglement and depth [8], [21], [22], and [11] tunes kernel bandwidth. Medical studies examine parts of the design: feature maps in [31] and embedding, depth and noise in [32]. No reviewed abstract varies qubit count, feature-map depth, entanglement, ansatz depth and training size together on one medical dataset. *Links to O3.*

**G4. Little study of noise effects on diagnostic recall.** Noise is studied for kernels in [10] and [13], for gradients in [21], and on hardware in [12]. In medical work, [32] tests depolarising noise on a VQC. No reviewed abstract reports malignant-class recall under a device noise model. *Links to O4.*

**G5. Diagnostic errors are rarely the headline result.** The abstracts of medical studies mostly report accuracy or AUC [29], [32], [33]. Missed cancers matter most clinically, but the reviewed abstracts do not report false-negative counts. This project reports them for every model. *Links to O2.*

**G6. Costs are rarely reported next to accuracy.** The quadratic cost of kernels is noted in [10] but left open, and speed claims can be hard to interpret [30], [45]. This project records training time, inference time, circuit counts and depth for each model. *Links to O5.*

## Proposed objectives for Section 4.3

- **O1.** Build QSVC and VQC pipelines in Qiskit and evaluate both on WDBC with the same preprocessing.
- **O2.** Compare both against classical models on identical reduced features using accuracy, precision, recall, specificity, F1, ROC-AUC and missed malignant cases, with 5-fold cross-validation.
- **O3.** Measure how qubit count, feature-map depth, entanglement, ansatz depth, optimiser and training size affect accuracy and cost.
- **O4.** Measure the effect of shot noise and a fake IBM device noise model on accuracy and recall.
- **O5.** Report the computational cost of each model and explain where quantum models stand today.
