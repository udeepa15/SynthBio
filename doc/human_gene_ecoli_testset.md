# Test set: human genes expressed in E. coli

Twelve human coding sequences fetched from NCBI on 2026-10-25 and scored with the
deployed artifact (`models/baseline_model.joblib`, XGBClassifier over GC% and
CAI). Seven are proteins routinely produced in *E. coli*, five are proteins that
are not. Sequences are in [`../data/testsets/human_genes_ecoli.fasta`](../data/testsets/human_genes_ecoli.fasta).

## Read this first: two things this table is not

**The outcome column is not a fetched measurement.** No public resource maps a
GenBank accession to an expression level, because expression is a property of a
construct in a host under an induction condition, not of a sequence. It is
published in each paper's own units in supplementary tables keyed to the
authors' variant names. The "documented outcome" column below is the
well-established production status of each protein, at the level found in any
biologics text: insulin and growth hormone are made in *E. coli*, erythropoietin
and factor VIII are not. It is a qualitative label, not a number, and it was not
verified here against a specific paper's table for each gene. For sequences with
real measured yields, use the Welch 2009 set in
`validation_set_known_expression.md`.

**None of these accessions can be pasted into the application's NCBI tab.** Every
RefSeq mRNA record includes the untranslated regions, so the fetched sequence
does not start at the ATG. `NM_000207` returns 465 nt against a 333 nt CDS and
fails with a missing start codon and internal stops; `NM_000605` returns 1144 nt
against 567 nt and fails on reading frame as well. Use the FASTA file above and
the Raw DNA tab.

## The table

Tier is the model's prediction, 0 (Very Low) to 5 (Very High). Confidence is its
own probability for that tier; with six tiers, chance is 17%. Driver is the
feature with the larger SHAP contribution, which can only ever be GC% or CAI
because those are the only two features the artifact uses.

| Accession | Gene | Documented in E. coli | CDS nt | Predicted tier | Conf. | Risk | GC% | CAI | Driver |
| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | :--- |
| NM_000605 | IFNA2, interferon alpha-2 | produced | 567 | 4 High | 66% | 56.6% | 48.9 | 0.426 | CAI |
| NM_000586 | IL2, interleukin-2 | produced | 462 | 4 High | 28% | 40.0% | 38.5 | 0.365 | CAI |
| NM_000454 | SOD1, superoxide dismutase | produced | 465 | 4 High | 28% | 71.9% | 49.5 | 0.470 | CAI |
| NM_000207 | INS, insulin | produced | 333 | 2 Medium-Low | 43% | 83.8% | 64.6 | 0.511 | GC% |
| NM_000515 | GH1, growth hormone | produced | 654 | 2 Medium-Low | 47% | 65.4% | 56.1 | 0.455 | CAI |
| NM_000759 | CSF3, G-CSF | produced | 624 | 2 Medium-Low | 42% | 71.9% | 65.1 | 0.474 | CAI |
| NM_021009 | UBC, ubiquitin | produced | 2058 | **1 Low** | 26% | 37.1% | 52.8 | 0.530 | CAI |
| NM_000132 | F8, factor VIII | **not produced** | 7056 | **4 High** | 31% | 38.7% | 42.6 | 0.400 | CAI |
| NM_000930 | PLAT, tPA | not produced | 1689 | 2 Medium-Low | 52% | 48.7% | 58.4 | 0.445 | GC% |
| NM_000799 | EPO, erythropoietin | not produced | 582 | 2 Medium-Low | 39% | 65.5% | 60.3 | 0.402 | CAI |
| NM_000546 | TP53, p53 | not produced | 1182 | 2 Medium-Low | 39% | 62.9% | 56.9 | 0.438 | CAI |
| NM_000492 | CFTR | not produced | 4443 | 0 Very Low | 27% | 45.9% | 41.5 | 0.383 | CAI |

## What the model got right and wrong

| | Mean predicted tier |
|---|---|
| Produced in E. coli (n=7) | 2.71 |
| Not produced (n=5) | 2.00 |

Ranking a produced protein above a non-produced one happens 63% of the time
(chance is 50%). The direction is right and the margin is small, which is
consistent with the 0.626 AUC measured on the Boël data and the negative
correlation found on Welch's.

The two outright misses are the informative ones:

- **Factor VIII scores tier 4, "High".** It is a 7,056 nt, heavily glycosylated,
  multi-domain protein that cannot be made in *E. coli* at all. Nothing in GC% or
  CAI reflects that, and the model has no notion of length.
- **Ubiquitin scores tier 1, "Low".** It expresses superbly in *E. coli* and is a
  standard fusion tag. Its CAI of 0.530 is the highest in the set, yet the model
  placed it near the bottom.

Mean confidence across the set is 39%.

## Why "which feature caused low expression" has a limited answer

The driver column is honest about what the model used, and that is the problem
rather than the answer. The actual reasons these five proteins are not made in
*E. coli* are:

| Gene | Real reason | Visible to the model? |
| :--- | :--- | :--- |
| EPO | Requires N- and O-glycosylation for activity and half-life | No |
| PLAT | 17 disulfide bonds plus glycosylation | No |
| F8 | 2,351 residues, multi-domain, glycosylated | No |
| CFTR | Polytopic membrane protein, needs a lipid bilayer | No |
| TP53 | Aggregation-prone, poorly soluble when overexpressed | Partly, via GRAVY, which the model does not use |

None of these are codon-usage or GC problems, so the model cannot in principle
attribute them correctly. The biological diagnostics in the application, which
read GRAVY and pI, are the part of the interface that can speak to aggregation
and inclusion bodies, and they run independently of the model.

A sequence-feature model of this kind answers "is this construct's codon usage
and base composition unfavourable", which is a narrower question than "will this
protein be made". The test set above is a fair way to keep that distinction
visible.

## Reproducing

The fetch-and-score script is not committed; the sequences it produced are. To
rescore after retraining, paste each record from the FASTA into the Raw DNA tab,
or load the file and call `predict_construct` directly.
