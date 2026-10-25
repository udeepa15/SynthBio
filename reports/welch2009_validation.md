# External validation against Welch et al. 2009

Welch M, Govindarajan S, Ness JE, Villalobos A, Gurney A, Minshull J, Gustafsson C. *Design parameters to control synthetic gene expression in Escherichia coli.* PLoS ONE 4(9):e7002. doi:10.1371/journal.pone.0007002

Artifact under test: `models/baseline_model.joblib`, an XGBClassifier over [np.str_('GC_percent'), np.str_('CAI')], predicting tiers [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4), np.int64(5)].

## 1. What the sheet actually contains

`Supporting Table 1` holds **81 variants** in two series: **39 polymerase** (Phi29 DNA polymerase, 575 codons) and **42 scFv** (single-chain antibody fragment, 281 codons).

**62 variants carry both a sequence and a number.** The remaining **19** are left-censored: the sheet records the literal string `nd` for both expression columns, against a stated detection limit of ~5 ug/ml at A600 = 3.

Deviations from what was expected going in:

- **The absolute-expression column is headed `Absolute Expression, mg/ml`, but the values are ug/ml.** The series maxima are 225 and 68; 225 mg/ml of recombinant protein per ml of culture is not physically plausible, and the two lowest measured values, 5.34 and 5.91, sit just above the paper's ~5 ug/ml detection limit while every `nd` falls below it. The header unit label is wrong; the column is read as ug/ml.
- **The 5' UTR is not per-variant.** It is a single 71 nt leader given once, in a trailing row labelled `5'-term RNA`, and shared by all 81 constructs. Only the coding sequence varies between variants, so no UTR-derived feature can rank them.
- **The ORFs carry no stop codon** (843 nt = 281 codons; 1725 nt = 575 codons exactly). The stops sit in two further trailing rows, `3'-term polymerase` and `3'-term scFv`.
- **The sheet already publishes GC%, CAI and four real folding energies** (`RNA MIN_ORF`, `RNA Avg_ORF`, `RNA Structure -42 to +50`, `RNA Structure -4 to +38`), which were not anticipated. They let the model be scored on Welch's own descriptors as well as on this project's recomputed ones, and both are reported below.
- **Relative expression is normalised to the best variant in each series = 3**, as expected (`A1_14` and `P19` both sit at exactly 3.0).

## 2. Rank correlation against measured yield

Spearman rho between predicted tier and measured ug/ml, over the 62 measurable variants. Accuracy is not reported: the tiers are Boel's relative 0-5 bins from a different assay, so only ordering is comparable.

| Scoring | Scope | n | Spearman rho | p | Permutation null mean &#124;rho&#124; | Null 95th pct |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Welch's own GC% + CAI | overall | 62 | -0.166 | 0.196 | 0.102 | 0.254 |
| Welch's own GC% + CAI | polymerase | 30 | -0.125 | 0.511 | 0.150 | 0.366 |
| Welch's own GC% + CAI | scFv | 32 | -0.028 | 0.878 | 0.145 | 0.349 |
| src/features.py GC% + CAI | overall | 62 | -0.289 | 0.023 | 0.102 | 0.249 |
| src/features.py GC% + CAI | polymerase | 30 | -0.370 | 0.044 | 0.148 | 0.362 |
| src/features.py GC% + CAI | scFv | 32 | -0.128 | 0.484 | 0.145 | 0.355 |

### Trivial and reference baselines

| Baseline | Scope | n | Spearman rho | Note |
| :--- | :--- | ---: | ---: | :--- |
| Always predict majority tier (5) | overall | 62 | undefined | A constant predictor has zero variance, so rho does not exist. It cannot rank anything, which is the point of printing it. |
| Welch GC% alone | overall | 62 | 0.719 | Single published descriptor, no model. |
| Welch CAI alone | overall | 62 | -0.157 | Single published descriptor, no model. |
| Welch CAI alone | polymerase | 30 | -0.451 | Single published descriptor, no model. |
| Welch CAI alone | scFv | 32 | -0.197 | Single published descriptor, no model. |

## 3. Do the undetectable variants receive low tiers?

Predicted tier against measured decile, over all 81 variants. The 19 censored variants form their own column rather than being forced into the bottom decile.

**Welch's own GC% + CAI**

| predicted tier | censored | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0 | 2 | 1 | 0 | 1 | 1 | 0 | 0 | 1 | 0 |
| 2 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 2 | 1 | 1 | 2 |
| 3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 | 9 | 0 | 2 | 0 | 1 | 0 | 2 | 1 | 1 | 3 | 2 |
| 5 | 7 | 6 | 1 | 5 | 5 | 5 | 3 | 3 | 4 | 1 | 3 |

Censored variants receive tiers [1, 2, 3, 4, 5] (mean 4.05); measurable variants average 4.03.

**src/features.py GC% + CAI**

| predicted tier | censored | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| 1 | 2 | 0 | 2 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 0 |
| 2 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 2 | 1 | 0 | 1 |
| 3 | 5 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 2 | 2 |
| 4 | 3 | 0 | 2 | 0 | 1 | 0 | 2 | 1 | 1 | 2 | 1 |
| 5 | 5 | 6 | 1 | 5 | 4 | 4 | 3 | 2 | 2 | 1 | 2 |

Censored variants receive tiers [0, 1, 2, 3, 4, 5] (mean 2.95); measurable variants average 3.71.

## 4. Scale check: Welch features against the training distribution

If the Welch features fall outside the range the model was fitted on, the predictions are extrapolations. A gradient-boosted tree has no slope beyond its outermost split: it repeats the nearest leaf, producing a confident-looking tier with nothing behind it. Everything in section 2 is then uninterpretable.

Training ranges are the observed min and max in `data/processed/feature_table_v2.csv`.

| Feature | Welch min | Welch max | Training min | Training max | Outside |
| :--- | ---: | ---: | ---: | ---: | ---: |
| GC% (Welch's own) | 37.97 | 58.48 | 21.57 | 78.52 | 0/81 |
| GC% (src/features.py) | 37.97 | 58.48 | 21.57 | 78.52 | 0/81 |
| CAI (Welch's own) | 0.35 | 0.95 | 0.33 | 0.83 | 8/81 |
| CAI (src/features.py) | 0.40 | 0.87 | 0.33 | 0.83 | 7/81 |
| 5' folding, -42..+50 (Welch's own) | -17.79 | -6.04 | -60.52 | -25.74 | 81/81 |
| 5' folding, UTR junction (src/features.py) | -9.00 | -7.90 | -60.52 | -25.74 | 81/81 |
| gene-wide folding avg (Welch's own) | -12.21 | -6.76 | -24.36 | -2.77 | 0/81 |
| ORF-head folding proxy (src/features.py) | -35.58 | -12.58 | -24.36 | -2.77 | 40/81 |

**Out-of-distribution features flagged:**

- CAI (Welch's own): 8 of 81 outside [0.33, 0.83]
- CAI (src/features.py): 7 of 81 outside [0.33, 0.83]
- 5' folding, -42..+50 (Welch's own): 81 of 81 outside [-60.52, -25.74]
- 5' folding, UTR junction (src/features.py): 81 of 81 outside [-60.52, -25.74]
- ORF-head folding proxy (src/features.py): 40 of 81 outside [-24.36, -2.77]

The UTR-junction folding feature takes **2 distinct value(s)** across 81 variants, because 42 of its 48 proxy nucleotides come from the shared leader. A feature that barely varies cannot order variants, which is why the ORF-only folding proxy is kept alongside it.
