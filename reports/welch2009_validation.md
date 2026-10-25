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
| src/features.py GC% + CAI | overall | 62 | -0.269 | 0.035 | 0.103 | 0.250 |
| src/features.py GC% + CAI | polymerase | 30 | -0.275 | 0.142 | 0.148 | 0.360 |
| src/features.py GC% + CAI | scFv | 32 | -0.166 | 0.363 | 0.145 | 0.351 |

### Trivial and reference baselines

| Baseline | Scope | n | Spearman rho | Note |
| :--- | :--- | ---: | ---: | :--- |
| Always predict majority tier (5) | overall | 62 | undefined | A constant predictor has zero variance, so rho does not exist. It cannot rank anything, which is the point of printing it. |
| GC% alone | overall | 62 | 0.719 | Raw descriptor. Pooled; see confound note. |
| GC% alone | polymerase | 30 | 0.531 | Raw descriptor, no model. |
| GC% alone | scFv | 32 | 0.380 | Raw descriptor, no model. |
| CAI alone | overall | 62 | -0.157 | Raw descriptor. Pooled; see confound note. |
| CAI alone | polymerase | 30 | -0.451 | Raw descriptor, no model. |
| CAI alone | scFv | 32 | -0.197 | Raw descriptor, no model. |
| GC% alone | pooled, within-series ranks | 62 | 0.449 | Confound removed: each variant ranked inside its own series first. |

**On the pooled GC% figure.** The two series differ in GC range (43.2-50.7% for polymerase versus 49.7-57.4% for scFv) and in yield scale (up to 68 versus 225 ug/ml), so a pooled correlation partly measures which gene a variant belongs to. The pooled 0.719 overstates the effect; the within-series figures (0.531 and 0.380) and the rank-normalised 0.449 are the honest ones. GC% carries real positive signal by any of them.

**CAI runs the wrong way in this dataset.** CAI correlates *negatively* with yield in both series. That is the paper's own headline result, not an artefact: the maximum-CAI scFv variant (`A_HiCAI`, CAI 0.883) yielded 20.8 ug/ml against a series best of 225.5. A model that treats rising CAI as a positive signal will rank this dataset backwards.

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
| 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| 1 | 1 | 0 | 2 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 0 |
| 2 | 2 | 0 | 1 | 0 | 0 | 0 | 1 | 3 | 1 | 1 | 1 |
| 3 | 6 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 3 | 2 |
| 4 | 3 | 0 | 2 | 0 | 0 | 0 | 1 | 0 | 1 | 2 | 0 |
| 5 | 6 | 6 | 1 | 5 | 5 | 5 | 4 | 1 | 3 | 0 | 3 |

Censored variants receive tiers [0, 1, 2, 3, 4, 5] (mean 3.42); measurable variants average 3.76.

## 4. Scale check: Welch features against the training distribution

If the Welch features fall outside the range the model was fitted on, the predictions are extrapolations. A gradient-boosted tree has no slope beyond its outermost split: it repeats the nearest leaf, producing a confident-looking tier with nothing behind it. Everything in section 2 is then uninterpretable.

Training ranges are the observed min and max in `data/processed/feature_table_v2.csv`.

| Feature | Welch min | Welch max | Training min | Training max | Outside |
| :--- | ---: | ---: | ---: | ---: | ---: |
| GC% (Welch's own) | 37.97 | 58.48 | 21.57 | 78.52 | 0/81 |
| GC% (src/features.py) | 37.97 | 58.48 | 21.57 | 78.52 | 0/81 |
| CAI (Welch's own) | 0.35 | 0.95 | 0.33 | 0.83 | 8/81 |
| CAI (src/features.py) | 0.41 | 0.87 | 0.33 | 0.83 | 6/81 |
| 5' folding, -42..+50 (Welch's own) | -17.79 | -6.04 | -60.52 | -25.74 | 81/81 |
| 5' folding, UTR junction (src/features.py) | -9.00 | -7.90 | -60.52 | -25.74 | 81/81 |
| gene-wide folding avg (Welch's own) | -12.21 | -6.76 | -24.36 | -2.77 | 0/81 |
| ORF-head folding proxy (src/features.py) | -35.58 | -12.58 | -24.36 | -2.77 | 40/81 |

**Out-of-distribution features flagged:**

- CAI (Welch's own): 8 of 81 outside [0.33, 0.83]
- CAI (src/features.py): 6 of 81 outside [0.33, 0.83]
- 5' folding, -42..+50 (Welch's own): 81 of 81 outside [-60.52, -25.74]
- 5' folding, UTR junction (src/features.py): 81 of 81 outside [-60.52, -25.74]
- ORF-head folding proxy (src/features.py): 40 of 81 outside [-24.36, -2.77]

The UTR-junction folding feature takes **2 distinct value(s)** across 81 variants, because 42 of its 48 proxy nucleotides come from the shared leader. A feature that barely varies cannot order variants, which is why the ORF-only folding proxy is kept alongside it.

The two features the artifact actually consumes, GC% and CAI, are **inside** the training range for 81 and 73 of 81 variants respectively. So the negative correlations in section 2 are not explained away as extrapolation: the model is being asked a question it was fitted for, and is getting the ordering wrong. The folding rows matter only for the four-feature model the metadata describes (see below), not for the artifact on disk.

### Artifact does not match its own metadata

`models/baseline_model_metadata.json` records four features (`GC_percent`, `CAI`, `head_folding_energy`, `genewide_folding_avg48`) and a sha256 for `data/processed/feature_table_v2.csv`. The artifact on disk takes **2**: [np.str_('GC_percent'), np.str_('CAI')]. The metadata's quoted feature ranges also disagree with that CSV (it gives `head_folding_energy` as -54.11 to -25.74 against the table's -60.52 to -25.74). The metadata describes a model that is not the one being served, so its reported 0.19 macro-F1 and its baselines do not characterise this file.

On the other hand `src/features.py` reproduces Welch's own descriptors well: computed GC% matches Welch's to within 0.01 points (rho = 1.00) and computed CAI ranks almost identically (rho = 0.99) despite using a different weight table and a lower absolute scale. The feature code is not the problem here.

## 5. Verdict

**The model fails this external validation.** It does not merely fail to beat a trivial baseline; it orders the Welch variants slightly backwards.

- Predicted tier against measured yield gives rho = -0.166 on Welch's own GC%/CAI and -0.269 on the project's recomputed ones. Both are negative. The -0.269 sits beyond its permutation null (95th percentile 0.250, p = 0.035), which makes it a reproducible *inversion* rather than noise.
- A single raw feature the model already consumes does better than the model: GC% alone reaches 0.531 within the polymerase series and 0.380 within scFv (0.449 rank-normalised across both). The signal is present in the inputs and the model destroys it.
- The likely mechanism is CAI. In Welch's data CAI is negatively associated with yield in both series, which is the paper's central finding; a model fitted where CAI reads as a positive signal will invert exactly this way.
- The censoring check is the most damaging. On Welch's own descriptors the 19 undetectable variants receive a mean predicted tier of 4.05 against 4.03 for the variants that did express, and 7 of them are assigned the top tier. The model cannot distinguish a construct that yields nothing from one that yields 225 ug/ml.
- This is not an out-of-distribution excuse. GC% is inside the training range for 81 of 81 variants and CAI for 73 of 81, so the model is operating where it was fitted.

What the exercise does establish is that the validation set itself is sound and reusable: 81 variants, 62 with a paired sequence and number, 19 informative left-censored cases, two independent genes, and the authors' own GC/CAI/folding descriptors for cross-checking. It is a usable external benchmark. The current artifact simply does not pass it, and the headline number to carry forward is the negative rho, not an accuracy figure.

Caveats worth stating. n = 62 measurable variants is small, so each rho carries roughly +/-0.25 of uncertainty; the two series are different proteins measured on different scales and have been treated separately wherever pooling would confound; and the tier-to-yield mapping is ordinal only, so nothing here speaks to calibration, only to ranking.
