# Improving SynthBio's Accuracy

A practical list of what stands between the current prototype and a tool whose predictions can be trusted. Ordered so the earlier items unblock the later ones.

## Progress

| Step | State |
|---|---|
| 1. Align the app's features with the training data | **Largely done** by the v2 table. Its CAI uses external K-12 weights rather than weights derived from the top-expressing genes, which removes a label leak, and GFP's computed CAI of 0.443 now falls inside the table's 0.33–0.82 range. Two real bugs in `calculate_cai` were fixed alongside: the serine family was not normalised to a peak of 1.0, and stop codons were entering the geometric mean. The folding columns remain unservable, see below. |
| 2. Confirm the target | **Partly done.** `doc/data.md` now describes the six tiers and their counts. How they were derived from the original measurements is still unknown, and the evidence below says the six-way split is the wrong question regardless. |
| 3. Add features with real signal | **Partly done.** v2 replaced the GC-derived folding proxy with Boël's measured energies and separated head folding (`Gz48v`) from the gene-wide average (`Gave48`). Head folding is the strongest single feature available, and the one the application cannot compute. |
| 4. Validate honestly | **Done.** `src/pipeline.py` now holds organisms out of each fold, keeps the test set for a single final scoring, and prints the trivial baselines beside the model. |
| 5. Calibration and scope | **Partly done.** Predictions now carry the model's own probability, constructs outside the training range are flagged in the app, and known-answer tests cover GFP. A reliability curve is still to do. |
| 6. Model refinements | Not started, deliberately. |
| 7. Production engineering | **Partly done.** Dependencies are pinned, and the pipeline writes a metadata sidecar recording the data hash, feature ranges, library versions and scores. Prediction logging is still to do. |

## Where it stands today

Measured on the v2 table, organisms held out of every fold, 6,348 genes:

| Feature set | 6-tier macro F1 | Binary AUC | Top-10% enrichment |
|---|---|---|---|
| Chance / majority | 0.079 | 0.500 | 49.5% |
| v1 (GC, leaky CAI, `Gave48`) | 0.161 | 0.600 | 64.5% |
| **v2, all four features** | **0.170** | **0.626** | **66.2%** |
| v2, head folding alone | 0.147 | 0.602 | 53.5% |
| v2, GC + CAI (what the app can compute) | 0.168 | 0.596 | 64.2% |

*Binary AUC* is "expresses at all" versus tier 0. *Top-10% enrichment* is the
share of genuinely high expressers (tier 4–5) among the constructs a regression
model ranks in its top 10%, against a 49.5% base rate.

Three conclusions follow, and they redirect the rest of this document:

1. **v2 is better than v1 on every framing**, so the label leak and the folding
   column were both real problems and both are now fixed.
2. **Six-tier classification is the wrong question.** It is the weakest framing in
   the table, and the served model's macro F1 of 0.169 does not beat guessing by
   class frequency (0.181). The pipeline now warns about exactly this.
3. **Ranking works where labelling does not.** A 1.34x enrichment in the top
   decile is a modest but genuine screening signal, and it is the claim the tool
   can actually support. Reporting a tier per construct is not.

The remaining structural problem is that the strongest feature cannot be served.
`head_folding_energy` is an RNAstructure energy over the vector 5' UTR plus the
first 48 nt, so it depends on the expression vector, not on the construct alone.
A model fitted on it loads in the app and then never runs, leaving the fixed
rules to answer while the interface still credits a trained model. `run_pipeline`
therefore trains on servable features by default and says which it dropped; pass
`servable_only=False` for a research run.

---

## 0. Serve the head-folding feature, or reframe the product

**Why this is now first:** the measurements above say the gain from the best
feature set over what the app can compute is real but small (AUC 0.626 against
0.596), and that no feature set supports a six-way tier label. Two choices
follow, and they are not both worth doing.

**Either** make head folding servable. That means computing an RNAstructure or
ViennaRNA energy over the vector 5' UTR plus the first 48 nt of the pasted
construct, which requires the user to say which expression vector they are using
and a stored UTR sequence for each supported vector. This is the honest version
of the current feature, and it is a real piece of work.

**Or** drop it and reframe. Keep GC and CAI, stop reporting a tier per
construct, and present the tool as what the numbers support: a ranking that
enriches for high expressers when screening a candidate set. Report the
enrichment, not an accuracy.

The second is cheaper and defensible today. The first is what raises the
ceiling. Pick one before investing further in the model.

---

## 1. Make the app's features match the training data

**Why first:** the model is currently fed values outside the range it trained on. In the CSV, CAI spans 0.61–0.92; the app computed 0.45 for GFP. Folding values in the app run roughly twice as negative as the CSV's for the same GC content. Nothing downstream can be evaluated while this holds.

**What to do:**
- Recover the script, notebook, or tool that produced `feature_table.csv`. Failing that, recover the raw sequences, and recompute every feature with `src/features.py` so training and inference share one code path.
- Add a test that computes features for a handful of genes in the CSV from their sequences and asserts the values match the stored columns.
- Add a guard at prediction time that compares each incoming feature against the training ranges, and surfaces a warning when a construct falls outside them.

**Done when:** feature values computed by the app for a training gene reproduce its CSV row.

---

## 2. Check that the target is what you think it is

**Why:** `doc/data.md` describes `expression_score` as three tiers (0=Low, 1=Medium, 2=High), while the CSV holds six (0–5). Tier 1 has 131 genes against tier 0's 1,754. Before investing in modelling, confirm what the six values mean and how they were derived from the underlying measurements.

**What to do:**
- Document the exact mapping from the original Boël et al. measurements to 0–5, and correct `doc/data.md`.
- Decide whether six tiers is the right question. Expression is a continuous quantity cut into bins; cutting it finely than the data supports creates classes the features cannot separate.
- Consider the alternatives: regression on the underlying continuous value, ordinal classification (which respects that tier 4 is closer to 5 than to 0), or a binary "expresses / does not express" split. The binary form already scores better than the six-way one (AUC 0.61 vs near-chance), which suggests the signal is there but coarse.

**Done when:** the target's provenance is written down, and the prediction task is chosen deliberately rather than inherited.

---

## 3. Add features that carry real signal

**Why:** this is the largest available gain. GC content, CAI and a GC-derived folding proxy are three views of roughly the same thing — GC and folding correlate at −0.96 in your data, so they are nearly the same feature twice. The literature on *E. coli* expression points to several stronger predictors.

**What to add, roughly in order of expected value:**

| Feature | What it captures | How |
|---|---|---|
| True 5' mRNA folding energy | Secondary structure blocking ribosome entry | ViennaRNA (`RNAfold`) over the RBS and first ~40 codons, replacing the GC proxy |
| Ribosome binding site strength | Translation initiation rate | The RBS Calculator, or a local implementation |
| Codon usage beyond CAI | Rare-codon runs, %MinMax, tAI | Computable from the sequence with a tRNA abundance table |
| Protein length | Strongly associated with expression success | Trivially available |
| Rare-codon clusters | Ribosome stalling | Sliding window over codon frequencies |
| GRAVY and pI | Aggregation and inclusion bodies | Already computed in the app, just not used by the model |
| Disorder, transmembrane segments | Toxicity, misfolding | IUPred, TMHMM, or similar |

Two of these — GRAVY and pI — you already compute and discard. Including them costs nothing beyond regenerating the feature table.

**Done when:** the feature table is regenerated with the expanded set, and each feature's contribution is measured rather than assumed.

---

## 4. Validate honestly

**Why:** the repo's own pipeline uses a plain random split, which lets genes from the same organism land on both sides. Since GC content is strongly organism-specific, that inflates scores without improving real predictions. The tuning I ran already holds out whole organisms; the committed pipeline does not.

**What to do:**
- Move the organism-grouped cross-validation into `src/pipeline.py` so every training run uses it.
- Keep a test set that is touched exactly once, at the end. Tuning against the test set is the most common way a model looks better than it is.
- Report macro-F1 and balanced accuracy alongside a stated baseline, not accuracy alone. With this class distribution, accuracy rewards guessing tier 5.
- If the goal is ranking candidates rather than labelling them, measure that directly: precision among the top N predictions is what a user screening constructs actually cares about.

**Done when:** one command reproduces the reported numbers, and the numbers are quoted next to the baseline they beat.

---

## 5. Calibrate, and say what the model does not know

**Why:** a tier shown without any indication of confidence invites the user to trust a coin flip. The GFP case — a protein known to express well in *E. coli*, predicted as Tier 0 — is exactly what an uncalibrated model looks like in use.

**What to do:**
- Calibrate the predicted probabilities (`CalibratedClassifierCV`, or isotonic regression on a validation split) and check them with a reliability curve.
- Show the predicted probability, or an interval, next to the tier.
- Withhold the prediction, or mark it as out of scope, when a construct's features fall outside the training distribution.
- Assemble a small set of constructs with known outcomes — GFP, mCherry, a few proteins known to form inclusion bodies — and keep them as a sanity check that runs on every model change. A model that fails on GFP is not ready regardless of its macro-F1.

**Done when:** every prediction carries a confidence, and known-answer cases are part of the test suite.

---

## 6. Then, and only then, improve the model itself

**Why last:** algorithm choice is worth a few points of F1. Features and a correct target are worth considerably more. With three weakly-informative features, XGBoost and LightGBM landed within 0.006 of each other, which is what a feature ceiling looks like.

**What to try once the inputs are sound:**
- Ordinal models, which respect the ordering of the tiers.
- Monotonic constraints, where biology dictates a direction (very low CAI should not increase predicted expression).
- An ensemble over several seeds, to reduce variance on the rare tiers.
- Per-organism or per-host models, if enough data exists per group.

---

## 7. Production engineering

Separate from accuracy, but needed before anyone relies on the output:

- **Version the artifacts.** Record with each model: training data hash, feature list, library versions, metrics, and the date. `src/provenance.py` already reports some of this; make it complete and enforced.
- **Pin dependencies.** `requirements.txt` uses `>=` throughout, so a fresh install can silently load a different scikit-learn or XGBoost than the model was built with.
- **Commit the training command.** `src/pipeline.py` trains with untuned defaults, which do not match the saved model. One command should reproduce the deployed artifact.
- **Keep a holdout the model never sees across retrains**, so successive versions remain comparable.
- **Log predictions** with their inputs, so real usage can be audited later against outcomes.
- **State the model's scope** in the UI: which organisms and sequence types it was trained on, and that it screens candidates rather than predicting outcomes.

---

## Honest expectation

Steps 1 and 2 are correctness work and should be done regardless. Step 3 is where accuracy actually comes from. Published models for *E. coli* expression, using RBS strength and true folding energies, reach useful but not decisive performance — this is a hard biological problem, and a well-built version of this tool will rank candidates for prioritisation rather than predict outcomes. That is still valuable; it is just a different claim than the current UI makes.

A reasonable sequence: fix the feature mismatch, confirm the target, regenerate the table with ViennaRNA folding plus the protein features you already compute, re-measure against the baseline, and only then tune. If accuracy remains near chance with a strong feature set, the limit is the data, and the answer is more or better-labelled constructs rather than more modelling.
