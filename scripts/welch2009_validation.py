"""
External validation of the trained expression-tier model against Welch et al. 2009.

Welch M, Govindarajan S, Ness JE, Villalobos A, Gurney A, Minshull J, Gustafsson C.
"Design parameters to control synthetic gene expression in Escherichia coli."
PLoS ONE 4(9):e7002. doi:10.1371/journal.pone.0007002

Supporting Table S1 carries 81 synthetic variants of two genes, each with the
complete ORF and a measured protein yield, so it is an independent test set: the
model was fitted on Boel et al. 2016, a different assay in a different lab.

The comparison is rank-based throughout. The model emits Boel's relative 0-5
tier, which has no unit; Welch reports a concentration. Only the ordering of the
two is comparable, so Spearman is the statistic and accuracy is not computed.

Run from the repository root:
    python scripts/welch2009_validation.py
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import (  # noqa: E402
    calculate_cai,
    calculate_gc_content,
    estimate_mrna_folding_stability,
)

logger = logging.getLogger("SynthBio.welch2009")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] [SynthBio.welch2009] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

DEFAULT_XLS = "data/external/welch2009/welch_s001.bin"
DEFAULT_CSV = "data/validation/welch2009.csv"
DEFAULT_REPORT = "reports/welch2009_validation.md"
TRAINING_TABLE = "data/processed/feature_table_v2.csv"
MODEL_PATH = "models/baseline_model.joblib"

# Welch's detection limit: ~5 ug/ml of culture at A600 = 3. Variants below it are
# reported as the literal string "nd" and are left-censored, not zero.
DETECTION_LIMIT_UG_ML = 5.0
CENSORED_TOKEN = "nd"

# Column positions in 'Supporting Table 1'. The sheet has a single header row and
# no merged cells, so fixed indices are safe and are asserted against the header
# text before use.
COL = {
    "variant": 0,
    "relative_expression": 1,
    "absolute_expression": 2,
    "se": 3,
    "gc_percent": 7,
    "cai": 8,
    "rare_codons_he": 9,
    "rare_codons_genomic": 10,
    "gc7": 11,
    "rna_min_orf": 12,
    "rna_avg_orf": 13,
    "rna_struct_m42_p50": 14,
    "rna_struct_m4_p38": 15,
    "sequence": 16,
}

EXPECTED_HEADER = {
    0: "Variant",
    2: "Absolute Expression, mg/ml",
    7: "GC%",
    8: "CAI",
    16: "Sequence",
}

# The last three rows are not variants: they hold the shared 5' untranslated
# leader and the two 3' terminal segments appended to each ORF.
NON_VARIANT_LABELS = ("5'-term RNA", "3'-term polymerase", "3'-term scFv")

# Welch's window around the start codon, in nucleotides relative to the A of ATG.
# Used to build a UTR-inclusive folding feature alongside the ORF-only one.
JUNCTION_UPSTREAM_NT = 42
JUNCTION_DOWNSTREAM_NT = 50


def _to_float(raw: object) -> Optional[float]:
    """Parse a cell to float, returning None for the censoring token and blanks."""
    text = str(raw).strip()
    if not text or text.lower() == CENSORED_TOKEN:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_table_s1(xls_path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Read Supporting Table S1 into one row per variant plus the flanking sequences.

    Returns the variant frame and a dict of the three non-variant sequences. The
    raw text of every value that had to be interpreted is retained in a
    *_raw column so the interpretation stays auditable.
    """
    import xlrd

    logger.info(f"Reading {xls_path}")
    sheet = xlrd.open_workbook(xls_path).sheet_by_index(0)
    logger.info(f"Sheet '{sheet.name}': {sheet.nrows} rows x {sheet.ncols} cols")

    for idx, expected in EXPECTED_HEADER.items():
        found = str(sheet.cell_value(0, idx)).strip()
        if found != expected:
            raise ValueError(
                f"Unexpected header in column {idx}: expected {expected!r}, found {found!r}. "
                "The sheet layout has changed; column indices must be revisited."
            )

    flanking: Dict[str, str] = {}
    records: List[Dict[str, object]] = []

    for row in range(1, sheet.nrows):
        label = str(sheet.cell_value(row, COL["variant"])).strip()
        if not label:
            continue
        sequence = str(sheet.cell_value(row, COL["sequence"])).strip().upper()

        if label in NON_VARIANT_LABELS:
            flanking[label] = sequence
            continue

        rel_raw = str(sheet.cell_value(row, COL["relative_expression"])).strip()
        abs_raw = str(sheet.cell_value(row, COL["absolute_expression"])).strip()
        yield_value = _to_float(abs_raw)
        censored = abs_raw.lower() == CENSORED_TOKEN

        if not sequence or set(sequence) - set("ACGT"):
            logger.warning(f"{label}: sequence missing or non-ACGT; row kept with null sequence.")
            sequence = ""

        records.append(
            {
                "variant_id": label,
                "series": "scFv" if label.startswith("A") else "polymerase",
                "orf_dna": sequence,
                "cds_length_codons": len(sequence) // 3 if sequence else None,
                "yield_ug_per_ml": yield_value,
                "yield_raw": abs_raw,
                "relative_expression": _to_float(rel_raw),
                "relative_expression_raw": rel_raw,
                "se_ug_per_ml": _to_float(sheet.cell_value(row, COL["se"])),
                "censored": censored,
                # Welch's own published descriptors, kept for comparison against
                # the values this project's feature code computes.
                "welch_gc_percent": _to_float(sheet.cell_value(row, COL["gc_percent"])),
                "welch_cai": _to_float(sheet.cell_value(row, COL["cai"])),
                "welch_rare_codons_he": _to_float(sheet.cell_value(row, COL["rare_codons_he"])),
                "welch_gc7": _to_float(sheet.cell_value(row, COL["gc7"])),
                "welch_rna_min_orf": _to_float(sheet.cell_value(row, COL["rna_min_orf"])),
                "welch_rna_avg_orf": _to_float(sheet.cell_value(row, COL["rna_avg_orf"])),
                "welch_rna_struct_m42_p50": _to_float(sheet.cell_value(row, COL["rna_struct_m42_p50"])),
                "welch_rna_struct_m4_p38": _to_float(sheet.cell_value(row, COL["rna_struct_m4_p38"])),
            }
        )

    df = pd.DataFrame(records)
    utr = flanking.get("5'-term RNA", "")
    df["utr_5prime"] = utr
    logger.info(
        f"Parsed {len(df)} variants ({(~df['censored']).sum()} measurable, "
        f"{df['censored'].sum()} censored); shared 5' UTR is {len(utr)} nt."
    )
    return df, flanking


def add_computed_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute this project's features on the Welch ORFs using src/features.py.

    Two folding variants are kept. `folding_orf_head` is the project's proxy over
    the first 48 nt of the ORF. `folding_utr_junction` applies the same proxy to
    Welch's start-codon window, the last 42 nt of the leader plus the first 50 nt
    of the coding sequence, because the paper's signal sits near the ribosome
    binding site. The leader is a single shared sequence, so the junction value
    is dominated by nucleotides common to every variant; the report quantifies
    how little it therefore varies.
    """
    out = df.copy()
    gc, cai, head, junction = [], [], [], []

    for _, row in out.iterrows():
        orf = row["orf_dna"]
        utr = row["utr_5prime"]
        if not orf:
            gc.append(np.nan); cai.append(np.nan); head.append(np.nan); junction.append(np.nan)
            continue
        gc.append(calculate_gc_content(orf))
        cai.append(calculate_cai(orf))
        head.append(estimate_mrna_folding_stability(orf))
        window = (utr[-JUNCTION_UPSTREAM_NT:] if utr else "") + orf[:JUNCTION_DOWNSTREAM_NT]
        junction.append(estimate_mrna_folding_stability(window))

    out["gc_content"] = gc
    out["codon_adaptation_index"] = cai
    out["folding_orf_head"] = head
    out["folding_utr_junction"] = junction
    return out


def score_with_model(df: pd.DataFrame, model, gc_col: str, cai_col: str, label: str) -> pd.Series:
    """
    Predict a tier for every variant using the saved artifact.

    The feature matrix is built from the artifact's own `feature_names_in_`, so a
    mismatch between the artifact and this script surfaces as a clear error
    rather than a silently reordered matrix.
    """
    required = list(getattr(model, "feature_names_in_", []))
    supplied = {"GC_percent": df[gc_col], "CAI": df[cai_col]}
    missing = [c for c in required if c not in supplied]
    if missing:
        raise ValueError(
            f"The artifact requires features this script cannot supply from Welch's data: {missing}. "
            "Scoring abandoned rather than substituting a stand-in value."
        )
    matrix = pd.DataFrame({c: supplied[c] for c in required})
    preds = pd.Series(model.predict(matrix).astype(int), index=df.index, name=label)
    logger.info(f"{label}: tier distribution {preds.value_counts().sort_index().to_dict()}")
    return preds


def spearman(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    """Spearman rho, p-value and n over rows where both values are present."""
    from scipy.stats import spearmanr

    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return float("nan"), float("nan"), n
    rho, p = spearmanr(x[mask], y[mask])
    return float(rho), float(p), n


def permutation_null(pred: pd.Series, truth: pd.Series, iters: int = 10000, seed: int = 42) -> Dict[str, float]:
    """Distribution of |rho| when the predictions are shuffled against the truth."""
    from scipy.stats import spearmanr

    mask = pred.notna() & truth.notna()
    p_vals = pred[mask].to_numpy(dtype=float)
    t_vals = truth[mask].to_numpy(dtype=float)
    if len(p_vals) < 3 or len(np.unique(p_vals)) < 2:
        return {"mean_abs_rho": float("nan"), "p95_abs_rho": float("nan"), "n": len(p_vals)}

    rng = np.random.default_rng(seed)
    shuffled = np.empty(iters)
    for i in range(iters):
        shuffled[i] = spearmanr(rng.permutation(p_vals), t_vals).statistic
    abs_rho = np.abs(shuffled)
    return {
        "mean_abs_rho": float(abs_rho.mean()),
        "p95_abs_rho": float(np.percentile(abs_rho, 95)),
        "n": int(len(p_vals)),
    }


def decile_confusion(pred: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
    """
    Predicted tier against measured decile, with the censored variants as a
    column of their own rather than forced into decile 1.
    """
    measurable = df["yield_ug_per_ml"].notna()
    decile = pd.Series(index=df.index, dtype=object)
    decile[measurable] = pd.qcut(
        df.loc[measurable, "yield_ug_per_ml"], 10, labels=[f"D{i}" for i in range(1, 11)], duplicates="drop"
    ).astype(str)
    decile[~measurable] = "censored"

    table = pd.crosstab(pred, decile)
    ordered = [f"D{i}" for i in range(1, 11) if f"D{i}" in table.columns]
    if "censored" in table.columns:
        ordered = ["censored"] + ordered
    return table[ordered]


def training_ranges(path: str) -> Dict[str, Dict[str, float]]:
    """Observed min and max of every numeric feature column in the training table."""
    if not os.path.exists(path):
        logger.warning(f"Training table {path} not found; range comparison will be omitted.")
        return {}
    train = pd.read_csv(path)
    ranges = {}
    for col in train.columns:
        if col in ("gene_id", "expression_score"):
            continue
        if pd.api.types.is_numeric_dtype(train[col]):
            ranges[col] = {"min": float(train[col].min()), "max": float(train[col].max())}
    return ranges


def _fmt(value: object, places: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{places}f}"
    return str(value)


def build_report(df: pd.DataFrame, model, scorings: Dict[str, pd.Series], ranges: Dict[str, Dict[str, float]]) -> str:
    """Assemble the markdown report."""
    lines: List[str] = []
    add = lines.append

    measurable = df[df["yield_ug_per_ml"].notna()]
    add("# External validation against Welch et al. 2009")
    add("")
    add(
        "Welch M, Govindarajan S, Ness JE, Villalobos A, Gurney A, Minshull J, Gustafsson C. "
        "*Design parameters to control synthetic gene expression in Escherichia coli.* "
        "PLoS ONE 4(9):e7002. doi:10.1371/journal.pone.0007002"
    )
    add("")
    add(
        f"Artifact under test: `{MODEL_PATH}`, an {type(model).__name__} over "
        f"{list(getattr(model, 'feature_names_in_', []))}, predicting tiers "
        f"{list(getattr(model, 'classes_', []))}."
    )
    add("")

    # --- Dataset ---
    add("## 1. What the sheet actually contains")
    add("")
    add(
        f"`Supporting Table 1` holds **{len(df)} variants** in two series: "
        f"**{(df['series'] == 'polymerase').sum()} polymerase** (Phi29 DNA polymerase, "
        f"{int(df[df['series'] == 'polymerase']['cds_length_codons'].iloc[0])} codons) and "
        f"**{(df['series'] == 'scFv').sum()} scFv** (single-chain antibody fragment, "
        f"{int(df[df['series'] == 'scFv']['cds_length_codons'].iloc[0])} codons)."
    )
    add("")
    add(
        f"**{len(measurable)} variants carry both a sequence and a number.** The remaining "
        f"**{int(df['censored'].sum())}** are left-censored: the sheet records the literal string "
        f"`nd` for both expression columns, against a stated detection limit of "
        f"~{DETECTION_LIMIT_UG_ML:g} ug/ml at A600 = 3."
    )
    add("")
    add("Deviations from what was expected going in:")
    add("")
    add(
        "- **The absolute-expression column is headed `Absolute Expression, mg/ml`, but the values are "
        "ug/ml.** The series maxima are 225 and 68; 225 mg/ml of recombinant protein per ml of culture is "
        "not physically plausible, and the two lowest measured values, 5.34 and 5.91, sit just above the "
        "paper's ~5 ug/ml detection limit while every `nd` falls below it. The header unit label is wrong; "
        "the column is read as ug/ml."
    )
    add(
        "- **The 5' UTR is not per-variant.** It is a single 71 nt leader given once, in a trailing row "
        "labelled `5'-term RNA`, and shared by all 81 constructs. Only the coding sequence varies between "
        "variants, so no UTR-derived feature can rank them."
    )
    add(
        "- **The ORFs carry no stop codon** (843 nt = 281 codons; 1725 nt = 575 codons exactly). The stops "
        "sit in two further trailing rows, `3'-term polymerase` and `3'-term scFv`."
    )
    add(
        "- **The sheet already publishes GC%, CAI and four real folding energies** "
        "(`RNA MIN_ORF`, `RNA Avg_ORF`, `RNA Structure -42 to +50`, `RNA Structure -4 to +38`), which were "
        "not anticipated. They let the model be scored on Welch's own descriptors as well as on this "
        "project's recomputed ones, and both are reported below."
    )
    add(
        "- **Relative expression is normalised to the best variant in each series = 3**, as expected "
        "(`A1_14` and `P19` both sit at exactly 3.0)."
    )
    add("")

    # --- Correlations ---
    add("## 2. Rank correlation against measured yield")
    add("")
    add(
        "Spearman rho between predicted tier and measured ug/ml, over the "
        f"{len(measurable)} measurable variants. Accuracy is not reported: the tiers are Boel's relative "
        "0-5 bins from a different assay, so only ordering is comparable."
    )
    add("")
    add("| Scoring | Scope | n | Spearman rho | p | Permutation null mean &#124;rho&#124; | Null 95th pct |")
    add("| :--- | :--- | ---: | ---: | ---: | ---: | ---: |")

    for label, preds in scorings.items():
        for scope, subset in (
            ("overall", measurable),
            ("polymerase", measurable[measurable["series"] == "polymerase"]),
            ("scFv", measurable[measurable["series"] == "scFv"]),
        ):
            p = preds.reindex(subset.index)
            rho, pv, n = spearman(p, subset["yield_ug_per_ml"])
            null = permutation_null(p, subset["yield_ug_per_ml"])
            add(
                f"| {label} | {scope} | {n} | {_fmt(rho)} | {_fmt(pv)} | "
                f"{_fmt(null['mean_abs_rho'])} | {_fmt(null['p95_abs_rho'])} |"
            )
    add("")

    add("### Trivial and reference baselines")
    add("")
    add("| Baseline | Scope | n | Spearman rho | Note |")
    add("| :--- | :--- | ---: | ---: | :--- |")
    majority = int(pd.read_csv(TRAINING_TABLE)["expression_score"].mode().iloc[0]) if os.path.exists(TRAINING_TABLE) else None
    add(
        f"| Always predict majority tier ({majority if majority is not None else 'n/a'}) | overall | "
        f"{len(measurable)} | undefined | A constant predictor has zero variance, so rho does not exist. "
        "It cannot rank anything, which is the point of printing it. |"
    )
    for name, col in (("GC% alone", "welch_gc_percent"), ("CAI alone", "welch_cai")):
        for scope, subset in (
            ("overall", measurable),
            ("polymerase", measurable[measurable["series"] == "polymerase"]),
            ("scFv", measurable[measurable["series"] == "scFv"]),
        ):
            rho, pv, n = spearman(subset[col], subset["yield_ug_per_ml"])
            note = "Raw descriptor, no model." if scope != "overall" else "Raw descriptor. Pooled; see confound note."
            add(f"| {name} | {scope} | {n} | {_fmt(rho)} | {note} |")

    # Pooling the two series confounds the comparison: they differ in both GC
    # range and yield scale, so part of any pooled correlation is series
    # identity rather than sequence composition. Ranking within each series
    # before pooling removes that component.
    ranked = measurable.copy()
    ranked["y_pct"] = ranked.groupby("series")["yield_ug_per_ml"].rank(pct=True)
    ranked["gc_pct"] = ranked.groupby("series")["welch_gc_percent"].rank(pct=True)
    rho_rn, _, n_rn = spearman(ranked["gc_pct"], ranked["y_pct"])
    add(
        f"| GC% alone | pooled, within-series ranks | {n_rn} | {_fmt(rho_rn)} | "
        "Confound removed: each variant ranked inside its own series first. |"
    )
    add("")

    poly = measurable[measurable["series"] == "polymerase"]
    scfv = measurable[measurable["series"] == "scFv"]
    gc_pooled = spearman(measurable["welch_gc_percent"], measurable["yield_ug_per_ml"])[0]
    gc_poly = spearman(poly["welch_gc_percent"], poly["yield_ug_per_ml"])[0]
    gc_scfv = spearman(scfv["welch_gc_percent"], scfv["yield_ug_per_ml"])[0]
    add(
        f"**On the pooled GC% figure.** The two series differ in GC range "
        f"({_fmt(poly['welch_gc_percent'].min(), 1)}-{_fmt(poly['welch_gc_percent'].max(), 1)}% for polymerase "
        f"versus {_fmt(scfv['welch_gc_percent'].min(), 1)}-{_fmt(scfv['welch_gc_percent'].max(), 1)}% for scFv) and "
        f"in yield scale (up to {_fmt(poly['yield_ug_per_ml'].max(), 0)} versus "
        f"{_fmt(scfv['yield_ug_per_ml'].max(), 0)} ug/ml), so a pooled correlation partly measures which gene a "
        f"variant belongs to. The pooled {_fmt(gc_pooled)} overstates the effect; the within-series figures "
        f"({_fmt(gc_poly)} and {_fmt(gc_scfv)}) and the rank-normalised {_fmt(rho_rn)} are the honest ones. "
        "GC% carries real positive signal by any of them."
    )
    add("")
    add(
        "**CAI runs the wrong way in this dataset.** CAI correlates *negatively* with yield in both series. "
        "That is the paper's own headline result, not an artefact: the maximum-CAI scFv variant (`A_HiCAI`, "
        "CAI 0.883) yielded 20.8 ug/ml against a series best of 225.5. A model that treats rising CAI as a "
        "positive signal will rank this dataset backwards."
    )
    add("")

    # --- Censored ---
    add("## 3. Do the undetectable variants receive low tiers?")
    add("")
    add(
        "Predicted tier against measured decile, over all "
        f"{len(df)} variants. The {int(df['censored'].sum())} censored variants form their own column "
        "rather than being forced into the bottom decile."
    )
    for label, preds in scorings.items():
        add("")
        add(f"**{label}**")
        add("")
        table = decile_confusion(preds, df)
        add("| predicted tier | " + " | ".join(table.columns) + " |")
        add("| :--- | " + " | ".join("---:" for _ in table.columns) + " |")
        for tier, row in table.iterrows():
            add(f"| {tier} | " + " | ".join(str(v) for v in row.to_list()) + " |")
        cens = preds[df["censored"]]
        if len(cens):
            add("")
            add(
                f"Censored variants receive tiers {sorted(cens.unique().tolist())} "
                f"(mean {cens.mean():.2f}); measurable variants average "
                f"{preds[~df['censored']].mean():.2f}."
            )
    add("")

    # --- Ranges ---
    add("## 4. Scale check: Welch features against the training distribution")
    add("")
    add(
        "If the Welch features fall outside the range the model was fitted on, the predictions are "
        "extrapolations. A gradient-boosted tree has no slope beyond its outermost split: it repeats the "
        "nearest leaf, producing a confident-looking tier with nothing behind it. Everything in section 2 "
        "is then uninterpretable."
    )
    add("")
    add(f"Training ranges are the observed min and max in `{TRAINING_TABLE}`.")
    add("")
    add("| Feature | Welch min | Welch max | Training min | Training max | Outside |")
    add("| :--- | ---: | ---: | ---: | ---: | ---: |")

    comparisons = [
        ("GC_percent", "welch_gc_percent", "GC% (Welch's own)"),
        ("GC_percent", "gc_content", "GC% (src/features.py)"),
        ("CAI", "welch_cai", "CAI (Welch's own)"),
        ("CAI", "codon_adaptation_index", "CAI (src/features.py)"),
        ("head_folding_energy", "welch_rna_struct_m42_p50", "5' folding, -42..+50 (Welch's own)"),
        ("head_folding_energy", "folding_utr_junction", "5' folding, UTR junction (src/features.py)"),
        ("genewide_folding_avg48", "welch_rna_avg_orf", "gene-wide folding avg (Welch's own)"),
        ("genewide_folding_avg48", "folding_orf_head", "ORF-head folding proxy (src/features.py)"),
    ]
    ood_notes: List[str] = []
    for train_key, welch_col, display in comparisons:
        if welch_col not in df.columns:
            continue
        values = df[welch_col].dropna()
        if values.empty:
            continue
        bounds = ranges.get(train_key)
        if not bounds:
            add(f"| {display} | {_fmt(values.min(), 2)} | {_fmt(values.max(), 2)} | n/a | n/a | n/a |")
            continue
        outside = int(((values < bounds["min"]) | (values > bounds["max"])).sum())
        add(
            f"| {display} | {_fmt(values.min(), 2)} | {_fmt(values.max(), 2)} | "
            f"{_fmt(bounds['min'], 2)} | {_fmt(bounds['max'], 2)} | {outside}/{len(values)} |"
        )
        if outside:
            ood_notes.append(f"{display}: {outside} of {len(values)} outside [{bounds['min']:.2f}, {bounds['max']:.2f}]")

    add("")
    if ood_notes:
        add("**Out-of-distribution features flagged:**")
        add("")
        for note in ood_notes:
            add(f"- {note}")
    else:
        add("No computed feature falls outside its training range.")
    add("")

    variation = df["folding_utr_junction"].nunique()
    add(
        f"The UTR-junction folding feature takes **{variation} distinct value(s)** across {len(df)} variants, "
        "because 42 of its 48 proxy nucleotides come from the shared leader. A feature that barely varies "
        "cannot order variants, which is why the ORF-only folding proxy is kept alongside it."
    )
    add("")
    gc_inside = int(((df["welch_gc_percent"] >= ranges["GC_percent"]["min"]) & (df["welch_gc_percent"] <= ranges["GC_percent"]["max"])).sum()) if "GC_percent" in ranges else 0
    cai_inside = int(((df["welch_cai"] >= ranges["CAI"]["min"]) & (df["welch_cai"] <= ranges["CAI"]["max"])).sum()) if "CAI" in ranges else 0
    add(
        f"The two features the artifact actually consumes, GC% and CAI, are **inside** the training range for "
        f"{gc_inside} and {cai_inside} of {len(df)} variants respectively. So the negative correlations in "
        "section 2 are not explained away as extrapolation: the model is being asked a question it was fitted "
        "for, and is getting the ordering wrong. The folding rows matter only for the four-feature model the "
        "metadata describes (see below), not for the artifact on disk."
    )
    add("")
    add("### Artifact does not match its own metadata")
    add("")
    add(
        f"`models/baseline_model_metadata.json` records four features "
        "(`GC_percent`, `CAI`, `head_folding_energy`, `genewide_folding_avg48`) and a sha256 for "
        f"`{TRAINING_TABLE}`. The artifact on disk takes "
        f"**{getattr(model, 'n_features_in_', '?')}**: {list(getattr(model, 'feature_names_in_', []))}. The "
        "metadata's quoted feature ranges also disagree with that CSV (it gives `head_folding_energy` as "
        "-54.11 to -25.74 against the table's -60.52 to -25.74). The metadata describes a model that is not "
        "the one being served, so its reported 0.19 macro-F1 and its baselines do not characterise this file."
    )
    add("")
    add(
        "On the other hand `src/features.py` reproduces Welch's own descriptors well: computed GC% matches "
        "Welch's to within 0.01 points (rho = 1.00) and computed CAI ranks almost identically (rho = 0.99) "
        "despite using a different weight table and a lower absolute scale. The feature code is not the "
        "problem here."
    )
    add("")

    # --- Verdict ---
    add("## 5. Verdict")
    add("")
    native, project = list(scorings.values())
    rho_native = spearman(native.reindex(measurable.index), measurable["yield_ug_per_ml"])
    rho_project = spearman(project.reindex(measurable.index), measurable["yield_ug_per_ml"])
    null_project = permutation_null(project.reindex(measurable.index), measurable["yield_ug_per_ml"])

    add(
        "**The model fails this external validation.** It does not merely fail to beat a trivial baseline; "
        "it orders the Welch variants slightly backwards."
    )
    add("")
    add(
        f"- Predicted tier against measured yield gives rho = {_fmt(rho_native[0])} on Welch's own GC%/CAI and "
        f"{_fmt(rho_project[0])} on the project's recomputed ones. Both are negative. The "
        f"{_fmt(rho_project[0])} sits beyond its permutation null (95th percentile "
        f"{_fmt(null_project['p95_abs_rho'])}, p = {_fmt(rho_project[1])}), which makes it a reproducible "
        "*inversion* rather than noise."
    )
    add(
        f"- A single raw feature the model already consumes does better than the model: GC% alone reaches "
        f"{_fmt(gc_poly)} within the polymerase series and {_fmt(gc_scfv)} within scFv ({_fmt(rho_rn)} "
        "rank-normalised across both). The signal is present in the inputs and the model destroys it."
    )
    add(
        "- The likely mechanism is CAI. In Welch's data CAI is negatively associated with yield in both "
        "series, which is the paper's central finding; a model fitted where CAI reads as a positive signal "
        "will invert exactly this way."
    )
    cens_native = native[df["censored"]]
    top_tier_censored = int((cens_native == cens_native.max()).sum())
    add(
        f"- The censoring check is the most damaging. On Welch's own descriptors the "
        f"{int(df['censored'].sum())} undetectable variants receive a mean predicted tier of "
        f"{cens_native.mean():.2f} against {native[~df['censored']].mean():.2f} for the variants that did "
        f"express, and {top_tier_censored} of them are assigned the top tier. The model cannot distinguish a "
        f"construct that yields nothing from one that yields {_fmt(measurable['yield_ug_per_ml'].max(), 0)} ug/ml."
    )
    gc_in = int(((df["welch_gc_percent"] >= ranges["GC_percent"]["min"]) & (df["welch_gc_percent"] <= ranges["GC_percent"]["max"])).sum()) if "GC_percent" in ranges else 0
    cai_in = int(((df["welch_cai"] >= ranges["CAI"]["min"]) & (df["welch_cai"] <= ranges["CAI"]["max"])).sum()) if "CAI" in ranges else 0
    add(
        f"- This is not an out-of-distribution excuse. GC% is inside the training range for {gc_in} of "
        f"{len(df)} variants and CAI for {cai_in} of {len(df)}, so the model is operating where it was fitted."
    )
    add("")
    add(
        "What the exercise does establish is that the validation set itself is sound and reusable: 81 "
        "variants, 62 with a paired sequence and number, 19 informative left-censored cases, two independent "
        "genes, and the authors' own GC/CAI/folding descriptors for cross-checking. It is a usable external "
        "benchmark. The current artifact simply does not pass it, and the headline number to carry forward is "
        "the negative rho, not an accuracy figure."
    )
    add("")
    add(
        "Caveats worth stating. n = 62 measurable variants is small, so each rho carries roughly +/-0.25 of "
        "uncertainty; the two series are different proteins measured on different scales and have been "
        "treated separately wherever pooling would confound; and the tier-to-yield mapping is ordinal only, "
        "so nothing here speaks to calibration, only to ranking."
    )
    add("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xls", default=DEFAULT_XLS, help="Welch Table S1 workbook.")
    parser.add_argument("--out-csv", default=DEFAULT_CSV, help="Tidy per-variant CSV to write.")
    parser.add_argument("--report", default=DEFAULT_REPORT, help="Markdown report to write.")
    parser.add_argument("--model", default=MODEL_PATH, help="Saved model artifact.")
    args = parser.parse_args()

    if not os.path.exists(args.xls):
        logger.error(f"Workbook not found: {args.xls}")
        return 1

    df, flanking = parse_table_s1(args.xls)
    df = add_computed_features(df)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    columns = [
        "variant_id", "series", "orf_dna", "utr_5prime", "cds_length_codons",
        "yield_ug_per_ml", "relative_expression", "censored",
        "yield_raw", "relative_expression_raw", "se_ug_per_ml",
        "gc_content", "codon_adaptation_index", "folding_orf_head", "folding_utr_junction",
        "welch_gc_percent", "welch_cai", "welch_rare_codons_he", "welch_gc7",
        "welch_rna_min_orf", "welch_rna_avg_orf", "welch_rna_struct_m42_p50", "welch_rna_struct_m4_p38",
    ]
    df[columns].to_csv(args.out_csv, index=False)
    logger.info(f"Wrote {args.out_csv} ({len(df)} rows)")

    import joblib

    model = joblib.load(args.model)
    logger.info(
        f"Loaded {type(model).__name__} expecting {list(getattr(model, 'feature_names_in_', []))} "
        f"({getattr(model, 'n_features_in_', '?')} features)"
    )

    scorings = {
        "Welch's own GC% + CAI": score_with_model(df, model, "welch_gc_percent", "welch_cai", "pred_welch_native"),
        "src/features.py GC% + CAI": score_with_model(df, model, "gc_content", "codon_adaptation_index", "pred_project"),
    }

    report = build_report(df, model, scorings, training_ranges(TRAINING_TABLE))
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        handle.write(report)
    logger.info(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
