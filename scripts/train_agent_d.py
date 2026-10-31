"""
Train Agent D, the initiation structure agent, and run Gate D on it.

    python scripts/train_agent_d.py

Agent D is the one agent in the plan blocked by a missing input rather than a
missing dataset. `feature_table_v4.csv` carries Boel's own RNAstructure
energies over the expression vector's 5' untranslated region, Gz48v and Gave48,
and the served model cannot use them because the application does not know
which vector a pasted sequence is going into. Declare the vector and the
transcript's initiation window becomes computable, so the strongest feature in
the Boel table becomes servable for the first time.

What this script fits. A binary classifier for the lowest expression band, over
features of the initiation region only: the folding energy of the leader plus
insert window, the folding of the insert head alone, the GC of the first 48
nucleotides and the rare codon fraction of the first 30 codons. It is
deliberately narrow. Agent A already answers which tier; this agent answers
only whether the 5' region folds so as to block initiation, and a narrow agent
is one that can be gated against a named baseline.

Gate D, from the plan's section 3, has two parts:

1. Beat the GC of the first 48 nucleotides alone at predicting the lowest
   expression band. This is the baseline that specifically catches the
   historical failure mode where a folding feature was really a GC feature
   wearing a different label.
2. Show that leader plus insert folding beats insert only folding on the same
   rows. If it does not, the vector input is not earning its place in the
   interface and should be dropped.

The split is organism grouped, not identity clustered, because MMseqs2 is not
installed. That is weaker than the gate asks for and the report says so.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import vectors  # noqa: E402
from src.agents.structure import (  # noqa: E402
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    STRUCTURE_FEATURES,
    structure_feature_frame,
)
from src.datasets import clustering  # noqa: E402

logger = logging.getLogger("SynthBio.train_agent_d")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] [SynthBio.train_agent_d] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

SEQUENCE_TABLE = "data/processed/feature_table_v3.csv"
FEATURE_CACHE = "data/external/agent_d_features.csv"

LOWEST_BAND = 0

MODEL_PARAMS = {
    "objective": "binary:logistic",
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 10,
    "random_state": 42,
    "tree_method": "hist",
}


def compute_features(frame: pd.DataFrame, leader: str, cache_path: Optional[str] = None) -> pd.DataFrame:
    """
    Initiation region features for every Boel construct, with and without the leader.

    Both folding values are computed over the same window shape, so the gate's
    second clause compares like with like: the only difference between them is
    whether the vector's leader is in the window.
    """
    if cache_path and os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        if list(cached.get("gene_id", [])) == list(frame["name"]):
            logger.info(f"Reusing cached features from '{cache_path}'.")
            return cached
        logger.info(f"Ignoring '{cache_path}': it does not match the rows being trained on.")

    from src.features import (
        FOLDING_WINDOW_NT,
        calculate_folding_energy,
        calculate_initiation_folding,
        calculate_rare_codon_fraction,
    )

    logger.info(f"Computing initiation features for {len(frame):,} constructs.")
    rows: List[Dict[str, float]] = []
    for index, (gene_id, dna) in enumerate(zip(frame["name"], frame["seq"].astype(str)), start=1):
        head = dna[:FOLDING_WINDOW_NT]
        rows.append({
            "gene_id": gene_id,
            "initiation_folding_leader": calculate_initiation_folding(leader, dna),
            "initiation_folding_insert": calculate_initiation_folding("", dna),
            "head_folding_48": calculate_folding_energy(head),
            "gc_head48": 100.0 * (head.count("G") + head.count("C")) / max(len(head), 1),
            "rare_codons_head30": calculate_rare_codon_fraction(dna, head_codons=30),
        })
        if index % 1000 == 0:
            logger.info(f"  {index:,} of {len(frame):,}")

    table = pd.DataFrame(rows)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        table.to_csv(cache_path, index=False)
        logger.info(f"Cached features to '{cache_path}'.")
    return table


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    from sklearn.metrics import roc_auc_score

    s = pd.Series(list(scores), dtype="float64").reset_index(drop=True)
    y = pd.Series(list(labels), dtype="int64").reset_index(drop=True)
    mask = s.notna()
    if y[mask].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y[mask], s[mask]))


def best_orientation_auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """
    A single descriptor's AUROC in whichever direction suits it.

    AUROC is direction sensitive, so a descriptor pointed the wrong way scores
    below 0.5 and looks like a weak baseline when it is really a sign error.
    Beating a mis-signed baseline is not beating the baseline, so each
    descriptor is given its better orientation before the comparison. On this
    data that matters: the GC of the first 48 nt runs the opposite way to the
    assumption, a GC rich 5' region being less likely to land in the lowest
    band rather than more.
    """
    value = auroc(scores, labels)
    if value != value:
        return value
    return max(value, 1.0 - value)


def paired_bootstrap(
    first: Sequence[float],
    second: Sequence[float],
    labels: Sequence[int],
    iterations: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    A confidence interval on the difference between two AUROCs on the same rows.

    Paired, resampling rows rather than scores, so the two models are always
    compared on the same bootstrap sample. Without this a difference of a few
    thousandths reads as a result; with it, an interval straddling zero says
    plainly that the comparison did not separate them. Gate D's second clause
    turns on exactly such a margin, which is why it is tested rather than eyed.
    """
    from sklearn.metrics import roc_auc_score

    a = np.asarray(list(first), dtype=float)
    b = np.asarray(list(second), dtype=float)
    y = np.asarray(list(labels), dtype=int)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b, y = a[mask], b[mask], y[mask]

    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations)
    n = len(y)
    for i in range(iterations):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            deltas[i] = np.nan
            continue
        deltas[i] = roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx])
    deltas = deltas[~np.isnan(deltas)]
    if deltas.size == 0:
        return {"delta": float("nan"), "low": float("nan"), "high": float("nan"), "separated": False}
    low, high = (float(v) for v in np.percentile(deltas, [2.5, 97.5]))
    return {
        "delta": float(np.mean(deltas)),
        "low": low,
        "high": high,
        "separated": bool(low > 0.0),
    }


def grouped_predictions(
    features: pd.DataFrame,
    labels: pd.Series,
    groups: Sequence[str],
    columns: Sequence[str],
    n_splits: int = 5,
) -> np.ndarray:
    """
    Out of fold probabilities for the lowest band, under a grouped split.

    Out of fold rather than in sample: a model scored on rows it was fitted on
    would report a number about memorisation. These are also exactly the
    predictions a critic would later be trained on, which is why they are
    produced this way from the start.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    from xgboost import XGBClassifier

    X = features[list(columns)].astype(float).reset_index(drop=True)
    y = labels.astype(int).reset_index(drop=True)
    out = np.full(len(X), np.nan)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, test_idx in splitter.split(X, y, groups=list(groups)):
        model = XGBClassifier(**MODEL_PARAMS).fit(X.iloc[train_idx], y.iloc[train_idx])
        out[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
    return out


def evaluate_gate_d(
    features: pd.DataFrame,
    labels: pd.Series,
    groups: Sequence[str],
) -> Dict[str, object]:
    """
    Both clauses of Gate D, measured out of fold.

    The ablation is the second clause: the same model, the same rows, the same
    folds, with the leader feature swapped for the insert only one. Anything
    else would confound the vector input with a change of model.
    """
    full = grouped_predictions(features, labels, groups, STRUCTURE_FEATURES)
    # The ablation drops the leader feature and keeps everything else,
    # including the insert only folding over the same window. So the two models
    # differ by exactly one thing: whether the vector's leader is available.
    insert_only_columns = [c for c in STRUCTURE_FEATURES if c != "initiation_folding_leader"]
    ablated = grouped_predictions(features, labels, groups, insert_only_columns)

    # GC and folding are scored negated, because a GC rich and tightly folded
    # 5' region is the one expected to land in the lowest band and AUROC is
    # direction sensitive.
    gc_scores = features["gc_head48"].to_numpy(dtype=float)
    gc_raw = auroc(gc_scores, labels)
    # Orient GC the way the data says before the paired test, so the interval
    # compares the agent against the baseline at its best rather than against a
    # sign error.
    oriented_gc = gc_scores if gc_raw >= 0.5 else -gc_scores

    results = {
        "agent_auroc": auroc(full, labels),
        "insert_only_auroc": auroc(ablated, labels),
        "gc_head48_auroc": best_orientation_auroc(gc_scores, labels),
        "gc_head48_orientation": (
            "a GC rich 5' region is more likely to land in the lowest band"
            if gc_raw >= 0.5
            else "a GC poor 5' region is more likely to land in the lowest band"
        ),
        "leader_folding_alone_auroc": best_orientation_auroc(
            features["initiation_folding_leader"], labels
        ),
        "insert_folding_alone_auroc": best_orientation_auroc(
            features["initiation_folding_insert"], labels
        ),
        "n": int(len(labels)),
        "n_lowest_band": int(labels.sum()),
    }

    results["agent_vs_gc"] = paired_bootstrap(full, oriented_gc, labels)
    results["leader_vs_insert"] = paired_bootstrap(full, ablated, labels)

    # Both clauses require the margin to be separated from zero, not merely
    # positive. A gate decided by a difference of a few thousandths would pass
    # or fail on the random seed.
    results["beats_gc_baseline"] = bool(results["agent_vs_gc"]["separated"])
    results["leader_beats_insert"] = bool(results["leader_vs_insert"]["separated"])
    results["passed"] = bool(results["beats_gc_baseline"] and results["leader_beats_insert"])
    return results


def summarise(results: Dict[str, object]) -> str:
    gc, insert = results["agent_vs_gc"], results["leader_vs_insert"]
    detail = (
        f"AUROC {results['agent_auroc']:.3f} against {results['gc_head48_auroc']:.3f} for the GC of "
        f"the first 48 nt alone, a difference of {gc['delta']:+.3f} with a bootstrap interval of "
        f"{gc['low']:+.3f} to {gc['high']:+.3f}; leader plus insert {results['agent_auroc']:.3f} "
        f"against insert only {results['insert_only_auroc']:.3f}, a difference of "
        f"{insert['delta']:+.3f} with an interval of {insert['low']:+.3f} to {insert['high']:+.3f}"
    )
    if results["passed"]:
        verdict = "Gate D passed"
    elif not results["beats_gc_baseline"]:
        verdict = "Gate D failed: the agent does not separate from the GC of the first 48 nt alone"
    else:
        verdict = (
            "Gate D failed: declaring the vector does not separate from the insert alone, so the "
            "vector input is not earning its place in the interface"
        )
    return f"{verdict}. {detail}."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", default=SEQUENCE_TABLE)
    parser.add_argument("--vector", default=vectors.DEFAULT_VECTOR)
    parser.add_argument("--model-out", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata-out", default=DEFAULT_METADATA_PATH)
    parser.add_argument("--features-cache", default=FEATURE_CACHE)
    args = parser.parse_args()

    if not os.path.exists(args.table):
        logger.error(
            f"'{args.table}' is not in the working tree, so there are no sequences to train on. "
            "Recover it with `python scripts/fetch_datasets.py --source boel`."
        )
        return 1

    vector = vectors.get(args.vector)
    if not vector.leader:
        logger.error(f"Vector '{args.vector}' declares no leader, so there is nothing to add to the insert.")
        return 1
    logger.info(f"Vector '{vector.key}': {vector.leader_length} nt leader. {vector.source}")

    frame = pd.read_csv(args.table)
    features = compute_features(frame, vector.leader, args.features_cache or None)

    # The critic pool is held back from training, tuning and gate reporting
    # alike, so it is removed before anything is measured rather than after.
    grouping = clustering.grouping_for(frame["name"].tolist())
    keep = pd.Series(grouping.in_scope, index=frame.index)
    frame = frame[keep].reset_index(drop=True)
    features = features[keep.to_numpy()].reset_index(drop=True)
    groups = [g for g, in_scope in zip(grouping.groups, grouping.in_scope) if in_scope]

    labels = (frame["E"].astype(int) == LOWEST_BAND).astype(int)
    logger.info(
        f"{len(frame):,} constructs in scope, {int(labels.sum()):,} in the lowest band; "
        f"split is {grouping.description}."
    )

    results = evaluate_gate_d(features, labels, groups)
    results["split"] = grouping.description
    results["identity_clustered"] = grouping.identity_clustered
    logger.info(summarise(results) + f" The split is {grouping.description}.")

    from xgboost import XGBClassifier

    model = XGBClassifier(**MODEL_PARAMS).fit(
        structure_feature_frame(features).astype(float), labels.astype(int)
    )

    import joblib

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(model, args.model_out)

    metadata = {
        "created_at": date.today().isoformat(),
        "model_class": type(model).__name__,
        "model_params": MODEL_PARAMS,
        "agent": "structure",
        "question": "Does the 5' region of the transcript fold so as to block initiation?",
        "features": list(STRUCTURE_FEATURES),
        "target": "lowest expression band, Boel tier 0",
        "vector": {"key": vector.key, "leader": vector.leader, "source": vector.source},
        "training_data": {
            "dataset": "Boël et al. 2016, NESG E. coli expression set",
            "path": args.table,
            "n_rows": int(len(frame)),
            "n_lowest_band": int(labels.sum()),
            "split": grouping.description,
            "identity_clustered": grouping.identity_clustered,
        },
        "feature_ranges": {
            column: {"min": float(features[column].min()), "max": float(features[column].max())}
            for column in STRUCTURE_FEATURES
        },
        "gate": {
            "name": "Gate D",
            "status": "passed" if results["passed"] else "failed",
            "summary": summarise(results),
            "measured_on": date.today().isoformat(),
            "report_path": "reports/gates.md",
            "baselines_beaten": (
                ["GC of the first 48 nt alone", "insert only folding"] if results["passed"] else []
            ),
        },
        "gate_detail": results,
    }
    with open(args.metadata_out, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    logger.info(f"Wrote {args.model_out} and {args.metadata_out}")

    if not results["passed"]:
        logger.warning(
            "Gate D did not pass, so Agent D keeps abstaining at runtime and the vector selector "
            "stays out of the scoring path. The artifact and its gate report are kept."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
