"""
Train Agent C, the within gene codon variant ranker, and run Gate C on it.

    python scripts/train_agent_c.py

The script is the whole of milestone M2 and it refuses to produce an ungated
artifact. It ingests whichever within gene sources are on disk, fits a ranking
model inside query groups, scores Welch 2009 per series, and writes the
artifact together with a metadata sidecar carrying the gate result. Agent C
reads that sidecar at load time and abstains unless it records a pass, so a
trained artifact can never be mistaken for a passed gate.

What it needs, and what it does when that is missing. The training sources are
Cambray 2018 and Goodman 2013, and neither is in the working tree. With no
source the script explains what it looked for and exits without writing
anything, which is the current behaviour. Drop either release into the paths
named by `src/datasets/cambray.py` or `src/datasets/goodman.py` and the same
command trains and gates without further changes.

Three constraints are enforced here rather than left to whoever runs it.

Welch is never trained on. It is the held out external test set and it is the
gate, so it is loaded only to score.

The feature matrix is the within gene allowlist from `src/agents/variant.py`.
GRAVY, pI and length in codons cannot change under a synonymous substitution,
so inside a query group they are constants; including them is how a within gene
ranker turns back into the between gene model whose failure it exists to fix.
The script asserts that no allowed feature is constant across every group,
which catches a group column that is not really grouping anything.

Groups are split by group, never by row. Two synonymous variants of one gene in
different folds would be the row level leak of this agent.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import gating  # noqa: E402
from src.agents.variant import (  # noqa: E402
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    VARIANT_FEATURES,
    VariantAgent,
    constant_within_group,
    variant_feature_frame,
)
from src.datasets.variant_sets import VariantSet, load_variant_sets  # noqa: E402

logger = logging.getLogger("SynthBio.train_agent_c")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] [SynthBio.train_agent_c] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Ranking objective and settings. Pairwise rather than listwise: the readouts
# are continuous measurements with no graded relevance labels, and a pairwise
# objective asks exactly the question the user asks, which of these two is
# better.
RANKER_PARAMS = {
    "objective": "rank:pairwise",
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 10,
    "random_state": 42,
    "tree_method": "hist",
}

# Share of query groups held out for the secondary part of Gate C, which checks
# that the model has not simply memorised the one or two scaffolds its training
# library is built on.
HELDOUT_GROUP_FRACTION = 0.2
SPLIT_SEED = 42

# Relevance levels a pairwise ranker is given per group. The readout is
# continuous, so it is binned within each group; ten levels keeps the ordering
# while tolerating the measurement noise in a fluorescence readout.
RELEVANCE_LEVELS = 10


def compute_features(frame: pd.DataFrame, cache_path: Optional[str] = None) -> List[Dict[str, float]]:
    """
    Every feature for every variant, through the application's own extractor.

    This is the slow step: the gene wide folding average evaluates a ViennaRNA
    fold for each 48 nt window of each sequence, which is about nine minutes
    for Cambray's 75,000 usable variants. The cache holds only the features
    this agent is allowed to see, keyed by variant id, and is rejected unless
    it covers exactly the rows being trained on, so a stale cache cannot
    silently pair one variant's features with another's readout.
    """
    from src.features import extract_features

    if cache_path and os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        if list(cached.get("variant_id", [])) == list(frame["variant_id"]):
            logger.info(f"Reusing cached features from '{cache_path}'.")
            return cached[list(VARIANT_FEATURES)].to_dict("records")
        logger.info(f"Ignoring '{cache_path}': it does not match the rows being trained on.")

    logger.info(f"Computing features for {len(frame):,} variants.")
    out = []
    for index, dna in enumerate(frame["dna"].astype(str), start=1):
        out.append(extract_features(dna))
        if index % 2500 == 0:
            logger.info(f"  {index:,} of {len(frame):,}")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        table = pd.DataFrame([{k: f.get(k) for k in VARIANT_FEATURES} for f in out])
        table.insert(0, "variant_id", frame["variant_id"].to_numpy())
        table.to_csv(cache_path, index=False)
        logger.info(f"Cached features to '{cache_path}'.")
    return out


def within_group_relevance(frame: pd.DataFrame, levels: int = RELEVANCE_LEVELS) -> np.ndarray:
    """
    The readout binned into ordered relevance levels inside each group.

    Binned within the group and never across groups, because the readouts of
    two different libraries are not on one scale and a ranker trained on a
    single pooled binning would be learning the difference between the
    libraries.
    """
    relevance = np.zeros(len(frame), dtype=int)
    for _, index in frame.groupby("group_id").groups.items():
        values = frame.loc[index, "readout"]
        n_bins = min(levels, max(2, values.nunique()))
        try:
            binned = pd.qcut(values, n_bins, labels=False, duplicates="drop")
        except ValueError:
            binned = pd.Series(np.zeros(len(values), dtype=int), index=values.index)
        relevance[frame.index.get_indexer(index)] = np.nan_to_num(binned.to_numpy(dtype=float)).astype(int)
    return relevance


def split_by_group(frame: pd.DataFrame, fraction: float = HELDOUT_GROUP_FRACTION, seed: int = SPLIT_SEED):
    """Hold out whole query groups, never rows inside a group."""
    import random

    groups = sorted(frame["group_id"].unique())
    if len(groups) < 2:
        logger.warning(
            f"Only {len(groups)} query group(s) in the training data, so no group can be held out. "
            "The secondary part of Gate C, Kendall tau on held out groups, cannot be run and the "
            "Welch gate is the only external evidence."
        )
        return frame, frame.iloc[0:0]
    rng = random.Random(seed)
    n_held = max(1, int(round(len(groups) * fraction)))
    held = set(rng.sample(groups, n_held))
    logger.info(f"Holding out {len(held)} of {len(groups)} query groups.")
    return frame[~frame["group_id"].isin(held)], frame[frame["group_id"].isin(held)]


def fit_ranker(frame: pd.DataFrame, feature_dicts: Sequence[Dict[str, float]]):
    """
    Fit the within group ranker, asserting the allowlist does its job first.

    A feature that is constant inside every query group carries no within group
    signal, so its presence means the allowlist or the grouping is wrong. The
    assertion is the cheapest available guard against silently rebuilding the
    between gene model.
    """
    from xgboost import XGBRanker

    # `frame` may be a subset of the full table, so its index labels are the
    # positions in `feature_dicts`. Indexing by label rather than by position
    # within the subset is what keeps each row with its own features.
    order = frame.sort_values("group_id").index
    ordered = frame.loc[order].reset_index(drop=True)
    features = variant_feature_frame([feature_dicts[i] for i in order])

    constant_everywhere = None
    for group_id, index in ordered.groupby("group_id").groups.items():
        rows = features.iloc[ordered.index.get_indexer(index)]
        constant = set(constant_within_group(rows))
        constant_everywhere = constant if constant_everywhere is None else (constant_everywhere & constant)
    if constant_everywhere:
        raise ValueError(
            "These features are constant inside every query group, so they carry no within gene "
            f"signal: {', '.join(sorted(constant_everywhere))}. Either the grouping column is not "
            "grouping synonymous variants of one gene, or the feature allowlist is wrong."
        )

    relevance = within_group_relevance(ordered)
    group_sizes = ordered.groupby("group_id", sort=True)["variant_id"].size().to_numpy()
    logger.info(
        f"Fitting {len(features):,} rows in {len(group_sizes):,} query groups with "
        f"objective '{RANKER_PARAMS['objective']}'."
    )
    ranker = XGBRanker(**RANKER_PARAMS)
    ranker.fit(features, relevance, group=group_sizes)
    return ranker, features


def heldout_tau(ranker, frame: pd.DataFrame, feature_dicts: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """
    Kendall tau within each held out group, for the model and for GC3 alone.

    This is the secondary part of Gate C. It catches the case where the model
    has memorised the two scaffolds of its training library rather than learning
    anything about codon choice.
    """
    from src.agents.variant import GC3Baseline

    if frame.empty:
        return {}
    model_taus, gc3_taus = [], []
    for _, index in frame.groupby("group_id").groups.items():
        dicts = [feature_dicts[i] for i in index]
        features = variant_feature_frame(dicts)
        scores = np.asarray(ranker.predict(features), dtype=float)
        truth = frame.loc[index, "readout"].to_numpy(dtype=float)
        tau, _ = gating.kendall_tau(scores, truth)
        gc3_tau, _ = gating.kendall_tau(GC3Baseline.score(dicts), truth)
        if tau == tau:
            model_taus.append(tau)
        if gc3_tau == gc3_tau:
            gc3_taus.append(gc3_tau)
    return {
        "model_mean_tau": float(np.mean(model_taus)) if model_taus else float("nan"),
        "gc3_mean_tau": float(np.mean(gc3_taus)) if gc3_taus else float("nan"),
        "n_groups": int(frame["group_id"].nunique()),
    }


def run_gate_c(ranker) -> Tuple[gating.GateCOutcome, Dict[str, gating.WelchScoring]]:
    """Score the candidate ranker on Welch, per series, against GC3 and GC per cent."""
    welch = gating.load_welch()
    feature_dicts = gating.welch_feature_dicts(welch)
    frame = variant_feature_frame(feature_dicts)
    scores = np.asarray(ranker.predict(frame), dtype=float)

    candidate = gating.score_welch(welch, list(scores), "Agent C, variant ranker")
    gc3 = gating.score_welch(welch, [f["gc3_content"] for f in feature_dicts], "GC3 alone")
    gc_percent = gating.score_welch(welch, [f["gc_content"] for f in feature_dicts], "GC per cent alone")
    return gating.evaluate_gate_c(candidate, gc3, gc_percent), {
        "candidate": candidate,
        "gc3": gc3,
        "gc_percent": gc_percent,
    }


def feature_ranges_from(features: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Observed min and max per feature, in the shape the domain check reads."""
    return {
        column: {"min": float(features[column].min()), "max": float(features[column].max())}
        for column in features.columns
        if features[column].notna().any()
    }


def write_artifact(
    ranker,
    sets: Sequence[VariantSet],
    frame: pd.DataFrame,
    outcome: gating.GateCOutcome,
    scorings: Dict[str, gating.WelchScoring],
    tau: Dict[str, float],
    model_path: str,
    metadata_path: str,
    feature_ranges: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    """
    Write the artifact and the sidecar that decides whether the agent speaks.

    The gate status in the sidecar is what `VariantAgent` reads. A failed gate
    is written down rather than discarded: the artifact stays in the repository
    with its gate report, and the agent abstains.
    """
    import joblib

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(ranker, model_path)

    metadata = {
        "created_at": date.today().isoformat(),
        "model_class": type(ranker).__name__,
        "model_params": RANKER_PARAMS,
        "agent": "variant",
        "question": "Of two synonymous versions of the same gene, which expresses better?",
        "features": list(VARIANT_FEATURES),
        "excluded_features": {
            "gravy_score": "constant under synonymous substitution",
            "pI": "constant under synonymous substitution",
            "length_codons": "constant under synonymous substitution",
            "folding_energy": "GC derived proxy, strictly monotonic in 5' GC",
        },
        "training_data": {
            "dataset": ", ".join(s.citation for s in sets),
            "sources": {s.name: int((frame["source_dataset"] == s.name).sum()) for s in sets},
            "n_rows": int(len(frame)),
            "n_groups": int(frame["group_id"].nunique()),
            "held_out": "Welch 2009 in full; see reports/gates.md",
            "sequence_length_nt": {
                "min": int(frame["dna"].str.len().min()),
                "median": int(frame["dna"].str.len().median()),
                "max": int(frame["dna"].str.len().max()),
            },
        },
        # The agent's competence domain. Recorded from the training matrix so
        # that a construct outside it is flagged rather than silently scored.
        # The length range matters more here than it does for the tier model:
        # Cambray's designed region is 96 nt, so a ranker fitted on it is being
        # asked about whole genes when the application calls it.
        "feature_ranges": feature_ranges,
        "gate": {
            "name": "Gate C",
            "status": "passed" if outcome.passed else "failed",
            "summary": outcome.summary,
            "measured_on": date.today().isoformat(),
            "report_path": "reports/gates.md",
            "baselines_beaten": list(outcome.baselines_beaten),
        },
        "welch": {
            name: {
                "overall_rho": scoring.overall[0],
                "polymerase_rho": scoring.rho("polymerase"),
                "scfv_rho": scoring.rho("scFv"),
            }
            for name, scoring in scorings.items()
        },
        "heldout_groups": tau,
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    logger.info(f"Wrote {model_path} and {metadata_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="*", default=[], help="Within gene sources to use; default is all available.")
    parser.add_argument("--max-rows", type=int, default=0, help="Sample this many rows before computing features.")
    parser.add_argument(
        "--features-cache",
        default="data/external/agent_c_features.csv",
        help="Where to cache the computed features; empty string disables the cache.",
    )
    parser.add_argument("--model-out", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata-out", default=DEFAULT_METADATA_PATH)
    args = parser.parse_args()

    sets = load_variant_sets(args.sources)
    if not sets:
        from src.datasets.variant_sets import available_variant_sets

        logger.error("No within gene training source is available, so Agent C cannot be trained.")
        for name, reason in available_variant_sets().items():
            logger.error(f"  {name}: {reason}")
        logger.error(
            "Nothing was written. Agent C stays abstaining and the redesign loop keeps ranking on "
            "the tier model with the GC3 corroboration check as its guard."
        )
        return 1

    for variant_set in sets:
        logger.info(variant_set.summary())
    frame = pd.concat([s.frame for s in sets], ignore_index=True)

    if args.max_rows and len(frame) > args.max_rows:
        # Sample whole groups, so that no group arrives partly represented.
        groups = frame["group_id"].drop_duplicates().tolist()
        kept, taken = [], 0
        for group_id in groups:
            rows = frame[frame["group_id"] == group_id]
            if taken and taken + len(rows) > args.max_rows:
                continue
            kept.append(rows)
            taken += len(rows)
        frame = pd.concat(kept, ignore_index=True)
        logger.info(f"Sampled {len(frame):,} rows in {frame['group_id'].nunique():,} whole groups.")

    frame = frame.reset_index(drop=True)
    feature_dicts = compute_features(frame, args.features_cache or None)

    train, held = split_by_group(frame)
    ranker, training_features = fit_ranker(train, feature_dicts)
    tau = heldout_tau(ranker, held, feature_dicts)
    if tau:
        logger.info(
            f"Held out groups: mean Kendall tau {tau['model_mean_tau']:+.3f} for the model against "
            f"{tau['gc3_mean_tau']:+.3f} for GC3 alone, over {tau['n_groups']} group(s)."
        )

    outcome, scorings = run_gate_c(ranker)
    logger.info(outcome.summary)
    write_artifact(
        ranker, sets, frame, outcome, scorings, tau, args.model_out, args.metadata_out,
        feature_ranges=feature_ranges_from(training_features),
    )

    if not outcome.passed:
        logger.warning(
            "Gate C did not pass, so Agent C will keep abstaining at runtime. The artifact and its "
            "gate report are kept rather than discarded: a failed gate is a result about the data, "
            "and for a within gene ranker trained on one scaffold it is a reportable finding about "
            "scaffold specific libraries. Rerun scripts/gates.py to refresh reports/gates.md."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
