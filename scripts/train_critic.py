"""
Train the critic, and decide on the evidence whether it is worth shipping.

    python scripts/train_critic.py

The critic takes the agents' opinions and returns what the user sees. This
script builds it the only way that does not leak, and then tests it against the
one baseline that matters: the best single agent.

How it is built, following the plan's section 4:

1. The critic pool is a set of clusters reserved in `reports/clusters.json`
   that no specialist ever sees, in training, in tuning or in gate reporting.
   It was reserved once, before any of this, and is not drawn again here.
2. On the remaining clusters, each specialist is run under identity clustered
   K fold, so every row's agent opinions come from a model that never saw that
   row. Those out of fold opinions are what the critic learns from.
3. The critic itself is deliberately small, a multinomial logistic regression
   over three bands, because it has few informative inputs and is the easiest
   place in the system to overfit.
4. It is evaluated once, on the pool, against the best single agent. If it does
   not beat that agent, the honest outcome is to ship the single agent and keep
   the critic for the agreement score alone. That is a legitimate result and
   this script records it as one rather than retrying until it passes.

Which agents feed it. Agent A, the expression tier model, and Agent D, the
initiation structure agent, both produce a distribution for a single submitted
sequence. Agent B does not exist. Agent C ranks variants within a gene and has
no opinion about a single sequence by construction, so it contributes an
abstention here, which is correct behaviour and not a gap.

The abstention policy is set the same way: the conflict threshold is chosen on
the pool by the trade off between how often the system answers and how well it
answers, and the curve is written into the report so the choice can be argued
with.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import gating  # noqa: E402
from src.agents.base import HIGH_BAND, LOW_BAND, MID_BAND, TIER_TO_BAND, binary_to_bands, tiers_to_bands  # noqa: E402
from src.agents.critic import jensen_shannon  # noqa: E402
from src.agents.structure import STRUCTURE_FEATURES  # noqa: E402
from src.datasets import clustering  # noqa: E402

logger = logging.getLogger("SynthBio.train_critic")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] [SynthBio.train_critic] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

DEFAULT_MODEL_PATH = "models/critic.joblib"
DEFAULT_METADATA_PATH = "models/critic_metadata.json"

FEATURE_TABLE = "data/processed/feature_table_v4.csv"
STRUCTURE_FEATURE_CACHE = "data/external/agent_d_features.csv"

BAND_NAMES = {LOW_BAND: "low", MID_BAND: "medium", HIGH_BAND: "high"}

# The share of inputs the system must still answer for an abstention policy to
# be worth having. A system that declines almost everything is accurate and
# useless; the plan puts a defensible operating point at declining 30 to 40 per
# cent, so the search is bounded well outside that to show the whole curve.
MIN_ANSWERED_FRACTION = 0.5


@dataclass
class AgentColumns:
    """Which columns in the opinion frame belong to which agent."""

    expression: Tuple[str, ...] = ("a_low", "a_mid", "a_high")
    structure: Tuple[str, ...] = ("d_low_prob",)
    derived: Tuple[str, ...] = ("divergence", "d_applies", "a_out_of_range")

    def all(self) -> List[str]:
        return list(self.expression) + list(self.structure) + list(self.derived)


COLUMNS = AgentColumns()


def load_rows() -> pd.DataFrame:
    """The Boel rows with both agents' features and the true band."""
    table = pd.read_csv(FEATURE_TABLE)
    if not os.path.exists(STRUCTURE_FEATURE_CACHE):
        raise FileNotFoundError(
            f"'{STRUCTURE_FEATURE_CACHE}' is missing, so Agent D cannot be run over the training "
            "rows. Build it with `python scripts/train_agent_d.py`."
        )
    structure = pd.read_csv(STRUCTURE_FEATURE_CACHE)
    if list(structure["gene_id"]) != list(table["gene_id"]):
        raise ValueError(
            "The structure feature cache does not line up with the feature table row for row. "
            "Rebuild it with `python scripts/train_agent_d.py`."
        )
    # `head_folding_48` and `rare_codons_head30` are in both tables and are the
    # same quantity computed the same way, so the structure cache's copies are
    # dropped rather than carried as duplicate columns.
    shared = [c for c in structure.columns if c != "gene_id" and c in table.columns]
    merged = pd.concat([table, structure.drop(columns=["gene_id", *shared])], axis=1)
    merged["band"] = merged["expression_score"].astype(int).map(TIER_TO_BAND)
    return merged


def out_of_fold_opinions(
    rows: pd.DataFrame,
    groups: Sequence[str],
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Each agent's opinion of every training row, from a model that never saw it.

    This is the step the whole design rests on. An in sample opinion would tell
    the critic that the agent is more accurate than it is, and the agreement
    score would then be confident exactly where it should not be.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    from xgboost import XGBClassifier

    expression_features = list(gating.MODEL_FEATURES)
    structure_features = list(STRUCTURE_FEATURES)
    tiers = rows["expression_score"].astype(int)
    lowest = (tiers == 0).astype(int)

    out = pd.DataFrame(index=rows.index, columns=COLUMNS.all(), dtype=float)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    expression_params = dict(gating.ServedArtifactRefit(_model_params()).params)
    expression_params.pop("objective", None)

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(rows[expression_features], tiers, groups=list(groups)), start=1
    ):
        tier_model = XGBClassifier(objective="multi:softprob", **expression_params).fit(
            rows.iloc[train_idx][expression_features], tiers.iloc[train_idx]
        )
        structure_model = XGBClassifier(
            objective="binary:logistic", n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=10, random_state=42, tree_method="hist",
        ).fit(rows.iloc[train_idx][structure_features], lowest.iloc[train_idx])

        tier_proba = tier_model.predict_proba(rows.iloc[test_idx][expression_features])
        classes = [int(c) for c in tier_model.classes_]
        low_proba = structure_model.predict_proba(rows.iloc[test_idx][structure_features])[:, 1]

        for position, row_index in enumerate(rows.index[test_idx]):
            bands = tiers_to_bands(dict(zip(classes, tier_proba[position])))
            structure_bands = binary_to_bands(1.0 - float(low_proba[position]))
            out.loc[row_index, list(COLUMNS.expression)] = [
                bands[LOW_BAND], bands[MID_BAND], bands[HIGH_BAND]
            ]
            out.loc[row_index, "d_low_prob"] = float(low_proba[position])
            out.loc[row_index, "divergence"] = jensen_shannon(bands, structure_bands)
            out.loc[row_index, "d_applies"] = 1.0
            out.loc[row_index, "a_out_of_range"] = 0.0
        logger.info(f"Out of fold opinions: fold {fold} of {n_splits}.")

    return out


def _model_params() -> Optional[Dict[str, object]]:
    if not os.path.exists(gating.METADATA_PATH):
        return None
    try:
        with open(gating.METADATA_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle).get("model_params")
    except Exception:
        return None


def fit_agents_on(rows: pd.DataFrame) -> Tuple[object, object]:
    """Fit both specialists on every training row, for scoring the pool."""
    from xgboost import XGBClassifier

    expression_params = dict(gating.ServedArtifactRefit(_model_params()).params)
    expression_params.pop("objective", None)
    tier_model = XGBClassifier(objective="multi:softprob", **expression_params).fit(
        rows[list(gating.MODEL_FEATURES)], rows["expression_score"].astype(int)
    )
    structure_model = XGBClassifier(
        objective="binary:logistic", n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=10, random_state=42, tree_method="hist",
    ).fit(rows[list(STRUCTURE_FEATURES)], (rows["expression_score"].astype(int) == 0).astype(int))
    return tier_model, structure_model


def opinions_for(rows: pd.DataFrame, tier_model, structure_model) -> pd.DataFrame:
    """Both agents' opinions of rows they have never seen."""
    tier_proba = tier_model.predict_proba(rows[list(gating.MODEL_FEATURES)])
    classes = [int(c) for c in tier_model.classes_]
    low_proba = structure_model.predict_proba(rows[list(STRUCTURE_FEATURES)])[:, 1]

    out = pd.DataFrame(index=rows.index, columns=COLUMNS.all(), dtype=float)
    for position, row_index in enumerate(rows.index):
        bands = tiers_to_bands(dict(zip(classes, tier_proba[position])))
        structure_bands = binary_to_bands(1.0 - float(low_proba[position]))
        out.loc[row_index, list(COLUMNS.expression)] = [
            bands[LOW_BAND], bands[MID_BAND], bands[HIGH_BAND]
        ]
        out.loc[row_index, "d_low_prob"] = float(low_proba[position])
        out.loc[row_index, "divergence"] = jensen_shannon(bands, structure_bands)
        out.loc[row_index, "d_applies"] = 1.0
        out.loc[row_index, "a_out_of_range"] = 0.0
    return out


def single_agent_bands(opinions: pd.DataFrame) -> np.ndarray:
    """The best single agent's answer: Agent A's band, which is the only one with three."""
    return opinions[list(COLUMNS.expression)].to_numpy(dtype=float).argmax(axis=1)


def answered_curve(
    probabilities: np.ndarray,
    truth: np.ndarray,
    steps: int = 40,
) -> List[Dict[str, float]]:
    """
    Accuracy on the answered subset against the fraction answered.

    The abstention policy is chosen from this curve rather than by taste. A
    system that answers 60 per cent of inputs well is a result; one that
    answers all of them at the baseline is not.
    """
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == truth).astype(float)
    curve = []
    for threshold in np.linspace(0.0, confidence.max(), steps):
        answered = confidence >= threshold
        fraction = float(answered.mean())
        if fraction <= 0:
            continue
        curve.append({
            "threshold": float(threshold),
            "answered_fraction": fraction,
            "accuracy_on_answered": float(correct[answered].mean()),
        })
    return curve


def divergence_curve(
    divergence: np.ndarray,
    predictions: np.ndarray,
    truth: np.ndarray,
    steps: int = 40,
) -> List[Dict[str, float]]:
    """
    What declining on disagreement actually buys, as a curve.

    This is the threshold the shipped critic uses, so it is the one worth
    fitting. The system answers when the agents' distributions are closer than
    the threshold and declines when they are further apart, and the answer is
    the single agent's band, because the trained critic did not earn the right
    to overrule it. Each point is how often that answers and how often it is
    right when it does.
    """
    correct = (predictions == truth).astype(float)
    finite = divergence[np.isfinite(divergence)]
    if finite.size == 0:
        return []
    curve = []
    for threshold in np.linspace(0.0, float(finite.max()), steps):
        answered = np.isfinite(divergence) & (divergence <= threshold)
        fraction = float(answered.mean())
        if fraction <= 0:
            continue
        curve.append({
            "threshold": float(threshold),
            "answered_fraction": fraction,
            "accuracy_on_answered": float(correct[answered].mean()),
        })
    return curve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-out", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata-out", default=DEFAULT_METADATA_PATH)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    assignment = clustering.load_assignment()
    if assignment is None:
        logger.error(
            "No cluster assignment is committed, so there is no reserved critic pool and a critic "
            "cannot be trained without leaking. Run `python scripts/cluster.py` first."
        )
        return 1

    rows = load_rows()
    grouping = clustering.grouping_for(rows["gene_id"].tolist())
    in_scope = pd.Series(grouping.in_scope, index=rows.index)

    training = rows[in_scope]
    pool = rows[~in_scope]
    training_groups = [g for g, keep in zip(grouping.groups, grouping.in_scope) if keep]
    logger.info(
        f"{len(training):,} training rows outside the pool, {len(pool):,} rows in the reserved "
        f"critic pool. {grouping.description}."
    )
    if pool.empty:
        logger.error("The critic pool is empty, so the critic cannot be evaluated.")
        return 1

    # 1. Out of fold opinions on the training rows.
    opinions = out_of_fold_opinions(training, training_groups, n_splits=args.folds)

    # 2. The critic, deliberately small.
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    critic = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42),
    ).fit(opinions[COLUMNS.all()], training["band"].astype(int))

    # 3. Score the pool with agents fitted on every training row.
    tier_model, structure_model = fit_agents_on(training)
    pool_opinions = opinions_for(pool, tier_model, structure_model)
    truth = pool["band"].astype(int).to_numpy()

    critic_proba = critic.predict_proba(pool_opinions[COLUMNS.all()])
    critic_bands = critic_proba.argmax(axis=1)
    agent_bands = single_agent_bands(pool_opinions)

    from sklearn.metrics import accuracy_score, f1_score

    results = {
        "critic_accuracy": float(accuracy_score(truth, critic_bands)),
        "critic_macro_f1": float(f1_score(truth, critic_bands, average="macro", zero_division=0)),
        "best_single_agent_accuracy": float(accuracy_score(truth, agent_bands)),
        "best_single_agent_macro_f1": float(
            f1_score(truth, agent_bands, average="macro", zero_division=0)
        ),
        "majority_accuracy": float(
            accuracy_score(truth, np.full_like(truth, pd.Series(truth).mode().iloc[0]))
        ),
        "n_pool": int(len(pool)),
        "n_training": int(len(training)),
    }
    results["beats_best_single_agent"] = bool(
        results["critic_macro_f1"] > results["best_single_agent_macro_f1"]
    )

    curve = answered_curve(critic_proba, truth)
    # The operating point: the most accurate the system can be while still
    # answering at least the stated share of inputs.
    usable = [p for p in curve if p["answered_fraction"] >= MIN_ANSWERED_FRACTION]
    operating = max(usable, key=lambda p: p["accuracy_on_answered"]) if usable else None

    # The threshold the shipped critic actually uses: decline when the agents'
    # distributions diverge by more than this, and otherwise answer with the
    # single agent's band.
    divergence = pool_opinions["divergence"].to_numpy(dtype=float)
    conflict_curve = divergence_curve(divergence, agent_bands, truth)
    conflict_usable = [p for p in conflict_curve if p["answered_fraction"] >= MIN_ANSWERED_FRACTION]
    conflict_point = (
        max(conflict_usable, key=lambda p: p["accuracy_on_answered"]) if conflict_usable else None
    )
    baseline_accuracy = results["best_single_agent_accuracy"]
    if conflict_point:
        conflict_point["lift_over_answering_everything"] = (
            conflict_point["accuracy_on_answered"] - baseline_accuracy
        )
        logger.info(
            f"Conflict threshold {conflict_point['threshold']:.3f}: answers "
            f"{conflict_point['answered_fraction']:.0%} of inputs at "
            f"{conflict_point['accuracy_on_answered']:.3f} accuracy, against "
            f"{baseline_accuracy:.3f} when answering everything."
        )

    logger.info(
        f"Critic on the pool: macro F1 {results['critic_macro_f1']:.3f} against "
        f"{results['best_single_agent_macro_f1']:.3f} for the best single agent; accuracy "
        f"{results['critic_accuracy']:.3f} against a majority rate of "
        f"{results['majority_accuracy']:.3f}."
    )
    if operating:
        logger.info(
            f"Operating point: answering {operating['answered_fraction']:.0%} of inputs at "
            f"{operating['accuracy_on_answered']:.3f} accuracy on those answered."
        )

    import joblib

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(critic, args.model_out)

    verdict = (
        "The critic beats the best single agent on the reserved pool, so it is used for the tier."
        if results["beats_best_single_agent"]
        else "The critic does not beat the best single agent on the reserved pool, so the single "
        "agent is shipped for the tier and the critic is kept for the agreement score and the "
        "abstention policy only. That is the outcome the plan names as legitimate."
    )
    metadata = {
        "created_at": date.today().isoformat(),
        "model_class": type(critic).__name__,
        "inputs": COLUMNS.all(),
        "agents": ["expression", "structure"],
        "band_names": {str(k): v for k, v in BAND_NAMES.items()},
        "evaluation": results,
        "verdict": verdict,
        "answered_curve": curve,
        "operating_point": operating,
        "conflict_curve": conflict_curve,
        "conflict_threshold": conflict_point,
        "min_answered_fraction": MIN_ANSWERED_FRACTION,
        "critic_pool": {
            "clusters": len(assignment.critic_pool),
            "rows": int(len(pool)),
            "source": clustering.CLUSTER_REPORT,
        },
        "split": grouping.description,
        "gate": {
            "name": "Critic",
            "status": "passed" if results["beats_best_single_agent"] else "failed",
            "summary": verdict,
            "measured_on": date.today().isoformat(),
            "report_path": "reports/gates.md",
        },
    }
    with open(args.metadata_out, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    logger.info(f"Wrote {args.model_out} and {args.metadata_out}")
    logger.info(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
