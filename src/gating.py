"""
The measurements behind every gate, in one place.

Each agent in the plan has one named gate and a named baseline it has to beat.
This module holds the measurements themselves, so that the gate harness,
`scripts/gates.py`, and the trainers that have to decide whether to record a
pass, cannot drift apart by computing the same gate two different ways.

Nothing in here records a pass on its own. A gate is a comparison against a
baseline, and the comparison is returned as data for the caller to write down.

The two gates that can be measured in this working tree:

Gate A, partly. The four baselines of the plan's section 3 are measurable on the
feature table, and so is the served artifact under organism grouped cross
validation. The identity clustered split that Gate A actually requires is not:
it needs the training sequences, and the feature table carries gene ids and
features only. So this module reports organism grouped figures and labels them
as insufficient, which they are.

Gate C, fully. Welch 2009 is on disk with both series, the exact coding
sequences and a measured yield per variant, so the per series gate can be run
the day an Agent C artifact exists, and the GC3 baseline it has to beat can be
measured today.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("SynthBio.gating")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] [SynthBio.gating] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

FEATURE_TABLE = "data/processed/feature_table_v4.csv"
WELCH_CSV = "data/validation/welch2009.csv"
MODEL_PATH = "models/baseline_model.joblib"
METADATA_PATH = "models/baseline_model_metadata.json"

# The ten servable features the artifact was fitted on.
MODEL_FEATURES = (
    "GC_percent", "GC3", "CAI", "rare_codon_fraction", "rare_codons_head30",
    "head_folding_48", "genewide_folding_avg", "gravy_score", "pI", "length_codons",
)

# The validated heuristic, from the Boel table: GRAVY above this flags a
# construct as likely to land in the lowest expression band.
GRAVY_RULE_THRESHOLD = 0.4

# Welch's two series, and the GC3 figures Gate C has to beat on each.
WELCH_SERIES = ("polymerase", "scFv")
GC3_PER_SERIES_TARGET = {"polymerase": 0.643, "scFv": 0.394}


# --- shared statistics ------------------------------------------------------


def spearman(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float, int]:
    """Spearman rho, p value and n, over rows where both values are present."""
    from scipy.stats import spearmanr

    xs = pd.Series(list(x), dtype="float64")
    ys = pd.Series(list(y), dtype="float64")
    mask = xs.notna() & ys.notna()
    n = int(mask.sum())
    if n < 3 or xs[mask].nunique() < 2 or ys[mask].nunique() < 2:
        return float("nan"), float("nan"), n
    result = spearmanr(xs[mask], ys[mask])
    return float(result.statistic), float(result.pvalue), n


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> Tuple[float, int]:
    """Kendall tau and n, used for the within group part of Gate C."""
    from scipy.stats import kendalltau

    xs = pd.Series(list(x), dtype="float64")
    ys = pd.Series(list(y), dtype="float64")
    mask = xs.notna() & ys.notna()
    n = int(mask.sum())
    if n < 3 or xs[mask].nunique() < 2 or ys[mask].nunique() < 2:
        return float("nan"), n
    return float(kendalltau(xs[mask], ys[mask]).statistic), n


# --- Gate A: the expression tier baselines ----------------------------------


@dataclass
class FoldMetrics:
    """What one split of the expression data says about one predictor."""

    macro_f1: float
    accuracy: float
    spearman_rho: float
    mean_absolute_tier_error: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "macro_f1": self.macro_f1,
            "accuracy": self.accuracy,
            "spearman_rho": self.spearman_rho,
            "mean_absolute_tier_error": self.mean_absolute_tier_error,
        }


@dataclass
class PredictorResult:
    """One predictor across every fold, with the spread between folds."""

    name: str
    folds: List[FoldMetrics] = field(default_factory=list)
    note: str = ""

    def mean(self, metric: str) -> float:
        values = [getattr(f, metric) for f in self.folds]
        values = [v for v in values if v == v]
        return float(np.mean(values)) if values else float("nan")

    def std(self, metric: str) -> float:
        values = [getattr(f, metric) for f in self.folds]
        values = [v for v in values if v == v]
        return float(np.std(values)) if values else float("nan")


def score_fold(y_true: Sequence[int], y_pred: Sequence[int], y_continuous: Sequence[float]) -> FoldMetrics:
    """
    The four numbers every expression predictor is scored on.

    Macro F1 alone hides the thing that matters most about an ordinal target: a
    model confusing tier 4 with tier 5 is far more useful than one confusing
    tier 0 with tier 5. So the rank correlation and the mean absolute tier
    error are reported beside it, and `y_continuous` carries the predictor's
    own continuous score where it has one, because a rank correlation computed
    on rounded classes throws away the ordering inside a class.
    """
    from sklearn.metrics import accuracy_score, f1_score

    rho, _, _ = spearman(y_continuous, y_true)
    return FoldMetrics(
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        spearman_rho=rho,
        mean_absolute_tier_error=float(np.mean(np.abs(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)))),
    )


class MajorityBaseline:
    """Always predicts the most common tier in the training fold."""

    name = "Majority class"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MajorityBaseline":
        self.tier_ = int(pd.Series(y).mode().iloc[0])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.tier_, dtype=int)

    def expected(self, X: pd.DataFrame) -> np.ndarray:
        # A constant predictor has no variance, so its rank correlation does not
        # exist. NaN rather than zero: it cannot rank anything at all, which is
        # the point of carrying it as a baseline.
        return np.full(len(X), np.nan)


class SingleFeatureBaseline:
    """
    Multinomial logistic regression on one feature.

    The plan asks for an ordinal logistic model. scikit-learn ships none and
    `mord` is not a dependency of this project, so this is a multinomial fit
    and is labelled as one. The ordinal information is not lost from the
    comparison: the expected tier under the fitted distribution is used for the
    rank correlation, which is an ordinal measure.
    """

    def __init__(self, feature: str, name: str):
        self.feature = feature
        self.name = name

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SingleFeatureBaseline":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.model_ = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=42),
        ).fit(X[[self.feature]], y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X[[self.feature]]).astype(int)

    def expected(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.model_.predict_proba(X[[self.feature]])
        return proba @ np.asarray(self.classes_, dtype=float)


class GravyRuleBaseline:
    """
    The validated heuristic as a predictor: GRAVY above 0.4 means the lowest band.

    High precision and low recall by construction, so its macro F1 is not
    flattering. It is here because it is the one rule in this project that has
    been checked against the data and held up, and because Agent B has to beat
    it.
    """

    name = "GRAVY rule alone"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GravyRuleBaseline":
        self.fallback_ = int(pd.Series(y).mode().iloc[0])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        flagged = X["gravy_score"].to_numpy(dtype=float) > GRAVY_RULE_THRESHOLD
        return np.where(flagged, 0, self.fallback_).astype(int)

    def expected(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict(X).astype(float)


class ServedArtifactRefit:
    """
    The served model's own configuration, refitted inside each fold.

    Refitted rather than loaded, because scoring the deployed artifact on folds
    it was fitted on would measure nothing. Its hyperparameters are read from
    the metadata sidecar so the comparison is against the model that is actually
    deployed and not against a fresh guess at its settings.
    """

    name = "Served model configuration"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ServedArtifactRefit":
        from xgboost import XGBClassifier

        params = dict(self.params)
        params.pop("objective", None)
        self.model_ = XGBClassifier(objective="multi:softprob", **params).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X).astype(int)

    def expected(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.model_.predict_proba(X)
        return proba @ np.asarray(self.classes_, dtype=float)


def expression_predictors(model_params: Optional[Dict[str, Any]] = None) -> List[Any]:
    """Every predictor Gate A compares, the artifact's configuration first."""
    return [
        ServedArtifactRefit(model_params),
        MajorityBaseline(),
        SingleFeatureBaseline("length_codons", "Length only"),
        SingleFeatureBaseline("GC_percent", "GC per cent only"),
        GravyRuleBaseline(),
    ]


def grouped_cv(
    table: pd.DataFrame,
    predictors: Sequence[Any],
    groups: Sequence[str],
    n_splits: int = 5,
    features: Sequence[str] = MODEL_FEATURES,
) -> List[PredictorResult]:
    """
    Every predictor under one grouped cross validation, folds shared between them.

    The folds are drawn once and reused, so the comparison between predictors is
    paired and the fold to fold standard deviation means what Gate A needs it to
    mean: the gap between a predictor and a baseline has to be larger than it.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    X = table[list(features)].astype(float)
    y = table["expression_score"].astype(int)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(splitter.split(X, y, groups=list(groups)))

    results = [PredictorResult(name=p.name) for p in predictors]
    for fold_index, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        for predictor, result in zip(predictors, results):
            fitted = predictor.fit(X_train, y_train)
            result.folds.append(
                score_fold(y_test.tolist(), fitted.predict(X_test), fitted.expected(X_test))
            )
        logger.info(f"Fold {fold_index} of {len(splits)} scored.")
    return results


# --- Gate C: within gene ranking against Welch ------------------------------


@dataclass
class WelchScoring:
    """One scorer's rank correlation against measured yield, per series."""

    name: str
    overall: Tuple[float, float, int]
    per_series: Dict[str, Tuple[float, float, int]]

    def rho(self, scope: str) -> float:
        if scope == "overall":
            return self.overall[0]
        return self.per_series.get(scope, (float("nan"), float("nan"), 0))[0]


def load_welch(path: str = WELCH_CSV) -> pd.DataFrame:
    """
    Welch's 81 variants, measurable rows only, with this project's features.

    Censored rows are dropped for the rank correlation: the sheet records the
    literal string `nd` against a detection limit of about 5 ug/ml, so they are
    left censored rather than zero and cannot be ordered against each other.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Welch data not found at '{path}'. Run scripts/welch2009_validation.py first.")
    frame = pd.read_csv(path)
    measurable = frame[frame["yield_ug_per_ml"].notna()].reset_index(drop=True)
    logger.info(f"Welch: {len(measurable)} measurable of {len(frame)} variants.")
    return measurable


def welch_feature_dicts(frame: pd.DataFrame) -> List[Dict[str, float]]:
    """
    This project's features for every Welch ORF.

    Computed with the same `extract_features` the application runs, so a gate
    result cannot come from a feature path the application does not use.
    """
    from src.features import extract_features

    logger.info(f"Computing features for {len(frame)} Welch ORFs; ViennaRNA folds every 48 nt window.")
    return [extract_features(seq) for seq in frame["orf_dna"].astype(str)]


def score_welch(
    frame: pd.DataFrame,
    scores: Sequence[float],
    name: str,
) -> WelchScoring:
    """Rank correlation of one scorer against measured yield, overall and per series."""
    series_results = {}
    for series in WELCH_SERIES:
        mask = frame["series"] == series
        series_results[series] = spearman(
            [s for s, keep in zip(scores, mask) if keep],
            frame.loc[mask, "yield_ug_per_ml"].tolist(),
        )
    scoring = WelchScoring(
        name=name,
        overall=spearman(scores, frame["yield_ug_per_ml"].tolist()),
        per_series=series_results,
    )
    logger.info(
        f"Welch, {name}: overall rho {scoring.overall[0]:.3f}, "
        + ", ".join(f"{s} {scoring.rho(s):+.3f}" for s in WELCH_SERIES)
    )
    return scoring


@dataclass
class GateCOutcome:
    """Whether Gate C passed, and on what evidence."""

    passed: bool
    summary: str
    candidate: Optional[WelchScoring]
    baseline: WelchScoring
    baselines_beaten: Tuple[str, ...] = ()


def evaluate_gate_c(candidate: Optional[WelchScoring], gc3: WelchScoring, gc_percent: WelchScoring) -> GateCOutcome:
    """
    Gate C, exactly as the plan states it.

    Pass requires a positive Spearman rho on both the polymerase and the scFv
    series, and beating GC3 alone on both. Beating GC per cent alone overall is
    a harder bar: it is reported but is not required, because the per series
    comparison is the use case.

    An agent that is positive on both series and merely competitive with GC3 is
    still a pass by the plan's wording. That judgement is not made here: the
    numeric condition is applied as written, and the margins are reported so a
    marginal pass can be read as one.
    """
    if candidate is None:
        return GateCOutcome(
            passed=False,
            summary=(
                "Gate C was not run: no variant ranker artifact exists, so there is nothing to "
                "score. The GC3 baseline it has to beat is measured and recorded below."
            ),
            candidate=None,
            baseline=gc3,
        )

    beaten: List[str] = []
    failures: List[str] = []
    for series in WELCH_SERIES:
        candidate_rho, baseline_rho = candidate.rho(series), gc3.rho(series)
        if not (candidate_rho == candidate_rho) or candidate_rho <= 0:
            failures.append(f"{series}: rho {candidate_rho:+.3f} is not positive")
            continue
        if candidate_rho <= baseline_rho:
            failures.append(
                f"{series}: rho {candidate_rho:+.3f} does not beat GC3 alone at {baseline_rho:+.3f}"
            )
            continue
        beaten.append(f"GC3 alone on {series} ({candidate_rho:+.3f} against {baseline_rho:+.3f})")

    if candidate.overall[0] > gc_percent.overall[0]:
        beaten.append(
            f"GC per cent alone overall ({candidate.overall[0]:+.3f} against {gc_percent.overall[0]:+.3f})"
        )

    if failures:
        return GateCOutcome(
            passed=False,
            summary="Gate C failed. " + "; ".join(failures) + ".",
            candidate=candidate,
            baseline=gc3,
            baselines_beaten=tuple(beaten),
        )
    return GateCOutcome(
        passed=True,
        summary=(
            "Gate C passed: positive on both Welch series and beating GC3 alone on both "
            f"(polymerase {candidate.rho('polymerase'):+.3f} against {gc3.rho('polymerase'):+.3f}, "
            f"scFv {candidate.rho('scFv'):+.3f} against {gc3.rho('scFv'):+.3f})."
        ),
        candidate=candidate,
        baseline=gc3,
        baselines_beaten=tuple(beaten),
    )
