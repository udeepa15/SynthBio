"""
Agent A, expression tier.

Question: will this protein express at a measurable level, and in which tier.

This agent wraps the existing artifact unchanged. It fits no new model, changes
no feature and retrains nothing, so the tier it returns is the tier the
application already shows. What is new is that the tier now travels with a
declared competence domain, a gate status and an abstention path.

Its gate has not been passed. Under organism grouped cross validation the
artifact reaches macro F1 0.209 plus or minus 0.010 against a majority class
macro F1 of 0.079, while its accuracy sits at the majority class rate, and on
Welch's synonymous variant series its predicted tier ranks variants backwards.
The gate the plan sets for it, in section 3, needs identity clustered splits,
which need the training sequences; those are not in the working tree. So the
gate status is `blocked` rather than `failed`, and the reason travels with it.

Gate A is milestone M4. Until it is passed this agent's opinion is reported with
its gate status attached, and the critic refuses to treat it as corroboration.
"""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.agents.base import (
    EXPRESSION,
    GATE_BLOCKED,
    AgentOpinion,
    AgentProvenance,
    AgentRequest,
    DomainCheck,
    GateResult,
    abstain,
)
from src.agents.domain import describe_domain, load_feature_ranges

logger = logging.getLogger("SynthBio.agents.expression")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] [SynthBio.agents.expression] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

DEFAULT_MODEL_PATH = "models/baseline_model.joblib"
DEFAULT_METADATA_PATH = "models/baseline_model_metadata.json"

TRAINING_DATASET = "Boël et al. 2016 (JMB), NESG E. coli expression set"

# The gate this agent has to pass, and why it cannot be run today. Carried in
# the interface so the per agent panel states it rather than leaving a blank.
GATE_A = GateResult(
    status=GATE_BLOCKED,
    summary=(
        "Gate A needs identity clustered splits at 30 per cent identity. The training table carries "
        "features and gene ids but no sequences, so the clustering cannot be computed and the gate "
        "has not been run. The organism grouped figure, macro F1 0.209 plus or minus 0.010 against "
        "a majority class 0.079, is not a substitute: paralogues inside one organism still leak."
    ),
    measured_on=None,
    report_path="reports/gates.md",
)


class ExpressionAgent:
    """
    Agent A. A thin wrapper over the served XGBClassifier.

    The model and its SHAP explainer are passed in rather than loaded here, so
    the application keeps its single cached load and this agent adds no second
    copy of a 6,348 row artifact to memory.
    """

    name = EXPRESSION

    def __init__(
        self,
        model: Any = None,
        explainer: Any = None,
        model_path: str = DEFAULT_MODEL_PATH,
        metadata_path: str = DEFAULT_METADATA_PATH,
        feature_ranges: Optional[Dict[str, Dict[str, float]]] = None,
        require_gate: bool = False,
    ):
        self.model = model
        self.explainer = explainer
        self.model_path = model_path
        self.feature_ranges = feature_ranges if feature_ranges is not None else load_feature_ranges(metadata_path)
        # Flipping this to True makes the agent abstain until Gate A passes,
        # which is what milestone M4 does once the gate can be run. It is False
        # at M1 so that the agent path reproduces today's behaviour exactly.
        self.require_gate = require_gate

    # -- provenance ---------------------------------------------------------

    def provenance(self) -> AgentProvenance:
        built = None
        if self.model_path and os.path.exists(self.model_path):
            try:
                built = date.fromtimestamp(os.stat(self.model_path).st_mtime)
            except OSError:
                built = None
        return AgentProvenance(
            dataset=TRAINING_DATASET if self.model is not None else None,
            artifact_path=self.model_path if self.model is not None else None,
            artifact_class=type(self.model).__name__ if self.model is not None else None,
            built_on=built,
            gate=GATE_A,
            notes=(
                "Separates one gene from another. It is not a chooser between synonymous codon "
                "variants of one gene: on Welch's measured variant series it ranks those backwards.",
            ),
        )

    # -- the one function ---------------------------------------------------

    def opine(self, request: AgentRequest) -> AgentOpinion:
        provenance = self.provenance()

        if self.model is None or not hasattr(self.model, "predict_proba"):
            return abstain(
                self.name,
                "No trained expression artifact is loaded, so this agent has no tier to offer.",
                provenance=provenance,
            )

        if self.require_gate and not provenance.gate.passed:
            return abstain(
                self.name,
                "This agent has not passed its gate, so it is not connected. " + provenance.gate.summary,
                provenance=provenance,
            )

        from src.predict import build_feature_row

        features = request.resolved_features()
        feature_row = build_feature_row(features)

        required = list(getattr(self.model, "feature_names_in_", feature_row.columns))
        unservable = [c for c in required if c not in feature_row.columns]
        if unservable:
            return abstain(
                self.name,
                "The loaded artifact needs features that cannot be computed from a pasted sequence: "
                + ", ".join(unservable)
                + ".",
                provenance=provenance,
                features=features,
            )

        matrix = feature_row[required]
        probabilities = self.model.predict_proba(matrix)[0]
        classes = [int(c) for c in getattr(self.model, "classes_", range(len(probabilities)))]
        proba = {c: float(p) for c, p in zip(classes, probabilities)}

        # The expected tier, not the argmax. It is the quantity the redesign
        # engine already optimises, and it orders two constructs that share a
        # modal tier.
        score = float(sum(c * p for c, p in proba.items()))

        domain = describe_domain(
            matrix,
            self.feature_ranges,
            sequence=request.sequence,
        )

        attribution = self._attribution(matrix)

        logger.info(
            f"Agent A: expected tier {score:.3f}, modal tier {max(proba, key=proba.get)}, "
            f"{len(domain.out_of_range)} feature(s) outside the training range."
        )
        return AgentOpinion(
            agent=self.name,
            applies=True,
            score=score,
            proba=proba,
            features={k: _as_float(v) for k, v in features.items()},
            attribution=attribution,
            domain=domain,
            provenance=provenance,
        )

    def _attribution(self, matrix: pd.DataFrame) -> Dict[str, float]:
        """
        Signed SHAP values for the predicted tier, keyed by model feature name.

        Returns an empty dict with no explainer, rather than the placeholder
        constants the older display path fell back on. An empty attribution is
        legible in the interface; a constant dressed as an attribution is not.
        """
        if self.explainer is None:
            return {}
        try:
            from src.model import explain_prediction_shap

            values = explain_prediction_shap(self.explainer, matrix)
            return {str(k): float(v) for k, v in (values or {}).items()}
        except Exception as err:
            logger.warning(f"SHAP attribution unavailable for Agent A: {err}")
            return {}


def _as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if not np.isnan(out) else float("nan")
