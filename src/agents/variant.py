"""
Agent C, codon variant ranking.

Question: of two synonymous versions of the same gene, which expresses better.

This is the one question the served tier model demonstrably gets wrong. On
Welch's two synonymous variant series the model's expected tier tracks measured
yield at plus 0.47 on the scFv variants and minus 0.41 on the polymerase
variants, so it ranks one of the two series backwards, while GC3 alone is
positive on both. The redesign engine optimises that same expected tier, which
is why it needs a corroboration check to stop it driving GC3 down.

Two things make this agent different in kind from the others.

It only has an opinion when there is something to compare. One sequence on its
own has no rank, so on the single sequence path this agent abstains, and that
abstention is correct behaviour rather than a gap.

Its features are restricted to quantities a synonymous substitution can
actually move. GRAVY, pI and length in codons are constant across synonymous
variants of one gene, so inside a query group they carry no signal at all;
including them would let the model latch onto between gene variation and
reproduce exactly the failure this agent exists to fix. The restriction is
enforced here, in `VARIANT_FEATURES`, rather than left to whoever trains the
model.

Gate C, from the plan's section 3: on Welch 2009, per series, positive Spearman
rho on both the polymerase and the scFv series, beating GC3 alone on both. The
figures to beat are GC3 at plus 0.39 on scFv and plus 0.64 on polymerase. Welch
is held out entirely and is never trained on.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.agents.base import (
    GATE_BLOCKED,
    GATE_FAILED,
    GATE_PASSED,
    GATE_UNGATED,
    VARIANT,
    AgentOpinion,
    AgentProvenance,
    AgentRequest,
    DomainCheck,
    GateResult,
    abstain,
)

logger = logging.getLogger("SynthBio.agents.variant")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] [SynthBio.agents.variant] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

DEFAULT_MODEL_PATH = "models/agent_c_variant_ranker.joblib"
DEFAULT_METADATA_PATH = "models/agent_c_variant_ranker_metadata.json"


# Features a synonymous codon substitution can move, and therefore the only
# features this agent is allowed to see.
#
# Keys are the names `src.features.extract_features` returns. The three
# deliberately absent quantities are gravy_score, pI and length_codons: all
# three are invariant under synonymous substitution, so within a query group
# they are constants.
#
# Codon pair bias and tAI are both named in the plan and are both absent here
# for the same reason: each needs a reference table the project does not have,
# codon pair frequencies for the first and tRNA gene copy numbers for the
# second. Adding either means adding its table and saying where the table came
# from, not deriving it from the codon usage weights already in src/features.py,
# which would produce a quantity that carries the name of a published measure
# without being one.
VARIANT_FEATURES: Tuple[str, ...] = (
    "gc_content",
    "gc3_content",
    "codon_adaptation_index",
    "rare_codon_fraction",
    "rare_codons_head30",
    "head_folding_48",
    "genewide_folding_avg",
)

# Features that must never enter this agent, asserted at train and at predict
# time. Named so that the check is a statement rather than a negation of the
# allowlist.
FORBIDDEN_FEATURES: Tuple[str, ...] = ("gravy_score", "pI", "length_codons", "mrna_folding_stability")

# The published per series figures Gate C has to beat, from
# reports/welch2009_validation.md and scripts/redesign_sanity_check.py.
GC3_BASELINE_RHO = {"polymerase": 0.643, "scFv": 0.394}

GATE_C_NOT_RUN = GateResult(
    status=GATE_BLOCKED,
    summary=(
        "Gate C has not been run because this agent has no artifact. Fetch the training data with "
        "`python scripts/fetch_datasets.py --source cambray`, then train and gate in one step with "
        "`python scripts/train_agent_c.py`. Welch 2009 is the held out test set and is never "
        "trained on."
    ),
    report_path="reports/gates.md",
)


def variant_feature_frame(feature_dicts: Sequence[Dict[str, float]]) -> pd.DataFrame:
    """
    The feature matrix this agent is allowed to see, in a fixed column order.

    Raises ValueError if a forbidden feature reaches the matrix, which is the
    one mistake that would silently turn this agent back into the between gene
    model it exists to replace.
    """
    frame = pd.DataFrame(
        [{name: _as_float(f.get(name)) for name in VARIANT_FEATURES} for f in feature_dicts],
        columns=list(VARIANT_FEATURES),
    )
    leaked = [c for c in frame.columns if c in FORBIDDEN_FEATURES]
    if leaked:
        raise ValueError(
            "Features that cannot change under a synonymous substitution reached the variant "
            f"ranker: {', '.join(leaked)}. They are constants within a query group and would let "
            "the model rank on between gene signal."
        )
    return frame


def constant_within_group(frame: pd.DataFrame, tolerance: float = 1e-9) -> List[str]:
    """
    Columns that do not vary across the rows of one query group.

    Run on every group at training time. A feature that is constant inside
    every group cannot contribute to a within group ranking, so its presence
    means either the allowlist is wrong or the group is not what it claims.
    """
    out = []
    for column in frame.columns:
        values = frame[column].dropna()
        if values.empty or float(values.max() - values.min()) <= tolerance:
            out.append(column)
    return out


@dataclass
class VariantRanking:
    """One ranked group: the sequences, their scores and the ordering."""

    scores: Tuple[float, ...]
    order: Tuple[int, ...]  # indices, best first
    feature_frame: pd.DataFrame

    @property
    def best_index(self) -> int:
        return self.order[0]


class GC3Baseline:
    """
    GC3 alone, as a ranker. The baseline Gate C has to beat.

    It is not an agent: it has no attribution, no competence domain and no
    probability. It exists so that the gate harness and the critic's rank
    disagreement term both score the same baseline, computed the same way, in
    one place.
    """

    name = "gc3_only"

    @staticmethod
    def score(feature_dicts: Sequence[Dict[str, float]]) -> List[float]:
        """Higher is better, so GC3 is used as it stands."""
        return [_as_float(f.get("gc3_content")) for f in feature_dicts]

    @staticmethod
    def order(feature_dicts: Sequence[Dict[str, float]]) -> List[int]:
        scores = GC3Baseline.score(feature_dicts)
        return sorted(range(len(scores)), key=lambda i: -scores[i])


class VariantAgent:
    """
    Agent C. A within group ranker over synonymous variants of one gene.

    With no artifact on disk it abstains with the reason, and the redesign loop
    falls back to the tier model with its GC3 corroboration check, which is
    today's behaviour. Once an artifact and a passed gate exist, the same
    interface starts returning an ordering and the redesign loop ranks on it.
    """

    name = VARIANT

    def __init__(
        self,
        ranker: Any = None,
        model_path: str = DEFAULT_MODEL_PATH,
        metadata_path: str = DEFAULT_METADATA_PATH,
        require_gate: bool = True,
        load: bool = True,
    ):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.ranker = ranker
        self.metadata: Dict[str, Any] = {}
        # Unlike Agent A, this agent is gate gated from the start. It has no
        # deployed history to preserve, and the whole point of it is that a
        # within gene claim is not made until the Welch gate supports it.
        self.require_gate = require_gate
        if self.ranker is None and load:
            self._load()

    def _load(self) -> None:
        if not os.path.exists(self.model_path):
            logger.info(f"No variant ranker artifact at '{self.model_path}'; Agent C will abstain.")
            return
        try:
            import joblib

            self.ranker = joblib.load(self.model_path)
            logger.info(f"Loaded variant ranker {type(self.ranker).__name__} from '{self.model_path}'.")
        except Exception as err:
            logger.error(f"Could not load the variant ranker from '{self.model_path}': {err}")
            self.ranker = None
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as handle:
                    self.metadata = json.load(handle)
            except (OSError, json.JSONDecodeError) as err:
                logger.warning(f"Could not read '{self.metadata_path}': {err}")

    # -- provenance ---------------------------------------------------------

    def gate(self) -> GateResult:
        """
        The gate result recorded beside the artifact, or the blocked gate.

        The status is read from the artifact's own metadata sidecar, so an
        artifact that was trained but never gated reports `ungated` and the
        agent still abstains. A trained artifact is not evidence of a passed
        gate and is not treated as one.
        """
        if not self.metadata:
            return GATE_C_NOT_RUN
        recorded = self.metadata.get("gate") or {}
        status = recorded.get("status", GATE_UNGATED)
        if status not in (GATE_PASSED, GATE_FAILED, GATE_BLOCKED, GATE_UNGATED):
            status = GATE_UNGATED
        measured = recorded.get("measured_on")
        try:
            measured_on = date.fromisoformat(measured) if measured else None
        except ValueError:
            measured_on = None
        return GateResult(
            status=status,
            summary=recorded.get("summary", "No gate summary was recorded beside this artifact."),
            measured_on=measured_on,
            report_path=recorded.get("report_path", "reports/gates.md"),
            baselines_beaten=tuple(recorded.get("baselines_beaten", ())),
        )

    def provenance(self) -> AgentProvenance:
        built = None
        if self.ranker is not None and os.path.exists(self.model_path):
            try:
                built = date.fromtimestamp(os.stat(self.model_path).st_mtime)
            except OSError:
                built = None
        return AgentProvenance(
            dataset=(self.metadata.get("training_data") or {}).get("dataset"),
            artifact_path=self.model_path if self.ranker is not None else None,
            artifact_class=type(self.ranker).__name__ if self.ranker is not None else None,
            built_on=built,
            gate=self.gate(),
            notes=(
                "Ranks synonymous variants of one gene against each other. It has no opinion about "
                "one gene against another and no absolute tier.",
            ),
        )

    # -- the one function ---------------------------------------------------

    def opine(self, request: AgentRequest) -> AgentOpinion:
        provenance = self.provenance()

        if not request.has_comparison:
            return abstain(
                self.name,
                "This agent compares synonymous variants of one gene, and only one sequence was "
                "submitted. There is nothing to rank.",
                provenance=provenance,
            )

        if self.ranker is None:
            return abstain(
                self.name,
                "No variant ranker artifact is on disk, so this agent cannot rank. " + provenance.gate.summary,
                provenance=provenance,
            )

        if self.require_gate and not provenance.gate.passed:
            return abstain(
                self.name,
                "This agent has not passed Gate C, so it is not connected. " + provenance.gate.summary,
                provenance=provenance,
            )

        sequences = self._group(request)
        ranking = self.rank(sequences)
        submitted = sequences.index(request.sequence)

        return AgentOpinion(
            agent=self.name,
            applies=True,
            score=float(ranking.scores[submitted]),
            proba=None,
            features=dict(zip(VARIANT_FEATURES, ranking.feature_frame.iloc[submitted].tolist())),
            attribution=self._attribution(ranking.feature_frame.iloc[[submitted]]),
            domain=self._domain(ranking.feature_frame.iloc[[submitted]], request.sequence),
            provenance=provenance,
            ranking=tuple(
                {
                    "sequence": sequences[i],
                    "score": float(ranking.scores[i]),
                    "rank": rank + 1,
                    "submitted": i == submitted,
                }
                for rank, i in enumerate(ranking.order)
            ),
        )

    @staticmethod
    def _group(request: AgentRequest) -> List[str]:
        """The submitted sequence first, then the comparison set, duplicates dropped."""
        group = [request.sequence]
        for candidate in request.comparison:
            if candidate not in group:
                group.append(candidate)
        return group

    def rank(self, sequences: Sequence[str], feature_dicts: Optional[Sequence[Dict[str, float]]] = None) -> VariantRanking:
        """
        Score and order one query group. Higher scores rank better.

        Every sequence in a group must encode the same protein. That is the
        caller's responsibility: the redesign engine already asserts it on every
        candidate it generates, and this agent does not translate thousands of
        candidates a second time.
        """
        if self.ranker is None:
            raise ValueError("Agent C has no artifact loaded, so it cannot rank. Check `opine` first.")

        if feature_dicts is None:
            from src.features import extract_features

            feature_dicts = [extract_features(seq) for seq in sequences]

        frame = variant_feature_frame(feature_dicts)
        required = list(getattr(self.ranker, "feature_names_in_", frame.columns))
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(
                "The variant ranker was fitted on features this agent does not supply: "
                + ", ".join(missing)
            )
        scores = np.asarray(self.ranker.predict(frame[required]), dtype=float).ravel()
        order = tuple(sorted(range(len(scores)), key=lambda i: -scores[i]))
        return VariantRanking(scores=tuple(float(s) for s in scores), order=order, feature_frame=frame)

    def _domain(self, row: pd.DataFrame, sequence: str) -> DomainCheck:
        """
        Whether this construct is the kind of thing the ranker was fitted on.

        Length is checked separately from the features, because it is the one
        that bites here. Cambray's designed region is 96 nt, so a ranker fitted
        on it has never seen a whole gene, and the application hands it whole
        genes. That is a real extrapolation and it is said out loud rather than
        left for the gate report to imply.

        The nearest neighbour identity check is deliberately not run: the
        training rows are 32 codon fragments of a reporter, so an identity
        comparison against them would answer a question nobody asked.
        """
        from src.agents.domain import describe_domain

        check = describe_domain(
            row,
            (self.metadata or {}).get("feature_ranges"),
            sequence=None,
            check_identity=False,
        )
        notes = list(check.notes)
        lengths = ((self.metadata or {}).get("training_data") or {}).get("sequence_length_nt") or {}
        if lengths:
            low, high = lengths.get("min"), lengths.get("max")
            notes.append(
                f"Trained on sequences of {low} to {high} nt; this construct is {len(sequence)} nt."
                + (
                    " It is longer than anything in the training set, so the ranking is an "
                    "extrapolation in sequence length."
                    if high is not None and len(sequence) > high
                    else ""
                )
            )
        notes.append(
            "No nearest neighbour identity check is run for this agent: its training rows are "
            "fragments of one reporter, so identity to them carries no information."
        )
        return DomainCheck(
            out_of_range=check.out_of_range,
            nearest_identity=None,
            identity_checked=False,
            notes=tuple(notes),
        )

    def _attribution(self, row: pd.DataFrame) -> Dict[str, float]:
        """SHAP values for one ranked sequence, empty when SHAP is unavailable."""
        try:
            import shap

            explainer = shap.TreeExplainer(self.ranker)
            values = np.asarray(explainer.shap_values(row)).reshape(-1)
            return {name: float(v) for name, v in zip(row.columns, values)}
        except Exception as err:
            logger.warning(f"SHAP attribution unavailable for Agent C: {err}")
            return {}


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
