"""
The one interface every agent in the multi agent predictor exposes.

Each agent answers a single question, declares the domain it is competent over,
and returns one `AgentOpinion`. The constraint that makes the system
incrementally buildable is that an agent which does not exist yet, or which has
not passed its gate, is simply an agent that returns an abstention. Nothing
downstream has to know the difference.

An agent is a narrow numeric predictor with a declared competence domain and a
structured opinion. It is not a prompt, and no language model sits anywhere in
the scoring path.

The fields of `AgentOpinion` follow the plan in
`docs/synthbio-multi-agent-plan.pdf`, section 2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

logger = logging.getLogger("SynthBio.agents")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] [SynthBio.agents] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# Agent names. Used as dict keys throughout, so they are defined once.
EXPRESSION = "expression"
SOLUBILITY = "solubility"
VARIANT = "variant"
STRUCTURE = "structure"

AGENT_ORDER = (EXPRESSION, SOLUBILITY, VARIANT, STRUCTURE)

# One sentence each, for the per agent panel in the interface.
AGENT_QUESTIONS = {
    EXPRESSION: "Will this protein express at a measurable level, and in which tier?",
    SOLUBILITY: "Given that it expresses, will it be soluble rather than inclusion bodies?",
    VARIANT: "Of two synonymous versions of the same gene, which expresses better?",
    STRUCTURE: "Does the 5' region of the transcript fold so as to block initiation?",
}


# Gate outcomes. `UNGATED` is the state of an agent whose gate has never been
# run, which is different from an agent that ran its gate and failed.
GATE_PASSED = "passed"
GATE_FAILED = "failed"
GATE_UNGATED = "ungated"
GATE_BLOCKED = "blocked"


@dataclass(frozen=True)
class GateResult:
    """
    The outcome of the one named gate an agent has to pass before it is
    connected to the critic.

    `status` is one of the four constants above. `blocked` means the gate could
    not be run at all, which on this project is usually a missing dataset, and
    it carries the reason so the interface can say why rather than showing a
    blank. An agent whose gate is anything other than `passed` must abstain.
    """

    status: str = GATE_UNGATED
    summary: str = "This agent's gate has not been run."
    measured_on: Optional[date] = None
    report_path: Optional[str] = None
    baselines_beaten: Tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status == GATE_PASSED

    def as_line(self) -> str:
        """One line for the per agent panel, always naming the date or its absence."""
        when = f", measured {self.measured_on:%Y-%m-%d}" if self.measured_on else ", never measured"
        return f"Gate {self.status}{when}. {self.summary}"


@dataclass(frozen=True)
class AgentProvenance:
    """
    Where one agent's number came from. The per agent counterpart of
    `src/provenance.py`, which describes the application as a whole.

    The gate result travels with the provenance rather than beside it, because
    an artifact without its gate is exactly what this project is trying not to
    ship.
    """

    dataset: Optional[str] = None
    artifact_path: Optional[str] = None
    artifact_class: Optional[str] = None
    built_on: Optional[date] = None
    gate: GateResult = field(default_factory=GateResult)
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class DomainCheck:
    """
    Whether this input is the kind of thing the agent was fitted on.

    `out_of_range` holds one entry per feature outside the agent's training
    range, in the shape `check_feature_ranges` in `src/predict.py` already
    returns, so the existing interface code can render it unchanged.

    `nearest_identity` is the sequence identity to the closest training row,
    which is the second half of a competence domain and is None until an agent
    carries the sequences needed to compute it. A None is reported as unknown
    and never as a pass.
    """

    out_of_range: Tuple[Dict[str, Any], ...] = ()
    nearest_identity: Optional[float] = None
    identity_checked: bool = False
    notes: Tuple[str, ...] = ()

    @property
    def in_training_range(self) -> bool:
        return not self.out_of_range

    @property
    def inside(self) -> bool:
        """
        True only when every check that ran passed and the identity check ran.

        An unrun identity check leaves this False, so an input is never called
        in domain on the strength of a range check alone.
        """
        return self.in_training_range and self.identity_checked and (
            self.nearest_identity is not None
        )


@dataclass(frozen=True)
class AgentOpinion:
    """
    One agent's structured answer. The only thing an agent ever returns.

    `score` is the agent's own scalar in its own units, so it is not comparable
    across agents; `proba` is the calibrated distribution where the agent is a
    classifier, and is None where it is not. The critic maps both onto a common
    scale itself rather than trusting the agents to agree on one.
    """

    agent: str
    applies: bool
    abstain_reason: Optional[str] = None
    score: Optional[float] = None
    proba: Optional[Dict[int, float]] = None
    features: Dict[str, float] = field(default_factory=dict)
    attribution: Dict[str, float] = field(default_factory=dict)
    domain: DomainCheck = field(default_factory=DomainCheck)
    provenance: AgentProvenance = field(default_factory=AgentProvenance)
    # Agent C ranks a group rather than scoring one sequence, so it needs
    # somewhere to put the ordering. Empty for every scoring agent.
    ranking: Tuple[Dict[str, Any], ...] = ()

    @property
    def question(self) -> str:
        return AGENT_QUESTIONS.get(self.agent, "")

    @property
    def gate(self) -> GateResult:
        return self.provenance.gate

    def top_attribution(self, n: int = 3) -> List[Tuple[str, float]]:
        """The features that moved this opinion most, largest absolute first."""
        return sorted(self.attribution.items(), key=lambda kv: -abs(kv[1]))[:n]

    def with_domain(self, domain: DomainCheck) -> "AgentOpinion":
        return replace(self, domain=domain)


def abstain(
    agent: str,
    reason: str,
    provenance: Optional[AgentProvenance] = None,
    features: Optional[Mapping[str, float]] = None,
    domain: Optional[DomainCheck] = None,
) -> AgentOpinion:
    """
    The abstention an agent returns when it has no opinion.

    Used for all four cases that are not failures: the agent's precondition is
    unmet, its artifact is absent, its gate has not been passed, or the input
    sits outside its competence domain. The reason is user facing, so it says
    what is missing rather than naming an exception.
    """
    logger.info(f"Agent '{agent}' abstains: {reason}")
    return AgentOpinion(
        agent=agent,
        applies=False,
        abstain_reason=reason,
        features=dict(features or {}),
        domain=domain or DomainCheck(),
        provenance=provenance or AgentProvenance(),
    )


@dataclass
class AgentRequest:
    """
    Everything the agents are given about one submitted construct.

    `features` caches `extract_features` so that four agents and a redesign loop
    do not each pay for the same ViennaRNA folds. `comparison` carries the
    synonymous variants to be ranked and is the only thing that lets Agent C
    apply: on the single sequence path it is empty and Agent C abstains, which
    is correct behaviour and not a gap.

    `leader_5prime` is the expression vector's 5' untranslated region, needed by
    Agent D before Boel's Gz48v and Gave48 become servable. It is None until the
    interface asks for a vector, which is milestone M6.
    """

    sequence: str
    features: Optional[Dict[str, float]] = None
    comparison: Sequence[str] = ()
    leader_5prime: Optional[str] = None

    def resolved_features(self) -> Dict[str, float]:
        """Features for the submitted sequence, computed once and cached."""
        if self.features is None:
            from src.features import extract_features

            self.features = extract_features(self.sequence)
        return self.features

    @property
    def has_comparison(self) -> bool:
        """True when there are at least two sequences to rank against each other."""
        return len({self.sequence, *self.comparison}) > 1


class Agent(Protocol):
    """
    The protocol every agent satisfies. One name, one question, one function.

    `opine` must not raise for an input it cannot handle: it abstains instead.
    A raised exception from an agent is a defect, not an answer, and the graph
    deliberately does not catch one.
    """

    name: str

    def opine(self, request: AgentRequest) -> AgentOpinion:  # pragma: no cover - protocol
        ...


# --- Mapping agents onto a common scale -------------------------------------
#
# The agents answer in different units: six ordinal tiers, a binary probability,
# a within group rank. The critic compares them on three bands, which is the
# coarsest scale that still carries the decision a user makes. Tier 1 holds 131
# of 6,348 training rows, so a six band comparison would rest on a class the
# model has barely seen.

LOW_BAND, MID_BAND, HIGH_BAND = 0, 1, 2
BAND_LABELS = {LOW_BAND: "low", MID_BAND: "medium", HIGH_BAND: "high"}

# Boel's six tiers onto the three bands.
TIER_TO_BAND = {0: LOW_BAND, 1: LOW_BAND, 2: MID_BAND, 3: MID_BAND, 4: HIGH_BAND, 5: HIGH_BAND}


def tiers_to_bands(proba: Mapping[int, float]) -> Dict[int, float]:
    """Collapse a distribution over Boel's six tiers onto the three bands."""
    bands = {LOW_BAND: 0.0, MID_BAND: 0.0, HIGH_BAND: 0.0}
    for tier, p in proba.items():
        band = TIER_TO_BAND.get(int(tier))
        if band is None:
            continue
        bands[band] += float(p)
    total = sum(bands.values())
    if total <= 0:
        return bands
    return {band: value / total for band, value in bands.items()}


def binary_to_bands(p_high: float) -> Dict[int, float]:
    """
    Put a binary agent's probability on the three band scale.

    The middle band takes the mass that neither extreme claims, so a binary
    agent at 0.5 reads as undecided rather than as a confident middle.
    """
    p_high = max(0.0, min(1.0, float(p_high)))
    confidence = abs(p_high - 0.5) * 2.0
    return {
        LOW_BAND: (1.0 - p_high) * confidence,
        MID_BAND: 1.0 - confidence,
        HIGH_BAND: p_high * confidence,
    }
