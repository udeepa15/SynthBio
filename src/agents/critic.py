"""
The critic: what the user is told, and how much the agents agree.

The critic takes the agents' opinions and returns one of four states. Three of
them show a number and one deliberately does not. A well calibrated refusal is
the product here: the application exists to stop someone spending money on a
gene synthesis order, and a confident guess is worse than a declared conflict.

Two things about this file are deliberately unfinished, and they are marked in
the output rather than papered over.

It is not a trained critic. The plan trains one on out of fold agent opinions
over a reserved pool of clusters that no specialist ever sees, which needs
identity clustering of the training sequences; those sequences are not in the
working tree. So this is the rule based critic of milestone M1: it passes a
single agent through, computes the agreement components that are computable,
and records that no critic model was fitted.

Its threshold is fitted when a critic has been trained, and declared when one
has not. `scripts/train_critic.py` chooses the conflict threshold on the
reserved pool by the trade off between how often the system answers and how
often it is right when it does, and writes it beside the critic artifact. This
module reads it from there, and every verdict carries `thresholds_fitted` so
that nothing downstream can present a tuned abstention rate that was never
tuned.

What that fit actually showed is worth knowing before leaning on it: declining
on disagreement raises accuracy on the answered subset by a few points while
declining about half the inputs. The mechanism works, the signal is weak, and
`reports/gates.md` carries the whole curve rather than the single point.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from src.agents.base import (
    BAND_LABELS,
    EXPRESSION,
    HIGH_BAND,
    LOW_BAND,
    MID_BAND,
    VARIANT,
    AgentOpinion,
    binary_to_bands,
    tiers_to_bands,
)

logger = logging.getLogger("SynthBio.agents.critic")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] [SynthBio.agents.critic] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# The four states. The interface must render them differently: a conflicted
# verdict has to look like a deliberate answer and not like an error.
AGREED = "agreed"
AGREED_OUT_OF_DOMAIN = "agreed_out_of_domain"
CONFLICTED = "conflicted"
NOT_ANSWERABLE = "not_answerable"

STATE_LABELS = {
    AGREED: "Agreed",
    AGREED_OUT_OF_DOMAIN: "Agreed, outside the training domain",
    CONFLICTED: "Conflicted, no tier shown",
    NOT_ANSWERABLE: "Not answerable from the trained agents",
}

# The declared fallback, used when no critic has been trained. Jensen Shannon
# divergence in bits between two distributions over three bands runs from 0 to
# 1. Two agents whose modal band differs and which are each moderately
# confident land around 0.3, which is where this sits.
DEFAULT_CONFLICT_JS_THRESHOLD = 0.30

CRITIC_METADATA_PATH = "models/critic_metadata.json"


@lru_cache(maxsize=1)
def fitted_threshold() -> Optional[Dict[str, float]]:
    """
    The conflict threshold chosen on the reserved pool, or None.

    Cached because it is read on every verdict and changes only when a critic
    is retrained.
    """
    if not os.path.exists(CRITIC_METADATA_PATH):
        return None
    try:
        with open(CRITIC_METADATA_PATH, "r", encoding="utf-8") as handle:
            recorded = (json.load(handle) or {}).get("conflict_threshold")
    except (OSError, json.JSONDecodeError) as err:
        logger.warning(f"Could not read '{CRITIC_METADATA_PATH}': {err}")
        return None
    if not recorded or "threshold" not in recorded:
        return None
    return {str(k): float(v) for k, v in recorded.items()}


def conflict_threshold() -> float:
    fitted = fitted_threshold()
    return float(fitted["threshold"]) if fitted else DEFAULT_CONFLICT_JS_THRESHOLD


def thresholds_are_fitted() -> bool:
    return fitted_threshold() is not None

# Which agents, if they abstain, make the question unanswerable rather than
# merely less supported. Agent C abstaining on a single sequence is normal.
TIER_BEARING_AGENTS = (EXPRESSION,)


def jensen_shannon(p: Mapping[int, float], q: Mapping[int, float]) -> float:
    """
    Jensen Shannon divergence in bits between two band distributions.

    Zero for identical distributions, one for distributions with disjoint
    support. Symmetric, which matters because no agent is the reference.
    """
    keys = sorted(set(p) | set(q))
    pv = [max(0.0, float(p.get(k, 0.0))) for k in keys]
    qv = [max(0.0, float(q.get(k, 0.0))) for k in keys]
    ps, qs = sum(pv), sum(qv)
    if ps <= 0 or qs <= 0:
        return float("nan")
    pv = [v / ps for v in pv]
    qv = [v / qs for v in qv]

    def kl(a: Sequence[float], b: Sequence[float]) -> float:
        total = 0.0
        for x, y in zip(a, b):
            if x > 0 and y > 0:
                total += x * math.log2(x / y)
        return total

    m = [(x + y) / 2.0 for x, y in zip(pv, qv)]
    return max(0.0, 0.5 * kl(pv, m) + 0.5 * kl(qv, m))


def bands_for(opinion: AgentOpinion) -> Optional[Dict[int, float]]:
    """
    One agent's answer on the common three band scale, or None.

    An agent with no probability vector has no distribution to compare, so it
    contributes to the domain component of the agreement score and to nothing
    else. Agent C is the case that matters: a within group rank has no absolute
    band at all, by construction.
    """
    if not opinion.applies or not opinion.proba:
        return None
    values = {int(k): float(v) for k, v in opinion.proba.items()}
    if set(values) <= {0, 1} and len(values) == 2:
        return binary_to_bands(values.get(1, 0.0))
    return tiers_to_bands(values)


@dataclass
class AgreementScore:
    """
    How much the agents agree, in three distinct quantities rather than one.

    Each component is None when it could not be computed, which is different
    from zero. A single applying agent produces no distributional disagreement
    because there is nothing to compare, and reporting that as perfect
    agreement would be the most misleading number the system could emit.
    """

    distributional: Optional[float] = None
    rank: Optional[float] = None
    domain: float = 0.0
    n_applying: int = 0
    n_abstained: int = 0
    pairs: Tuple[Dict[str, object], ...] = ()
    notes: Tuple[str, ...] = ()

    @property
    def conflicted(self) -> bool:
        if self.distributional is not None and self.distributional > conflict_threshold():
            return True
        # A negative rank component means the ranker and GC3 order the same
        # pair in opposite directions, which is the known failure mode.
        return self.rank is not None and self.rank < 0

    def as_line(self) -> str:
        parts = []
        if self.distributional is None:
            parts.append("distributional agreement not measurable, fewer than two agents with a distribution")
        else:
            parts.append(f"mean pairwise divergence {self.distributional:.3f} of a possible 1.000")
        if self.rank is not None:
            parts.append(f"rank agreement with GC3 {self.rank:+.3f}")
        parts.append(f"{self.n_applying} agent(s) applied, {self.n_abstained} abstained")
        return "; ".join(parts)


@dataclass
class CriticVerdict:
    """What the interface renders. One state, and a number only in three of them."""

    state: str
    agreement: AgreementScore
    opinions: Dict[str, AgentOpinion] = field(default_factory=dict)
    tier: Optional[int] = None
    expected_tier: Optional[float] = None
    bands: Optional[Dict[int, float]] = None
    risk: Optional[float] = None
    risk_basis: str = ""
    explanation: str = ""
    conflicts: Tuple[str, ...] = ()
    thresholds_fitted: bool = field(default_factory=thresholds_are_fitted)
    critic_model: Optional[str] = None

    @property
    def state_label(self) -> str:
        return STATE_LABELS.get(self.state, self.state)

    @property
    def shows_tier(self) -> bool:
        return self.state in (AGREED, AGREED_OUT_OF_DOMAIN) and self.tier is not None

    @property
    def applying(self) -> List[AgentOpinion]:
        return [o for o in self.opinions.values() if o.applies]

    @property
    def abstaining(self) -> List[AgentOpinion]:
        return [o for o in self.opinions.values() if not o.applies]

    @property
    def band_label(self) -> Optional[str]:
        if not self.bands:
            return None
        return BAND_LABELS.get(max(self.bands, key=self.bands.get))


def _domain_pressure(opinions: Sequence[AgentOpinion]) -> Tuple[float, List[str]]:
    """
    How far this input sits outside what the applying agents know, in 0 to 1.

    Counts features outside the training range and an unavailable identity
    check. An unavailable check raises the pressure rather than leaving it
    unchanged, because not knowing whether anything resembles the input is a
    reason for less confidence, not the same as knowing it does.
    """
    applying = [o for o in opinions if o.applies]
    if not applying:
        return 1.0, ["No agent applied, so nothing is known about this construct's domain."]

    notes: List[str] = []
    pressure = 0.0
    for opinion in applying:
        if opinion.domain.out_of_range:
            names = ", ".join(str(o["feature"]) for o in opinion.domain.out_of_range)
            notes.append(f"{opinion.agent}: outside the training range on {names}.")
            pressure += min(1.0, len(opinion.domain.out_of_range) / 3.0)
        if opinion.domain.identity_applicable and not opinion.domain.identity_checked:
            pressure += 0.5
            notes.extend(n for n in opinion.domain.notes if n)
    return min(1.0, pressure / len(applying)), notes


def score_agreement(opinions: Sequence[AgentOpinion], rank_agreement: Optional[float] = None) -> AgreementScore:
    """
    The three part agreement score of the plan's section 4.

    `rank_agreement` is supplied by the redesign loop, where a within group
    ordering exists: positive when Agent C and GC3 order a pair the same way,
    negative when they disagree. It is None on the single sequence path, where
    there is no ordering to compare.
    """
    applying = [o for o in opinions if o.applies]
    abstained = [o for o in opinions if not o.applies]

    distributions = [(o.agent, bands_for(o)) for o in applying]
    distributions = [(name, bands) for name, bands in distributions if bands]

    pairs: List[Dict[str, object]] = []
    divergences: List[float] = []
    for i in range(len(distributions)):
        for j in range(i + 1, len(distributions)):
            (a_name, a_bands), (b_name, b_bands) = distributions[i], distributions[j]
            js = jensen_shannon(a_bands, b_bands)
            if not math.isnan(js):
                divergences.append(js)
                pairs.append({
                    "agents": (a_name, b_name),
                    "divergence": js,
                    "modal": (
                        BAND_LABELS.get(max(a_bands, key=a_bands.get)),
                        BAND_LABELS.get(max(b_bands, key=b_bands.get)),
                    ),
                })

    pressure, notes = _domain_pressure(opinions)
    if len(distributions) < 2:
        notes.append(
            "Fewer than two agents returned a distribution, so there is no distributional "
            "agreement to measure. This is not agreement."
        )

    return AgreementScore(
        distributional=(sum(divergences) / len(divergences)) if divergences else None,
        rank=rank_agreement,
        domain=pressure,
        n_applying=len(applying),
        n_abstained=len(abstained),
        pairs=tuple(pairs),
        notes=tuple(notes),
    )


def _risk_from_bands(
    bands: Mapping[int, float],
    pressure: float,
    contributors: Sequence[AgentOpinion] = (),
) -> Tuple[float, str]:
    """
    Failure risk from the agents' own distributions, not from a parallel rule.

    This is the critic output that resolves the open question of how failure
    risk is defined. Risk is the probability mass the agents put on the low
    band, raised by the domain pressure, so a tier and a risk can no longer
    contradict each other: they come from the same distribution.

    The biophysical penalty sum in `calculate_failure_risk` survives as agent
    input features, which is where it belongs. It is not consulted here.

    The figure inherits the calibration of the agents feeding it. With Agent A
    uncalibrated and its gate unrun, the basis string says so, and the caller is
    expected to show that string next to the number.
    """
    low = float(bands.get(LOW_BAND, 0.0))
    risk = 100.0 * min(1.0, low + 0.25 * pressure * (1.0 - low))

    # The caveat is assembled from the agents that actually contributed, so it
    # cannot go stale when one of them passes its gate.
    gated = [o.agent for o in contributors if o.gate.passed]
    ungated = [o.agent for o in contributors if not o.gate.passed]
    provenance = []
    if gated:
        provenance.append(f"{', '.join(gated)} has passed its gate")
    if ungated:
        provenance.append(f"{', '.join(ungated)} has not")
    basis = (
        "Probability the applying agents place on the low expression band, raised for distance "
        "from their training domain. "
        + ("; ".join(provenance) + ". " if provenance else "")
        + "No agent feeding it has been calibrated, so read it as an ordering rather than as a "
        "percentage."
    )
    return round(min(risk, 99.9), 1), basis


def combine(
    opinions: Sequence[AgentOpinion],
    rank_agreement: Optional[float] = None,
) -> CriticVerdict:
    """
    Turn the agents' opinions into the one verdict the user sees.

    At M1 this is a pass through for a single agent plus the domain checks, as
    the plan specifies. It becomes a trained combination once there is a critic
    pool to train and evaluate one on, and `critic_model` names the model in
    use so the interface never implies one that does not exist.
    """
    by_name = {o.agent: o for o in opinions}
    agreement = score_agreement(opinions, rank_agreement=rank_agreement)
    applying = [o for o in opinions if o.applies]

    if not applying:
        reasons = "; ".join(f"{o.agent}: {o.abstain_reason}" for o in opinions if o.abstain_reason)
        logger.info("Critic: not answerable, every agent abstained.")
        return CriticVerdict(
            state=NOT_ANSWERABLE,
            agreement=agreement,
            opinions=by_name,
            explanation=(
                "No agent was able to answer, so no tier is shown. The validated heuristic is the "
                "only thing left: GRAVY above 0.4 flags a construct as likely to land in the lowest "
                "expression band, at high precision and low recall. "
                + (f"Why each agent abstained: {reasons}" if reasons else "")
            ),
            conflicts=(),
        )

    distributions = {o.agent: bands_for(o) for o in applying}
    distributions = {name: bands for name, bands in distributions.items() if bands}

    if not distributions:
        # Agents applied but none of them carries an absolute band. A within
        # group ranking alone cannot produce a tier, which is the redesign path.
        logger.info("Critic: agents applied but none returned an absolute band.")
        return CriticVerdict(
            state=NOT_ANSWERABLE,
            agreement=agreement,
            opinions=by_name,
            explanation=(
                "The agents that applied rank constructs against each other but none of them "
                "assigns an absolute expression tier, so there is an ordering here and no tier."
            ),
        )

    # Mean of the available distributions. An unweighted mean is deliberate:
    # weighting needs a critic trained on a held out pool, and an invented
    # weight would be the easiest place in the system to smuggle in confidence.
    bands = {band: 0.0 for band in (LOW_BAND, MID_BAND, HIGH_BAND)}
    for values in distributions.values():
        for band, p in values.items():
            bands[band] += p / len(distributions)

    conflicts: List[str] = []
    if agreement.conflicted:
        for pair in agreement.pairs:
            if float(pair["divergence"]) > conflict_threshold():
                a, b = pair["agents"]
                ma, mb = pair["modal"]
                conflicts.append(
                    f"{a} reads this construct as {ma} while {b} reads it as {mb} "
                    f"(divergence {float(pair['divergence']):.3f})."
                )
        if agreement.rank is not None and agreement.rank < 0:
            conflicts.append(
                "The variant ranker and GC3 order these synonymous variants in opposite "
                "directions. On Welch's measured series GC3 is positive in both, so the ranker "
                "is the signal in doubt here."
            )

    pressure = agreement.domain
    contributors = [o for o in applying if o.agent in distributions]
    risk, risk_basis = _risk_from_bands(bands, pressure, contributors)
    expression = by_name.get(EXPRESSION)
    expected_tier = expression.score if expression is not None and expression.applies else None
    tier = None
    if expression is not None and expression.applies and expression.proba:
        tier = int(max(expression.proba, key=expression.proba.get))

    if conflicts:
        logger.info(f"Critic: conflicted, {len(conflicts)} disagreement(s); no tier shown.")
        return CriticVerdict(
            state=CONFLICTED,
            agreement=agreement,
            opinions=by_name,
            bands=bands,
            risk=risk,
            risk_basis=risk_basis,
            explanation=(
                "The agents disagree beyond the declared threshold, so no tier is shown. "
                "What would break the tie is a measurement, not another model: express one "
                "construct and read the yield."
            ),
            conflicts=tuple(conflicts),
        )

    # Out of domain means a feature outside the range an applying agent was
    # fitted on, which is a known extrapolation. An identity check that could
    # not run is a different thing: it leaves the question open rather than
    # answering it against the construct. It raises the risk figure and is
    # stated in the explanation, but it does not relabel the verdict, because a
    # state that fires on every input carries no information.
    out_of_domain = any(o.domain.out_of_range for o in applying)
    identity_unknown = any(
        o.domain.identity_applicable and not o.domain.identity_checked for o in applying
    )
    state = AGREED_OUT_OF_DOMAIN if out_of_domain else AGREED
    explanation = (
        "The agents that apply are concordant."
        if not out_of_domain
        else "The agents that apply are concordant, but this construct sits outside the feature "
        "range they were fitted on, so the tier is an extrapolation."
    )
    if identity_unknown:
        explanation += (
            " Whether anything in the training set resembles this construct is unknown: the "
            "training sequences are not in the working tree, so only the feature ranges were "
            "checked."
        )
    if len(distributions) == 1:
        explanation += (
            " One agent answered, so the agents were not actually compared: concordance here means "
            "nothing contradicted it."
        )

    logger.info(f"Critic: {state}, tier {tier}, risk {risk}, domain pressure {pressure:.2f}.")
    return CriticVerdict(
        state=state,
        agreement=agreement,
        opinions=by_name,
        tier=tier,
        expected_tier=expected_tier,
        bands=bands,
        risk=risk,
        risk_basis=risk_basis,
        explanation=explanation,
    )
