"""
The critic. The thing being tested is mostly what it refuses to say.

A system that answers 60 per cent of inputs well is a result. A system that
answers all of them at the majority baseline is not, and the states below are
how the difference reaches the screen.
"""

import math

from src.agents.base import (
    EXPRESSION,
    SOLUBILITY,
    VARIANT,
    AgentOpinion,
    AgentProvenance,
    DomainCheck,
    abstain,
)
from src.agents.critic import (
    AGREED,
    AGREED_OUT_OF_DOMAIN,
    CONFLICTED,
    NOT_ANSWERABLE,
    combine,
    jensen_shannon,
    score_agreement,
)


def tier_opinion(proba, out_of_range=(), identity_checked=True, agent=EXPRESSION):
    return AgentOpinion(
        agent=agent,
        applies=True,
        score=float(sum(k * v for k, v in proba.items())),
        proba=proba,
        domain=DomainCheck(
            out_of_range=tuple(out_of_range),
            nearest_identity=0.9 if identity_checked else None,
            identity_checked=identity_checked,
        ),
        provenance=AgentProvenance(),
    )


HIGH = {0: 0.02, 1: 0.02, 2: 0.04, 3: 0.07, 4: 0.35, 5: 0.50}
LOW = {0: 0.55, 1: 0.20, 2: 0.10, 3: 0.08, 4: 0.04, 5: 0.03}


def test_no_agent_applying_is_not_answerable():
    verdict = combine([abstain(EXPRESSION, "no artifact"), abstain(VARIANT, "one sequence only")])
    assert verdict.state == NOT_ANSWERABLE
    assert verdict.tier is None
    assert "GRAVY" in verdict.explanation, "the fallback must name the validated heuristic"


def test_one_agent_passes_through_and_says_it_was_not_corroborated():
    verdict = combine([tier_opinion(HIGH), abstain(VARIANT, "one sequence only")])
    assert verdict.state == AGREED
    assert verdict.tier == 5
    assert verdict.agreement.distributional is None, "one distribution is not agreement"
    assert "not actually compared" in verdict.explanation


def test_a_feature_outside_the_training_range_labels_the_tier_an_extrapolation():
    outside = [{"feature": "GC_percent", "value": 95.0, "min": 20.0, "max": 80.0, "side": "above"}]
    verdict = combine([tier_opinion(HIGH, out_of_range=outside)])
    assert verdict.state == AGREED_OUT_OF_DOMAIN
    assert "extrapolation" in verdict.explanation


def test_an_unavailable_identity_check_is_stated_but_does_not_relabel_the_verdict():
    verdict = combine([tier_opinion(HIGH, identity_checked=False)])
    assert verdict.state == AGREED
    assert "unknown" in verdict.explanation
    assert verdict.agreement.domain > 0.0, "not knowing must raise the domain pressure"


def test_two_agents_reading_the_construct_differently_is_conflicted_with_no_tier():
    verdict = combine([tier_opinion(HIGH), tier_opinion(LOW, agent=SOLUBILITY)])
    assert verdict.state == CONFLICTED
    assert verdict.tier is None, "a conflicted verdict must not show a tier"
    assert verdict.conflicts, "a conflict must name which agents disagree and how"
    assert "measure" in verdict.explanation


def test_two_agents_that_agree_are_not_conflicted():
    verdict = combine([tier_opinion(HIGH), tier_opinion(HIGH, agent=SOLUBILITY)])
    assert verdict.state in (AGREED, AGREED_OUT_OF_DOMAIN)
    assert verdict.agreement.distributional is not None
    assert verdict.agreement.distributional < 0.05


def test_a_ranker_pointing_against_gc3_is_a_conflict_on_its_own():
    verdict = combine([tier_opinion(HIGH)], rank_agreement=-0.6)
    assert verdict.state == CONFLICTED
    assert any("opposite directions" in c for c in verdict.conflicts)


def test_risk_comes_from_the_agents_and_not_from_a_parallel_rule():
    high = combine([tier_opinion(HIGH)])
    low = combine([tier_opinion(LOW)])
    assert low.risk > high.risk, "a construct the agents put in the low band must carry more risk"
    assert "calibrated" in low.risk_basis, "the risk figure must say what it is not"


def test_the_critic_never_claims_fitted_thresholds_or_a_trained_model():
    verdict = combine([tier_opinion(HIGH)])
    assert verdict.thresholds_fitted is False
    assert verdict.critic_model is None


def test_jensen_shannon_is_bounded_and_symmetric():
    p, q = {0: 1.0, 1: 0.0}, {0: 0.0, 1: 1.0}
    assert math.isclose(jensen_shannon(p, q), 1.0, rel_tol=1e-6)
    assert math.isclose(jensen_shannon(p, p), 0.0, abs_tol=1e-9)
    a, b = {0: 0.7, 1: 0.3}, {0: 0.2, 1: 0.8}
    assert math.isclose(jensen_shannon(a, b), jensen_shannon(b, a), rel_tol=1e-9)


def test_agreement_counts_abstentions_separately_from_disagreement():
    score = score_agreement([tier_opinion(HIGH), abstain(VARIANT, "one sequence only")])
    assert score.n_applying == 1
    assert score.n_abstained == 1
    assert score.conflicted is False
