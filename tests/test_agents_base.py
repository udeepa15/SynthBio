"""
The agent interface. These tests are about the contract rather than any model.

An agent that does not exist, or has not passed its gate, must behave exactly
like an agent that has no opinion about this particular input, because that is
what makes the system buildable one agent at a time.
"""

import math

import pandas as pd
import pytest

from src.agents.base import (
    AGENT_QUESTIONS,
    EXPRESSION,
    GATE_BLOCKED,
    GATE_PASSED,
    HIGH_BAND,
    LOW_BAND,
    MID_BAND,
    AgentOpinion,
    AgentProvenance,
    AgentRequest,
    DomainCheck,
    GateResult,
    abstain,
    binary_to_bands,
    tiers_to_bands,
)
from src.agents.domain import describe_domain
from src.agents.expression import ExpressionAgent
from src.agents.solubility import SolubilityAgent
from src.agents.structure import StructureAgent
from src.agents.variant import VariantAgent

SEQ = "ATGAAACGCATTAGCCTGACCCTGCTGAGCGCGGCGATTCTGCTGACCGCGCTGAGCGCGTAA"


def test_abstention_is_an_answer_not_an_error():
    opinion = abstain(EXPRESSION, "no artifact")
    assert opinion.applies is False
    assert opinion.abstain_reason == "no artifact"
    assert opinion.score is None and opinion.proba is None
    assert opinion.question == AGENT_QUESTIONS[EXPRESSION]


def test_every_agent_abstains_rather_than_raising_without_its_artifact():
    request = AgentRequest(sequence=SEQ)
    for agent in (ExpressionAgent(model=None), SolubilityAgent(), StructureAgent(), VariantAgent(load=False)):
        opinion = agent.opine(request)
        assert opinion.applies is False
        assert opinion.abstain_reason, f"{agent.name} abstained without saying why"


def test_a_gate_that_has_not_been_run_is_not_a_pass():
    assert GateResult().passed is False
    assert GateResult(status=GATE_BLOCKED).passed is False
    assert GateResult(status=GATE_PASSED).passed is True
    # The displayed line must always say whether a date exists.
    assert "never measured" in GateResult().as_line()


def test_declared_agents_carry_a_written_down_gate():
    for agent in (SolubilityAgent(), StructureAgent()):
        gate = agent.provenance().gate
        assert gate.passed is False
        assert len(gate.summary) > 40, "a declared agent must say what its gate requires"


def test_variant_agent_abstains_on_a_single_sequence():
    opinion = VariantAgent(load=False).opine(AgentRequest(sequence=SEQ))
    assert opinion.applies is False
    assert "one sequence" in opinion.abstain_reason


def test_structure_agent_distinguishes_a_missing_input_from_a_missing_model():
    without = StructureAgent().opine(AgentRequest(sequence=SEQ))
    with_leader = StructureAgent().opine(AgentRequest(sequence=SEQ, leader_5prime="GGAGA" * 10))
    assert without.abstain_reason != with_leader.abstain_reason
    assert "vector" in without.abstain_reason


def test_domain_is_not_inside_when_the_identity_check_could_not_run():
    check = DomainCheck(out_of_range=(), nearest_identity=None, identity_checked=False)
    assert check.in_training_range is True
    assert check.inside is False, "an unrun identity check must not count as in domain"


def test_describe_domain_reports_a_feature_outside_its_range():
    row = pd.DataFrame([{"GC_percent": 95.0}])
    check = describe_domain(row, {"GC_percent": {"min": 20.0, "max": 80.0}}, sequence=None)
    assert [o["feature"] for o in check.out_of_range] == ["GC_percent"]
    assert check.in_training_range is False


def test_three_band_mapping_keeps_the_mass():
    bands = tiers_to_bands({0: 0.1, 1: 0.1, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2})
    assert math.isclose(sum(bands.values()), 1.0)
    assert math.isclose(bands[LOW_BAND], 0.2)
    assert math.isclose(bands[MID_BAND], 0.4)
    assert math.isclose(bands[HIGH_BAND], 0.4)


def test_an_undecided_binary_agent_reads_as_undecided_not_as_middle_confidence():
    undecided = binary_to_bands(0.5)
    assert math.isclose(undecided[MID_BAND], 1.0)
    confident = binary_to_bands(1.0)
    assert math.isclose(confident[HIGH_BAND], 1.0)
    assert math.isclose(confident[MID_BAND], 0.0)


def test_features_are_computed_once_per_request():
    request = AgentRequest(sequence=SEQ)
    first = request.resolved_features()
    assert request.resolved_features() is first


def test_top_attribution_orders_by_magnitude_including_negatives():
    opinion = AgentOpinion(
        agent=EXPRESSION,
        applies=True,
        attribution={"a": 0.1, "b": -0.9, "c": 0.4},
        provenance=AgentProvenance(),
    )
    assert [name for name, _ in opinion.top_attribution(2)] == ["b", "c"]
