"""
Agent D, initiation structure, and the vector leader it depends on.

The thing under test is mostly the precondition. Agent D is blocked by a
missing input rather than a missing dataset, so the behaviour that matters is
what it does when the input is absent, when it is present but wrong, and when
the gate that justifies asking for it has not been passed.
"""

import json
import os

import pandas as pd
import pytest

from src import vectors
from src.agents.base import AgentRequest
from src.agents.structure import (
    DEFAULT_METADATA_PATH,
    STRUCTURE_FEATURES,
    StructureAgent,
    structure_feature_frame,
)
from src.features import (
    calculate_folding_energy,
    calculate_initiation_folding,
    initiation_window,
)

SEQ = (
    "ATGAAACGCATTAGCCTGACCCTGCTGAGCGCGGCGATTCTGCTGACCGCGCTGAGCGCG"
    "GAACTGAGCCTGAAAGCGCTGGATGAAATTCGCAAAGCGCTGGAAGGCGCGAAAGATTAA"
)


# --- the leader -------------------------------------------------------------


def test_the_default_vector_carries_a_real_leader_with_a_source():
    vector = vectors.get(vectors.DEFAULT_VECTOR)
    assert vector.leader, "the default vector must declare a leader"
    assert set(vector.leader) <= set("ACGT")
    assert "Welch" in vector.source, "a leader without a stated source is a guess"


def test_an_undeclared_vector_declares_no_leader():
    assert vectors.get(vectors.NO_VECTOR).leader == ""
    assert vectors.get(None).leader == ""
    assert vectors.get("a vector nobody has heard of").leader == ""


def test_a_supplied_leader_beats_a_named_vector_and_is_cleaned():
    supplied = vectors.resolve_leader(vectors.DEFAULT_VECTOR, "  ggag uua\n")
    assert supplied == "GGAGTTA"
    assert vectors.resolve_leader(vectors.DEFAULT_VECTOR, "") == vectors.PET_T7_LEADER
    assert vectors.resolve_leader(vectors.DEFAULT_VECTOR, "!!!") == vectors.PET_T7_LEADER


# --- the window -------------------------------------------------------------


def test_the_initiation_window_is_leader_tail_plus_insert_head():
    window = initiation_window("A" * 100, SEQ, upstream=42, downstream=50)
    assert len(window) == 92
    assert window[:42] == "A" * 42
    assert window[42:] == SEQ[:50]


def test_without_a_leader_the_window_is_the_insert_head_alone():
    assert initiation_window("", SEQ, upstream=42, downstream=50) == SEQ[:50]


def test_the_leader_changes_the_folding_energy():
    with_leader = calculate_initiation_folding(vectors.PET_T7_LEADER, SEQ)
    without = calculate_initiation_folding("", SEQ)
    if with_leader is None or without is None:
        pytest.skip("ViennaRNA is unavailable.")
    assert with_leader != without, (
        "if the leader cannot move the folding energy there is no point asking for it"
    )
    assert with_leader <= without, "a longer window cannot fold less tightly than its own suffix"


def test_initiation_folding_is_a_real_fold_not_the_gc_proxy():
    energy = calculate_initiation_folding("", SEQ)
    direct = calculate_folding_energy(SEQ[:50])
    if energy is None:
        pytest.skip("ViennaRNA is unavailable.")
    assert energy == direct


# --- the agent --------------------------------------------------------------


def test_no_leader_means_no_opinion():
    opinion = StructureAgent(load=False).opine(AgentRequest(sequence=SEQ))
    assert opinion.applies is False
    assert "vector" in opinion.abstain_reason
    assert "5' leader" in opinion.abstain_reason


def test_a_leader_without_an_artifact_gives_a_different_reason():
    without = StructureAgent(load=False).opine(AgentRequest(sequence=SEQ))
    with_leader = StructureAgent(load=False).opine(
        AgentRequest(sequence=SEQ, leader_5prime=vectors.PET_T7_LEADER)
    )
    assert with_leader.applies is False
    assert with_leader.abstain_reason != without.abstain_reason
    assert "artifact" in with_leader.abstain_reason


def test_the_feature_set_is_the_initiation_region_and_nothing_else():
    for excluded in ("gravy_score", "pI", "length_codons", "CAI", "folding_energy"):
        assert excluded not in STRUCTURE_FEATURES
    assert "initiation_folding_leader" in STRUCTURE_FEATURES
    assert "gc_head48" in STRUCTURE_FEATURES


def test_the_features_the_agent_computes_are_the_ones_it_was_fitted_on():
    features = StructureAgent.initiation_features(SEQ, vectors.PET_T7_LEADER)
    assert set(features) == set(STRUCTURE_FEATURES)
    frame = structure_feature_frame(pd.DataFrame([features]))
    assert list(frame.columns) == list(STRUCTURE_FEATURES)


def test_structure_feature_frame_refuses_a_matrix_missing_a_feature():
    with pytest.raises(ValueError, match="missing"):
        structure_feature_frame(pd.DataFrame([{"gc_head48": 50.0}]))


class _FakeModel:
    """Predicts the lowest band whenever the leader window folds tightly."""

    def predict_proba(self, frame):
        import numpy as np

        tight = (frame["initiation_folding_leader"].to_numpy(dtype=float) < -10.0).astype(float)
        return np.column_stack([1.0 - tight, tight])


def test_an_ungated_artifact_does_not_speak():
    agent = StructureAgent(model=_FakeModel(), require_gate=True)
    agent.metadata = {"gate": {"status": "failed", "summary": "Gate D failed on the ablation."}}
    opinion = agent.opine(AgentRequest(sequence=SEQ, leader_5prime=vectors.PET_T7_LEADER))
    assert opinion.applies is False
    assert "Gate D" in opinion.abstain_reason


def test_a_gated_agent_returns_a_binary_probability_and_its_features():
    agent = StructureAgent(model=_FakeModel(), require_gate=False)
    opinion = agent.opine(AgentRequest(sequence=SEQ, leader_5prime=vectors.PET_T7_LEADER))
    assert opinion.applies is True
    assert set(opinion.proba) == {0, 1}
    assert abs(sum(opinion.proba.values()) - 1.0) < 1e-9
    assert opinion.score == opinion.proba[1]
    assert set(opinion.features) == set(STRUCTURE_FEATURES)


def test_a_leader_other_than_the_fitted_one_is_called_out_in_the_domain():
    agent = StructureAgent(model=_FakeModel(), require_gate=False)
    agent.metadata = {"vector": {"key": "pET", "leader": vectors.PET_T7_LEADER, "source": "test"}}
    opinion = agent.opine(AgentRequest(sequence=SEQ, leader_5prime="GGAGGAGGAGGA"))
    notes = " ".join(opinion.domain.notes)
    assert "differs" in notes
    assert opinion.domain.identity_checked is False


# --- the gate that was actually measured ------------------------------------


@pytest.mark.skipif(not os.path.exists(DEFAULT_METADATA_PATH), reason="Agent D has not been trained here.")
def test_the_recorded_gate_decides_whether_the_vector_input_is_worth_asking_for():
    """
    The second clause of Gate D is a decision about the interface.

    If declaring the vector does not separate from the insert alone, the vector
    selector should not be added to the application, and the agent must abstain
    rather than quietly using a leader the data does not justify asking for.
    """
    with open(DEFAULT_METADATA_PATH, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    detail = metadata.get("gate_detail") or {}
    assert "leader_vs_insert" in detail, "the ablation must be recorded, pass or fail"

    separated = detail["leader_vs_insert"]["separated"]
    agent = StructureAgent()
    if not separated:
        assert agent.gate().passed is False
        opinion = agent.opine(AgentRequest(sequence=SEQ, leader_5prime=vectors.PET_T7_LEADER))
        assert opinion.applies is False
