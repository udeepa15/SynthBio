"""Tests for input sequence validation."""

import pytest

from src.validation import (
    validate_sequence,
    normalise_sequence,
    is_scorable,
    blocking_issues,
    advisory_issues,
)
from src.features import FOLDING_WINDOW_NT

VALID = "ATGAAACGCATTAGCCTGACCCTGCTGAGCGCGGCGATTCTGCTGACCGCGCTGAGCGCG"


def codes(issues):
    return {i.code for i in issues}


def test_genome_length_input_is_reported_as_such():
    """
    A whole-genome accession must be named, not described by its symptoms.

    Pasting one used to produce a frame error, a missing start codon and a count
    of tens of thousands of internal stops, none of which said the input was a
    genome rather than a gene.
    """
    from src.validation import MAX_CDS_NT

    genome = "ATG" + "AGC" * ((MAX_CDS_NT // 3) + 10)
    _, issues = validate_sequence(genome)

    assert [i.code for i in issues] == ["not_a_coding_sequence"]
    assert issues[0].blocking
    assert not is_scorable(issues)


def test_a_long_but_plausible_cds_is_still_scored():
    """The length guard must not reject an unusually long but legitimate gene."""
    long_cds = "ATG" + "GCT" * 3000 + "TAA"
    _, issues = validate_sequence(long_cds)

    assert "not_a_coding_sequence" not in codes(issues)
    assert is_scorable(issues)


def test_valid_sequence_has_no_issues():
    seq, issues = validate_sequence(VALID)
    assert seq == VALID
    assert issues == []
    assert is_scorable(issues)


def test_empty_input_is_blocking():
    _, issues = validate_sequence("   \n  ")
    assert codes(issues) == {"empty"}
    assert not is_scorable(issues)


def test_non_nucleotide_characters_are_blocking():
    _, issues = validate_sequence(VALID[:30] + "ZZZ" + VALID[33:])
    assert "invalid_characters" in codes(issues)
    assert not is_scorable(issues)


def test_frame_not_multiple_of_three_is_blocking():
    _, issues = validate_sequence(VALID[:-1])
    assert "frame" in codes(issues)
    assert not is_scorable(issues)


def test_missing_start_codon_is_blocking():
    _, issues = validate_sequence("GTG" + VALID[3:])
    assert "start_codon" in codes(issues)
    assert not is_scorable(issues)


def test_internal_stop_codon_is_blocking():
    seq = "ATG" + "AAA" + "TAA" + "CTG" * 15
    _, issues = validate_sequence(seq)
    assert "internal_stop" in codes(issues)
    assert not is_scorable(issues)


def test_trailing_stop_codon_is_allowed():
    seq = "ATG" + "CTG" * 20 + "TAA"
    _, issues = validate_sequence(seq)
    assert "internal_stop" not in codes(issues)


def test_sequence_shorter_than_folding_window_is_blocking():
    seq = "ATG" + "CTG" * 5
    assert len(seq) < FOLDING_WINDOW_NT
    _, issues = validate_sequence(seq)
    assert "short_for_folding" in codes(issues)
    assert not is_scorable(issues)


def test_ambiguity_codes_are_advisory_not_blocking():
    seq = VALID[:-3] + "NNN"
    _, issues = validate_sequence(seq)
    assert "ambiguity_codes" in codes(issues)
    assert is_scorable(issues)
    assert len(advisory_issues(issues)) >= 1
    assert blocking_issues(issues) == []


def test_fasta_header_and_whitespace_are_stripped():
    raw = f">test sequence description\n{VALID[:30]}\n{VALID[30:]}\n"
    seq, issues = validate_sequence(raw)
    assert seq == VALID
    assert issues == []


def test_normalise_removes_digits_and_case():
    assert normalise_sequence("1 atg  20 ccc\n") == "ATGCCC"


@pytest.mark.parametrize("issue_field", ["title", "detail", "remedy"])
def test_every_issue_explains_itself(issue_field):
    """Each issue must carry a human-readable explanation and a remedy."""
    _, issues = validate_sequence("GTGZZ")
    assert issues
    for issue in issues:
        text = getattr(issue, issue_field)
        assert isinstance(text, str) and text.strip()
