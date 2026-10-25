import pytest
from src.features import calculate_gc_content, calculate_cai, estimate_mrna_folding_stability, extract_features, fetch_ncbi_sequence

def test_calculate_gc_content():
    # ATCG -> 50% GC
    assert calculate_gc_content("ATCG") == 50.0
    # GGGG -> 100% GC
    assert calculate_gc_content("GGGG") == 100.0
    # AAAA -> 0% GC
    assert calculate_gc_content("AAAA") == 0.0

def test_calculate_cai():
    # Test valid CDS sequence
    seq = "ATGCTGACCGCG"
    cai = calculate_cai(seq)
    assert 0.0 <= cai <= 1.0

def test_extract_features():
    seq = "ATGAAACGCATTAGCCTGACCCTG"
    features = extract_features(seq)
    assert "gc_content" in features
    assert "codon_adaptation_index" in features
    assert "mrna_folding_stability" in features

def test_parse_fasta_records_splits_multiple_entries():
    """The CDS fetch reads multi-record FASTA, which is how NCBI returns CDSs."""
    from src.features import _parse_fasta_records

    text = ">lcl|one [gene=A]\nATGAAA\nTTTTAA\n>lcl|two [gene=B]\nATGCCC\n"
    records = _parse_fasta_records(text)

    assert [h.split()[0] for h in [r[0] for r in records]] == ["lcl|one", "lcl|two"]
    assert [r[1] for r in records] == ["ATGAAATTTTAA", "ATGCCC"]


def test_parse_fasta_records_handles_empty_input():
    from src.features import _parse_fasta_records

    assert _parse_fasta_records("") == []


def test_fetch_ncbi_cds_rejects_empty_accession():
    from src.features import fetch_ncbi_cds

    with pytest.raises(ValueError):
        fetch_ncbi_cds("   ")


def test_fetch_ncbi_sequence_invalid():
    # Test empty or invalid accession raises ValueError
    with pytest.raises(ValueError):
        fetch_ncbi_sequence("")
