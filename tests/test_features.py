import pytest
from src.features import calculate_gc_content, calculate_cai, estimate_mrna_folding_stability, extract_features

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
