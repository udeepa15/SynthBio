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

def test_gc3_reads_only_the_wobble_position():
    """GC3 must see the third base of each codon and nothing else."""
    from src.features import calculate_gc3

    # Third bases: G, C, G, C -> 100%. First two bases are AT-rich throughout.
    assert calculate_gc3("AAGTTCAAGTTC") == 100.0
    # Third bases: A, T, A, T -> 0%, even though the codons are GC-rich overall.
    assert calculate_gc3("GGAGGTGGAGGT") == 0.0
    assert calculate_gc3("") == 0.0


def test_gc3_is_independent_of_the_protein():
    """Synonymous recoding must move GC3 while leaving the protein alone."""
    from src.features import calculate_gc3
    from Bio.Seq import Seq

    low, high = "GGAGGAGGA", "GGCGGCGGC"  # both encode Gly-Gly-Gly
    assert str(Seq(low).translate()) == str(Seq(high).translate())
    assert calculate_gc3(high) > calculate_gc3(low)


def test_rare_codon_fraction_counts_and_windows():
    from src.features import calculate_rare_codon_fraction

    seq = "AGGAGGATGATG"  # two rare (AGG, AGG), two common (ATG, ATG)
    assert calculate_rare_codon_fraction(seq) == 0.5
    assert calculate_rare_codon_fraction(seq, head_codons=2) == 1.0
    assert calculate_rare_codon_fraction("") == 0.0


def test_real_folding_energy_is_negative_and_tracks_structure():
    """A hairpin must fold more tightly than an unstructured sequence."""
    from src.features import calculate_folding_energy

    hairpin = calculate_folding_energy("GGGGGGGGGGAAAACCCCCCCCCC")
    unstructured = calculate_folding_energy("AAAAAAAAAAAAAAAAAAAAAAAA")
    if hairpin is None or unstructured is None:
        pytest.skip("ViennaRNA not installed.")

    assert hairpin < 0
    assert hairpin < unstructured


def test_genewide_folding_average_handles_short_sequences():
    from src.features import calculate_genewide_folding_avg, FOLDING_AVG_WINDOW_NT

    short = "AUGC" * 4
    assert len(short) < FOLDING_AVG_WINDOW_NT
    result = calculate_genewide_folding_avg(short)
    if result is None:
        pytest.skip("ViennaRNA not installed.")
    assert isinstance(result, float)


def test_extract_features_exposes_the_new_measures():
    from src.features import extract_features

    f = extract_features("ATGGCTGAAAAACTGCGTGATCTGGCTGAAAAACTGCGTGATCTGTAA")
    for key in ("gc3_content", "rare_codon_fraction", "genewide_folding_avg"):
        assert key in f


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
