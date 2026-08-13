import os
import math
import pytest
import pandas as pd
import numpy as np
from src.features import calculate_protein_features, extract_features
from src.data_processing import add_protein_features_to_df, process_protein_features_dataset

def test_calculate_protein_features_valid():
    # Valid DNA sequence starting with ATG
    seq = "ATGAAACGCATTAGCCTGACCCTGCTGAGCGCGGCGATTCTGCTGACCGCGCTGAGCGCG"
    res = calculate_protein_features(seq)
    
    assert "gravy_score" in res
    assert "pI" in res
    assert not math.isnan(res["gravy_score"])
    assert not math.isnan(res["pI"])
    assert 0.0 <= res["pI"] <= 14.0

def test_calculate_protein_features_with_stop_codon():
    # DNA sequence with stop codon TAA in the middle
    seq = "ATGGCATACCCATACGACGTCCCAGACTACGCTTAAAGCTGA"
    res = calculate_protein_features(seq)
    
    assert "gravy_score" in res
    assert "pI" in res
    assert not math.isnan(res["gravy_score"])
    assert not math.isnan(res["pI"])

def test_calculate_protein_features_invalid_graceful():
    # Malformed sequence / non-DNA characters
    invalid_seq = "INVALID_12345_XYZ"
    res = calculate_protein_features(invalid_seq)
    assert math.isnan(res["gravy_score"])
    assert math.isnan(res["pI"])

    # Empty sequence
    empty_res = calculate_protein_features("")
    assert math.isnan(empty_res["gravy_score"])
    assert math.isnan(empty_res["pI"])

def test_extract_features_includes_protein_metrics():
    seq = "ATGAAACGCATTAGCCTGACCCTGCTGAGCGCGGCGATTCTGCTGACCGCGCTGAGCGCG"
    feats = extract_features(seq)
    assert "gc_content" in feats
    assert "codon_adaptation_index" in feats
    assert "mrna_folding_stability" in feats
    assert "gravy_score" in feats
    assert "pI" in feats

def test_add_protein_features_to_df():
    df = pd.DataFrame({
        "gene_id": ["G1", "G2"],
        "sequence": [
            "ATGAAACGCATTAGCCTGACCCTGCTGAGCGCGGCGATTCTGCTGACCGCGCTGAGCGCG",
            "INVALID_MALFORMED_DNA"
        ]
    })
    updated_df = add_protein_features_to_df(df, sequence_col="sequence")
    assert "gravy_score" in updated_df.columns
    assert "pI" in updated_df.columns
    assert not math.isnan(updated_df.loc[0, "gravy_score"])
    assert math.isnan(updated_df.loc[1, "gravy_score"])

def test_process_protein_features_dataset_export(tmp_path):
    input_csv = os.path.join(tmp_path, "test_input.csv")
    output_csv = os.path.join(tmp_path, "feature_table_boel_5features.csv")
    
    sample_df = pd.DataFrame({
        "gene_id": ["G01", "G02"],
        "sequence": [
            "ATGAAACGCATTAGCCTGACCCTGCTGAGCGCGGCGATTCTGCTGACCGCGCTGAGCGCG",
            "ATGTTTGTTTTTCTTGTTTTATTGCCACTAGTCTCTAGTCAGTGTGTTAATCTTACAACC"
        ]
    })
    sample_df.to_csv(input_csv, index=False)
    
    res_df = process_protein_features_dataset(input_csv_path=input_csv, output_csv_path=output_csv)
    assert os.path.exists(output_csv)
    assert "gravy_score" in res_df.columns
    assert "pI" in res_df.columns
    assert len(res_df) == 2
