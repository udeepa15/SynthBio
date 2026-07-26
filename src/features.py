"""
Feature engineering module for computing 3 core construct features:
1. GC Content (%)
2. Codon Adaptation Index (CAI) for E. coli
3. mRNA folding stability score (MFE delta G)
"""

import math
from typing import Dict, Union, List
import pandas as pd

# E. coli K-12 codon usage relative adaptiveness (simplified representative table)
ECOLI_CODON_W: Dict[str, float] = {
    'TTT': 0.58, 'TTC': 1.00, 'TTA': 0.14, 'TTG': 0.13,
    'CTT': 0.12, 'CTC': 0.10, 'CTA': 0.04, 'CTG': 1.00,
    'ATT': 0.49, 'ATC': 1.00, 'ATA': 0.08, 'ATG': 1.00,
    'GTT': 0.54, 'GTC': 0.38, 'GTA': 0.23, 'GTG': 1.00,
    'TCT': 0.86, 'TCC': 0.74, 'TCA': 0.14, 'TCG': 0.14,
    'CCT': 0.18, 'CCC': 0.13, 'CCA': 0.20, 'CCG': 1.00,
    'ACT': 0.89, 'ACC': 1.00, 'ACA': 0.14, 'ACG': 0.25,
    'GCT': 1.00, 'GCC': 0.26, 'GCA': 0.58, 'GCG': 0.38,
    'TAT': 0.43, 'TAC': 1.00, 'TAA': 1.00, 'TAG': 0.08,
    'CAT': 0.57, 'CAC': 1.00, 'CAA': 0.34, 'CAG': 1.00,
    'AAT': 0.45, 'AAC': 1.00, 'AAA': 1.00, 'AAG': 0.25,
    'GAT': 0.63, 'GAC': 1.00, 'GAA': 1.00, 'GAG': 0.31,
    'TGT': 0.46, 'TGC': 1.00, 'TGA': 0.30, 'TGG': 1.00,
    'CGT': 1.00, 'CGC': 0.64, 'CGA': 0.06, 'CGG': 0.08,
    'AGT': 0.16, 'AGC': 0.41, 'AGA': 0.04, 'AGG': 0.02,
    'GGT': 1.00, 'GGC': 0.72, 'GGA': 0.13, 'GGG': 0.15,
}

def calculate_gc_content(sequence: str) -> float:
    """Calculate GC percentage of a DNA/RNA sequence."""
    seq = sequence.upper().replace(" ", "").replace("\n", "")
    if not seq:
        return 0.0
    gc_count = seq.count("G") + seq.count("C")
    return round((gc_count / len(seq)) * 100.0, 2)


def calculate_cai(sequence: str, codon_table: Dict[str, float] = ECOLI_CODON_W) -> float:
    """Calculate Codon Adaptation Index (CAI) for an open reading frame sequence."""
    seq = sequence.upper().replace(" ", "").replace("\n", "")
    if len(seq) < 3:
        return 0.0
    
    codons = [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]
    valid_w_values = []
    
    for codon in codons:
        if codon in codon_table and codon_table[codon] > 0:
            valid_w_values.append(math.log(codon_table[codon]))
            
    if not valid_w_values:
        return 0.0
        
    return round(math.exp(sum(valid_w_values) / len(valid_w_values)), 3)


def estimate_mrna_folding_stability(sequence: str) -> float:
    """
    Estimate 5' mRNA secondary structure minimum free energy (MFE in kcal/mol).
    Note: Lower (more negative) values indicate tighter folding (potential translation blockage).
    """
    seq = sequence.upper().replace(" ", "").replace("\n", "")
    if not seq:
        return 0.0
    
    # Heuristic estimation based on GC count and GC dynamic runs for early 5' region
    region_5prime = seq[:48]
    gc_fraction = (region_5prime.count("G") + region_5prime.count("C")) / max(len(region_5prime), 1)
    
    # Base estimated MFE (kcal/mol)
    mfe = -1.5 * len(region_5prime) * (gc_fraction ** 1.5)
    return round(mfe, 2)


def extract_features(sequence: str) -> Dict[str, float]:
    """Extract all 3 core computed features for a given gene sequence."""
    return {
        "gc_content": calculate_gc_content(sequence),
        "codon_adaptation_index": calculate_cai(sequence),
        "mrna_folding_stability": estimate_mrna_folding_stability(sequence)
    }


def extract_features_df(df: pd.DataFrame, sequence_col: str = "sequence") -> pd.DataFrame:
    """Batch extract core features for a pandas DataFrame of sequences."""
    features_list = df[sequence_col].apply(extract_features).tolist()
    return pd.DataFrame(features_list)
