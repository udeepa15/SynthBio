"""
Feature engineering module for computing construct biophysical & protein features:
1. GC Content (%)
2. Codon Adaptation Index (CAI) for E. coli
3. mRNA folding stability score (MFE delta G)
4. GRAVY Score (Hydrophobicity)
5. Isoelectric Point (pI)
"""

import math
import logging
from typing import Dict, Union, List
import pandas as pd

# Configure logger
logger = logging.getLogger("SynthBio.features")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [SynthBio.features] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

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
        logger.warning("Empty sequence provided for GC calculation.")
        return 0.0
    gc_count = seq.count("G") + seq.count("C")
    gc = round((gc_count / len(seq)) * 100.0, 2)
    return gc


def calculate_cai(sequence: str, codon_table: Dict[str, float] = ECOLI_CODON_W) -> float:
    """Calculate Codon Adaptation Index (CAI) for an open reading frame sequence."""
    seq = sequence.upper().replace(" ", "").replace("\n", "")
    if len(seq) < 3:
        logger.warning(f"Sequence length ({len(seq)} bp) too short for CAI codon analysis.")
        return 0.0
    
    codons = [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]
    valid_w_values = []
    
    for codon in codons:
        if codon in codon_table and codon_table[codon] > 0:
            valid_w_values.append(math.log(codon_table[codon]))
            
    if not valid_w_values:
        logger.warning("No valid codons found matching E. coli CAI adaptiveness table.")
        return 0.0
        
    cai = round(math.exp(sum(valid_w_values) / len(valid_w_values)), 3)
    return cai


def estimate_mrna_folding_stability(sequence: str) -> float:
    """
    Estimate 5' mRNA secondary structure minimum free energy (MFE in kcal/mol).
    Lower (more negative) values indicate tighter folding (potential translation blockage).
    """
    seq = sequence.upper().replace(" ", "").replace("\n", "")
    if not seq:
        logger.warning("Empty sequence provided for MFE folding stability calculation.")
        return 0.0
    
    region_5prime = seq[:48]
    gc_fraction = (region_5prime.count("G") + region_5prime.count("C")) / max(len(region_5prime), 1)
    mfe = -1.5 * len(region_5prime) * (gc_fraction ** 1.5)
    return round(mfe, 2)


def calculate_protein_features(sequence: str) -> Dict[str, float]:
    """
    Translate DNA sequence to amino acid sequence (stopping at first stop codon)
    and compute GRAVY score (hydrophobicity) and Isoelectric Point (pI).
    """
    if not isinstance(sequence, str) or not sequence.strip():
        logger.warning("Invalid or empty string passed for protein feature extraction.")
        return {"gravy_score": float("nan"), "pI": float("nan")}

    try:
        from Bio.Seq import Seq
        from Bio.SeqUtils.ProtParam import ProteinAnalysis

        seq = sequence.upper().replace(" ", "").replace("\n", "")
        seq_obj = Seq(seq)
        protein_seq = str(seq_obj.translate(to_stop=True))

        if not protein_seq or any(c in protein_seq for c in ("*", "X", "J", "O", "U", "B", "Z")):
            clean_protein = "".join([c for c in protein_seq if c in "ACDEFGHIKLMNPQRSTVWY"])
            if not clean_protein:
                logger.warning(f"Could not translate DNA to standard amino acids. Sequence length: {len(seq)} bp.")
                return {"gravy_score": float("nan"), "pI": float("nan")}
            protein_seq = clean_protein

        analyzer = ProteinAnalysis(protein_seq)
        gravy = round(analyzer.gravy(), 4)
        pi = round(analyzer.isoelectric_point(), 4)
        return {"gravy_score": gravy, "pI": pi}
    except Exception as err:
        logger.error(f"Protein feature extraction failed: {str(err)}", exc_info=False)
        return {"gravy_score": float("nan"), "pI": float("nan")}


def extract_features(sequence: str) -> Dict[str, float]:
    """Extract all 5 computed features (GC%, CAI, MFE, GRAVY, pI) for a given sequence."""
    logger.info(f"Extracting features for sequence length {len(sequence)} bp...")
    features = {
        "gc_content": calculate_gc_content(sequence),
        "codon_adaptation_index": calculate_cai(sequence),
        "mrna_folding_stability": estimate_mrna_folding_stability(sequence)
    }
    protein_feats = calculate_protein_features(sequence)
    features.update(protein_feats)
    logger.info(
        f"Feature Extraction Summary -> GC: {features['gc_content']}%, CAI: {features['codon_adaptation_index']}, "
        f"MFE: {features['mrna_folding_stability']} kcal/mol, GRAVY: {features.get('gravy_score')}, pI: {features.get('pI')}"
    )
    return features


def extract_features_df(df: pd.DataFrame, sequence_col: str = "sequence") -> pd.DataFrame:
    """Batch extract core features for a pandas DataFrame of sequences."""
    logger.info(f"Batch extracting features for DataFrame of {len(df)} constructs...")
    features_list = df[sequence_col].apply(extract_features).tolist()
    return pd.DataFrame(features_list)


def fetch_ncbi_sequence(accession_id: str, email: str = "synthbio@example.com") -> str:
    """Fetch nucleotide FASTA sequence from NCBI Nucleotide database using accession number."""
    acc = accession_id.strip().upper()
    if not acc:
        logger.error("Empty NCBI Accession ID provided.")
        raise ValueError("NCBI Accession ID cannot be empty.")
    
    logger.info(f"Attempting NCBI Entrez fetch for Accession '{acc}'...")
    
    # Try BioPython Entrez first
    try:
        from Bio import Entrez, SeqIO
        Entrez.email = email
        handle = Entrez.efetch(db="nuccore", id=acc, rettype="fasta", retmode="text")
        record = SeqIO.read(handle, "fasta")
        handle.close()
        seq = str(record.seq).upper()
        if seq:
            logger.info(f"Successfully fetched {len(seq)} bp from NCBI via Bio.Entrez for '{acc}'.")
            return seq
    except Exception as err:
        logger.warning(f"Bio.Entrez fetch failed for '{acc}': {str(err)}. Attempting HTTP eutils fallback...")

    # Fallback to direct HTTP request to NCBI eutils API
    try:
        import urllib.request
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={acc}&rettype=fasta&retmode=text"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            fasta_data = response.read().decode('utf-8')
            lines = fasta_data.strip().split('\n')
            seq = "".join(line.strip() for line in lines if not line.startswith(">"))
            if not seq:
                raise ValueError(f"No sequence data returned from NCBI for accession '{acc}'.")
            logger.info(f"Successfully fetched {len(seq)} bp from NCBI via HTTP eutils for '{acc}'.")
            return seq.upper()
    except Exception as e:
        logger.error(f"Failed to fetch sequence from NCBI for accession '{acc}': {str(e)}")
        raise ValueError(f"Failed to fetch sequence from NCBI for accession '{acc}': {str(e)}")
