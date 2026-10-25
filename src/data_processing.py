from typing import Optional, Tuple
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.features import extract_features_df, calculate_protein_features

def categorize_expression(yield_val: float) -> str:
    """Categorize continuous expression yield into High, Medium, or Low tier."""
    if yield_val >= 70.0:
        return "High"
    elif yield_val >= 30.0:
        return "Medium"
    else:
        return "Low"

def add_protein_features_to_df(df: pd.DataFrame, sequence_col: str = "sequence") -> pd.DataFrame:
    """
    Translates DNA sequence column into amino acid sequences (stopping at the first stop codon),
    calculates GRAVY score and Isoelectric Point (pI) via Biopython, and appends `gravy_score` and `pI`.
    Handles translation errors or malformed sequences gracefully by assigning NaN.
    """
    if sequence_col not in df.columns:
        raise KeyError(f"Sequence column '{sequence_col}' not found in DataFrame columns: {list(df.columns)}")
        
    protein_feats = df[sequence_col].apply(calculate_protein_features).tolist()
    protein_df = pd.DataFrame(protein_feats)
    df["gravy_score"] = protein_df["gravy_score"]
    df["pI"] = protein_df["pI"]
    return df

def process_protein_features_dataset(
    input_csv_path: str,
    output_csv_path: str = "data/processed/feature_table_boel_5features.csv",
    sequence_col: str = "sequence"
) -> pd.DataFrame:
    """
    Reads an existing CSV containing a DNA sequence column, calculates GRAVY score and pI,
    appends `gravy_score` and `pI` as two new columns, and outputs the updated dataset to CSV.
    """
    print(f"Reading input dataset from '{input_csv_path}'...")
    df = pd.read_csv(input_csv_path)
    
    print(f"Translating sequences in column '{sequence_col}' & computing GRAVY and pI...")
    updated_df = add_protein_features_to_df(df, sequence_col=sequence_col)
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    updated_df.to_csv(output_csv_path, index=False)
    print(f"Successfully saved updated dataset with 5 features to '{output_csv_path}'.")
    return updated_df

def prepare_dataset(
    raw_csv_path: str,
    sequence_col: str = "sequence",
    target_col: str = "expression_yield"
) -> pd.DataFrame:
    """Load raw dataset and build final processed feature dataframe."""
    df = pd.read_csv(raw_csv_path)
    
    # Extract core computed features
    features_df = extract_features_df(df, sequence_col=sequence_col)
    
    processed_df = pd.concat([df, features_df], axis=1)
    
    if target_col in processed_df.columns:
        processed_df["expression_tier"] = processed_df[target_col].apply(categorize_expression)
        
    return processed_df

def organism_groups(gene_ids: pd.Series) -> pd.Series:
    """
    Group identifier for each gene, taken from the letter prefix of its gene_id.

    NESG target ids carry a source-organism prefix (`AaR17`, `BtR40`), and GC
    content is strongly organism-specific. Grouping folds by this prefix keeps
    relatives of a gene out of the fold that scores it, so a model cannot score
    well merely by recognising the organism. Ids with no letter prefix fall into
    a single `_other` group.
    """
    return gene_ids.astype(str).str.extract(r"^([A-Za-z]+)")[0].fillna("_other").astype(str)


def feature_ranges(X: pd.DataFrame) -> dict:
    """Per-feature min/max of the training data, for out-of-distribution checks."""
    return {col: {"min": float(X[col].min()), "max": float(X[col].max())} for col in X.columns}


def validate_dataset(df: pd.DataFrame, feature_cols: list, target_col: str, id_col: str = "gene_id") -> bool:
    """Validate dataset for shape, missing values, and numeric dtypes of feature columns."""
    print("=" * 60)
    print("[DATASET VALIDATION REPORT]")
    print("=" * 60)
    print(f"- DataFrame Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Check missing values
    missing = df.isnull().sum()
    total_missing = missing.sum()
    print(f"\n- Missing Values Check (Total Nulls: {total_missing}):")
    for col, count in missing.items():
        print(f"  - {col}: {count} missing")
        
    # Verify feature columns are numeric
    print(f"\n- Feature Column Dtype Verification:")
    all_numeric = True
    for col in feature_cols:
        if col in df.columns:
            is_num = pd.api.types.is_numeric_dtype(df[col])
            print(f"  - {col}: {df[col].dtype} | Numeric: {'YES' if is_num else 'NO'}")
            if not is_num:
                all_numeric = False
        else:
            print(f"  - {col}: MISSING")
            all_numeric = False
            
    print("=" * 60)
    return total_missing == 0 and all_numeric

# Every feature column any revision of the table may carry. The list is a
# superset on purpose: whichever of these a given CSV holds becomes the feature
# set, so a table revision does not require a code change.
#
#   head_folding_energy      Boel's Gz48v: folding energy of the vector 5' UTR
#                            plus the first 48 nt of the gene. The strongest
#                            single feature available (Spearman 0.177).
#   genewide_folding_avg48   Boel's Gave48: folding energy averaged over sliding
#                            48 nt windows across the rest of the gene (0.077).
#   folding_energy           The v1 table's single folding column, which held
#                            Gave48 values. Superseded by the two above.
FEATURE_CANDIDATES = [
    "GC_percent",
    "CAI",
    "head_folding_energy",
    "genewide_folding_avg48",
    "folding_energy",
    "gravy_score",
    "pI",
]

# Newest first. The v1 table's CAI was computed from weights derived over the
# highest-expressing genes, which leaked the target label into a feature, and its
# folding column was the weaker gene-wide average; v2 fixes both and strips the
# vector His-tag before computing GC and CAI.
DATASET_PREFERENCE = [
    "data/processed/feature_table_v2.csv",
    "data/processed/feature_table_boel_5features.csv",
    "data/processed/feature_table.csv",
]


def resolve_dataset_path(csv_path: Optional[str] = None) -> Optional[str]:
    """The dataset to train on: the requested one, else the newest available."""
    if csv_path and os.path.exists(csv_path):
        return csv_path
    for candidate in DATASET_PREFERENCE:
        if os.path.exists(candidate):
            return candidate
    return None


def load_and_prep_feature_table(
    csv_path: str = None,
    feature_cols: list = None,
    target_col: str = "expression_score",
    id_col: str = "gene_id"
) -> Tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Loads CSV, validates columns, separates into identifiers, features (X), and target (y),
    and prints summary statistics.
    """
    csv_path = resolve_dataset_path(csv_path) or csv_path
    feature_cols = FEATURE_CANDIDATES if feature_cols is None else feature_cols

    df = pd.read_csv(csv_path)

    # Filter feature_cols to those present in df
    actual_feature_cols = [c for c in feature_cols if c in df.columns]
    
    # Validate dataset
    is_valid = validate_dataset(df, feature_cols=actual_feature_cols, target_col=target_col, id_col=id_col)
    
    # Separate into identifiers, X, and y
    gene_ids = df[id_col] if id_col in df.columns else df.index
    X = df[actual_feature_cols]
    y = df[target_col] if target_col in df.columns else None
    
    # Summary outputs
    print("\n--- DataFrame Head (First 5 rows) ---")
    print(df.head())
    
    print("\n--- Summary Statistics (df.describe()) ---")
    print(df.describe())
    
    return gene_ids, X, y, df

def split_data(
    df: pd.DataFrame,
    feature_cols: list = None,
    target_col: str = "expression_score",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets for model training.

    The split is stratified on the target whenever the target is discrete, so
    that tier 1, which holds 131 of 6348 rows, is represented on both sides. An
    earlier revision stratified only for object dtype, which silently skipped an
    integer target and left the rare tiers to chance.
    """
    feature_cols = FEATURE_CANDIDATES if feature_cols is None else feature_cols
    actual_feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[actual_feature_cols]
    y = df[target_col]
    stratify = y if y.nunique() < max(2, len(y) // 20) else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process gene sequences and calculate GRAVY & pI protein features.")
    parser.add_argument("--input", type=str, default=None, help="Path to input CSV containing 'sequence' column.")
    parser.add_argument("--output", type=str, default="data/processed/feature_table_boel_5features.csv", help="Path to output CSV.")
    args = parser.parse_args()

    input_path = args.input
    if input_path is None:
        # Check potential default raw paths
        potential_inputs = [
            "data/raw/dataset.csv",
            "data/raw/sequences.csv",
            "data/raw/sample_sequences.csv"
        ]
        for p in potential_inputs:
            if os.path.exists(p):
                input_path = p
                break
                
    if input_path is None or not os.path.exists(input_path):
        print("No existing input CSV provided or found. Creating a synthetic sample CSV with sequence column for processing...")
        sample_df = pd.DataFrame({
            "gene_id": [f"Gene_{i:03d}" for i in range(1, 11)],
            "sequence": [
                "ATGAAACGCATTAGCCTGACCCTGCTGAGCGCGGCGATTCTGCTGACCGCGCTGAGCGCG",
                "ATGTTTGTTTTTCTTGTTTTATTGCCACTAGTCTCTAGTCAGTGTGTTAATCTTACAACC",
                "ATGGCATACCCATACGACGTCCCAGACTACGCTTAA", # contains stop codon TAA
                "ATGAAAGCAATTTTCGTACTGAAAGGTTAG",
                "INVALID_MALFORMED_DNA_SEQUENCE_12345", # malformed sequence -> NaN
                "", # empty string -> NaN
                "ATGGCGCGCTAGGCGCGC", # stop codon inside
                "ATGGCCATTATCGAAGTGGAACACCAACAGCTGATCACCGAAGTGTAA",
                "ATGAAACGTATTCTTGCCCTGGCAGCGGCAATTCTG",
                "ATGAAGAAAATCTGGTTAGCGCTGGCCGGCTTAACC"
            ],
            "expression_score": [0, 5, 2, 3, 0, 1, 4, 5, 3, 2]
        })
        os.makedirs("data/raw", exist_ok=True)
        input_path = "data/raw/sample_sequences.csv"
        sample_df.to_csv(input_path, index=False)
        print(f"Created sample input file: {input_path}")

    process_protein_features_dataset(input_csv_path=input_path, output_csv_path=args.output)


