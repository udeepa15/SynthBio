"""
Data processing module: loads raw gene construct datasets, cleans sequence columns,
computes features, assigns target labels, and splits train/test datasets.
"""

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from src.features import extract_features_df

def categorize_expression(yield_val: float) -> str:
    """Categorize continuous expression yield into High, Medium, or Low tier."""
    if yield_val >= 70.0:
        return "High"
    elif yield_val >= 30.0:
        return "Medium"
    else:
        return "Low"

def prepare_dataset(
    raw_csv_path: str,
    sequence_col: str = "sequence",
    target_col: str = "expression_yield"
) -> pd.DataFrame:
    """Load raw dataset and build final processed feature dataframe."""
    df = pd.read_csv(raw_csv_path)
    
    # Extract 3 core computed features
    features_df = extract_features_df(df, sequence_col=sequence_col)
    
    processed_df = pd.concat([df, features_df], axis=1)
    
    if target_col in processed_df.columns:
        processed_df["expression_tier"] = processed_df[target_col].apply(categorize_expression)
        
    return processed_df

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
        is_num = pd.api.types.is_numeric_dtype(df[col])
        print(f"  - {col}: {df[col].dtype} | Numeric: {'YES' if is_num else 'NO'}")
        if not is_num:
            all_numeric = False
            
    print("=" * 60)
    return total_missing == 0 and all_numeric

def load_and_prep_feature_table(
    csv_path: str = "data/processed/feature_table.csv",
    feature_cols: list = ["GC_percent", "CAI", "folding_energy"],
    target_col: str = "expression_score",
    id_col: str = "gene_id"
) -> Tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Loads CSV, validates columns, separates into identifiers, features (X), and target (y),
    and prints summary statistics.
    """
    df = pd.read_csv(csv_path)
    
    # Validate dataset
    is_valid = validate_dataset(df, feature_cols=feature_cols, target_col=target_col, id_col=id_col)
    
    # Separate into identifiers, X, and y
    gene_ids = df[id_col]
    X = df[feature_cols]
    y = df[target_col]
    
    # Summary outputs
    print("\n--- DataFrame Head (First 5 rows) ---")
    print(df.head())
    
    print("\n--- Summary Statistics (df.describe()) ---")
    print(df.describe())
    
    return gene_ids, X, y, df

def split_data(
    df: pd.DataFrame,
    feature_cols: list = ["GC_percent", "CAI", "folding_energy"],
    target_col: str = "expression_score",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets for model training."""
    X = df[feature_cols]
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if y.dtype == 'object' else None)

