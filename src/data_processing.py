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

def split_data(
    df: pd.DataFrame,
    feature_cols: list = ["gc_content", "codon_adaptation_index", "mrna_folding_stability"],
    target_col: str = "expression_tier",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets for model training."""
    X = df[feature_cols]
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if y.dtype == 'object' else None)
