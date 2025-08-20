import pandas as pd
from typing import Tuple, List

def load_csv(path: str) -> pd.DataFrame:
    """
    Load CSV to DataFrame.
    Expect at least a `text` column (configurable).
    """
    df = pd.read_csv(path)
    return df

def filter_common_labels(df: pd.DataFrame, label_col: str, min_count: int) -> pd.DataFrame:
    """
    Keep only labels that appear at least `min_count` times.
    """
    counts = df[label_col].value_counts()
    commons = counts[counts >= min_count].index.tolist()
    return df[df[label_col].isin(commons)].reset_index(drop=True)