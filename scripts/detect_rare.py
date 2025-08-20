"""
Detect rare samples from incoming data:
- embed
- umap transform
- sequential iforest filtering
- save rare candidates to data/rare_candidates.csv
"""
import os
import pandas as pd
from src.config import DATA_DIR, UMAP_MODEL_PATH, IFOREST_DIR, EMBEDDING_MODEL, EMBEDDING_USE_MOCK, TEXT_COL
from src.rare_detection import detect_rare_emails

def main():
    # input: data/new_emails.csv with column 'text'
    input_path = DATA_DIR / "new_emails.csv"
    if not input_path.exists():
        raise FileNotFoundError("Please provide incoming data at data/new_emails.csv with column 'text'")

    df = pd.read_csv(str(input_path))
    if TEXT_COL not in df.columns:
        raise ValueError(f"Input file must contain column '{TEXT_COL}'")

    texts = df[TEXT_COL].astype(str).tolist()
    rare_idx = detect_rare_emails(texts, str(UMAP_MODEL_PATH), str(IFOREST_DIR), EMBEDDING_MODEL, use_mock=EMBEDDING_USE_MOCK)
    rare_df = df.iloc[rare_idx].reset_index(drop=True)
    out_path = DATA_DIR / "rare_candidates.csv"
    rare_df.to_csv(str(out_path), index=False)
    print(f"Found {len(rare_df)} rare candidates. Saved to {out_path}")

if __name__ == "__main__":
    main()