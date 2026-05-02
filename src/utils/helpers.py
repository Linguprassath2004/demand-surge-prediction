import pandas as pd
from pathlib import Path

def save_dataframe(df, path):
    # Ensure the parent directory exists before saving
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)