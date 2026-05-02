from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW = BASE_DIR / "data" / "raw" / "train.csv"
DATA_PROCESSED = BASE_DIR / "data" / "processed" / "features.csv"

MODEL_PATH = BASE_DIR / "models" / "lightgbm_model.txt"