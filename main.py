from src.config import DATA_RAW, DATA_PROCESSED, MODEL_PATH

from src.data.load_data import load_data
from src.data.preprocess import preprocess
from src.features.build_features import build_features
from src.models.train_model import train
from src.utils.helpers import save_dataframe

def main():
    print("Loading data...")
    df = load_data(DATA_RAW)

    print("Preprocessing...")
    df = preprocess(df)

    print("Building features...")
    df = build_features(df)

    print("Saving processed data...")
    save_dataframe(df, DATA_PROCESSED)

    print("Training model...")
    model = train(df, MODEL_PATH)

    print("Done!")

if __name__ == "__main__":
    main()