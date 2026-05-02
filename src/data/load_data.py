import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    # Convert and ensure correct parsing without formatting warnings
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    return df