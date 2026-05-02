import numpy as np

def build_features(df):
    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Lag features
    df = df.sort_values(by=['geohash6', 'timestamp'])

    for lag in [1, 2, 3]:
        df[f'lag_{lag}'] = df.groupby('geohash6')['demand'].shift(lag)

    # Rolling features
    df['rolling_mean_3'] = df.groupby('geohash6')['demand'] \
        .transform(lambda x: x.shift(1).rolling(3).mean())

    df = df.dropna()

    return df