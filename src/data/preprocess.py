def preprocess(df):
    df = df.sort_values(by=['geohash6', 'timestamp'])
    
    # Drop redundant columns if exist
    if 'day' in df.columns:
        df = df.drop(columns=['day'])
    
    return df