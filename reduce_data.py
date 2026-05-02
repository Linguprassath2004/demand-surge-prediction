import pandas as pd

# 1. Load the dataset
df = pd.read_csv('data/raw/train.csv')

# 2. Keep only the last 200,000 rows to ensure size is well below 100 MB
df_small = df.tail(200000)

# 3. Save the reduced dataset
df_small.to_csv('data/raw/train.csv', index=False)
print(f"Reduced size: {len(df_small)} rows.")