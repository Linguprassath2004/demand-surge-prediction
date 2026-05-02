import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from pathlib import Path

def train(df, model_path):
    features = [col for col in df.columns if col not in ['timestamp', 'geohash6', 'demand']]

    X = df[features]
    y = df['demand']

    # Time-based split
    split_time = df['timestamp'].quantile(0.8)

    X_train = X[df['timestamp'] <= split_time]
    X_test  = X[df['timestamp'] > split_time]

    y_train = y[df['timestamp'] <= split_time]
    y_test  = y[df['timestamp'] > split_time]

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=50
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    print(f"RMSE: {rmse:.4f}")

    # Ensure the models directory exists before writing
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    # Save model
    model.booster_.save_model(model_path)

    return model