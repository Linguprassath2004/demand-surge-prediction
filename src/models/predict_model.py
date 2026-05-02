import lightgbm as lgb

def load_model(model_path):
    return lgb.Booster(model_file=model_path)

def predict(model, X):
    return model.predict(X)