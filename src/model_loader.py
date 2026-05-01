
import os, joblib
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
def load_model(name="catboost"):
    path = os.path.join(MODELS_DIR, f"{name}_health_risk_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return joblib.load(path)
