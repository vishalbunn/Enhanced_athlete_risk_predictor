
import os, json
import pandas as pd
from .model_loader import load_model
from .utils import prepare_input
import joblib

BASE_DIR = os.path.dirname(__file__)
with open(os.path.join(BASE_DIR, "template_columns.json"), "r") as f:
    TEMPLATE_COLUMNS = json.load(f)

le_risk = joblib.load(os.path.join(BASE_DIR, "..", "models", "le_risk_encoder.pkl"))

def predict_health_risk_from_dict(case_dict, model_name="catboost"):
    model = load_model(name=model_name)
    df = pd.DataFrame([case_dict])
    df_ready = prepare_input(df, TEMPLATE_COLUMNS)
    df_ready = df_ready[TEMPLATE_COLUMNS].astype(float)
    proba = model.predict_proba(df_ready)[0]
    pred_idx = int(proba.argmax())
    pred_label = le_risk.classes_[pred_idx]
    return pred_label, dict(zip(le_risk.classes_, proba.tolist()))
