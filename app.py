from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import json

# Load model and encoder
model = joblib.load("models/catboost_health_risk_model.pkl")
le_risk = joblib.load("models/le_risk_encoder.pkl")

with open("src/template_columns.json", "r") as f:
    TEMPLATE_COLUMNS = json.load(f)

app = FastAPI(title="Enhanced Athlete Risk Predictor")

class AthleteInput(BaseModel):
    age: int
    weight_kg: float
    bf_percent: float
    training_vol_hr_wk: float
    sleep_h: float
    testosterone_total: float
    estradiol: float
    ALT: float
    AST: float
    HDL: float
    LDL: float
    hematocrit: float
    creatinine: float
    mood_score: float
    libido_score: float
    enhancement_load: float
    sex: str
    status: str
    goal: str

def prepare_input(df_raw):
    df_enc = pd.get_dummies(df_raw)
    df_enc = df_enc.reindex(columns=TEMPLATE_COLUMNS, fill_value=0)
    return df_enc.astype(float)

@app.get("/")
def home():
    return {"message": "API is running successfully"}

@app.post("/predict")
def predict(data: AthleteInput):
    df = pd.DataFrame([data.dict()])
    df_ready = prepare_input(df)

    proba = model.predict_proba(df_ready)[0]
    pred_idx = proba.argmax()
    pred_label = le_risk.classes_[pred_idx]

    return {
        "predicted_risk": pred_label,
        "probabilities": dict(zip(le_risk.classes_, proba.tolist()))
    }
