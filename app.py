from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import json
from typing import Literal

# ===============================
# Load model and encoder
# ===============================

model = joblib.load("models/catboost_health_risk_model.pkl")
le_risk = joblib.load("models/le_risk_encoder.pkl")

with open("src/template_columns.json", "r") as f:
    TEMPLATE_COLUMNS = json.load(f)

# ===============================
# FastAPI App
# ===============================

app = FastAPI(
    title="Enhanced Athlete Risk Predictor",
    description="Advanced physiological health risk classification",
    version="2.0.0"
)

# ===============================
# Input Schema
# ===============================

class AthleteInput(BaseModel):
    age: int = Field(..., ge=18, le=60)
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
    sex: Literal["male", "female"]
    status: Literal["off", "on", "pct", "cruise"]
    goal: Literal["bulk", "cut", "recomp", "maintenance"]

# ===============================
# Helper Function
# ===============================

def prepare_input(df_raw):
    df_enc = pd.get_dummies(df_raw)
    df_enc = df_enc.reindex(columns=TEMPLATE_COLUMNS, fill_value=0)
    return df_enc.astype(float)

# ===============================
# Routes
# ===============================

# ✅ Serve UI directly (NO Jinja)
@app.get("/", response_class=HTMLResponse)
def home():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True
    }

@app.post("/predict")
def predict(data: AthleteInput):
    df = pd.DataFrame([data.dict()])
    df_ready = prepare_input(df)

    proba = model.predict_proba(df_ready)[0]
    pred_idx = proba.argmax()
    pred_label = le_risk.classes_[pred_idx]

    return {
        "prediction": pred_label,
        "confidence": float(max(proba)),
        "probabilities": dict(zip(le_risk.classes_, proba.tolist()))
    }
