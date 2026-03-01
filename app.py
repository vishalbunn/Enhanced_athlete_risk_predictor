from fastapi import FastAPI
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
# App Metadata (Improved)
# ===============================

app = FastAPI(
    title="Enhanced Athlete Risk Predictor",
    description="Enhancement-aware physiological health risk classification API using CatBoost",
    version="1.0.0"
)

# ===============================
# Input Schema (Validated + Examples)
# ===============================

class AthleteInput(BaseModel):
    age: int = Field(..., ge=18, le=60, example=29)
    weight_kg: float = Field(..., example=88)
    bf_percent: float = Field(..., ge=3, le=50, example=14)
    training_vol_hr_wk: float = Field(..., example=14)
    sleep_h: float = Field(..., ge=0, le=12, example=6)
    testosterone_total: float = Field(..., example=1100)
    estradiol: float = Field(..., example=40)
    ALT: float = Field(..., example=55)
    AST: float = Field(..., example=50)
    HDL: float = Field(..., example=35)
    LDL: float = Field(..., example=170)
    hematocrit: float = Field(..., example=53)
    creatinine: float = Field(..., example=1.35)
    mood_score: float = Field(..., example=7)
    libido_score: float = Field(..., example=9)
    enhancement_load: float = Field(..., example=1.45)
    sex: Literal["male", "female"] = Field(..., example="male")
    status: Literal["off", "on", "pct", "cruise"] = Field(..., example="on")
    goal: Literal["bulk", "cut", "recomp", "maintenance"] = Field(..., example="bulk")

# ===============================
# Helper Function
# ===============================

def prepare_input(df_raw):
    df_enc = pd.get_dummies(df_raw)
    df_enc = df_enc.reindex(columns=TEMPLATE_COLUMNS, fill_value=0)
    return df_enc.astype(float)

# ===============================
# Endpoints
# ===============================

@app.get("/")
def home():
    return {
        "message": "Enhanced Athlete Risk Predictor API is running",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": "CatBoost",
        "version": "1.0.0"
    }

@app.post("/predict")
def predict(data: AthleteInput):
    df = pd.DataFrame([data.dict()])
    df_ready = prepare_input(df)

    proba = model.predict_proba(df_ready)[0]
    pred_idx = proba.argmax()
    pred_label = le_risk.classes_[pred_idx]

    return {
        "model": "CatBoost v1.0",
        "prediction": pred_label,
        "confidence": float(max(proba)),
        "probabilities": dict(zip(le_risk.classes_, proba.tolist())),
        "metadata": {
            "framework": "CatBoost",
            "deployment": "Render",
            "version": "1.0.0"
        }
    }
