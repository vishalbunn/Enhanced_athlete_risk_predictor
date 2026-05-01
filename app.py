"""
app.py  (v2.1 — Full Backend Stack)
===================================
FastAPI server exposing the complete Day 1-7 research pipeline.

SETUP (one-time)
----------------
Run this from the project root to generate conformal_state.pkl:

    python -c "import sys; sys.path.insert(0,'src'); import joblib, pandas as pd, json; from conformal import ConformalPredictor; cal=pd.read_csv('data/splits/calibration.csv'); le=joblib.load('models/le_risk_encoder.pkl'); cols=json.load(open('src/template_columns.json')); F=['age','weight_kg','bf_percent','training_vol_hr_wk','sleep_h','testosterone_total','estradiol','ALT','AST','HDL','LDL','hematocrit','creatinine','mood_score','libido_score','enhancement_load']; Xc=pd.get_dummies(cal[F+['sex','status','goal']]).reindex(columns=cols,fill_value=0).values; yc=le.transform(cal['risk']); model=joblib.load('models/catboost_health_risk_model.pkl'); cp=ConformalPredictor(alpha=0.05); cp.calibrate(model,Xc,yc,le); joblib.dump({'q_hat': float(cp.q_hat), 'alpha': float(cp.alpha), 'classes': list(cp.classes_), 'n_cal': int(cp.n_cal)}, 'models/conformal_state.pkl'); print('Saved conformal_state.pkl')"

RUN SERVER
----------
    uvicorn app:app --reload --port 8000

Then open http://localhost:8000
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from typing import Literal

from fastapi                 import FastAPI, Request
from fastapi.responses       import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic                import BaseModel, Field

# ── Path resolution ──────────────────────────────────────────────────────────
HERE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
SRC_DIR    = os.path.join(HERE, "src")
TEMPLATES  = os.path.join(HERE, "templates")

sys.path.insert(0, SRC_DIR)
from conformal import ConformalPredictor  # noqa: E402

MODEL_PATH     = os.path.join(MODELS_DIR, "catboost_health_risk_model.pkl")
ENCODER_PATH   = os.path.join(MODELS_DIR, "le_risk_encoder.pkl")
CONFORMAL_PATH = os.path.join(MODELS_DIR, "conformal_state.pkl")
COLS_PATH      = os.path.join(SRC_DIR,    "template_columns.json")
INDEX_HTML     = os.path.join(TEMPLATES,  "index.html")

# ── Startup verification ─────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  ARPX — FastAPI startup (v2.1)")
print("=" * 55)

for label, path in [
    ("CatBoost model",      MODEL_PATH),
    ("Label encoder",       ENCODER_PATH),
    ("Conformal state",     CONFORMAL_PATH),
    ("Template columns",    COLS_PATH),
    ("index.html",          INDEX_HTML),
]:
    ok = os.path.exists(path)
    print(f"  {'OK ' if ok else 'MISS'}  {label:22s}  {path}")
    if not ok:
        if label == "Conformal state":
            raise FileNotFoundError(
                f"\n{path} not found.\n"
                f"Run the one-liner from the docstring at top of app.py to generate it.\n"
            )
        raise FileNotFoundError(path)

# ── Load model + encoder ─────────────────────────────────────────────────────
model   = joblib.load(MODEL_PATH)
le_risk = joblib.load(ENCODER_PATH)

# ── Load conformal state as dict, rebuild predictor object ───────────────────
_cstate   = joblib.load(CONFORMAL_PATH)
conformal = ConformalPredictor(alpha=_cstate["alpha"])
conformal.q_hat    = _cstate["q_hat"]
conformal.n_cal    = _cstate["n_cal"]
conformal.classes_ = np.array(_cstate["classes"])

with open(COLS_PATH, "r") as f:
    TEMPLATE_COLUMNS = json.load(f)

print(f"\n  Conformal q_hat:   {conformal.q_hat:.4f}")
print(f"  Coverage target:   {(1-conformal.alpha)*100:.0f}%")
print(f"  Classes:           {list(conformal.classes_)}")

# ── SHAP explainer ───────────────────────────────────────────────────────────
print("\n  Building SHAP explainer...")
import shap
SHAP_EXPLAINER = shap.TreeExplainer(model)
print("  SHAP explainer ready.\n")

# ── Hardcoded Day 4/5 stats ──────────────────────────────────────────────────
CALIBRATION_STATS = {
    "ece_mean":           0.0253,
    "brier_mean":         0.0373,
    "rating":             "well-calibrated",
    "target_coverage":    0.95,
    "empirical_coverage": 0.9578,
}

HONEST_NUMBERS = {
    "cv_accuracy_mean":   0.9484,
    "cv_accuracy_std":    0.0117,
    "test_accuracy":      0.9267,
    "auc_roc":            0.9867,
    "high_risk_recall":   0.84,
    "calibration_ece":    0.0253,
    "conformal_coverage": 0.9578,
    "n_training":         1800,
    "n_test":             450,
    "n_features":         19,
    "disclosure": (
        "Accuracy measured against benchmark scoring rule, not clinical "
        "outcomes. Cross-validated on 5 stratified folds."
    ),
}

BIOMARKER_RANGES = {
    "ALT":                (7,   56,   "U/L",     "liver enzyme"),
    "AST":                (10,  40,   "U/L",     "liver enzyme"),
    "HDL":                (40,  80,   "mg/dL",   "good cholesterol"),
    "LDL":                (0,   130,  "mg/dL",   "bad cholesterol"),
    "hematocrit":         (36,  50,   "%",       "red blood cell volume"),
    "creatinine":         (0.7, 1.2,  "mg/dL",   "kidney function"),
    "testosterone_total": (270, 1070, "ng/dL",   "testosterone"),
    "estradiol":          (10,  60,   "pg/mL",   "estradiol"),
}

CLINICAL_ACTIONS = {
    "hematocrit":         ("decrease", "Therapeutic phlebotomy + hydration"),
    "HDL":                ("increase", "Cardio + omega-3 supplementation"),
    "LDL":                ("decrease", "Diet adjustment + lipid review"),
    "ALT":                ("decrease", "Reduce oral AAS + hepatoprotection"),
    "AST":                ("decrease", "Liver recovery protocol"),
    "enhancement_load":   ("decrease", "Lower total PED dose"),
    "testosterone_total": ("decrease", "Consider dose reduction or cruise"),
    "creatinine":         ("decrease", "Hydration + reduce nephrotoxic load"),
}

# ═════════════════════════════════════════════════════════════════════════════
# FastAPI app
# ═════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="ARPX Enhanced Athlete Risk Predictor", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t = time.time()
    response = await call_next(request)
    ms = (time.time() - t) * 1000
    print(f"  {response.status_code} {request.method:6s} {request.url.path:20s}  {ms:.0f}ms")
    return response


class AthleteInput(BaseModel):
    age:                int   = Field(..., ge=18, le=60)
    weight_kg:          float
    bf_percent:         float
    training_vol_hr_wk: float
    sleep_h:            float
    testosterone_total: float
    estradiol:          float
    ALT:                float
    AST:                float
    HDL:                float
    LDL:                float
    hematocrit:         float
    creatinine:         float
    mood_score:         float
    libido_score:       float
    enhancement_load:   float
    sex:    Literal["male", "female"]
    status: Literal["off", "on", "pct", "cruise"]
    goal:   Literal["bulk", "cut", "recomp", "maintenance"]


def prepare_input(df_raw):
    df_enc = pd.get_dummies(df_raw)
    df_enc = df_enc.reindex(columns=TEMPLATE_COLUMNS, fill_value=0)
    return df_enc.astype(float)


def compute_shap_drivers(df_ready, pred_class_idx, raw_input, top_k=5):
    shap_vals = SHAP_EXPLAINER.shap_values(df_ready)
    sv = np.array(shap_vals)

    if sv.ndim == 3:
        if sv.shape[0] == 1:
            class_shap = sv[0, :, pred_class_idx]
        else:
            class_shap = sv[pred_class_idx, 0, :]
    else:
        class_shap = sv[0]

    pairs = list(zip(TEMPLATE_COLUMNS, class_shap))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    drivers = []
    for feat, impact in pairs[:top_k]:
        raw_val = raw_input.get(feat)
        if raw_val is None:
            for k, v in raw_input.items():
                if feat == f"{k}_{v}":
                    raw_val = "yes"
                    break
            if raw_val is None:
                raw_val = "—"

        drivers.append({
            "feature":   feat,
            "impact":    round(float(impact), 4),
            "direction": "increases" if impact > 0 else "decreases",
            "value":     raw_val if not isinstance(raw_val, float) else round(raw_val, 2),
        })
    return drivers


def find_counterfactual(df_ready, raw_input, target_class_idx):
    from scipy.optimize import differential_evolution

    x = df_ready.values[0].copy()

    mutable_feats = [
        "hematocrit", "HDL", "LDL", "ALT", "AST",
        "creatinine", "testosterone_total", "estradiol", "enhancement_load",
    ]
    feat_to_idx = {c: i for i, c in enumerate(TEMPLATE_COLUMNS)}
    mutable_idx = [feat_to_idx[f] for f in mutable_feats if f in feat_to_idx]

    perturbation_bounds = {
        "hematocrit":         (30.0, 55.0),
        "HDL":                (30.0, 85.0),
        "LDL":                (60.0, 200.0),
        "ALT":                (10.0, 80.0),
        "AST":                (10.0, 60.0),
        "creatinine":         (0.6,  1.8),
        "testosterone_total": (300,  1500),
        "estradiol":          (15,   80),
        "enhancement_load":   (0.5,  1.6),
    }
    bounds = [perturbation_bounds[f] for f in mutable_feats
              if f in feat_to_idx and f in perturbation_bounds]

    def loss(mut_vals):
        trial = x.copy()
        for val, idx in zip(mut_vals, mutable_idx):
            trial[idx] = val
        proba = model.predict_proba(trial.reshape(1, -1))[0]
        pred_loss = (1.0 - proba[target_class_idx]) ** 2
        delta_norm = sum(
            abs(trial[idx] - x[idx]) / max(abs(x[idx]), 1e-3)
            for idx in mutable_idx
        )
        return pred_loss + 0.02 * delta_norm

    result = differential_evolution(
        loss, bounds,
        seed=42, maxiter=25, popsize=8, tol=0.002,
        mutation=(0.5, 1.2), recombination=0.75,
        workers=1,
    )

    best_x = x.copy()
    for val, idx in zip(result.x, mutable_idx):
        best_x[idx] = val

    final_proba = model.predict_proba(best_x.reshape(1, -1))[0]
    flipped     = int(final_proba.argmax()) == target_class_idx

    changes = []
    for feat in mutable_feats:
        if feat not in feat_to_idx:
            continue
        idx    = feat_to_idx[feat]
        before = df_ready.values[0, idx]
        after  = best_x[idx]
        delta  = after - before
        if abs(delta) > abs(before) * 0.01:
            _, action = CLINICAL_ACTIONS.get(feat, ("modify", "Lifestyle adjustment"))
            changes.append({
                "feature":         feat,
                "before":          round(float(before), 2),
                "after":           round(float(after),  2),
                "delta_pct":       round(float(delta / max(abs(before), 1e-3)) * 100, 1),
                "clinical_action": action,
            })

    changes.sort(key=lambda c: abs(c["delta_pct"]), reverse=True)

    return {
        "success":               flipped,
        "target_class":          le_risk.classes_[target_class_idx],
        "projected_probability": round(float(final_proba[target_class_idx]), 4),
        "changes":               changes[:4],
    }


def flag_biomarkers(raw_input):
    flags = []
    for feat, (lo, hi, unit, meaning) in BIOMARKER_RANGES.items():
        if feat not in raw_input:
            continue
        val = raw_input[feat]
        if val < lo:
            flags.append({"feature": feat, "value": val, "unit": unit,
                          "status": "below", "range": f"{lo}-{hi}",
                          "meaning": meaning})
        elif val > hi:
            flags.append({"feature": feat, "value": val, "unit": unit,
                          "status": "above", "range": f"{lo}-{hi}",
                          "meaning": meaning})
    return flags


# ═════════════════════════════════════════════════════════════════════════════
# Routes
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
def home():
    with open(INDEX_HTML, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    return {
        "status":          "ok",
        "model_loaded":    True,
        "conformal_ready": conformal.q_hat is not None,
        "shap_ready":      True,
        "classes":         list(le_risk.classes_),
        "honest_numbers":  HONEST_NUMBERS,
    }


@app.get("/stats")
def stats():
    return HONEST_NUMBERS


@app.post("/predict")
def predict(data: AthleteInput):
    try:
        raw_input = data.dict()
        df        = pd.DataFrame([raw_input])
        df_ready  = prepare_input(df)

        # 1. Point prediction + probabilities
        proba    = model.predict_proba(df_ready)[0]
        pred_idx = int(proba.argmax())
        pred_lbl = le_risk.classes_[pred_idx]

        # 2. Conformal prediction set
        pred_sets, _ = conformal.predict_set(model, df_ready.values)
        conformal_classes = [le_risk.classes_[i]
                              for i, flag in enumerate(pred_sets[0]) if flag]

        # 3. SHAP top drivers
        drivers = compute_shap_drivers(df_ready, pred_idx, raw_input, top_k=5)

        # 4. Counterfactual (only for high-risk)
        counterfactual = None
        if pred_lbl == "high":
            mod_idx = int(list(le_risk.classes_).index("moderate"))
            counterfactual = find_counterfactual(df_ready, raw_input, mod_idx)

        # 5. Biomarker flags
        biomarker_flags = flag_biomarkers(raw_input)

        return {
            "prediction":           pred_lbl,
            "confidence":           float(max(proba)),
            "probabilities":        dict(zip(le_risk.classes_, proba.tolist())),
            "conformal_set":        conformal_classes,
            "is_uncertain":         len(conformal_classes) > 1,
            "coverage_guarantee":   CALIBRATION_STATS["target_coverage"],
            "calibration": {
                "ece":    CALIBRATION_STATS["ece_mean"],
                "brier":  CALIBRATION_STATS["brier_mean"],
                "rating": CALIBRATION_STATS["rating"],
            },
            "top_drivers":          drivers,
            "counterfactual":       counterfactual,
            "biomarker_flags":      biomarker_flags,
            "honest_accuracy_note": HONEST_NUMBERS["disclosure"],
        }

    except Exception as e:
        print(f"  !! Prediction error: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500,
                            content={"detail": f"Prediction failed: {e}"})


@app.options("/predict")
def predict_options():
    return {"ok": True}