

# 🚀 Enhanced Athlete Health Risk Predictor

<p align="center">
  <img src="https://img.shields.io/badge/Model-CatBoost%20%7C%20XGBoost%20%7C%20LightGBM-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Accuracy-99.6%25-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Dataset-Synthetic%20Enhanced%20Athlete-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Active-success?style=flat-square" />
</p>

---

## 📌 Overview

**Enhanced Athlete Health Risk Predictor** is a machine learning system designed to classify **health risk levels** (low, moderate, high) for enhanced athletes based on:

* Training volume
* Sleep, mood, libido
* Blood biomarkers
* Body composition
* **Cycle phases: on, off, cruise, pct**
* Fitness goals (bulk, cut, recomp, maintenance)

The dataset is fully synthetic but modeled after real physiological patterns across enhancement phases — an area with almost **no existing research papers**, making this project novel and research-worthy.

The best-performing model (**CatBoost**) achieves:

> **99.67% accuracy** on multi-class classification.

---

## 🌐 Repository

**GitHub:** [https://github.com/vishalbunn/Enhanced_athlete_risk_predictor](https://github.com/vishalbunn/Enhanced_athlete_risk_predictor)

---

## 🧬 Features

* 3000+ synthetic athlete profiles
* Multi-class health risk prediction
* Models: CatBoost, XGBoost, LightGBM, MLP
* SHAP interpretability
* Clean modular code (API-ready)
* Fully reproducible notebook
* Ideal for research & deployment

---

## 📂 Project Structure

```
Enhanced_athlete_risk_predictor/
│
├── data/
│   └── synthetic_athlete_health_risk.csv
│
├── models/
│   ├── catboost_health_risk_model.pkl
│   ├── xgboost_health_risk_model.pkl
│   ├── lightgbm_health_risk_model.pkl
│   ├── mlp_health_risk_model.pkl
│   └── le_risk_encoder.pkl
│
├── outputs/
│   └── fig_shap_bar.png
│
├── src/
│   ├── predict.py
│   ├── utils.py
│   ├── model_loader.py
│   └── template_columns.json
│
├── notebooks/
│   └── HealthRiskPredictor.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🧠 Model Performance

| Model        | Accuracy   |
| ------------ | ---------- |
| **CatBoost** | **0.9967** |
| XGBoost      | 0.9883     |
| LightGBM     | 0.9866     |
| MLP          | 0.5616     |

---

## 🎯 Example Prediction

```python
from src.predict import predict_health_risk_from_dict

case = {
    "age": 29,
    "sex": "male",
    "status": "on",
    "goal": "bulk",
    "weight_kg": 88,
    "bf_percent": 14,
    "training_vol_hr_wk": 14,
    "sleep_h": 6,
    "testosterone_total": 1100,
    "estradiol": 40,
    "ALT": 55,
    "AST": 50,
    "HDL": 35,
    "LDL": 170,
    "hematocrit": 53,
    "creatinine": 1.35,
    "mood_score": 7,
    "libido_score": 9,
    "enhancement_load": 1.45
}

label, probs = predict_health_risk_from_dict(case, model_name="catboost")

print("Predicted Risk:", label)
print("Probabilities:", probs)
```

### Example Output:

```
Predicted Risk: high
Probabilities: {'high': 0.91, 'low': 0.03, 'moderate': 0.06}
```

---


### Key Influential Biomarkers

* Hematocrit
* Creatinine
* LDL/HDL profile
* ALT & AST (liver markers)
* Enhancement load
* Testosterone
* Cycle status

---

## 🏗️ Installation

```bash
pip install -r requirements.txt
```

---

## 🧩 Using the Prediction API

```python
from src.predict import predict_health_risk_from_dict
```

---

## 🔬 Research Motivation

This project contributes to athlete risk analytics by introducing:

* A synthetic physiological model of enhancement phases
* A biomarker-based risk-scoring algorithm
* Multi-model benchmarking
* SHAP interpretability linked to biological reasoning

A formal **research publication** is currently being written based on this work.

---

## 🚀 Roadmap

* Full research paper (LaTeX)
* FastAPI + Docker deployment
* Streamlit dashboard
* Time-series biomarker forecasting
* Longitudinal athlete modeling

---

## ⭐ Support

If you find this project valuable, please ⭐ the repository:

**[https://github.com/vishalbunn/Enhanced_athlete_risk_predictor](https://github.com/vishalbunn/Enhanced_athlete_risk_predictor)**

---

## 👤 Author

**Vishal Hota**
AI Research • Athlete Analytics • Applied ML


