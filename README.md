# ARPX — Enhanced Athlete Risk Predictor

> Uncertainty-aware, counterfactually explainable risk stratification
> for pharmacologically enhanced athletes — a medically underserved
> population whose biomarker profiles fall outside standard clinical
> reference ranges.

**🔗 [Live demo](https://enhanced-athlete-risk-predictor.onrender.com)** ·
**📄 [Paper draft](./paper/main.pdf)** ·
**📊 [Model card](./MODEL_CARD.md)** ·
**📁 [Data card](./DATA_CARD.md)**

---

## What this is

A research framework that does three things existing athlete-monitoring
ML doesn't:

1. **Quantifies its own uncertainty.** Conformal prediction produces
   prediction *sets* like `{moderate, high}` with a mathematical
   95% coverage guarantee, instead of a single confident class.
2. **Tells the user what to change.** Counterfactual explanations
   identify the minimum biomarker modifications needed to flip a
   high-risk profile to moderate, mapped to clinical actions.
3. **Discloses its own limits.** Synthetic-data circularity is
   acknowledged; an in-progress blind clinician rating study (n=200)
   provides partial mitigation.

The framework is positioned as a harm-reduction instrument, not an
enforcement tool.

---

## Results (honest numbers, sealed test set)

| Metric                    | Value             | Notes                              |
|---------------------------|-------------------|------------------------------------|
| CV accuracy (CatBoost)    | **94.84% ± 1.17%**| 5-fold stratified, SMOTE-balanced  |
| Held-out test accuracy    | 92.67%            | n=450, never seen during training  |
| AUC-ROC                   | 0.9867            | one-vs-rest                        |
| Expected Calibration Error| 0.0253            | well-calibrated (threshold 0.05)   |
| Conformal coverage @ α=0.05| 95.78% empirical | distribution-free guarantee        |
| High-risk recall          | 0.838             | clinically critical class          |
| Counterfactual success    | 100% (5/5 cases)  | minimum-change flip to moderate    |

The original v1 of this project reported 99.67% accuracy, which was
the model recovering the rule that generated the labels (a depth-8
decision tree achieves 99.97% on the same task by rote memorization).
v2 dropped that misleading framing and reports the regularized
performance with explicit circularity disclosure.

---

## Stack

| Layer        | Tools                                                |
|--------------|------------------------------------------------------|
| ML           | CatBoost, XGBoost, LightGBM, MLP                     |
| Uncertainty  | Split conformal prediction (LAC nonconformity)       |
| Explanation  | SHAP TreeExplainer + Wachter et al. counterfactuals  |
| Backend      | FastAPI + uvicorn                                    |
| Frontend     | Vanilla HTML/CSS/JS (no framework)                   |
| Deploy       | Render (free tier, kept warm via UptimeRobot)        |

---

## Run locally

Requirements: Python 3.11+, ~600MB disk for dependencies.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the conformal calibration state (one-time)
python -c "import sys; sys.path.insert(0,'src'); import joblib, pandas as pd, json; from conformal import ConformalPredictor; cal=pd.read_csv('data/splits/calibration.csv'); le=joblib.load('models/le_risk_encoder.pkl'); cols=json.load(open('src/template_columns.json')); F=['age','weight_kg','bf_percent','training_vol_hr_wk','sleep_h','testosterone_total','estradiol','ALT','AST','HDL','LDL','hematocrit','creatinine','mood_score','libido_score','enhancement_load']; Xc=pd.get_dummies(cal[F+['sex','status','goal']]).reindex(columns=cols,fill_value=0).values; yc=le.transform(cal['risk']); model=joblib.load('models/catboost_health_risk_model.pkl'); cp=ConformalPredictor(alpha=0.05); cp.calibrate(model,Xc,yc,le); joblib.dump({'q_hat': float(cp.q_hat), 'alpha': float(cp.alpha), 'classes': list(cp.classes_), 'n_cal': int(cp.n_cal)}, 'models/conformal_state.pkl'); print('done')"

# 3. Launch
uvicorn app:app --reload --port 8000
```

Open http://localhost:8000.

---

## Reproducing the experiments

```bash
# Regenerate the synthetic dataset (3000 samples, seed=42)
python src/data_generator.py

# Train all 4 classifiers with 5-fold CV
python src/train.py

# Evaluate calibration
python src/evaluate.py

# Apply conformal prediction layer
python src/conformal.py

# Generate counterfactual case studies
python src/counterfactual.py

# NHANES external validation
python src/build_references_table.py
```

All scripts use a fixed seed (42). Outputs are deterministic.

---

## Repository structure

```
.
├── app.py                    # FastAPI server
├── templates/index.html      # Frontend
├── src/
│   ├── data_generator.py     # Synthetic dataset construction
│   ├── train.py              # Model training pipeline
│   ├── evaluate.py           # Calibration + per-class metrics
│   ├── conformal.py          # Split conformal prediction
│   ├── counterfactual.py     # Wachter et al. counterfactuals
│   ├── audit.py              # v1 dataset forensic audit
│   └── template_columns.json # Feature column ordering
├── data/
│   ├── synthetic_athlete_health_risk_v2.csv
│   ├── splits/               # train/val/calibration/test
│   └── references/           # NHANES validation data
├── models/                   # Trained models + conformal state
├── outputs/                  # All figures + per-class metrics
├── paper/                    # LaTeX source + bibliography
├── MODEL_CARD.md
├── DATA_CARD.md
└── requirements.txt
```

---

## Limitations

Disclosed in the paper (Section VII) and worth surfacing here:

- **Synthetic data only.** No real athlete records. Validation against
  NHANES marginals is partial.
- **Label circularity.** Risk labels are a deterministic function of
  the biomarkers, not clinical adjudication. A blind clinician rating
  study (n=200) is in progress.
- **Demographic skew.** ~85% male, US/European reference ranges.
  South Asian, female, and pediatric athletes need separate
  recalibration.
- **Cross-sectional only.** Does not model biomarker trajectories
  over time. A longitudinal extension is planned future work.
- **Counterfactuals lack causal realism.** Suggestions assume
  features can be modified independently; in practice, clinical
  interventions affect multiple biomarkers simultaneously.

---

## Citation

If this work informs further research, please cite the in-progress
paper (EMBC 2026 submission):

```
Hota, V. (2026). Uncertainty-Aware and Counterfactually Explainable
Risk Stratification for Enhanced Athletes: A Methodological Framework
on Synthetic Physiological Data. Submitted to IEEE EMBC 2026.
```

---

## License

Code: MIT. Synthetic dataset: CC BY-NC-SA 4.0 (research and
education only; commercial or enforcement use prohibited).

---

## Author

**Vishal Hota** — B.Tech CSE (AI/ML), KL University Hyderabad
Targeting MS programs at Carnegie Mellon, UCLA, Georgia Tech.

GitHub: [vishalbunn](https://github.com/vishalbunn) ·
Email: vishalhota9@gmail.com
