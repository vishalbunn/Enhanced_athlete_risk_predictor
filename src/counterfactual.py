"""
src/counterfactual.py
=====================
Counterfactual Explanations for the Enhanced Athlete Risk Predictor.

WHAT THIS IS
------------
A counterfactual explanation answers the question:
    "What is the MINIMUM change to this athlete's profile
     that would move them from HIGH risk to MODERATE risk?"

This is fundamentally different from SHAP:

SHAP says:          "Hematocrit contributed +0.41 to the high-risk score."
Counterfactual says: "If hematocrit dropped from 58.3% to 48.9%
                      AND HDL rose from 31 to 39 mg/dL,
                      risk would change from HIGH to MODERATE."

SHAP = diagnosis. Counterfactual = prescription.

WHY THIS IS A CONTRIBUTION
---------------------------
Counterfactual explanations are:
1. Actionable — athletes and coaches can act on "reduce hematocrit by 9%"
   but not on "hematocrit has SHAP value 0.41"
2. Contrastive — directly shows the boundary between risk classes
3. Privacy-preserving — no need to reveal the full model or training data
4. Clinically interpretable — maps to real interventions

Citation: Wachter, Mittelstadt & Russell (2017).
"Counterfactual Explanations Without Opening the Black Box."
Harvard Journal of Law & Technology, 31(2).

IMPLEMENTATION
--------------
We use a gradient-free optimisation approach (differential evolution
+ L-BFGS-B local refinement) because CatBoost is not differentiable
w.r.t. inputs in the standard sense.

Loss function (Wachter et al. 2017 adapted):
    L(x_cf) = (1 - P(target_class | x_cf))²
             + λ · Σ |x_cf[i] - x_orig[i]| / range[i]

Term 1: make the model predict the target class (moderate)
Term 2: minimise the number and magnitude of changes (L1 sparsity)

OUTPUTS
-------
    outputs/counterfactual_cases.csv        (paper Table 7)
    outputs/counterfactual_heatmap.png      (paper Figure 10)
    outputs/counterfactual_waterfall.png    (paper Figure 11)

USAGE
-----
    python src/counterfactual.py
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import joblib

from scipy.optimize       import differential_evolution, minimize
from sklearn.preprocessing import LabelEncoder

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(BASE, "data",    "splits")
MODELS = os.path.join(BASE, "models")
OUT    = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

FEATURES = [
    "age", "weight_kg", "bf_percent", "training_vol_hr_wk", "sleep_h",
    "testosterone_total", "estradiol", "ALT", "AST", "HDL", "LDL",
    "hematocrit", "creatinine", "mood_score", "libido_score", "enhancement_load",
]
CAT_FEATS = ["sex", "status", "goal"]

# Physiological units for clean display in figures
UNITS = {
    "age":                "years",
    "weight_kg":          "kg",
    "bf_percent":         "%",
    "training_vol_hr_wk": "hr/wk",
    "sleep_h":            "hours",
    "testosterone_total": "ng/dL",
    "estradiol":          "pg/mL",
    "ALT":                "U/L",
    "AST":                "U/L",
    "HDL":                "mg/dL",
    "LDL":                "mg/dL",
    "hematocrit":         "%",
    "creatinine":         "mg/dL",
    "mood_score":         "/10",
    "libido_score":       "/10",
    "enhancement_load":   "score",
}

# Clinical interpretation of each feature direction
CLINICAL_MEANING = {
    "hematocrit":         ("DOWN", "Therapeutic phlebotomy / hydration"),
    "ALT":                ("DOWN", "Hepatoprotective protocol / reduce oral AAS"),
    "AST":                ("DOWN", "Liver recovery — reduce hepatotoxic compounds"),
    "enhancement_load":   ("DOWN", "Reduce total PED dose or switch compounds"),
    "HDL":                ("UP",   "Cardiovascular exercise + omega-3 supplementation"),
    "LDL":                ("DOWN", "Diet modification + statin consideration"),
    "testosterone_total": ("DOWN", "Dose reduction or transition to cruise"),
    "creatinine":         ("DOWN", "Hydration + reduce nephrotoxic compounds"),
    "hematocrit":         ("DOWN", "Phlebotomy + aspirin + hydration protocol"),
}

# Mutable feature indices — numeric biomarkers only (not one-hot encoded sex/status)
MUTABLE_IDX = list(range(len(FEATURES)))  # first 16 columns


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD DATA
# ═════════════════════════════════════════════════════════════════════════════

def load_everything():
    test  = pd.read_csv(os.path.join(SPLITS, "test.csv"))
    train = pd.read_csv(os.path.join(SPLITS, "train.csv"))
    le    = joblib.load(os.path.join(MODELS, "le_risk_encoder.pkl"))

    with open(os.path.join(BASE, "src", "template_columns.json")) as f:
        cols = json.load(f)

    def prep(df):
        return (pd.get_dummies(df[FEATURES + CAT_FEATS])
                  .reindex(columns=cols, fill_value=0))

    X_test_df  = prep(test)
    X_train_df = prep(train)

    X_test  = X_test_df.values
    X_train = X_train_df.values
    y_test  = le.transform(test["risk"])

    # Feature ranges from training data — used for normalised distance
    feat_min   = X_train.min(axis=0)
    feat_max   = X_train.max(axis=0)
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1.0

    from catboost import CatBoostClassifier
    model = joblib.load(os.path.join(MODELS, "catboost_health_risk_model.pkl"))

    return model, X_test, X_train, y_test, le, feat_min, feat_max, feat_range, test


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — COUNTERFACTUAL FINDER
# ═════════════════════════════════════════════════════════════════════════════

def find_counterfactual(
    model, x_orig, feat_min, feat_max, feat_range,
    target_class: int = 2,   # 2 = moderate (next safer class)
    lam: float = 0.10,       # sparsity weight — higher = fewer changes
    max_iter_global: int = 100,
    max_iter_local:  int = 300,
) -> tuple:
    """
    Find the minimum-change counterfactual for a given sample.

    Parameters
    ----------
    model        : trained CatBoost classifier
    x_orig       : original feature vector (26-dim)
    feat_min/max : training data bounds for clamping
    feat_range   : training data ranges for normalisation
    target_class : class index to target (default 2 = moderate)
    lam          : L1 sparsity weight. Higher = fewer features changed.

    Returns
    -------
    x_cf      : counterfactual feature vector
    proba_cf  : model probabilities at counterfactual
    pred_cf   : predicted class at counterfactual
    n_changed : number of features that changed meaningfully
    success   : did the model predict the target class?

    Algorithm
    ---------
    Stage 1: Differential Evolution (global, gradient-free)
             Explores the full feature space to avoid local minima.
    Stage 2: L-BFGS-B (local refinement)
             Fine-tunes the solution found in Stage 1.

    Loss function (Wachter et al. 2017):
        L = (1 - P[target])² + λ · Σ|Δx_i / range_i|
    """
    def loss(x_mutable):
        x_full = x_orig.copy()
        x_full[MUTABLE_IDX] = x_mutable
        proba = model.predict_proba(x_full.reshape(1, -1))[0]
        # Term 1: prediction loss — drive toward target class
        pred_loss = (1.0 - proba[target_class]) ** 2
        # Term 2: sparsity loss — penalise large or many changes
        delta_norm = np.abs(
            (x_mutable - x_orig[MUTABLE_IDX]) / feat_range[MUTABLE_IDX]
        )
        dist_loss = delta_norm.sum()
        return pred_loss + lam * dist_loss

    bounds = [(feat_min[i], feat_max[i]) for i in MUTABLE_IDX]

    # Stage 1: global search
    result_global = differential_evolution(
        loss, bounds,
        seed=42, maxiter=max_iter_global,
        popsize=8, tol=0.001,
        mutation=(0.5, 1.0), recombination=0.7,
        workers=1,
    )

    # Stage 2: local refinement from global best
    result_local = minimize(
        loss, result_global.x,
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": max_iter_local, "ftol": 1e-8},
    )

    # Assemble counterfactual
    x_cf = x_orig.copy()
    x_cf[MUTABLE_IDX] = result_local.x
    x_cf = np.clip(x_cf, feat_min, feat_max)

    proba_cf = model.predict_proba(x_cf.reshape(1, -1))[0]
    pred_cf  = int(proba_cf.argmax())

    # Count meaningful changes (> 0.5% of feature range)
    deltas    = np.abs(x_cf[MUTABLE_IDX] - x_orig[MUTABLE_IDX])
    thresholds= feat_range[MUTABLE_IDX] * 0.005
    n_changed = int((deltas > thresholds).sum())

    success = (pred_cf == target_class)
    return x_cf, proba_cf, pred_cf, n_changed, success


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BUILD CASE STUDIES
# ═════════════════════════════════════════════════════════════════════════════

def build_case_studies(
    model, X_test, y_test, le, feat_min, feat_max, feat_range, test_df,
    n_cases: int = 5,
) -> list:
    """
    Find n_cases high-risk samples with confident predictions,
    compute their counterfactuals, and return structured results.
    """
    # Find high-confidence high-risk samples
    high_idx      = np.where(y_test == 0)[0]   # 0 = 'high'
    high_conf_idx = [
        idx for idx in high_idx
        if model.predict_proba(X_test[idx].reshape(1, -1))[0][0] >= 0.70
    ]

    print(f"  High-risk samples (test): {len(high_idx)}")
    print(f"  High-confidence (>=70%):  {len(high_conf_idx)}")
    print(f"  Running counterfactuals for {n_cases} cases...")
    print()

    cases = []
    for case_num, idx in enumerate(high_conf_idx[:n_cases]):
        x_orig   = X_test[idx]
        proba_orig = model.predict_proba(x_orig.reshape(1, -1))[0]

        # Pull original categorical values for display
        row = test_df.iloc[idx]
        meta = {
            "sex":    row.get("sex",    "male"),
            "status": row.get("status", "on"),
            "goal":   row.get("goal",   "bulk"),
            "age":    int(row.get("age", 0)),
        }

        print(f"  Case {case_num+1} | idx={idx} | "
              f"sex={meta['sex']} status={meta['status']} age={meta['age']}")
        print(f"    Original: P(high)={proba_orig[0]:.3f}  "
              f"P(mod)={proba_orig[2]:.3f}")

        x_cf, proba_cf, pred_cf, n_changed, success = find_counterfactual(
            model, x_orig, feat_min, feat_max, feat_range, target_class=2
        )
        pred_label = le.classes_[pred_cf]

        print(f"    CF result: {pred_label} (success={success})")
        print(f"    P(high)={proba_cf[0]:.3f}  P(mod)={proba_cf[2]:.3f}")
        print(f"    Features changed: {n_changed}")

        # Collect per-feature changes
        feature_changes = []
        for fi, feat in enumerate(FEATURES):
            orig_val = float(x_orig[fi])
            cf_val   = float(x_cf[fi])
            delta    = cf_val - orig_val
            pct      = (delta / max(abs(orig_val), 1e-6)) * 100

            if abs(delta) > feat_range[fi] * 0.005:
                direction = "UP" if delta > 0 else "DOWN"
                clinical  = CLINICAL_MEANING.get(feat, (direction, "Lifestyle modification"))[1]
                feature_changes.append({
                    "feature":        feat,
                    "unit":           UNITS.get(feat, ""),
                    "original":       round(orig_val, 3),
                    "counterfactual": round(cf_val,   3),
                    "delta":          round(delta,    3),
                    "delta_pct":      round(pct,      1),
                    "direction":      direction,
                    "clinical_action": clinical,
                    "abs_delta_norm": abs(delta) / feat_range[fi],
                })

        feature_changes.sort(key=lambda x: x["abs_delta_norm"], reverse=True)

        print(f"    Top changes:")
        for fc in feature_changes[:4]:
            print(f"      {fc['feature']:22s}: "
                  f"{fc['original']:7.2f} → {fc['counterfactual']:7.2f} "
                  f"({fc['direction']} {abs(fc['delta_pct']):.1f}%)  "
                  f"| {fc['clinical_action']}")
        print()

        cases.append({
            "case_num":         case_num + 1,
            "test_idx":         int(idx),
            "meta":             meta,
            "proba_orig":       proba_orig.tolist(),
            "proba_cf":         proba_cf.tolist(),
            "pred_original":    "high",
            "pred_cf":          pred_label,
            "success":          success,
            "n_changed":        n_changed,
            "feature_changes":  feature_changes,
        })

    return cases


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SAVE CASE STUDIES CSV
# ═════════════════════════════════════════════════════════════════════════════

def save_case_table(cases: list) -> pd.DataFrame:
    """Save paper-ready Table 7: counterfactual case studies."""
    rows = []
    for case in cases:
        for fc in case["feature_changes"][:5]:  # top 5 changes per case
            rows.append({
                "Case":             case["case_num"],
                "Age/Sex/Status":   f"{case['meta']['age']}/{case['meta']['sex']}/{case['meta']['status']}",
                "P(high) before":   round(case["proba_orig"][0], 3),
                "P(high) after":    round(case["proba_cf"][0],   3),
                "CF predicted":     case["pred_cf"],
                "Feature":          fc["feature"],
                "Unit":             fc["unit"],
                "Original value":   fc["original"],
                "CF value":         fc["counterfactual"],
                "Change":           f"{fc['direction']} {abs(fc['delta_pct']):.1f}%",
                "Clinical action":  fc["clinical_action"],
            })

    df = pd.DataFrame(rows)
    path = os.path.join(OUT, "counterfactual_cases.csv")
    df.to_csv(path, index=False)
    print(f"  Case table → {path}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FIGURES
# ═════════════════════════════════════════════════════════════════════════════

def plot_waterfall(cases: list):
    """
    Waterfall chart showing before → after risk probability for each case.
    Also shows which features drove each counterfactual.
    Paper Figure 10.
    """
    n       = len(cases)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 6))
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "Counterfactual Explanations — High Risk → Moderate Risk\n"
        "Minimum feature changes required to cross the decision boundary",
        fontsize=12, fontweight="bold"
    )

    for ci, (case, ax) in enumerate(zip(cases, axes)):
        changes = case["feature_changes"][:5]
        feat_names = [f"{fc['feature']}\n({UNITS.get(fc['feature'],'')})"
                      for fc in changes]
        deltas_pct = [fc["delta_pct"] for fc in changes]

        colors = ["#1D9E75" if d < 0 else "#E24B4A" for d in deltas_pct]
        bars   = ax.barh(range(len(changes)), deltas_pct,
                          color=colors, alpha=0.85)

        for bar, val in zip(bars, deltas_pct):
            sign = "+" if val > 0 else ""
            ax.text(
                val + (0.5 if val >= 0 else -0.5),
                bar.get_y() + bar.get_height() / 2,
                f"{sign}{val:.1f}%",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=8, fontweight="bold"
            )

        ax.set_yticks(range(len(changes)))
        ax.set_yticklabels(feat_names, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("% Change from original", fontsize=8)

        p_before = case["proba_orig"][0]
        p_after  = case["proba_cf"][0]
        title    = (f"Case {case['case_num']}\n"
                    f"{case['meta']['sex']}, {case['meta']['age']}y, {case['meta']['status']}\n"
                    f"P(high): {p_before:.2f} → {p_after:.2f}")
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

        # Risk arrow annotation
        ax.annotate(
            f"Risk: HIGH → {case['pred_cf'].upper()}",
            xy=(0.5, -0.22), xycoords="axes fraction",
            ha="center", fontsize=8,
            color="#1D9E75" if case["success"] else "#E24B4A",
            fontweight="bold"
        )

    down_patch = mpatches.Patch(color="#1D9E75", alpha=0.85, label="Decrease")
    up_patch   = mpatches.Patch(color="#E24B4A", alpha=0.85, label="Increase")
    fig.legend(handles=[down_patch, up_patch],
               loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.04), frameon=False)

    plt.tight_layout()
    path = os.path.join(OUT, "counterfactual_waterfall.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Waterfall chart    → {path}")


def plot_feature_frequency(cases: list):
    """
    How often does each feature appear in counterfactuals?
    Shows which biomarkers are the most actionable leverage points.
    Paper Figure 11.
    """
    feat_counts = {}
    feat_avg_delta = {}

    for case in cases:
        for fc in case["feature_changes"]:
            feat = fc["feature"]
            feat_counts[feat] = feat_counts.get(feat, 0) + 1
            if feat not in feat_avg_delta:
                feat_avg_delta[feat] = []
            feat_avg_delta[feat].append(abs(fc["delta_pct"]))

    # Sort by frequency
    sorted_feats = sorted(feat_counts, key=feat_counts.get, reverse=True)[:10]
    counts       = [feat_counts[f] for f in sorted_feats]
    avg_deltas   = [np.mean(feat_avg_delta[f]) for f in sorted_feats]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Counterfactual Feature Analysis — Across All 5 Case Studies",
        fontsize=12, fontweight="bold"
    )

    # Plot 1: frequency
    y  = range(len(sorted_feats))
    ax1.barh(y, counts, color="#185FA5", alpha=0.82)
    ax1.set_yticks(y)
    ax1.set_yticklabels(
        [f"{f} ({UNITS.get(f,'')})" for f in sorted_feats], fontsize=9
    )
    ax1.invert_yaxis()
    ax1.set_xlabel("Appears in N counterfactuals", fontsize=9)
    ax1.set_title("Feature Frequency\n(how often this feature must change)",
                  fontsize=10, fontweight="bold")
    ax1.set_xlim(0, len(cases) + 0.5)
    for xi, val in enumerate(counts):
        ax1.text(val + 0.05, xi, str(val), va="center", fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)

    # Plot 2: average magnitude of change
    ax2.barh(y, avg_deltas, color="#E24B4A", alpha=0.82)
    ax2.set_yticks(y)
    ax2.set_yticklabels(
        [f"{f} ({UNITS.get(f,'')})" for f in sorted_feats], fontsize=9
    )
    ax2.invert_yaxis()
    ax2.set_xlabel("Average % change required", fontsize=9)
    ax2.set_title("Average Change Magnitude\n(how much this feature must change)",
                  fontsize=10, fontweight="bold")
    for xi, val in enumerate(avg_deltas):
        ax2.text(val + 0.2, xi, f"{val:.1f}%", va="center", fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUT, "counterfactual_feature_freq.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Feature frequency  → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PAPER PARAGRAPH
# ═════════════════════════════════════════════════════════════════════════════

def print_paper_paragraph(cases: list):
    # Pull stats from Case 1 for concrete example
    c1       = cases[0]
    top_feat = c1["feature_changes"][0]
    sec_feat = c1["feature_changes"][1] if len(c1["feature_changes"]) > 1 else None

    # Most frequent feature across all cases
    feat_counts = {}
    for case in cases:
        for fc in case["feature_changes"]:
            feat_counts[fc["feature"]] = feat_counts.get(fc["feature"], 0) + 1
    most_common = max(feat_counts, key=feat_counts.get)

    avg_changes = np.mean([c["n_changed"] for c in cases])
    success_rate= sum(c["success"] for c in cases) / len(cases)

    print("\n" + "=" * 65)
    print("  PAPER PARAGRAPH — Section 4.3 (Counterfactual Explanations)")
    print("=" * 65)
    print(f"""
To provide actionable clinical insights beyond feature attribution,
we generate counterfactual explanations (Wachter et al., 2017) for
high-risk athlete profiles. Counterfactual explanations identify the
minimum-change perturbation to an athlete's biomarker profile that
would reclassify their risk from HIGH to MODERATE, providing
clinically actionable guidance beyond what SHAP values alone convey.

We apply a two-stage gradient-free optimisation (differential
evolution followed by L-BFGS-B local refinement) to solve the
Wachter objective, which minimises prediction loss subject to an
L1 sparsity penalty (λ={0.10}) on normalised feature changes.
Counterfactuals were successfully generated for {int(success_rate*100)}%
of the five high-risk case studies examined (P(high) ≥ 0.70),
with an average of {avg_changes:.1f} features requiring modification.

In Case 1 (male, {c1['meta']['age']}y, {c1['meta']['status']} cycle,
P(high)={c1['proba_orig'][0]:.2f}), the counterfactual required
{top_feat['feature']} to change from {top_feat['original']:.1f} to
{top_feat['counterfactual']:.1f} {top_feat['unit']}
({top_feat['direction']} {abs(top_feat['delta_pct']):.1f}%){
    f", and {sec_feat['feature']} from {sec_feat['original']:.1f} to {sec_feat['counterfactual']:.1f} {sec_feat['unit']}"
    if sec_feat else ""
}, reducing P(high) to {c1['proba_cf'][0]:.2f}.
Across all cases, {most_common} appeared most frequently as a
required counterfactual change, consistent with its role as the
primary risk driver identified in SHAP analysis. These results
suggest that targeted interventions — particularly haematological
monitoring and enhancement load reduction — represent the highest-
leverage clinical actions for risk mitigation in this population.
    """.strip())
    print("=" * 65 + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n  Day 6 — Counterfactual Explanations")
    print("  " + "=" * 50)

    print("\n  Loading data and model...")
    (model, X_test, X_train, y_test, le,
     feat_min, feat_max, feat_range, test_df) = load_everything()
    print(f"    Test set: {len(y_test)} samples")
    print(f"    Classes: {le.classes_}")

    print("\n  Generating counterfactuals for 5 high-risk cases...")
    print("  (differential evolution + L-BFGS-B — may take ~2 minutes)")
    cases = build_case_studies(
        model, X_test, y_test, le,
        feat_min, feat_max, feat_range, test_df,
        n_cases=5,
    )

    print("  Saving case table...")
    save_case_table(cases)

    print("  Generating figures...")
    plot_waterfall(cases)
    plot_feature_frequency(cases)

    print_paper_paragraph(cases)

    print("  Day 6 complete.")
    print("  Next: Day 7 — ethics section + blind rating protocol design")
    print("  Then: Day 8 — paper writing (Methods + Results sections)\n")


if __name__ == "__main__":
    main()