# Data Card — Enhanced Athlete Synthetic Health Risk Dataset v2

## Overview
| Field | Value |
|---|---|
| Version | 2.0 |
| Created | 2025 |
| Samples | 3,000 |
| Features | 19 (16 numeric + sex, status, goal) |
| Label | risk (low / moderate / high) |
| Generator | data/data_generator.py |
| Seed | 42 |

## Motivation
Enhanced athletes (individuals using performance-enhancing drugs) present
physiological profiles outside standard clinical reference ranges. No
publicly available labeled dataset exists for this population. This
synthetic benchmark is designed to enable development and evaluation of
ML-based risk stratification methods, pending real-world validation.

## Label Generation — MUST DISCLOSE IN PAPER
Risk labels are assigned via a deterministic scoring function
(`compute_risk_score` in data_generator.py), NOT from physician
diagnosis or clinical outcomes. The scoring function awards points
for clinically motivated thresholds:
- Hematocrit > 52% (polycythemia risk)
- ALT > 50 U/L (hepatotoxicity)
- Enhancement load > 1.25 (high PED burden)
- HDL < 35 mg/dL (cardiovascular risk)
- LDL > 160 mg/dL (cardiovascular risk)
- Creatinine > 1.4 (renal strain)

Score >= 7 → high | Score 3-6 → moderate | Score < 3 → low

## Class Distribution
| Class | Count | Proportion |
|---|---|---|
| low | 1583 | 52.8% |
| moderate | 1169 | 39.0% |
| high | 248 | 8.3% |

## Splits
| Split | N | Purpose |
|---|---|---|
| train | 1800 | Model training |
| val | 450 | Hyperparameter tuning |
| calibration | 300 | Conformal prediction calibration ONLY |
| test | 450 | Final held-out evaluation — do not touch until paper |

## Features
| Feature | Type | Unit | Physiological Range | Clinical Significance |
|---|---|---|---|---|
| age | int | years | 18-55 | Age-related hormonal decline |
| sex | cat | male/female | — | Sex-specific reference ranges |
| status | cat | on/off/cruise/pct | — | Cycle phase |
| goal | cat | bulk/cut/recomp/maintenance | — | Training objective |
| weight_kg | float | kg | 55-140 | Body mass |
| bf_percent | float | % | 4-30 | Body fat |
| training_vol_hr_wk | float | hr/week | 2-28 | Training load |
| sleep_h | float | hours | 4-10 | Recovery marker |
| testosterone_total | float | ng/dL | 50-2500 | Primary AAS marker |
| estradiol | float | pg/mL | 10-200 | Aromatization marker |
| ALT | float | U/L | 5-200 | Hepatotoxicity marker |
| AST | float | U/L | 5-200 | Hepatotoxicity marker |
| HDL | float | mg/dL | 15-100 | Cardiovascular risk (inverse) |
| LDL | float | mg/dL | 50-300 | Cardiovascular risk |
| hematocrit | float | % | 28-65 | Polycythemia / thrombosis risk |
| creatinine | float | mg/dL | 0.4-3.5 | Renal function |
| mood_score | float | 1-10 | 1-10 | Subjective wellbeing |
| libido_score | float | 1-10 | 1-10 | Hormonal suppression proxy |
| enhancement_load | float | composite | 0.3-2.0 | Total PED burden |

## Physiological Validation
Biomarker distributions were compared against:
- NHANES 2017-2018 (HDL: N=6,738)
- Bhasin et al. 2010, NEJM (testosterone)
- Stout et al. 2017, JAGS (hematocrit)
- Prati et al. 2002, Ann Intern Med (ALT/AST)
- Grundy et al. 2018, Circulation (LDL)
- Levey et al. 2009, Ann Intern Med (creatinine)

See outputs/nhanes_validation_comparison.png and
data/references/references_table.csv for full comparison.

## Known Limitations
1. Synthetic labels — not validated by physician diagnosis
2. 85% male cohort — female athlete profiles underrepresented
3. Single time-point — no longitudinal trajectories
4. No genetic, nutrition, or sleep quality data
5. Western biomarker reference ranges may not generalize globally

## Intended Use
Research and benchmarking only. NOT for clinical deployment.
Any real-world use requires validation on clinical data with
appropriate ethics approval.

## License
CC BY 4.0 — free to use with attribution.