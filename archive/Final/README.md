# SafeSpan: Predictive Bridge Analytics

SafeSpan is a machine learning project for predicting the structural condition of U.S. bridges using historical NBI inspection data. The goal is to move from reactive maintenance scheduling to a proactive, data-driven triage framework that prioritizes bridges by risk before inspectors arrive.

---

## Project Overview

The U.S. maintains approximately 617,000 public road bridges. Many are approaching or exceeding their original 50-year design life. This project builds a complete ML pipeline on **4.8 million bridge-year records** (2018–2025) to classify each bridge into one of four condition states: **Critical**, **Poor**, **Fair**, or **Good**, and produce a traffic-weighted risk ranking for inspection prioritization.

The final model is a LightGBM classifier trained on 4.2M rows with class-balanced weighting, evaluated on a strict 2025 temporal holdout of 600K bridges. The output is not just a predicted label but a calibrated probability score per class, combined with average daily traffic to generate a ranked inspection priority list.

---

## Results Summary

| Metric | Value |
|---|---|
| Macro F1 (2025 test) | 0.3148 |
| ROC-AUC (OvR macro) | 0.8046 |
| Critical Recall | 96.1% |
| Critical to Good Error | 2.23% |
| 5-fold CV Macro F1 | 0.3448 +/- 0.003 |

**Benchmark (500K sample, 5 models):**

| Model | Macro F1 | AUC | Crit-Good% |
|---|---|---|---|
| Logistic Regression | 0.242 | 0.569 | 29.8% |
| Gaussian Naive Bayes | 0.250 | 0.577 | 2.7% |
| Random Forest | **0.562** | **0.802** | **2.7%** |
| LightGBM | 0.385 | 0.684 | 14.6% |
| SVM (RBF, 15K) | 0.205 | 0.552 | 21.0% |

Random Forest achieved the strongest benchmark F1 and AUC. LightGBM was selected for the final model because it produces well-calibrated probabilities on the full 4.2M training set, which the risk-ranking output depends on.

---

## Repository Structure

```
SafeSpan/
├── SafeSpan_Notebook_Submit.ipynb   Main project notebook (45 cells, fully executed)
├── SafeSpan_Final_Presentation_v4.pptx   12-slide presentation (navy/teal, real plots)
├── SafeSpan_Midway_Report.docx      IEEE-format midway report (6 pages, two-column)
├── data_collection.py               Script to download and aggregate NBI ASCII files from FHWA
├── outputs/
│   ├── plots/                       35 generated figures (EDA, drift, SHAP, KM, Cox, dashboard)
│   ├── benchmark_comparison.csv     5-model benchmark results
│   ├── test_2025_metrics.csv        Final model test metrics
│   ├── bridge_risk_ranking.csv      600K bridges ranked by WEIGHTED_RISK
│   ├── cox_hazard_summary.csv       Cox PH hazard ratio table
│   ├── survival_dataset.csv         475,966 bridges, 7,161 events
│   ├── psi_numeric_drift.csv        PSI values by feature and year
│   ├── js_categorical_drift.csv     JS distance for categorical features
│   ├── stratified_cv_macro_f1.csv   5-fold CV results
│   └── feature_importance_gain.csv  LightGBM native importance scores
└── README.md
```

---

## Dataset

Source: [Federal Highway Administration National Bridge Inventory](https://www.fhwa.dot.gov/bridge/nbi/ascii.cfm)

- **Raw records**: 5M+ bridge-year observations across 2018-2025
- **Working dataset**: 4.8M rows after filtering to bridges present in all 8 years (consistent panel)
- **Test set**: 600,000 bridges from the 2025 inspection cycle (strict temporal holdout)
- **Features**: 48 raw NBI attributes reduced to 27 after leakage removal, near-zero variance filtering, and Pearson correlation pruning at rho = 0.90
- **Target**: BRIDGE_CONDITION = min(DECK, SUPERSTRUCTURE, SUBSTRUCTURE) rating, mapped to Critical / Poor / Fair / Good

Due to size (~2.1GB raw), the NBI CSV files are not included. Use `data_collection.py` to download and assemble them locally.

---

## Notebook Sections

| Section | Content |
|---|---|
| 1 | Data loading, manifest, column inventory |
| 2 | EDA: class distribution, age by condition, material, traffic, state rates |
| 3 | Feature engineering: 10 domain features including TRAFFIC_DENSITY, AGE_X_SPANS, LOG_ADT |
| 4 | PSI drift analysis: numeric features vs 2018 baseline, COVID/LRFR/aging patterns |
| 5 | JS distance: categorical drift (all 0.0 - consistent panel expected) |
| 5.5 | Multi-model benchmark: 5 families on 500K sample |
| 6 | LightGBM champion training: 2000 trees, class_weight=balanced, 4.2M rows |
| 6.1 | Stratified 5-fold CV on 250K subset |
| 7 | 2025 test set evaluation, confusion matrix, per-class report |
| 7.5 | Risk ranking: WEIGHTED_RISK = RISK_SCORE x log(1+ADT) |
| 8 | SHAP global importance (10K sample), beeswarm, native gain, PDP |
| 8.5 | LIME local explanations for Critical-to-Good misclassification cases |
| 9 | Performance drift: model trained 2018-2022, evaluated on 2023/2024/2025 |
| 10 | Survival analysis setup: consistent-panel bridge-level deterioration events |
| 11 | Kaplan-Meier curves: overall, by age group, by material, by rating gap |
| 12 | Cox Proportional Hazards: hazard ratios, PH assumption check |
| 13 | Final dashboard (22x14 multi-panel figure) and complete results summary |

---

## Key Technical Choices

**Why class_weight=balanced on LightGBM?**
The Critical class is only 6.1% of the data. Without reweighting, the model learns to mostly ignore it. Balanced weighting amplifies gradient updates for rare classes, achieving 96.1% Critical recall. The tradeoff is lower macro F1 — expected and intentional for a safety-critical triage framework.

**Why LightGBM over Random Forest for the final model?**
Random Forest achieved higher benchmark F1 (0.562 vs 0.385 on 500K). LightGBM was selected for the final 4.2M-row training run because it scales more efficiently to large data and produces calibrated probabilities via Platt scaling, which the risk-ranking output requires. The ROC-AUC of 0.8046 confirms strong discriminative ability.

**Why PSI and JS Distance for drift?**
PSI is the industry-standard metric for numeric drift monitoring. It measures practical magnitude rather than statistical significance — important because at 600K rows, standard tests (KS, chi-square) flag trivial differences as significant. JS Distance is used for categorical features because it is symmetric, bounded 0 to 1, and handles zero-frequency categories without undefined behavior.

**Why Kaplan-Meier and Cox PH?**
KM estimates time-to-deterioration survival curves without distributional assumptions, handling 98.5% censored observations (bridges that stayed healthy through 2025). Cox PH quantifies which features drive deterioration hazard. Both independently confirm bridge age as the dominant risk factor, consistent with the SHAP results.

---

## Data Drift Findings

All features showed PSI below 0.10 across 2019-2025. Three drift patterns identified:

- **2020 traffic drop (PSI spike then recovery)**: COVID-19 reduced ADT by ~25% nationally. Recovered by 2022.
- **2021 load rating shift (one-time step)**: AASHTO LRFR methodology adoption recalibrated operating and inventory ratings.
- **Annual age increase (slow linear climb)**: BRIDGE_AGE increases exactly 1 year per inspection cycle. Expected, non-random.

All categorical features showed JS Distance = 0.0. This is the correct result for a consistent panel where the same 600K bridge IDs appear every year.

---

## Survival Analysis Findings

| Metric | Value |
|---|---|
| Bridges in survival dataset | 475,966 |
| Observed deterioration events | 7,161 (1.5%) |
| Censored (stayed healthy) | 468,805 (98.5%) |
| KM median survival time | Infinity (most bridges never deteriorated in 8 years) |
| Log-rank p-value (age groups) | 7.43 x 10^-99 |
| Cox PH assumption | Passed |

Cox hazard ratios confirm: bridge age, ADT, and scour vulnerability increase deterioration hazard. Higher operating ratings and more frequent inspections reduce it.

---

## Team

Akashkumar Senthilkumar, Pramod Satya Dindukurthi, Shriram Dundigalla, Shruthi Thirukumaran

**Course**: DATA 245 - Machine Learning Technologies, San Jose State University, Spring 2026
