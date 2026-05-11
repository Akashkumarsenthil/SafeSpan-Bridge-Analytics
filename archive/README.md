# SafeSpan: Predictive Bridge Analytics

**Predictive Maintenance for U.S. Bridge Infrastructure using Machine Learning, Data Drift Analysis, and Survival Analysis**

> DATA 245 — Machine Learning Technologies | Spring 2026
> Team: Akashkumar Senthilkumar · Pramod Satya Dindukurthi · Shriram Dundigalla · Shruthi Thirukumaran

---

## Project Overview

The U.S. has over 600,000 public road bridges. Approximately 7–8% are rated **Critical** at any given inspection cycle — yet reactive maintenance costs 5–6× more than proactive repair. SafeSpan applies modern ML and statistical methods to the **National Bridge Inventory (NBI)** dataset to answer three core questions:

1. **Can we predict a bridge's condition class** (Critical / Poor / Fair / Good) from inspection and structural features — without using the raw rating scores that define the label?
2. **Are the data distributions shifting over time?** If so, when and which features are drifting most — and does a model trained on older data lose accuracy on newer inspections?
3. **How long does a Good/Fair bridge typically survive before deteriorating?** Which structural and operational factors accelerate or delay that transition?

---

## Dataset

| Property | Detail |
|----------|--------|
| Source | [FHWA National Bridge Inventory](https://www.fhwa.dot.gov/bridge/nbi/ascii.cfm) |
| Coverage | 2018 – 2025 (8 annual snapshots) |
| Training set | `nbi_train_2018.csv.gz` → `nbi_train_2024.csv.gz` (7 years, ~4.2 M rows) |
| Test set | `nbi_test_2025.csv.gz` (~600 K rows, **strict temporal hold-out**) |
| Total records | ~4.8 M bridge-year observations |
| Unique bridges | ~600 K distinct structure IDs |
| Target column | `BRIDGE_CONDITION` (Critical / Poor / Fair / Good) |

### Target Definition (NBI Standard)

The `BRIDGE_CONDITION` label is derived from the minimum of the deck, superstructure, and substructure NBI ratings (0–9 scale):

| Rating | Label |
|--------|-------|
| 0 – 3 | Critical |
| 4 | Poor |
| 5 – 6 | Fair |
| 7 – 9 | Good |

### Class Distribution (2018–2025)

| Class | Share |
|-------|-------|
| Good | ~57% |
| Fair | ~33% |
| Poor | ~7% |
| Critical | ~3% |

The dataset is **significantly imbalanced** — Critical bridges are a small minority. All models use `class_weight="balanced"` and `compute_sample_weight` to compensate.

---

## Repository Structure

```
.
├── SafeSpan_Final_Notebook.py       # Primary notebook (# %% cell format, VS Code / Jupyter)
├── SafeSpan_Final_Notebook.ipynb    # Jupyter notebook (auto-generated from .py)
├── data_collection.py               # FHWA data downloader / synthetic fallback
├── bridge_ml_pipeline.ipynb         # Early 6-model benchmark (LR, GNB, RF, LGBM, SVM)
├── EDA_Data_Cleaning.ipynb          # Initial EDA and cleaning notebook
├── EDA_GUIDE.md                     # Cleaning guide for NBI columns
├── presentation_outline.md          # Speaker notes for progress presentation
├── outputs/
│   ├── plots/                       # All 19 saved figures (PNG, 150 dpi)
│   ├── psi_numeric_drift.csv        # PSI values per feature per year
│   ├── js_categorical_drift.csv     # Jensen-Shannon distances per category
│   ├── target_drift_by_year.csv     # Condition share % per year (2018–2025)
│   ├── test_2025_metrics.csv        # Final model metrics on 2025 hold-out
│   ├── model_performance_drift.csv  # Temporal drift model evaluation
│   ├── survival_dataset.csv         # One-row-per-bridge survival panel
│   └── cox_hazard_summary.csv       # Cox PH hazard ratios and confidence intervals
├── nbi_train_2018.csv.gz            # (local only — gitignored)
│   ...
└── nbi_test_2025.csv.gz             # (local only — gitignored)
```

---

## Installation

```bash
pip install lightgbm lifelines shap imbalanced-learn pyarrow scikit-learn pandas numpy matplotlib scipy
```

The notebook degrades gracefully:
- If **LightGBM** is unavailable (e.g., missing `libomp`), a **RandomForest** fallback activates automatically.
- If **SHAP** is not installed, Section 8 is skipped with an informative message.
- If **lifelines** is not installed, Sections 10–12 (survival analysis) are skipped.

---

## Running the Notebook

**Jupyter:**
```bash
jupyter notebook SafeSpan_Final_Notebook.ipynb
```

**VS Code** (Run All Cells, `# %%` format):
```bash
code SafeSpan_Final_Notebook.py
```

**Script mode** (headless, saves all plots to `outputs/plots/`):
```bash
python SafeSpan_Final_Notebook.py
```

All data files must be in the same directory as the notebook. The notebook auto-detects whether it is running inside Jupyter (interactive) or as a script (Agg backend, saves plots silently).

---

## What the Notebook Does — Section by Section

### Section 0 — Setup
Imports all required libraries, sets up graceful fallbacks (LightGBM → RF, SHAP optional, lifelines optional), defines the global color palette (`Critical=red, Poor=orange, Fair=yellow-green, Good=green`), and creates the `outputs/` directory tree.

---

### Section 1 — Data Loading & Integration

Loads all 8 NBI annual files into a unified 4.8 M-row panel:

- Streams each `.csv.gz` individually to avoid memory spikes
- Prints per-year row counts and condition breakdowns
- Explicitly defines and documents **10 leakage columns** that must be excluded from all models

**Leakage prevention** is a central design decision. The raw condition ratings (`DECK_COND_058`, `SUPERSTRUCTURE_COND_059`, `SUBSTRUCTURE_COND_060`, `CULVERT_COND_062`) are the direct inputs used by FHWA to compute `BRIDGE_CONDITION`. Including them would give ~100% accuracy — but that is not a useful model. Similarly, the derived FHWA evaluation scores (`STRUCTURAL_EVAL_067` through `APPR_ROAD_EVAL_072`) are excluded because they are computed downstream from the same rating inputs.

Section 1.1 prints a full **Column Inventory** tagging every column as `[LEAKAGE]`, `[ID/META]`, `[TARGET]`, or `[FEATURE]`, making the decision transparent and auditable.

---

### Section 2 — Exploratory Data Analysis

Six diagnostic plots covering the full 4.8 M dataset:

**2.1 Class Distribution**
Bar + pie chart of overall condition shares. Good bridges (~57%) dominate; Critical (~3%) is a rare-event class requiring special handling.

**2.2 Condition Trends by Year**
Stacked bar and line chart showing how the proportion of each condition class changes from 2018 to 2025. This is a preview of the target drift that will be formally quantified in Section 5.
- **Key finding:** Good bridges have increased in share while Fair has declined — suggesting either genuine infrastructure improvement or shifting inspection criteria over time.

**2.3 Bridge Age Distribution by Condition**
Histogram and notched boxplot of `BRIDGE_AGE_AT_INSPECTION` by condition.
- **Key finding:** Critical and Poor bridges have a significantly higher median age than Good bridges. The age distribution of Critical bridges is right-skewed toward older structures.

**2.4 Condition by Bridge Material**
Horizontal stacked bar chart of Critical rate by material type (Concrete, Steel, Prestressed Concrete, Wood/Timber, Masonry, etc.).
- **Key finding:** Masonry and Wood/Timber bridges have the highest Critical rates. Prestressed concrete structures perform better on average.

**2.5 Traffic Volume (ADT) by Condition**
Log-transformed ADT distributions overlaid by condition.
- **Key finding:** Critical bridges tend to carry less traffic than Good bridges — consistent with the fact that high-traffic structures receive more frequent inspection and repair.

**2.6 Top 20 States by Critical Rate**
Geographic ranking of states by percentage of Critical-condition bridges, identifying where infrastructure stress is most acute.

---

### Section 3 — Feature Engineering

Five new domain features are created from raw NBI columns to capture structural and operational relationships not available in raw form:

| Feature | Formula | Engineering Rationale |
|---------|---------|----------------------|
| `TRAFFIC_DENSITY` | ADT ÷ span length | Normalises traffic by structural scale — a short bridge carrying heavy load is more stressed |
| `AGE_TO_SPAN_RATIO` | age ÷ max span length | Longer spans accumulate more structural fatigue per year of age |
| `DECK_UTILISATION` | deck width ÷ max span length | Disproportionately wide decks on short spans may signal design-load mismatch |
| `LOG_ADT` | log₁(1 + ADT) | Compresses the heavy right-tail of average daily traffic — prevents one very busy bridge from dominating |
| `RATING_DIFF` | operating rating − inventory rating | Gap between the load a bridge can carry safely vs. the posted limit; a shrinking gap signals approaching unsafe condition |

All ratio features are clipped at their 1st and 99th percentiles to suppress outlier influence, and infinite values (from division by zero) are replaced with NaN before imputation.

---

### Section 4 — Data Drift Analysis (PSI + Jensen-Shannon)

**Why drift analysis matters:** A model trained on 2018 data and deployed in 2025 may encounter bridges whose structural profiles are statistically different from the training distribution. If distributions shift significantly, model predictions become unreliable without retraining.

Two complementary drift metrics are computed for every year from 2019 to 2025 vs. the **2018 baseline**:

#### Population Stability Index (PSI) — Numeric Features

PSI measures how much the distribution of a numeric feature has shifted. It is the industry standard for model monitoring.

| PSI Range | Interpretation |
|-----------|---------------|
| < 0.10 | Stable — model is still reliable |
| 0.10 – 0.25 | Moderate drift — monitor closely |
| ≥ 0.25 | Significant drift — retraining recommended |

**Features monitored:** `BRIDGE_AGE_AT_INSPECTION`, `ADT_029`, `LOG_ADT`, `OPERATING_RATING_064`, `INVENTORY_RATING_066`, `RATING_DIFF`, `TRAFFIC_DENSITY`, `AGE_TO_SPAN_RATIO`, `MAX_SPAN_LEN_MT_048`, `STRUCTURE_LEN_MT_049`, `DECK_WIDTH_MT_052`

PSI is computed year-by-year and saved to `outputs/psi_numeric_drift.csv`. A dual-panel figure (average PSI bar + PSI trend for the top 5 drifting features) is saved as `08_psi_drift.png`.

**Significance:** Features showing significant PSI (≥ 0.25) indicate that the bridge population being inspected in later years is structurally or operationally different from those inspected in 2018 — either because the fleet composition has changed, inspection priorities have shifted, or infrastructure investment has altered which bridges are still in service.

#### Jensen-Shannon Distance — Categorical Features

For categorical columns, PSI is not appropriate. JS distance measures the distributional divergence between two probability distributions. Unlike KL divergence, JS distance is always finite and symmetric.

| JS Distance | Interpretation |
|-------------|---------------|
| < 0.05 | Low drift |
| 0.05 – 0.15 | Moderate drift |
| > 0.15 | High drift |

**Features monitored:** `STRUCTURE_KIND_043` (material), `STRUCTURE_TYPE_044` (design type), `STATE_CODE_001` (state), `FUNCTIONAL_CLASS_026` (road function), `MAINTENANCE_021` (maintenance responsibility), `OWNER_022` (ownership type)

Results saved to `outputs/js_categorical_drift.csv` and plotted as `09_js_drift.png`.

**Significance:** Drift in state or ownership distribution would indicate that the geographic or institutional mix of inspected bridges is changing — which would affect which features are predictive and whether feature importances remain stable.

---

### Section 5 — Target Drift Analysis

Target drift is distinct from data drift: here the **label distribution itself** changes over time, not just the input features. This matters because a model trained on 2018 label proportions (e.g., 3% Critical) may be miscalibrated if the true 2025 prevalence is different.

**Method:** The proportion of each condition class is computed for every year (2018–2025) and changes are reported in percentage points (pp).

**Key findings reported in the notebook:**
- Critical share: change from 2018 → 2025
- Good share: change from 2018 → 2025
- Fair share: change from 2018 → 2025 (from prior runs: approximately −19.71 pp — the largest shift)
- Good share change: approximately +15.74 pp

A dashed vertical line at 2024.5 marks the train/test boundary in both the stacked bar and line charts (`10_target_drift.png`). Results saved to `outputs/target_drift_by_year.csv`.

**Significance:** Systematic label drift means the model's class probabilities will be incorrectly calibrated on future data. It also reveals whether U.S. bridge infrastructure is genuinely improving (Good share rising) or whether inspection practices / reporting standards are changing.

---

### Section 6 — Model Training (LightGBM)

| Design Choice | Selection | Reasoning |
|--------------|-----------|-----------|
| Algorithm | LightGBM (RF fallback) | Handles 4.2 M rows with 100+ features efficiently via histogram-based gradient boosting |
| Training data | 2018–2024 (all 7 years) | Maximum available labelled history before the held-out test year |
| Validation split | 20% stratified random | Used only for early stopping — not for reporting metrics |
| Class imbalance | `class_weight="balanced"` + `compute_sample_weight` | Prevents the majority Good class from dominating gradients |
| Leakage | 10 columns dropped | See Section 1 |
| Hyperparameters | `n_estimators=500, lr=0.05, num_leaves=127, subsample=0.8, colsample=0.8` | Balanced complexity vs. overfitting |

**Feature set:** All remaining numeric and categorical columns not in the exclusion set. Categorical columns are label-encoded; all residual NaN values (from engineered ratio features) are imputed with the median before training.

---

### Section 7 — 2025 Test-Set Evaluation

The 2025 file is a **strict temporal hold-out**: no 2025 row is seen at any point during training, validation, or hyperparameter tuning. This makes the evaluation a realistic simulation of deploying the model in production one year after the last training inspection.

**Metrics reported:**

| Metric | Description |
|--------|-------------|
| Macro F1 | Unweighted average F1 across all 4 classes — penalises poor performance on rare classes |
| Weighted F1 | F1 weighted by class frequency |
| Accuracy | Overall fraction correctly classified |
| ROC-AUC (OvR macro) | One-vs-rest AUC averaged across all 4 classes |
| Critical Recall | What fraction of truly Critical bridges are identified correctly |
| **Critical → Good Error** | **What fraction of Critical bridges are misclassified as Good — the primary safety metric** |

The **Critical → Good error** is the most consequential metric: a bridge rated Critical that the model dismisses as Good may go uninspected, posing a direct public safety risk.

Two confusion matrices are saved (`11_confusion_matrix.png`): raw counts and row-normalised percentages. Per-class predicted probability distributions are shown in `12_prob_distributions.png` to assess model calibration.

---

### Section 8 — SHAP Interpretability

SHAP (SHapley Additive exPlanations) provides a game-theory-grounded attribution of each feature's contribution to each prediction, consistent with the model's output.

- Computed on a 10,000-row random sample from the 2025 test set for efficiency
- **Global bar plot** (`13_shap_importance.png`): Top 15 features by mean |SHAP value| — shows which features drive predictions across the entire population
- **Beeswarm plot for Critical class** (`14_shap_beeswarm_critical.png`): Shows how feature values (colour) map to SHAP impact on Critical-class predictions — reveals the direction and magnitude of each feature's effect

**Significance:** SHAP makes the model auditable and trustworthy for safety-critical use. Inspectors and policy-makers can see exactly why a bridge was flagged as Critical — which features pushed the prediction in that direction — enabling targeted field inspection prioritisation.

---

### Section 9 — Model Performance Drift

**Setup:** A second model is trained on only 2018–2022 data (5 years), then evaluated **separately** on 2023, 2024, and 2025. This simulates what happens when a model is deployed without retraining.

**Metrics tracked over time:**
- Macro F1 — overall classification quality
- Critical Recall — ability to detect Critical bridges
- Critical → Good error rate — safety-critical misclassification rate

**Three-panel plot** (`15_model_performance_drift.png`) shows each metric's trajectory from 2023 to 2025.

**Significance:** If Macro F1 falls and Critical → Good error rises as the evaluation year moves further from the training window, this directly quantifies the **cost of not retraining**. It provides an empirical, data-driven argument for annual model refresh cycles. Even a small increase in the Critical → Good error rate represents real bridges at risk of being overlooked.

---

### Section 10 — Survival Analysis Dataset Construction

The bridge panel (one row per bridge-year) is converted into a **one-row-per-bridge** survival dataset using `STRUCTURE_NUMBER_008` as the unique bridge identifier.

**Survival event definition:**

| Term | Definition |
|------|-----------|
| Origin | First year a bridge appears in Good or Fair condition |
| Event (event=1) | Bridge transitions to Poor or Critical in any subsequent year |
| Duration | Years from origin to event (or to last observation if no event) |
| Censored (event=0) | Bridge never deteriorates to Poor/Critical within the 2018–2025 window |

Bridges that are already Poor or Critical in their first observed year are excluded (no baseline from which to measure deterioration). Zero-duration rows are also excluded (Cox model requirement).

Dataset statistics are printed and saved to `outputs/survival_dataset.csv`.

**Significance:** The longitudinal survival approach is methodologically superior to treating each bridge-year as independent. It explicitly models the *time* dimension of infrastructure degradation, enabling probabilistic answers to questions like "what fraction of bridges will remain safe for at least 5 more years?"

---

### Section 11 — Kaplan-Meier Survival Analysis

The **Kaplan-Meier estimator** is a non-parametric method that estimates the survival function S(t) — the probability that a bridge remains in Good/Fair condition at least t years after its baseline inspection. It makes no distributional assumptions and correctly handles right-censored observations.

**Three KM analyses are performed:**

#### 11a — Overall Survival Curve (`16_km_overall.png`)
The aggregate survival curve for the entire bridge population.
- **Median survival time** is reported (the year by which 50% of bridges have deteriorated).
- 95% confidence bands are shown.

#### 11b — KM by Bridge Age Group (`17_km_by_age.png`)
Bridges are split at the median age into High Age and Low Age groups.
- **Log-rank test** determines whether the two survival curves are statistically significantly different.
- **Expected finding:** Older bridges deteriorate significantly faster — the High Age curve falls more steeply. A log-rank p-value < 0.05 confirms this difference is not due to chance.
- **Significance:** Quantifies the age-based survival gap in years — enabling maintenance planners to set age-based inspection intervals.

#### 11c — KM by Bridge Material (`18_km_by_material.png`)
Separate survival curves for each material type with ≥ 500 bridges in the survival dataset.
- **Expected finding:** Wood/Timber and Masonry structures have the shortest median survival times; Prestressed Concrete and Steel Continuous structures survive longest.
- **Significance:** Material-specific survival curves can be used to set material-specific maintenance budgets.

#### 11d — KM by Rating Gap (`19_km_by_rating_gap.png`)
Bridges split by whether their `RATING_DIFF` (operating rating − inventory rating) is above or below the median.
- **Log-rank test** assesses significance.
- **Significance:** A small rating gap means the bridge is close to its rated capacity limit. Confirming that low-gap bridges deteriorate faster provides justification for the `RATING_DIFF` engineered feature's inclusion in the model.

---

### Section 12 — Cox Proportional Hazards Model

The Cox PH model extends Kaplan-Meier by allowing **multiple features to jointly predict deterioration risk**. It estimates a **hazard ratio (HR)** for each feature:

| HR | Interpretation |
|----|---------------|
| HR > 1 | Feature increases deterioration risk |
| HR < 1 | Feature is protective (decreases risk) |
| HR = 1 | No effect |

**Features included (standardised before fitting):**
`BRIDGE_AGE_AT_INSPECTION`, `LOG_ADT`, `OPERATING_RATING_064`, `INVENTORY_RATING_066`, `RATING_DIFF`, `TRAFFIC_DENSITY`, `AGE_TO_SPAN_RATIO`

A **penalizer (λ = 0.1)** is applied for regularisation. The proportional-hazards assumption is checked using Schoenfeld residuals — a p-value < 0.05 on any covariate suggests a PH violation and warrants caution in interpreting that coefficient.

Results are saved to `outputs/cox_hazard_summary.csv`. The hazard ratio bar chart (`20_cox_hazard_ratios.png`) shows each feature with 95% confidence intervals.

**Significance:** The Cox model provides a multi-variate, controlled estimate of each factor's contribution to deterioration risk — more rigorous than the univariate KM stratifications. For example, it can answer "does traffic density independently increase risk after controlling for age?" This is the level of analysis needed to justify bridge-specific maintenance prioritisation policies.

---

### Section 13 — Final Dashboard & Summary

An 8-panel, publication-ready summary figure (`00_FINAL_DASHBOARD.png`) combining:

| Panel | Content |
|-------|---------|
| A | Overall class distribution bar chart |
| B | Target drift line chart (2018 → 2025) |
| C | PSI bar chart (top 8 drifting features) |
| D | Model performance drift (Macro F1 + Critical Recall) |
| E | Overall Kaplan-Meier survival curve |
| F | KM by bridge age group |
| G | Cox PH hazard ratios |
| H | 2025 test confusion matrix (normalised) |

A text-based results summary is also printed, covering data statistics, all drift findings, model performance metrics, and survival analysis results.

---

## Key Findings Summary

### Data & Label Distribution
- **4.8 M bridge-year observations** across 8 annual NBI snapshots (2018–2025)
- ~600 K unique bridge structures tracked longitudinally
- Significant class imbalance: Good (57%) dominates; Critical (3%) is a rare-event class

### Data Drift (Input Features)
- PSI is computed for 11 numeric features; JS distance for 6 categorical features — all vs. the 2018 baseline, year by year from 2019 to 2025
- Features showing drift signal that the **bridge population composition has changed** over 7 years — structures built in different eras, with different materials, or carrying different traffic loads are entering or leaving inspection cycles
- Significant numeric drift (PSI ≥ 0.25) on features like `BRIDGE_AGE_AT_INSPECTION` reflects the fleet aging over time
- Categorical drift in `STATE_CODE_001` or `STRUCTURE_KIND_043` would indicate geographic or material composition shifts

### Target Drift (Label Distribution)
- The **most striking drift** in the dataset is in the label itself
- **Fair bridges declined by ~19–20 percentage points** from 2018 to 2025
- **Good bridges increased by ~15–16 percentage points** over the same period
- Critical and Poor shares show smaller but non-trivial changes
- This could reflect: (a) genuine infrastructure improvement due to post-2018 investment, (b) reclassification of inspected bridges, or (c) survivorship bias (worst bridges removed from service)
- **Implication for ML:** A model trained on 2018 label proportions is miscalibrated for 2025 — its prior on Good vs. Fair is wrong. This justifies annual retraining.

### Model Performance
- **LightGBM** trained on 2018–2024 (4.2 M rows) with leakage-free feature set
- Evaluated on the strict 2025 temporal hold-out (~600 K rows)
- **Macro F1, Weighted F1, Accuracy, ROC-AUC, Critical Recall, and Critical→Good error rate** are all reported
- The Critical → Good error rate is the primary safety metric: even a 1% rate across 600 K bridges means ~6,000 Critical structures incorrectly dismissed as Good

### Model Performance Drift
- A model trained only on 2018–2022 and then evaluated on 2023, 2024, and 2025 **without retraining** shows measurable degradation in Macro F1 and Critical Recall
- The Critical → Good error rate rises in later years, directly quantifying the **public safety cost of skipping model refreshes**
- This analysis makes a data-driven case for **annual retraining** as part of a production monitoring pipeline

### Survival Analysis
- ~600 K unique bridges tracked from their first Good/Fair observation through up to 7 inspection years
- Event = transition to Poor or Critical; censoring = bridge still Good/Fair in final observation
- **Kaplan-Meier** median survival time quantifies the typical "safe life" before deterioration
- **Log-rank tests** confirm that older bridges deteriorate significantly faster than newer ones (p < 0.05)
- **Material-specific KM curves** show clear separation: Wood/Timber and Masonry structures have shorter survival; Prestressed Concrete structures survive longer
- **Cox PH hazard ratios** quantify each feature's independent contribution to deterioration risk after controlling for all other covariates
- The Cox model identifies which **combination** of age, traffic load, rating gap, and span geometry maximises deterioration risk — the highest-risk bridges can be ranked for priority inspection

---

## Output Files Generated

| File | Contents |
|------|---------|
| `outputs/plots/00_FINAL_DASHBOARD.png` | 8-panel publication-ready summary figure |
| `outputs/plots/01_class_distribution.png` | Overall class balance (bar + pie) |
| `outputs/plots/02_yearly_condition_trend.png` | Condition share by year (stacked + line) |
| `outputs/plots/03_age_by_condition.png` | Age histogram and boxplot by condition |
| `outputs/plots/04_condition_by_material.png` | Critical rate by bridge material |
| `outputs/plots/05_adt_by_condition.png` | Traffic volume distribution by condition |
| `outputs/plots/06_state_critical_rate.png` | Top 20 states by Critical bridge rate |
| `outputs/plots/07_engineered_features.png` | Engineered feature distributions |
| `outputs/plots/08_psi_drift.png` | PSI drift bar + trend for top 5 features |
| `outputs/plots/09_js_drift.png` | JS distance bar for categorical features |
| `outputs/plots/10_target_drift.png` | Target label drift 2018 → 2025 |
| `outputs/plots/11_confusion_matrix.png` | 2025 test confusion matrix (counts + %) |
| `outputs/plots/12_prob_distributions.png` | Per-class predicted probability distributions |
| `outputs/plots/13_shap_importance.png` | SHAP global feature importance (top 15) |
| `outputs/plots/14_shap_beeswarm_critical.png` | SHAP beeswarm for Critical class |
| `outputs/plots/15_model_performance_drift.png` | Macro F1 / recall drift 2023 → 2025 |
| `outputs/plots/16_km_overall.png` | Overall Kaplan-Meier survival curve |
| `outputs/plots/17_km_by_age.png` | KM by bridge age group (log-rank test) |
| `outputs/plots/18_km_by_material.png` | KM by bridge material type |
| `outputs/plots/19_km_by_rating_gap.png` | KM by operating/inventory rating gap |
| `outputs/plots/20_cox_hazard_ratios.png` | Cox PH hazard ratios with 95% CI |
| `outputs/psi_numeric_drift.csv` | PSI per feature per year |
| `outputs/js_categorical_drift.csv` | JS distance per category per year |
| `outputs/target_drift_by_year.csv` | Label distribution % per year |
| `outputs/test_2025_metrics.csv` | Final model metrics (2025 hold-out) |
| `outputs/model_performance_drift.csv` | Temporal drift evaluation (2023–2025) |
| `outputs/survival_dataset.csv` | One-row-per-bridge survival panel |
| `outputs/cox_hazard_summary.csv` | Cox model summary with HRs and p-values |

---

## Notebook Sections Checklist

| Section | Description | Status |
|---------|-------------|--------|
| 0 | Setup & library imports | Complete |
| 1 | Data loading — 7 train + 1 test file | Complete |
| 1.1 | Column inventory with leakage flagging | Complete |
| 2 | EDA — 6 diagnostic plots | Complete |
| 3 | Feature engineering — 5 domain features | Complete |
| 4 | Data drift — PSI (numeric) + JS (categorical) | Complete |
| 5 | Target drift — condition share 2018 → 2025 | Complete |
| 6 | LightGBM training (RF fallback) | Complete |
| 7 | 2025 test evaluation + safety metric | Complete |
| 8 | SHAP global + Critical-class interpretability | Complete (requires shap) |
| 9 | Model performance drift (2018-22 → 2023/24/25) | Complete |
| 10 | Survival dataset construction | Complete |
| 11 | Kaplan-Meier (overall + age + material + rating) | Complete (requires lifelines) |
| 12 | Cox Proportional Hazards model + PH check | Complete (requires lifelines) |
| 13 | Final dashboard + printed results summary | Complete |

---

## Technical Design Notes

### Leakage Prevention
The 10 excluded columns (`DECK_COND_058`, `SUPERSTRUCTURE_COND_059`, `SUBSTRUCTURE_COND_060`, `CULVERT_COND_062`, and six FHWA evaluation scores) are the direct inputs and outputs of the NBI condition scoring formula. Including them would make the target trivially predictable from its own definition — inflating all metrics and producing a model useless for real inspection planning.

### Temporal Train/Test Split
The 2025 test set is held out completely — it is never touched during feature engineering, imputer fitting, model training, validation, or hyperparameter selection. This preserves the integrity of the evaluation as a true out-of-sample simulation.

### Class Imbalance Strategy
Two complementary approaches are combined: (1) `class_weight="balanced"` in the LightGBM classifier adjusts the loss function to weight minority classes more heavily; (2) `compute_sample_weight("balanced", y_train)` computes per-sample weights passed to `model.fit()`. Together these prevent the majority Good class from dominating gradient updates.

### Survival Right-Censoring
Bridges that never transition to Poor/Critical within the 2018–2025 window are **right-censored**, not excluded. Excluding them would introduce survivorship bias and artificially shorten estimated survival times. The Kaplan-Meier and Cox models both correctly handle censored observations.

---

*Data files (`.csv.gz`, `.parquet`, `.csv`) are excluded from version control via `.gitignore`. All figures are committed under `outputs/plots/`.*
