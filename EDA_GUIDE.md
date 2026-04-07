# EDA & Data Cleaning Guide — NBI Bridge Deterioration Project

**Dataset**: `nbi_5million.csv` — **5,000,000 rows × 139 columns** (2.16 GB)  
**Years covered**: 2018–2025 (real FHWA data) + tiny synthetic tail  
**Target**: `TARGET_CONDITION` → `Critical (0-3)`, `Poor (4)`, `Fair (5-6)`, `Good (7-9)`

---

## 1. Data Cleaning Checklist

### 1a. Drop Useless Columns (18 columns to remove immediately)
These are either 100% missing, free-text identifiers, or date strings that add zero predictive value:

| Column | Reason |
|--------|--------|
| `CRITICAL_FACILITY_006B` | 100% null |
| `REMARKS` | 100% null, free text |
| `STEP_CODE`, `SPECIAL_CODE` | 100% null |
| `OTHER_STATE_CODE_098A` | 99.9% null |
| `TEMP_STRUCTURE_103` | 99.6% null |
| `FRACTURE_LAST_DATE_093A` | 96% null |
| `UNDWATER_LAST_DATE_093B` | 97% null |
| `SPEC_LAST_DATE_093C` | 93% null |
| `PIER_PROTECTION_111` | 97% null |
| `DTL_TYPE_OF_IMP`, `PROGRAM_CODE`, `PROJ_NO`, `NBI_TYPE_OF_IMP`, `PROJ_SUFFIX` | 93% null |
| `STRUCTURE_NUMBER_008` | Unique bridge ID (identifier, not feature) |
| `FEATURES_DESC_006A`, `FACILITY_CARRIED_007`, `LOCATION_009` | Free-text descriptions |
| `DATE_LAST_UPDATE`, `DATE_OF_INSPECT_090` | Date strings (extract year if needed) |
| `FED_AGENCY`, `LRS_INV_ROUTE_013A` | Sparse identifiers |

### 1b. ⚠️ CRITICAL: Remove Data Leakage Columns
These columns are **derived from the target** or directly encode the condition rating. Using them would be cheating:

| Column | Why It Leaks |
|--------|-------------|
| `DECK_COND_058` | **This IS the target** (raw rating 0-9) |
| `SUPERSTRUCTURE_COND_059` | **This IS the target** |
| `SUBSTRUCTURE_COND_060` | **This IS the target** |
| `CHANNEL_COND_061` | Directly used to compute target |
| `CULVERT_COND_062` | Directly used to compute target |
| `STRUCTURAL_EVAL_067` | Derived from condition ratings |
| `SUFFICIENCY_RATING` | Composite score using condition ratings |
| `STATUS_WITH_10YR_RULE` | Based on sufficiency rating |
| `STATUS_NO_10YR_RULE` | Based on sufficiency rating |
| `CAT10`, `CAT23`, `CAT29` | Derived categorizations |

### 1c. Handle Mixed Types
Several columns look numeric but contain the character `'N'` (meaning "Not Applicable"):
- `SCOUR_CRITICAL_113` → values: `[0-9, N, U, T]` — encode `N`→NaN, `U`→NaN, `T`→NaN, then cast to int
- Condition rating columns (above, already dropped for leakage)

### 1d. Missing Value Strategy
Per your proposal: **Median imputation** for numerics, **Mode imputation** for categoricals, plus add **missing indicator flags** (`_MISSING` binary columns) so the model can learn that missingness itself is informative.

### 1e. Drop Rows Where Target = `'Unknown'`
The synthetic tail records and any bridges without valid condition ratings will have `TARGET_CONDITION = 'Unknown'`. Drop these.

---

## 2. Key Features to Highlight in EDA

### 🏗️ Structural / Physical Features
| Feature | What It Is | EDA Action |
|---------|-----------|------------|
| `YEAR_BUILT_027` | Year bridge was constructed | **Top predictor.** Histogram + box plot by target class. Older = worse condition. |
| `YEAR_RECONSTRUCTED_106` | Year of last major rehab | Many are 0 (never reconstructed). Create binary: `WAS_RECONSTRUCTED` |
| `STRUCTURE_KIND_043A` | Material type (concrete, steel, wood, etc.) | Bar chart by target. Steel vs concrete deterioration rates differ. |
| `STRUCTURE_TYPE_043B` | Design type (slab, stringer, truss, etc.) | Bar chart by target class |
| `MAX_SPAN_LEN_MT_048` | Longest span in meters | Distribution plot, check for outliers |
| `STRUCTURE_LEN_MT_049` | Total bridge length | Correlation with condition |
| `DECK_WIDTH_MT_052` | Deck width | Narrow decks may correlate with older, poorer bridges |
| `DEGREES_SKEW_034` | Angle of skew | Skewed bridges experience more stress |
| `MAIN_UNIT_SPANS_045` | Number of main spans | Multi-span bridges may degrade differently |

### 🚗 Traffic Features
| Feature | What It Is | EDA Action |
|---------|-----------|------------|
| `ADT_029` | Average Daily Traffic | **High correlation.** Log-transform (heavily right-skewed). Scatter vs condition. |
| `FUTURE_ADT_114` | Projected future traffic | Compare with current ADT to see growth |
| `PERCENT_ADT_TRUCK_109` | % of traffic that's trucks | Heavy trucks = faster deterioration |
| `TRAFFIC_LANES_ON_028A` | Number of traffic lanes | Bar chart by target |

### 🌊 Environmental / Risk Features
| Feature | What It Is | EDA Action |
|---------|-----------|------------|
| `SCOUR_CRITICAL_113` | Scour vulnerability rating | **Critical feature.** Bridges near water are at risk. Bar chart by target. |
| `WATERWAY_EVAL_071` | Waterway adequacy | Related to flood/scour risk |
| `CHANNEL_COND_061` | Channel condition (if not dropped for leakage) | Only use historical lag if available |

### 📍 Geographic Features
| Feature | What It Is | EDA Action |
|---------|-----------|------------|
| `STATE_CODE_001` | State (1-56) | **Heatmap of condition by state.** States have very different maintenance budgets. |
| `COUNTY_CODE_003` | County code | Too granular alone; combine with state |
| `LAT_016`, `LONG_017` | Latitude/Longitude | Scatter map colored by condition (geography matters for climate) |
| `FUNCTIONAL_CLASS_026` | Road functional class (interstate, arterial, local) | Interstate bridges get more maintenance |

### 🔧 Maintenance / Operational Features
| Feature | What It Is | EDA Action |
|---------|-----------|------------|
| `OWNER_022` | Bridge owner (state, county, city, federal) | Different owners = different budgets |
| `MAINTENANCE_021` | Who maintains it | Same as above |
| `DESIGN_LOAD_031` | Original design load capacity | Older design standards = weaker bridges |
| `OPERATING_RATING_064` | Current load rating (tons) | Distribution by target class |
| `INVENTORY_RATING_066` | Inventory load rating | Same |
| `TOLL_020` | Toll bridge? | Toll bridges get more funding |
| `OPEN_CLOSED_POSTED_041` | Open/closed/posted status | May indicate known issues |

---

## 3. Feature Engineering (Must-Do)

These are the derived features from your proposal that you MUST create:

```python
df['BRIDGE_AGE'] = df['YEAR'] - df['YEAR_BUILT_027']
df['AGE_TO_SPAN_RATIO'] = df['BRIDGE_AGE'] / df['MAX_SPAN_LEN_MT_048']
df['TRAFFIC_DENSITY'] = df['ADT_029'] / df['STRUCTURE_LEN_MT_049']
df['TIME_SINCE_RECONSTRUCTION'] = df['YEAR'] - df['YEAR_RECONSTRUCTED_106']
df['DECK_TO_ROADWAY_RATIO'] = df['DECK_WIDTH_MT_052'] / df['ROADWAY_WIDTH_MT_051']
df['ADT_GROWTH_RATIO'] = df['FUTURE_ADT_114'] / df['ADT_029']
df['WAS_RECONSTRUCTED'] = (df['YEAR_RECONSTRUCTED_106'] > 0).astype(int)
```

After encoding `SCOUR_CRITICAL_113` to numeric:
```python
df['AGE_X_SCOUR'] = df['BRIDGE_AGE'] * df['SCOUR_CRITICAL_113']
```

---

## 4. Class Imbalance Analysis

From our sample, the approximate distribution is:

| Class | % of Data | Count (approx 5M) |
|-------|-----------|-------------------|
| Fair | ~73% | ~3,650,000 |
| Good | ~19% | ~950,000 |
| Poor | ~6% | ~300,000 |
| Critical | ~2% | ~100,000 |

**Actions**:
- Show this as a bar chart in your presentation (Slide 2)
- Plan for SMOTE / SMOTETomek during modeling
- Use class weights in tree-based models
- Evaluate with **Macro F1** and **per-class recall**, NOT accuracy

---

## 5. EDA Visualizations Checklist

| # | Plot | Purpose |
|---|------|---------|
| 1 | Target class distribution bar chart | Show the severe imbalance |
| 2 | Bridge age histogram colored by target | Older bridges = worse condition |
| 3 | Correlation heatmap (top 20 numeric features) | Identify multicollinearity |
| 4 | ADT (traffic) box plot by target class | Traffic impact on deterioration |
| 5 | State-level heatmap of % Critical bridges | Geographic patterns |
| 6 | Structure material type vs condition (stacked bar) | Material matters |
| 7 | Scour rating vs condition (stacked bar) | Environmental risk factor |
| 8 | Year-over-year trend of condition classes | Temporal degradation patterns |
| 9 | Missing value heatmap | Show data quality across columns |
| 10 | Pair plot of top 5 engineered features | Visualize separability |

---

## 6. Correlation Filtering

From our analysis, the top correlated features with target (after removing leakage):

| Feature | Correlation |
|---------|-------------|
| `YEAR_BUILT_027` (Bridge Age proxy) | 0.36 |
| `POSTING_EVAL_070` | 0.33 |
| `OPERATING_RATING_064` | 0.30 |
| `INSPECT_FREQ_MONTHS_091` | 0.25 |
| `INVENTORY_RATING_066` | 0.24 |
| `APPR_ROAD_EVAL_072` | 0.24 |
| `STRUCTURE_TYPE_043B` | 0.18 |
| `TRAFFIC_DIRECTION_102` | 0.14 |
| `APPR_WIDTH_MT_032` | 0.13 |

Also check inter-feature correlations: drop one of any pair with |r| > 0.90 (e.g., `OPERATING_RATING_064` and `INVENTORY_RATING_066` will likely be highly correlated).

---

## 7. Quick Reference — Work Division

| Team Member | EDA Responsibility |
|------------|-------------------|
| **Pramod** | Data cleaning, missing values, leakage detection, all visualizations |
| **Akash** | Feature engineering, correlation analysis, tree-based model prep |
| **Shruthi** | Encoding strategies (one-hot for categorical), scaling (Z-score), baseline prep |
| **Shriram** | Class imbalance analysis, evaluation framework setup, interpretability prep |
