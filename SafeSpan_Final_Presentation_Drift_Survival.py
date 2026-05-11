# %% [markdown]
# # SafeSpan Bridge Analytics — Final Presentation Notebook
#
# **Predictive Maintenance for U.S. Bridge Infrastructure** (DATA 245)
#
# This notebook extends prior SafeSpan work with:
#
# 1. **Data drift** — numeric (PSI) and categorical (Jensen–Shannon) vs. a baseline year.
# 2. **Target drift** — how `TARGET_CONDITION` mix changes by year.
# 3. **Model performance drift** — temporal train/val/test and metrics by evaluation year.
# 4. **Survival analysis** — time-to-deterioration (Poor/Critical), Kaplan–Meier, and Cox PH.
#
# **Input:** `safe_span_cleaned.csv` (bridge-year panel, 2018–2025).
#
# **Optional installs:** `pip install lightgbm lifelines`
#
# Run in VS Code with the Jupyter extension, or convert to `.ipynb` with `jupytext` / manual import.

# %% [markdown]
# ## Section 1 — Setup and data loading

# %%
# If imports fail, install: pip install lightgbm lifelines pandas numpy matplotlib scipy scikit-learn

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import jensenshannon

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams["figure.dpi"] = 120
plt.rcParams["figure.figsize"] = (9, 4.5)

# LightGBM optional
try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except (ImportError, OSError):
    HAS_LIGHTGBM = False
    print(
        "Note: LightGBM unavailable (missing package or OpenMP/libomp). "
        "Section 4 will use RandomForestClassifier. pip install lightgbm"
    )

# lifelines required for survival sections
try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test

    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    print("ERROR: lifelines is required for Sections 6–7. pip install lifelines")

# %%
# Resolve data path: prefer safe_span_cleaned.csv; fall back to project CSV if present
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
CANDIDATE_PATHS = [
    SCRIPT_DIR / "safe_span_cleaned.csv",
    Path("safe_span_cleaned.csv"),
    SCRIPT_DIR / "nbi_cleaned.csv",
]

DATA_PATH = None
for p in CANDIDATE_PATHS:
    if p.exists():
        DATA_PATH = p
        break

if DATA_PATH is None:
    raise FileNotFoundError(
        "Could not find safe_span_cleaned.csv. Place it next to this script or set DATA_PATH manually.\n"
        f"Tried: {[str(p) for p in CANDIDATE_PATHS]}"
    )

print(f"Loading: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, low_memory=False)

# Optional subsample for faster iteration (set to None for full-data production runs)
MAX_ROWS = None  # e.g. 150_000
if MAX_ROWS is not None and len(df) > MAX_ROWS:
    df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
    print(f"Subsampled to MAX_ROWS={MAX_ROWS}")

print("Shape:", df.shape)
print("Columns (first 30):", list(df.columns[:30]), "...")

# %%
# --- Column detection (defensive) ---

YEAR_CANDIDATES = ["YEAR", "INSPECTION_YEAR", "INSPECTION_YR", "DATA_YEAR"]
BRIDGE_ID_CANDIDATES = ["STRUCTURE_NUMBER_008", "BRIDGE_ID", "STRUCTURE_NUMBER"]
TARGET_CANDIDATES = ["TARGET_CONDITION", "target_condition"]


def first_existing(cols, dataframe):
    for c in cols:
        if c in dataframe.columns:
            return c
    return None


year_col = first_existing(YEAR_CANDIDATES, df)
bridge_col = first_existing(BRIDGE_ID_CANDIDATES, df)
target_col = first_existing(TARGET_CANDIDATES, df)

SYNTH_BRIDGE_COL = "__SAFE_SPAN_SYNTH_BRIDGE_ID__"
# Drop rows without a usable target label (after target_col is known)
if target_col and target_col in df.columns:
    _t0 = len(df)
    df = df[df[target_col].notna()].copy()
    df = df[df[target_col].astype(str).str.strip().str.lower() != "nan"].copy()
    df = df[df[target_col].astype(str) != "Unknown"].copy()
    if len(df) < _t0:
        print(f"Dropped {_t0 - len(df):,} rows with missing/Unknown target.")

if bridge_col is None:
    KEY_PARTS = [
        "STATE_CODE_001",
        "COUNTY_CODE_003",
        "ROUTE_PREFIX_005B",
        "ROUTE_NUMBER_005D",
        "KILOPOINT_011",
        "DIRECTION_005E",
    ]
    parts = [c for c in KEY_PARTS if c in df.columns]
    if len(parts) >= 3:
        df[SYNTH_BRIDGE_COL] = df[parts].astype(str).agg("|".join, axis=1)
        bridge_col = SYNTH_BRIDGE_COL
        print("INFO: No bridge ID column found — using composite key from:", parts)
    else:
        print(
            "WARNING: Cannot build composite bridge key — survival sections need "
            "STRUCTURE_NUMBER_008 or route/location columns."
        )

if year_col is None:
    print("WARNING: No year column found among", YEAR_CANDIDATES, "— drift and temporal splits will be skipped.")
if bridge_col is None:
    print("WARNING: No bridge ID column found — survival analysis (Sections 5–7) cannot run.")
if target_col is None:
    print("WARNING: No TARGET_CONDITION column — target drift and supervised sections will be limited.")

if year_col:
    yrs = pd.to_numeric(df[year_col], errors="coerce").dropna()
    uy = sorted(yrs.unique().astype(int).tolist())
    print("Years available (numeric):", uy)
    if len(uy) < 2 and len(df) > 50_000:
        print(
            "NOTE: Only one calendar year in this dataframe. Many NBI extracts are sorted by YEAR, "
            "so loading the first N rows can hide later years. For cross-year drift and temporal "
            "evaluation, use `safe_span_cleaned.csv` with all years or e.g. "
            "`df = pd.read_csv(DATA_PATH, low_memory=False).sample(n=250_000, random_state=42)`."
        )
if target_col:
    print("\nTarget distribution:\n", df[target_col].value_counts(dropna=False))

# %%
# Ordinal label order for SafeSpan (worst → best)
CONDITION_ORDER = ["Critical", "Poor", "Fair", "Good"]
LABEL_TO_CODE = {lab: i for i, lab in enumerate(CONDITION_ORDER)}


def critical_to_good_error_rate(y_true, y_pred):
    """
    Safety-critical rate: fraction of truly Critical bridges predicted as Good.
    y_true / y_pred can be string labels or integer codes (0=Critical ... 3=Good).
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.size == 0:
        return np.nan
    if yt.dtype == object or isinstance(yt.flat[0], str):
        crit_mask = yt == "Critical"
        bad = crit_mask & (yp == "Good")
    else:
        crit_mask = yt == 0
        bad = crit_mask & (yp == 3)
    n_crit = int(crit_mask.sum())
    if n_crit == 0:
        return np.nan
    return float(bad.sum() / n_crit)


# %% [markdown]
# ## Section 2 — Data drift analysis (numeric PSI + categorical JS)
#
# We compare each year to the **earliest calendar year** in the file (typically 2018).
#
# **PSI interpretation**
# - PSI \< 0.10: no significant drift
# - 0.10 ≤ PSI \< 0.25: moderate drift
# - PSI ≥ 0.25: significant drift

# %%
NUMERIC_DRIFT_FEATURES = [
    "BRIDGE_AGE",
    "TRAFFIC_DENSITY",
    "TIME_SINCE_RECONSTRUCTION",
    "AGE_X_SCOUR",
    "AGE_TO_SPAN_RATIO",
    "DECK_TO_ROADWAY_RATIO",
    "ADT_GROWTH_RATIO",
    "INVENTORY_RATING_066",
    "OPERATING_RATING_064",
]

CAT_DRIFT_FEATURES = [
    "SCOUR_CRITICAL_113",
    "STRUCTURE_KIND_043A",
    "DESIGN_LOAD_031",
    "SERVICE_ON_042A",
    "STATE_CODE_001",
]


def psi_numeric(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index between baseline (expected) and comparison (actual).
    Returns np.nan if PSI cannot be computed safely.
    """
    exp = np.asarray(expected, dtype=float)
    act = np.asarray(actual, dtype=float)
    exp = exp[~np.isnan(exp)]
    act = act[~np.isnan(act)]
    if len(exp) < n_bins or len(act) < n_bins:
        return np.nan
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(exp, qs))
    if len(edges) < 3:
        return np.nan
    exp_counts, _ = np.histogram(exp, bins=edges)
    act_counts, _ = np.histogram(act, bins=edges)
    exp_pct = exp_counts / max(exp_counts.sum(), 1)
    act_pct = act_counts / max(act_counts.sum(), 1)
    eps = 1e-6
    exp_pct = np.clip(exp_pct, eps, 1.0)
    act_pct = np.clip(act_pct, eps, 1.0)
    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    if not np.isfinite(psi):
        return np.nan
    return psi


def psi_drift_level(psi: float) -> str:
    if psi is None or (isinstance(psi, float) and np.isnan(psi)):
        return "unknown"
    if psi < 0.10:
        return "no significant drift"
    if psi < 0.25:
        return "moderate drift"
    return "significant drift"


def js_drift_level(js: float) -> str:
    if js is None or (isinstance(js, float) and np.isnan(js)):
        return "unknown"
    if js < 0.05:
        return "low drift"
    if js <= 0.15:
        return "moderate drift"
    return "high drift"


def categorical_js_distance(a: pd.Series, b: pd.Series) -> float:
    """Jensen–Shannon distance (sqrt of JS divergence) between two categorical samples."""
    a = a.astype("string").fillna("__NA__")
    b = b.astype("string").fillna("__NA__")
    cats = sorted(set(a.unique()) | set(b.unique()))
    if len(cats) < 2 and len(set(cats)) < 2:
        return 0.0
    p = a.value_counts(normalize=True).reindex(cats, fill_value=0.0).values.astype(float)
    q = b.value_counts(normalize=True).reindex(cats, fill_value=0.0).values.astype(float)
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(jensenshannon(p, q, base=2))


psi_rows = []
if year_col:
    df["_year_int"] = pd.to_numeric(df[year_col], errors="coerce")
    years_sorted = sorted(df["_year_int"].dropna().unique().astype(int).tolist())
    baseline_year = years_sorted[0] if years_sorted else None
    num_feats_present = [c for c in NUMERIC_DRIFT_FEATURES if c in df.columns]
    missing_nf = [c for c in NUMERIC_DRIFT_FEATURES if c not in df.columns]
    if missing_nf:
        print("Numeric drift features not in data (skipped):", missing_nf)

    if baseline_year is not None and num_feats_present:
        base_mask = df["_year_int"] == baseline_year
        for y in years_sorted:
            if y == baseline_year:
                continue
            comp_mask = df["_year_int"] == y
            for feat in num_feats_present:
                psi_val = psi_numeric(df.loc[base_mask, feat].values, df.loc[comp_mask, feat].values)
                psi_rows.append(
                    {
                        "feature": feat,
                        "baseline_year": baseline_year,
                        "comparison_year": y,
                        "psi": psi_val,
                        "drift_level": psi_drift_level(psi_val),
                    }
                )
    df.drop(columns=["_year_int"], inplace=True, errors="ignore")
else:
    baseline_year = None
    num_feats_present = []

psi_df = pd.DataFrame(psi_rows)
if len(psi_df):
    psi_df.to_csv(SCRIPT_DIR / "safespan_data_drift_results.csv", index=False)
    print("Saved safespan_data_drift_results.csv")
    mean_psi = psi_df.groupby("feature")["psi"].mean().sort_values(ascending=False)
    print("\nTop drifting numeric features (mean PSI across years):\n", mean_psi.head(10))
else:
    print("No PSI results — check year column and numeric features.")

# %%
# PSI plots
if len(psi_df):
    top5 = psi_df.groupby("feature")["psi"].mean().nlargest(5).index.tolist()
    fig, ax = plt.subplots(figsize=(9, 5))
    for feat in top5:
        sub = psi_df[psi_df["feature"] == feat]
        ax.plot(sub["comparison_year"], sub["psi"], marker="o", label=feat)
    ax.axhline(0.10, color="orange", ls="--", label="PSI 0.10")
    ax.axhline(0.25, color="red", ls="--", label="PSI 0.25")
    ax.set_xlabel("Comparison year")
    ax.set_ylabel("PSI vs baseline")
    ax.set_title("PSI trend by year — top 5 drifting numeric features")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / "safespan_psi_trends_top5.png", bbox_inches="tight")
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    mean_psi_plot = psi_df.groupby("feature")["psi"].mean().sort_values(ascending=True)
    mean_psi_plot.plot(kind="barh", ax=ax2, color="steelblue")
    ax2.set_xlabel("Mean PSI across comparison years")
    ax2.set_title("Average PSI by feature")
    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / "safespan_psi_mean_by_feature.png", bbox_inches="tight")
    plt.show()

# %%
# Categorical drift (Jensen–Shannon)
cat_rows = []
if year_col:
    df["_year_int"] = pd.to_numeric(df[year_col], errors="coerce")
    years_sorted = sorted(df["_year_int"].dropna().unique().astype(int).tolist())
    baseline_year = years_sorted[0] if years_sorted else None
    cat_present = [c for c in CAT_DRIFT_FEATURES if c in df.columns]
    missing_cf = [c for c in CAT_DRIFT_FEATURES if c not in df.columns]
    if missing_cf:
        print("Categorical drift features not in data (skipped):", missing_cf)
    if baseline_year is not None and cat_present:
        base_mask = df["_year_int"] == baseline_year
        for y in years_sorted:
            if y == baseline_year:
                continue
            comp_mask = df["_year_int"] == y
            for feat in cat_present:
                js = categorical_js_distance(df.loc[base_mask, feat], df.loc[comp_mask, feat])
                cat_rows.append(
                    {
                        "feature": feat,
                        "baseline_year": baseline_year,
                        "comparison_year": y,
                        "js_distance": js,
                        "drift_level": js_drift_level(js),
                    }
                )
    df.drop(columns=["_year_int"], inplace=True, errors="ignore")

cat_df = pd.DataFrame(cat_rows)
if len(cat_df):
    cat_df.to_csv(SCRIPT_DIR / "safespan_categorical_drift_results.csv", index=False)
    print("Saved safespan_categorical_drift_results.csv")
    top_cat = cat_df.groupby("feature")["js_distance"].mean().sort_values(ascending=False)
    print("\nTop categorical drift (mean JS):\n", top_cat)
else:
    print("No categorical drift table — check year / categorical columns.")

# %%
if len(cat_df):
    top_cat_feats = cat_df.groupby("feature")["js_distance"].mean().nlargest(min(3, cat_df["feature"].nunique()))
    fig3, axes = plt.subplots(1, len(top_cat_feats), figsize=(4 * len(top_cat_feats), 4))
    if len(top_cat_feats) == 1:
        axes = [axes]
    for ax, (feat, _) in zip(axes, top_cat_feats.items()):
        sub = cat_df[cat_df["feature"] == feat]
        ax.bar(sub["comparison_year"].astype(str), sub["js_distance"], color="coral")
        ax.axhline(0.05, color="green", ls="--", alpha=0.7)
        ax.axhline(0.15, color="red", ls="--", alpha=0.7)
        ax.set_title(f"JS distance — {feat}")
        ax.set_ylabel("JS distance")
        ax.tick_params(axis="x", rotation=45)
    plt.suptitle("Categorical drift vs baseline year", y=1.02)
    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / "safespan_categorical_drift_top.png", bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## Section 3 — Target drift analysis
#
# We tabulate `TARGET_CONDITION` mix by year and visualize a stacked bar chart.

# %%
target_drift_df = pd.DataFrame()
if year_col and target_col:
    df["_year_int"] = pd.to_numeric(df[year_col], errors="coerce")
    t = (
        df.dropna(subset=["_year_int", target_col])
        .groupby(["_year_int", target_col])
        .size()
        .unstack(fill_value=0)
    )
    target_drift_df = t.div(t.sum(axis=1), axis=0) * 100
    target_drift_df = target_drift_df.reset_index().rename(columns={"_year_int": "year"})
    target_drift_df.to_csv(SCRIPT_DIR / "safespan_target_drift_by_year.csv", index=False)
    print("Saved safespan_target_drift_by_year.csv")
    print(target_drift_df.to_string(index=False))

    plot_cols = [c for c in CONDITION_ORDER if c in target_drift_df.columns]
    if plot_cols:
        fig, ax = plt.subplots(figsize=(9, 5))
        bottoms = None
        colors = {"Critical": "#c0392b", "Poor": "#e67e22", "Fair": "#3498db", "Good": "#27ae60"}
        x = target_drift_df["year"].values
        for lab in plot_cols:
            vals = target_drift_df[lab].values
            ax.bar(x.astype(str), vals, bottom=bottoms, label=lab, color=colors.get(lab, "#7f8c8d"))
            bottoms = vals if bottoms is None else bottoms + vals
        ax.set_ylabel("Percent of records (%)")
        ax.set_xlabel("Year")
        ax.set_title("TARGET_CONDITION distribution by year")
        ax.legend(title="Condition")
        plt.tight_layout()
        plt.savefig(SCRIPT_DIR / "safespan_target_drift_stacked.png", bbox_inches="tight")
        plt.show()
else:
    print("Skipping target drift — need year and target columns.")

# %% [markdown]
# ## Section 4 — Model performance drift (temporal evaluation)
#
# - **Preferred split:** train on years ≤ 2022, validate on 2023, evaluate on 2024–2025 when present.
# - **Fallback:** earliest 70% of distinct years → train, next 15% → validation, latest 15% → test.
# - **No SMOTE** on validation/test. Metrics computed **per evaluation year**.

# %%


def temporal_year_split(years_sorted):
    """Returns (train_years, val_years, test_years) as lists of int."""
    ys = sorted(set(years_sorted))
    if not ys:
        return [], [], []
    # Preferred: train <= 2022, val = 2023, test >= 2024 (when those years exist in data)
    if min(ys) <= 2022:
        train = [y for y in ys if y <= 2022]
        val = [y for y in ys if y == 2023]
        test = [y for y in ys if y >= 2024]
        if train and (val or test):
            if not val and test:
                val = [test[0]]
                test = test[1:]
            return train, val, test
        # Example: only 2018–2022 in file — no 2023/2024+ for eval; use ratio split below
    # Fallback: earliest 70% / next 15% / latest 15% of distinct years
    n = len(ys)
    i70 = max(1, int(np.floor(0.70 * n)))
    i85 = min(n, max(i70 + 1, int(np.floor(0.85 * n))))
    train = ys[:i70]
    val = ys[i70:i85]
    test = ys[i85:]
    if not val and len(train) >= 2:
        val = [train[-1]]
        train = train[:-1]
    if not test and val:
        test = [val[-1]]
        val = val[:-1]
    return train, val, test


def build_feature_matrix(df_in, feature_cols, numeric_cols, categorical_cols, numeric_medians=None):
    """Median imputation (optional fixed medians from train) + Unknown for categoricals + one-hot."""
    X = df_in[feature_cols].copy()
    for c in numeric_cols:
        if c not in X.columns:
            continue
        med = numeric_medians.get(c, X[c].median()) if numeric_medians is not None else X[c].median()
        X[c] = X[c].fillna(med)
    for c in categorical_cols:
        if c in X.columns:
            X[c] = X[c].astype("string").fillna("Unknown")
    return pd.get_dummies(X, columns=categorical_cols, dummy_na=False)


performance_rows = []

if year_col and target_col:
    work = df.copy()
    work["_y"] = pd.to_numeric(work[year_col], errors="coerce")
    work = work.dropna(subset=["_y", target_col])
    work[target_col] = work[target_col].astype(str)
    work = work[work[target_col].isin(CONDITION_ORDER)]

    years_u = sorted(work["_y"].unique().astype(int).tolist())
    train_years, val_years, test_years = temporal_year_split(years_u)
    print("Temporal split — train years:", train_years, "val:", val_years, "test/future:", test_years)

    exclude_from_X = set([target_col, year_col, "TARGET_NUM"] + [c for c in work.columns if c.startswith("Unnamed")])
    if bridge_col:
        exclude_from_X.add(bridge_col)
    candidate_features = [c for c in work.columns if c not in exclude_from_X and c != "_y"]
    numeric_cols = work[candidate_features].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = work[candidate_features].select_dtypes(exclude=[np.number]).columns.tolist()

    train_mask = work["_y"].isin(train_years)
    X_train_df = work.loc[train_mask, candidate_features]
    y_train = work.loc[train_mask, target_col].map(LABEL_TO_CODE).values

    numeric_medians = {c: X_train_df[c].median() for c in numeric_cols if c in X_train_df.columns}
    X_train = build_feature_matrix(
        X_train_df, candidate_features, numeric_cols, categorical_cols, numeric_medians=numeric_medians
    )

    eval_years = sorted(set(val_years + test_years))

    def align_X(X_raw):
        Xb = build_feature_matrix(
            X_raw, candidate_features, numeric_cols, categorical_cols, numeric_medians=numeric_medians
        )
        return Xb.reindex(columns=X_train.columns, fill_value=0)

    if HAS_LIGHTGBM and len(np.unique(y_train)) >= 2:
        clf = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=63,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    clf.fit(X_train, y_train)

    for ey in eval_years:
        m = work["_y"] == ey
        if m.sum() == 0:
            continue
        X_e = align_X(work.loc[m, candidate_features])
        y_e = work.loc[m, target_col].map(LABEL_TO_CODE).values
        pred = clf.predict(X_e)
        pred_labels = np.array([CONDITION_ORDER[i] for i in pred])
        true_labels = work.loc[m, target_col].values

        present_labels = sorted(set(y_e) | set(pred))
        crit_recall = np.nan
        if 0 in present_labels:
            per_class = recall_score(
                y_e, pred, labels=[0, 1, 2, 3], average=None, zero_division=0
            )
            crit_recall = float(per_class[0])

        performance_rows.append(
            {
                "eval_year": ey,
                "n_samples": int(m.sum()),
                "macro_f1": f1_score(y_e, pred, average="macro", zero_division=0),
                "weighted_f1": f1_score(y_e, pred, average="weighted", zero_division=0),
                "accuracy": accuracy_score(y_e, pred),
                "critical_recall": crit_recall,
                "critical_to_good_error_rate": critical_to_good_error_rate(true_labels, pred_labels),
            }
        )

    perf_df = pd.DataFrame(performance_rows)
    if len(perf_df):
        perf_df.to_csv(SCRIPT_DIR / "safespan_model_performance_drift.csv", index=False)
        print("Saved safespan_model_performance_drift.csv\n", perf_df.to_string(index=False))

        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(perf_df["eval_year"], perf_df["macro_f1"], "o-", label="Macro F1")
        ax1.set_xlabel("Evaluation year")
        ax1.set_ylabel("Macro F1")
        ax1.set_title("Model performance drift — Macro F1 by year")
        plt.tight_layout()
        plt.savefig(SCRIPT_DIR / "safespan_perf_macro_f1_by_year.png", bbox_inches="tight")
        plt.show()

        fig, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(perf_df["eval_year"], perf_df["critical_recall"], "s-", color="darkred", label="Critical recall")
        ax2.set_xlabel("Evaluation year")
        ax2.set_ylabel("Critical recall")
        ax2.set_title("Critical-class recall by evaluation year")
        plt.tight_layout()
        plt.savefig(SCRIPT_DIR / "safespan_perf_critical_recall_by_year.png", bbox_inches="tight")
        plt.show()

        fig, ax3 = plt.subplots(figsize=(8, 4))
        ax3.plot(
            perf_df["eval_year"],
            perf_df["critical_to_good_error_rate"],
            "^-",
            color="purple",
            label="Critical→Good error rate",
        )
        ax3.set_xlabel("Evaluation year")
        ax3.set_ylabel("Error rate")
        ax3.set_title("Safety-critical drift — Critical→Good error rate")
        plt.tight_layout()
        plt.savefig(SCRIPT_DIR / "safespan_perf_critical_to_good_by_year.png", bbox_inches="tight")
        plt.show()
    else:
        print("No performance rows — insufficient temporal split or data.")
else:
    print("Skipping model performance drift — requires both year and TARGET_CONDITION columns.")

# %% [markdown]
# **Interpretation (performance drift):** If macro F1 falls in later years while the **Critical → Good** error rate rises, the classifier trained on older inspections is becoming less reliable on recent data. That motivates **year-aware retraining**, **recalibration**, or **monitoring** in production.

# %% [markdown]
# ## Section 5 — Survival dataset (time to Poor/Critical)
#
# - **Entry:** first year observed in **Good** or **Fair**.
# - **Event:** first later year reaching **Poor** or **Critical**.
# - **Censoring:** never reaches Poor/Critical in available history.
# - **Duration:** integer years from baseline to event or last observation.

# %%
survival_df = pd.DataFrame()
dropped_nonpositive = 0

if HAS_LIFELINES and year_col and bridge_col and target_col:
    s = df[[bridge_col, year_col, target_col]].copy()
    s["_y"] = pd.to_numeric(s[year_col], errors="coerce")
    s = s.dropna(subset=["_y"])
    s[target_col] = s[target_col].astype(str)

    good_fair = {"Good", "Fair"}
    poor_crit = {"Poor", "Critical"}

    rows_out = []
    for bid, grp in s.groupby(bridge_col):
        grp = grp.sort_values("_y")
        years = grp["_y"].values
        conds = grp[target_col].values
        idx0 = None
        for i, c in enumerate(conds):
            if c in good_fair:
                idx0 = i
                break
        if idx0 is None:
            continue
        base_y = int(years[idx0])
        event = 0
        event_y = None
        for j in range(idx0 + 1, len(grp)):
            if conds[j] in poor_crit:
                event = 1
                event_y = int(years[j])
                break
        last_y = int(years[-1])
        if event:
            duration = event_y - base_y
        else:
            duration = last_y - base_y

        same_id = df[bridge_col].astype(str) == str(bid)
        yr_num = pd.to_numeric(df[year_col], errors="coerce")
        feat_row = df[same_id & (yr_num == base_y)]
        if len(feat_row) == 0:
            feat_row = df[same_id].sort_values(year_col).iloc[[0]]
        base_features = feat_row.iloc[0].to_dict()
        base_features["bridge_id"] = bid
        base_features["baseline_year"] = base_y
        base_features["duration"] = duration
        base_features["event"] = event
        rows_out.append(base_features)

    survival_df = pd.DataFrame(rows_out)
    if len(survival_df):
        dropped_nonpositive = int((survival_df["duration"] <= 0).sum())
        survival_df = survival_df[survival_df["duration"] > 0].copy()
        if len(survival_df) == 0:
            print(
                "WARNING: All candidate survival rows had duration <= 0. "
                "This usually means each bridge appears for only one calendar year "
                "(no gap between baseline Good/Fair and a later Poor/Critical). "
                "Use a multi-year bridge-year panel for survival analysis."
            )
        else:
            survival_df.to_csv(SCRIPT_DIR / "safespan_survival_dataset.csv", index=False)
            print("Saved safespan_survival_dataset.csv")
            print("Bridges (rows):", len(survival_df))
            print("Events:", int(survival_df["event"].sum()))
            print("Censored:", int((survival_df["event"] == 0).sum()))
            print("Median duration (years):", float(survival_df["duration"].median()))
            print("Event rate:", float(survival_df["event"].mean()))
            print("Rows dropped (duration <= 0):", dropped_nonpositive)
else:
    print("Skipping survival dataset — need lifelines, year, bridge ID, and target.")

# %% [markdown]
# ## Section 6 — Kaplan–Meier survival curves

# %%
if HAS_LIFELINES and len(survival_df):
    kmf = KaplanMeierFitter()
    kmf.fit(survival_df["duration"], survival_df["event"], label="All bridges")
    ax = kmf.plot_survival_function()
    ax.set_xlabel("Years since baseline (Good/Fair)")
    ax.set_ylabel("Probability of remaining Good/Fair (no Poor/Critical yet)")
    ax.set_title("Kaplan–Meier Survival Curve: Time Until Poor/Critical Condition")
    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / "safespan_kaplan_meier_overall.png", bbox_inches="tight")
    plt.show()

    def km_group_plot(sdf, mask_high, mask_low, label_high, label_low, title, fname):
        plt.figure(figsize=(8, 4.5))
        km1, km2 = KaplanMeierFitter(), KaplanMeierFitter()
        km1.fit(sdf.loc[mask_high, "duration"], sdf.loc[mask_high, "event"], label=label_high)
        km2.fit(sdf.loc[mask_low, "duration"], sdf.loc[mask_low, "event"], label=label_low)
        km1.plot_survival_function()
        km2.plot_survival_function()
        plt.xlabel("Years since baseline")
        plt.ylabel("Survival probability")
        plt.title(title)
        res = logrank_test(
            sdf.loc[mask_high, "duration"],
            sdf.loc[mask_low, "duration"],
            sdf.loc[mask_high, "event"],
            sdf.loc[mask_low, "event"],
        )
        print(f"Log-rank {title}: p = {res.p_value:.4g}")
        plt.tight_layout()
        plt.savefig(SCRIPT_DIR / fname, bbox_inches="tight")
        plt.show()

    # 1) Bridge age split
    if "BRIDGE_AGE" in survival_df.columns:
        med_age = survival_df["BRIDGE_AGE"].median()
        hi = survival_df["BRIDGE_AGE"] >= med_age
        lo = survival_df["BRIDGE_AGE"] < med_age
        if hi.sum() > 5 and lo.sum() > 5:
            km_group_plot(
                survival_df,
                hi,
                lo,
                f"High age (≥{med_age:.0f})",
                f"Low age (<{med_age:.0f})",
                "KM by bridge age (baseline)",
                "safespan_km_by_age.png",
            )

    # 2) Scour risk: numeric SCOUR_CRITICAL_113 — high risk if <= 3 (NBI convention)
    scour_col = "SCOUR_CRITICAL_113"
    if scour_col in survival_df.columns and pd.api.types.is_numeric_dtype(survival_df[scour_col]):
        hi_scour = survival_df[scour_col] <= 3
        lo_scour = survival_df[scour_col] > 3
        if hi_scour.sum() > 5 and lo_scour.sum() > 5:
            km_group_plot(
                survival_df,
                hi_scour,
                lo_scour,
                "High scour risk (≤3)",
                "Low scour risk (>3)",
                "KM by scour criticality",
                "safespan_km_by_scour.png",
            )

    # 3) AGE_X_SCOUR median split
    if "AGE_X_SCOUR" in survival_df.columns and survival_df["AGE_X_SCOUR"].notna().sum() > 10:
        med_axs = survival_df["AGE_X_SCOUR"].median()
        hi2 = survival_df["AGE_X_SCOUR"] >= med_axs
        lo2 = survival_df["AGE_X_SCOUR"] < med_axs
        if hi2.sum() > 5 and lo2.sum() > 5:
            km_group_plot(
                survival_df,
                hi2,
                lo2,
                f"High AGE×SCOUR (≥{med_axs:.2g})",
                f"Low AGE×SCOUR (<{med_axs:.2g})",
                "KM by AGE×SCOUR",
                "safespan_km_by_age_x_scour.png",
            )
else:
    if not HAS_LIFELINES:
        print("Skipping Kaplan–Meier — install lifelines (pip install lifelines).")
    else:
        print("Skipping Kaplan–Meier — empty survival dataset (see Section 5 warnings).")

# %% [markdown]
# ## Section 7 — Cox proportional hazards
#
# Hazard ratios \> 1 imply **higher** hazard of transitioning to Poor/Critical (faster deterioration in this discrete-time approximation).

# %%
cox_summary_path = SCRIPT_DIR / "safespan_cox_hazard_summary.csv"

if HAS_LIFELINES and len(survival_df):
    COX_FEATURES = [
        "BRIDGE_AGE",
        "TRAFFIC_DENSITY",
        "TIME_SINCE_RECONSTRUCTION",
        "AGE_X_SCOUR",
        "INVENTORY_RATING_066",
        "OPERATING_RATING_064",
        "SCOUR_CRITICAL_113",
    ]
    cox_feats = [c for c in COX_FEATURES if c in survival_df.columns]
    cox_feats = [c for c in cox_feats if pd.api.types.is_numeric_dtype(survival_df[c])]
    if not cox_feats:
        print("No numeric Cox features available.")
    else:
        cox_df = survival_df[["duration", "event"] + cox_feats].copy()
        for c in cox_feats:
            cox_df[c] = cox_df[c].fillna(cox_df[c].median())
        # Standardize for stability
        for c in cox_feats:
            std = cox_df[c].std()
            if std and std > 0:
                cox_df[c] = (cox_df[c] - cox_df[c].mean()) / std

        cph = CoxPHFitter(penalizer=0.01)
        try:
            cph.fit(cox_df, duration_col="duration", event_col="event")
            cph.print_summary()
            summ = cph.summary
            summ.to_csv(cox_summary_path)
            print("Saved", cox_summary_path)

            hr = np.exp(cph.params_)
            fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(hr))))
            y_pos = np.arange(len(hr))
            ax.barh(y_pos, hr.values, color="teal", alpha=0.85)
            ax.axvline(1.0, color="black", lw=1)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(hr.index)
            ax.set_xlabel("Hazard ratio (exp(coef))")
            ax.set_title("Cox model — hazard ratios (>1 = higher deterioration risk)")
            plt.tight_layout()
            plt.savefig(SCRIPT_DIR / "safespan_cox_hazard_ratios.png", bbox_inches="tight")
            plt.show()

            print("\n--- Proportional hazards check (may warn with small / discrete data) ---")
            try:
                cph.check_assumptions(cox_df, p_value_threshold=0.05, show_plots=False)
            except Exception as e:
                print("check_assumptions failed or skipped:", e)
        except Exception as e:
            print("Cox fit failed:", e)
else:
    if not HAS_LIFELINES:
        print("Skipping Cox PH — install lifelines (pip install lifelines).")
    else:
        print("Skipping Cox PH — empty survival dataset.")

# %% [markdown]
# **Caution:** The Cox model assumes **proportional hazards**. With **annual** NBI snapshots (2018–2025), durations are coarse and censoring is heavy — use survival results as a **complement** to the classifier, not a replacement for engineering inspection.

# %% [markdown]
# ## Section 8 — Integration with SafeSpan (summary)
#
# | Module | Question it answers |
# |--------|---------------------|
# | **Numeric PSI drift** | Do key engineering and traffic features shift relative to a baseline year? |
# | **Categorical JS drift** | Do scour, structure type, load rating, or state mix change over time? |
# | **Target drift** | Is the national (or sample) mix of Critical/Poor/Fair/Good changing year to year? |
# | **Performance drift** | Does a model trained on earlier years keep macro F1 and safe Critical recall on later years? |
# | **Survival (KM + Cox)** | Instead of “what condition is this year?”, “how long until Poor/Critical from a Good/Fair start?” |
#
# Together, these pieces extend SafeSpan from a **static multi-class risk score** toward a **temporal maintenance and monitoring** narrative suitable for prioritization and governance discussion.

# %%
print("\nDone. Outputs written to:", SCRIPT_DIR)
print(
    "Artifacts: safespan_data_drift_results.csv, safespan_categorical_drift_results.csv, "
    "safespan_target_drift_by_year.csv, safespan_model_performance_drift.csv, "
    "safespan_survival_dataset.csv, safespan_cox_hazard_summary.csv, and PNG plots."
)
