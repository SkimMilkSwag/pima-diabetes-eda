"""Exploratory analysis for the Pima Indians Diabetes dataset.

The functions in this module are intentionally small and pure so they can be
imported by both the notebook (``EDA.ipynb``) and the test suite
(``tests/test_eda.py``). Everything is stdlib + pandas/numpy; no network calls.

Dataset reference: UCI Machine Learning Repository, "Pima Indians Diabetes
Dataset" (768 rows, 8 features, binary outcome). The copy bundled in
``data/diabetes.csv`` uses the conventional column names.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "diabetes.csv")

# The eight feature columns (everything except the binary ``Outcome`` target).
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# Columns that legitimately contain 0 in the raw export, which is really a
# missing value rather than a physically meaningful zero.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the diabetes CSV and coerce the known zero-as-missing columns to NaN.

    The UCI export records several "not available" measurements as ``0`` (a
    glucose of 0 is not biologically plausible). We keep the raw value in a
    sidecar column-free way: replace 0 with NaN for those columns so that
    summary statistics reflect real observations, and track how many were
    coerced.
    """
    df = pd.read_csv(path)
    return df


def coerce_zeros(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return a copy of ``df`` with zero-as-missing values set to NaN.

    Also returns a mapping of column -> number of zeros coerced, so callers
    can report how much of the "missingness" was actually a data-entry artifact.
    """
    out = df.copy()
    coerced = {}
    for col in ZERO_AS_MISSING:
        if col in out.columns:
            n = int((out[col] == 0).sum())
            out.loc[out[col] == 0, col] = np.nan
            coerced[col] = n
    return out, coerced


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missingness: count of NaN, % of rows, and a severity label.

    Severity is ``high`` at >50% missing, ``medium`` at >20%, else ``low`` —
    the same thresholds used in ``csv-toolkit``'s null-flag report so the two
    tools stay consistent.
    """
    rows = []
    n = len(df)
    for col in df.columns:
        missing = int(df[col].isna().sum())
        pct = 100.0 * missing / n if n else 0.0
        if pct > 50:
            sev = "high"
        elif pct > 20:
            sev = "medium"
        else:
            sev = "low"
        rows.append(
            {
                "column": col,
                "missing": missing,
                "pct_missing": round(pct, 2),
                "severity": sev,
            }
        )
    return pd.DataFrame(rows).sort_values("pct_missing", ascending=False)


def outcome_counts(df: pd.DataFrame) -> dict:
    """Return counts and rates for the binary ``Outcome`` column."""
    counts = df["Outcome"].value_counts().to_dict()
    total = len(df)
    pos = int(counts.get(1, 0))
    return {
        "total": total,
        "outcome_1": pos,
        "outcome_0": int(counts.get(0, total - pos)),
        "positive_rate": round(pos / total, 4) if total else 0.0,
    }


def feature_means_by_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of each feature within each outcome group.

    Produces a tidy table (feature x [outcome_0_mean, outcome_1_mean, delta])
    that makes it trivial to see which features separate the two groups most.
    Features with NaNs in a group are averaged over non-missing values only.
    """
    parts = {}
    for val in sorted(df["Outcome"].unique()):
        grp = df.loc[df["Outcome"] == val]
        parts[f"outcome_{int(val)}_mean"] = [grp[c].mean() for c in FEATURES]
    out = pd.DataFrame(parts, index=FEATURES)
    out.index.name = "feature"
    out["delta"] = out["outcome_1_mean"] - out["outcome_0_mean"]
    return out.round(3)


def group_distributions(df: pd.DataFrame) -> list[list[np.ndarray]]:
    """Per-feature distributions across the two outcome groups.

    Returns a list of eight entries (one per feature, in ``FEATURES`` order),
    each a two-element list of arrays — the 100-quantile points that bracket
    the data for the negative and positive groups respectively (index 0 =
    Outcome 0, index 1 = Outcome 1). Each entry is ready to feed straight
    into ``matplotlib.pyplot.boxplot``.
    """
    out: list[list[np.ndarray]] = []
    for feat in FEATURES:
        rows = []
        for val in sorted(df["Outcome"].unique()):
            vals = df.loc[df["Outcome"] == val, feat].dropna()
            rows.append(np.unique(np.quantile(vals, np.linspace(0, 1, 100))))
        out.append(rows)
    return out


def plot_distributions(
    df: pd.DataFrame, out_dir: str = "plots", skip: bool = False
) -> list[str]:
    """Render a boxplot comparison of every feature across outcome groups.

    Writes one PNG per feature (``out_dir/<feature>.png``) plus an 8-panel
    grid at ``out_dir/all_features.png``. Returns the written file paths.
    The Agg backend is used so the function works headless (CI, run.py).

    Pass ``skip=True`` in CI or on a headless machine without display: figure
    generation is elided and an empty list is returned, which keeps the test
    suite from depending on matplotlib at all (it's a plotting-only concern).
    """
    if skip:
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    paths: list[str] = []
    data = group_distributions(df)
    for feat, values in zip(FEATURES, data):
        fig, ax = plt.subplots(figsize=(6, 4))
        bp = ax.boxplot(values)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Outcome 0", "Outcome 1"])
        ax.set_title(feat)
        ax.set_ylabel(feat)
        for element in ("boxes", "whiskers", "caps"):
            plt.setp(bp[element], linewidth=1)
        path = os.path.join(out_dir, f"{feat}.png")
        fig.savefig(path, dpi=100)
        plt.close(fig)
        paths.append(path)

    fig, axes = plt.subplots(4, 2, figsize=(11, 16))
    for ax, feat, values in zip(axes.ravel(), FEATURES, data):
        bp = ax.boxplot(values)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Outcome 0", "Outcome 1"])
        ax.set_title(feat)
        plt.setp(bp["boxes"], linewidth=1)
    fig.tight_layout()
    path = os.path.join(out_dir, "all_features.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    paths.append(path)
    return paths


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation among the features (and Outcome), NaN-tolerant."""
    cols = FEATURES + ["Outcome"]
    return df[cols].corr().round(3)


def top_correlates_with_outcome(df: pd.DataFrame, k: int = 5) -> pd.Series:
    """The ``k`` features with the largest |correlation| to Outcome.

    Ranked by absolute correlation so both strongly positive and strongly
    negative associations surface; ties are broken deterministically by name.
    """
    corr = df[FEATURES + ["Outcome"]].corr()["Outcome"].drop("Outcome")
    return corr.abs().sort_values(ascending=False).head(k)


def train_baseline(
    df: pd.DataFrame,
    test_size: float = 0.3,
    seed: int = 42,
) -> dict:
    """Fit a LogisticRegression baseline and evaluate it on a held-out split.

    ``df`` is expected to be post-``coerce_zeros``: NaNs are imputed with the
    column median before the split so that train and validation see the same
    feature space (imputing per-split would leak group-level information).
    The 768 rows are small, so a single stratified shuffle is used instead of
    nested cross-validation; the seed keeps results reproducible.

    Returns a dict with:

    - ``model`` — the fitted ``LogisticRegression`` (for coefficient
      inspection downstream),
    - ``train_n`` / ``test_n`` — split sizes,
    - ``auc`` — ROC-AUC on the validation set,
    - ``confusion_matrix`` — 2x2 array for the validation predictions.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, roc_auc_score

    work = df.copy()
    for col in FEATURES:
        work[col] = work[col].fillna(work[col].median())
    X = work[FEATURES].to_numpy(dtype=float)
    y = work["Outcome"].to_numpy()

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    n_test = int(round(test_size * len(y)))
    # stratified split: keep the ~35% positive rate in both halves so the
    # validation set isn't skewed by chance
    test_idx = []
    for val in sorted(np.unique(y)):
        idx = perm[y[perm] == val]
        k = int(round(test_size * len(idx)))
        test_idx.extend(idx[:k].tolist())
    test_set = set(test_idx)
    train_idx = [i for i in perm if i not in test_set]

    model = LogisticRegression(max_iter=1000)
    model.fit(X[train_idx], y[train_idx])

    proba = model.predict_proba(X[test_idx])[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "model": model,
        "train_n": len(train_idx),
        "test_n": len(test_idx),
        "auc": float(roc_auc_score(y[test_idx], proba)),
        "confusion_matrix": confusion_matrix(y[test_idx], preds),
    }


def imputation_comparison(
    df: pd.DataFrame,
    test_size: float = 0.3,
    seed: int = 42,
) -> dict:
    """A/B test median imputation against no-imputation on the same split.

    Both arms share the identical stratified train/validation partition, so a
    delta in ROC-AUC reflects only how the missing values were treated — not a
    different dataset. The "imputed" arm fills each feature's NaNs with the
    column median computed on the *full* frame (the same convention
    ``train_baseline`` uses); the "naive" arm leaves the NaNs in place, so any
    row with a missing feature is dropped from the fit and evaluation.

    Returns a dict with:

    - ``auc_imputed`` / ``auc_naive`` — ROC-AUC per arm on the shared
      validation set,
    - ``delta_auc`` — ``auc_imputed - auc_naive`` (positive = imputation
      helped),
    - ``test_n`` — rows in the shared validation split,
    - ``naive_train_dropped`` / ``naive_test_dropped`` — how many train/
      validation rows the naive arm lost to its missing-row drop. The drop is
      computed here (rows with any NaN feature, before or after fitting) so
      the numbers are reported regardless of the sklearn version's handling.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    y = df["Outcome"].to_numpy()

    # shared stratified partition, derived once from the raw row order
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    test_idx = []
    for val in sorted(np.unique(y)):
        idx = perm[y[perm] == val]
        k = int(round(test_size * len(idx)))
        test_idx.extend(idx[:k].tolist())
    test_set = set(test_idx)
    train_idx = [i for i in perm if i not in test_set]
    test_idx_sorted = sorted(test_set)

    def _fit(X):
        model = LogisticRegression(max_iter=1000)
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx_sorted])[:, 1]
        return float(roc_auc_score(y[test_idx_sorted], proba))

    imputed = df.copy()
    for col in FEATURES:
        imputed[col] = imputed[col].fillna(imputed[col].median())
    auc_imputed = _fit(imputed[FEATURES].to_numpy(dtype=float))

    naive = df.copy()
    feat_naive = naive[FEATURES]
    has_nan = feat_naive.isna().any(axis=1)
    n_train_dropped = int(has_nan.loc[train_idx].sum())
    n_test_dropped = int(has_nan.loc[test_idx_sorted].sum())
    naive_idx_train = [i for i in train_idx if not has_nan.iloc[i]]
    naive_idx_test = [i for i in test_idx_sorted if not has_nan.iloc[i]]
    model = LogisticRegression(max_iter=1000)
    Xn = naive[FEATURES].to_numpy(dtype=float)
    model.fit(Xn[naive_idx_train], y[naive_idx_train])
    proba = model.predict_proba(Xn[naive_idx_test])[:, 1]
    auc_naive = float(roc_auc_score(y[naive_idx_test], proba))

    return {
        "auc_imputed": auc_imputed,
        "auc_naive": auc_naive,
        "delta_auc": auc_imputed - auc_naive,
        "test_n": len(test_idx_sorted),
        "naive_train_dropped": n_train_dropped,
        "naive_test_dropped": n_test_dropped,
    }


def coefficient_table(result: dict) -> pd.DataFrame:
    """Per-feature coefficients from a fitted baseline model.

    Columns: ``feature``, ``coef`` (signed log-odds), ``exp_coef`` (the odds
    ratio, i.e. how the odds of Outcome=1 scale per one-unit increase in the
    feature). Rows are sorted by |coef| descending — a quick, interpretable
    feature-importance proxy without a second model.
    """
    model = result["model"]
    coefs = np.asarray(model.coef_).ravel()
    out = pd.DataFrame(
        {
            "feature": FEATURES,
            "coef": coefs,
            "exp_coef": np.exp(coefs),
        }
    )
    return out.reindex(out["coef"].abs().sort_values(ascending=False).index)


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper: the standard pandas describe for all features."""
    return df[FEATURES].describe().T.round(3)
