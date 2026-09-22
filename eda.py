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


def plot_distributions(df: pd.DataFrame, out_dir: str = "plots") -> list[str]:
    """Render a boxplot comparison of every feature across outcome groups.

    Writes one PNG per feature (``out_dir/<feature>.png``) plus an 8-panel
    grid at ``out_dir/all_features.png``. Returns the written file paths.
    The Agg backend is used so the function works headless (CI, run.py).
    """
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


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper: the standard pandas describe for all features."""
    return df[FEATURES].describe().T.round(3)
