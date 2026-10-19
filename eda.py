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


def train_random_forest(
    df: pd.DataFrame,
    test_size: float = 0.3,
    seed: int = 42,
) -> dict:
    """Fit a RandomForestClassifier on the same shared split as the baseline.

    Mirrors ``train_baseline``'s median imputation and stratified
    train/validation partition exactly, so ROC-AUC differences between the
    two models reflect only the learner — not a different dataset or a
    different handling of missing values.

    Returns a dict with:

    - ``model`` — the fitted ``RandomForestClassifier``,
    - ``train_n`` / ``test_n`` — split sizes (identical to the baseline's),
    - ``auc`` — ROC-AUC on the validation set,
    - ``feature_importances`` — normalized impurity-decrease importances for
      each feature in ``FEATURES`` order.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    work = df.copy()
    for col in FEATURES:
        work[col] = work[col].fillna(work[col].median())
    X = work[FEATURES].to_numpy(dtype=float)
    y = work["Outcome"].to_numpy()

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    test_idx = []
    for val in sorted(np.unique(y)):
        idx = perm[y[perm] == val]
        k = int(round(test_size * len(idx)))
        test_idx.extend(idx[:k].tolist())
    test_set = set(test_idx)
    train_idx = [i for i in perm if i not in test_set]

    model = RandomForestClassifier(n_estimators=200, random_state=seed)
    model.fit(X[train_idx], y[train_idx])
    proba = model.predict_proba(X[sorted(test_set)])[:, 1]
    return {
        "model": model,
        "train_n": len(train_idx),
        "test_n": len(test_idx),
        "auc": float(roc_auc_score(y[sorted(test_set)], proba)),
        "feature_importances": model.feature_importances_,
    }


def rf_importance_table(result: dict) -> pd.DataFrame:
    """RandomForest feature importances as a tidy table, ranked.

    Columns: ``feature``, ``importance`` (normalized impurity decrease; the
    column sums to 1), and ``rank`` (1 = most important). Rows are sorted by
    importance descending — the tree-model counterpart of the
    ``coefficient_table`` that ranks the LR baseline's signed coefficients.
    """
    importances = np.asarray(result["feature_importances"], dtype=float)
    order = np.argsort(-importances, kind="stable")
    out = pd.DataFrame(
        {
            "feature": [FEATURES[i] for i in order],
            "importance": [float(importances[i]) for i in order],
            "rank": range(1, len(FEATURES) + 1),
        }
    )
    return out


def plot_rf_importance(
    df: pd.DataFrame, out_dir: str = "plots", skip: bool = False
) -> list[str]:
    """Render the RandomForest feature-importance bar chart.

    Writes ``out_dir/rf_importance.png`` and returns its path (an empty list
    when ``skip=True``). The Agg backend keeps the function headless-friendly,
    matching ``plot_distributions``.
    """
    if skip:
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = rf_importance_table(train_random_forest(df))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        table["feature"][::-1],
        table["importance"][::-1],
        color="#4c72b0",
    )
    ax.set_xlabel("Importance (normalized impurity decrease)")
    ax.set_title("RandomForest feature importances — Pima diabetes")
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "rf_importance.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return [path]


def threshold_sweep(
    df: pd.DataFrame,
    cutoffs: tuple[float, ...] | list[float] = (0.3, 0.4, 0.5, 0.6, 0.7),
    test_size: float = 0.3,
    seed: int = 42,
) -> dict:
    """Sweep decision thresholds on the baseline model's validation probabilities.

    The same LogisticRegression and the same stratified train/validation split
    as ``train_baseline`` are refit once; each row of the returned table then
    re-scores the *same* validation probabilities at a different cutoff, so the
    sweep isolates the threshold choice rather than any dataset difference.

    Returns a dict with:

    - ``table`` — a DataFrame indexed by ``cutoff`` (ascending) with columns
      ``precision``, ``recall``, ``f1``, ``tp``, ``fp``, ``fn``, ``tn``.
      A cutoff of 0.5 reproduces the baseline's hard predictions exactly.
    - ``best_cutoff`` — the cutoff maximizing F1; ties break toward the lower
      threshold (the conservative one, which favors recall).
    - ``auc`` — ROC-AUC of the underlying probabilities, included so callers
      can report it next to the table without refitting.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    work = df.copy()
    for col in FEATURES:
        work[col] = work[col].fillna(work[col].median())
    X = work[FEATURES].to_numpy(dtype=float)
    y = work["Outcome"].to_numpy()

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    test_idx = []
    for val in sorted(np.unique(y)):
        idx = perm[y[perm] == val]
        k = int(round(test_size * len(idx)))
        test_idx.extend(idx[:k].tolist())
    test_set = set(test_idx)
    train_idx = [i for i in perm if i not in test_set]

    model = LogisticRegression(max_iter=1000)
    model.fit(X[train_idx], y[train_idx])
    proba = model.predict_proba(X[sorted(test_set)])[:, 1]
    y_true = y[sorted(test_set)]

    rows = []
    best_key = None
    for cutoff in cutoffs:
        preds = (proba >= cutoff).astype(int)
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))
        tn = int(np.sum((preds == 0) & (y_true == 0)))
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        rows.append(
            {
                "cutoff": cutoff,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
        # ties on F1 break toward the lower cutoff (max over an ascending list
        # keeps the first occurrence), which is the recall-favoring choice
        if best_key is None or f1 > best_key[0]:
            best_key = (f1, cutoff)

    table = pd.DataFrame(rows).set_index("cutoff")
    return {
        "table": table,
        "best_cutoff": float(best_key[1]),
        "auc": float(roc_auc_score(y_true, proba)),
    }


def calibration_check(
    df: pd.DataFrame,
    n_bins: int = 10,
    test_size: float = 0.3,
    seed: int = 42,
) -> dict:
    """Calibration check: predicted vs observed positive rate per probability decile.

    The same LogisticRegression and stratified train/validation split as
    ``train_baseline`` are refit once; the validation probabilities are then
    binned into ``n_bins`` equal-width buckets (deciles by default) and each
    bin reports what the model predicted versus what actually happened — the
    classic reliability-diagram data.

    Returns a dict with:

    - ``table`` — a DataFrame indexed by decile (0 = lowest predicted
      probability) with columns ``n`` (rows in the bin), ``mean_predicted``
      (average model probability), ``observed_rate`` (actual positive rate).
      Empty bins are dropped; on 768 rows with ~35% positives, all ten
      deciles normally have members.
    - ``calibration_slope`` / ``intercept`` — the fit of observed vs
      predicted across the populated bins' bin centers. A well-calibrated
      model sits near slope=1, intercept=0 (the 45-degree line).
    - ``auc`` — ROC-AUC of the underlying probabilities, included so callers
      can report discrimination next to calibration without refitting.

    Notes: with ``n_bins > 10`` the bins are finer than deciles; with
    ``n_bins == 1`` everything lands in a single bin (slope/intercept need
    at least two populated points and come back NaN).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    work = df.copy()
    for col in FEATURES:
        work[col] = work[col].fillna(work[col].median())
    X = work[FEATURES].to_numpy(dtype=float)
    y = work["Outcome"].to_numpy()

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    test_idx = []
    for val in sorted(np.unique(y)):
        idx = perm[y[perm] == val]
        k = int(round(test_size * len(idx)))
        test_idx.extend(idx[:k].tolist())
    test_set = set(test_idx)
    train_idx = [i for i in perm if i not in test_set]

    model = LogisticRegression(max_iter=1000)
    model.fit(X[train_idx], y[train_idx])
    proba = model.predict_proba(X[sorted(test_set)])[:, 1]
    y_true = y[sorted(test_set)]

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    # proba is 1-D; compare against the bin edges to produce a (n, n_bins)
    # membership matrix where labels[:, i] selects the rows in bin i.
    labels = (proba[:, None] >= bins[:-1]) & (proba[:, None] < bins[1:])
    # probabilities exactly at 1.0 fall into the last bin
    if n_bins > 1:
        labels[:, -1] |= proba >= bins[-1]

    rows = []
    for i in range(n_bins):
        mask = labels[:, i]
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        pred = float(proba[mask].mean())
        obs = float(y_true[mask].mean())
        rows.append(
            {
                "decile": i + 1,
                "n": n_bin,
                "mean_predicted": round(pred, 4),
                "observed_rate": round(obs, 4),
            }
        )

    table = pd.DataFrame(rows).set_index("decile") if rows else pd.DataFrame(
        columns=["n", "mean_predicted", "observed_rate"]
    ).rename_axis("decile")

    slope, intercept = float("nan"), float("nan")
    if len(table) >= 2:
        # least-squares fit of observed vs predicted at each bin's center —
        # the calibration curve's linear summary (slope ~1, intercept ~0 is
        # well calibrated)
        x = table["mean_predicted"].to_numpy(dtype=float)
        v = table["observed_rate"].to_numpy(dtype=float)
        denom = float(((x - x.mean()) ** 2).sum())
        if denom > 0:
            slope = float(((x - x.mean()) * (v - v.mean())).sum() / denom)
            intercept = float(v.mean() - slope * x.mean())

    return {
        "table": table,
        "calibration_slope": round(slope, 4) if not np.isnan(slope) else None,
        "intercept": round(intercept, 4) if not np.isnan(intercept) else None,
        "auc": float(roc_auc_score(y_true, proba)),
    }


def plot_threshold_sweep(
    df: pd.DataFrame, out_dir: str = "plots", skip: bool = False
) -> list[str]:
    """Render the threshold sweep as a precision/recall/F1 line chart.

    Writes ``out_dir/threshold_sweep.png`` and returns its path (an empty list
    when ``skip=True``). The Agg backend keeps the function headless-friendly,
    matching ``plot_distributions`` and ``plot_rf_importance``. The cutoff that
    maximizes F1 is marked with a vertical line so the optimal operating point
    reads directly off the figure.
    """
    if skip:
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep = threshold_sweep(df)
    table = sweep["table"]
    best = sweep["best_cutoff"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, style in (("precision", "-o"), ("recall", "-s"), ("f1", "-^")):
        ax.plot(table.index, table[col], style, label=col)
    ax.axvline(best, color="#00000030", linestyle="--")
    ax.set_xlabel("Decision threshold (predict positive if P >= cutoff)")
    ax.set_ylabel("Score")
    ax.set_title(
        "Baseline LogisticRegression — precision / recall / F1 by threshold "
        f"(best F1 at {best:.1f})"
    )
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "threshold_sweep.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return [path]


def plot_calibration(
    df: pd.DataFrame, out_dir: str = "plots", skip: bool = False
) -> list[str]:
    """Render the calibration check as a reliability diagram.

    Writes ``out_dir/calibration.png`` and returns its path (an empty list
    when ``skip=True``). The Agg backend keeps the function headless-friendly,
    matching the other plot helpers. Each decile is plotted at its mean
    predicted probability against its observed rate; a 45-degree reference
    line marks perfect calibration so mis-calibration reads directly off the
    figure.
    """
    if skip:
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cal = calibration_check(df)
    table = cal["table"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        [0, 1], [0, 1], color="#888888", linestyle="--", label="perfectly calibrated"
    )
    ax.scatter(
        table["mean_predicted"],
        table["observed_rate"],
        s=table["n"] * 3,
        c="#4c72b0",
        edgecolor="#1f3b57",
        alpha=0.85,
        label="deciles (size ~ n)",
    )
    if cal["calibration_slope"] is not None:
        xs = np.linspace(0, 1, 50)
        ys = cal["intercept"] + cal["calibration_slope"] * xs
        ax.plot(
            xs,
            ys,
            color="#c44e52",
            linestyle=":",
            label=f"fit (slope {cal['calibration_slope']:.2f}, intercept {cal['intercept']:+.2f})",
        )
    ax.set_xlabel("Mean predicted probability per decile")
    ax.set_ylabel("Observed positive rate")
    ax.set_title(
        "Calibration — baseline LogisticRegression, "
        f"{len(table)} populated deciles (ROC-AUC {cal['auc']:.3f})"
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "calibration.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return [path]


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper: the standard pandas describe for all features."""
    return df[FEATURES].describe().T.round(3)
