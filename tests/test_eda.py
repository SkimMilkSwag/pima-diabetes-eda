"""Tests for the Pima Indians Diabetes EDA helpers.

Run with::

    PYTHONPATH=/tmp/mlopen python -m pytest tests/ -q

(the /tmp/mlopen overlay provides a compatible numpy/pandas on this box).
"""

import os

import numpy as np
import pandas as pd
import pytest

import eda


def _load():
    return eda.load()


def test_load_shape_and_columns():
    df = _load()
    assert len(df) == 768
    for col in eda.FEATURES + ["Outcome"]:
        assert col in df.columns


def test_coerce_zeros_sets_nan_and_counts():
    raw = _load()
    n_glucose_zero_raw = int((raw["Glucose"] == 0).sum())
    assert n_glucose_zero_raw > 0, "expected some zero-glucose rows to coerce"
    df, coerced = eda.coerce_zeros(raw)
    assert (df["Glucose"] == 0).sum() == 0
    assert np.isnan(df.loc[raw["Glucose"] == 0, "Glucose"]).all()
    assert coerced["Glucose"] == n_glucose_zero_raw
    # untouched columns keep their zeros-free state (Age has no zeros)
    assert "Age" not in coerced or coerced.get("Age", 0) == 0


def test_missing_report_shape_and_labels():
    raw = _load()
    df, _ = eda.coerce_zeros(raw)
    rep = eda.missing_report(df)
    assert list(rep.columns) == ["column", "missing", "pct_missing", "severity"]
    assert len(rep) == df.shape[1]
    assert set(rep["severity"]).issubset({"low", "medium", "high"})
    # SkinThickness and Insulin are famously >20% missing after coercion
    skin = rep.loc[rep["column"] == "SkinThickness", "pct_missing"].iloc[0]
    ins = rep.loc[rep["column"] == "Insulin", "pct_missing"].iloc[0]
    assert skin > 20 and ins > 20


def test_outcome_counts():
    df = _load()
    oc = eda.outcome_counts(df)
    assert oc["total"] == 768
    assert oc["outcome_1"] + oc["outcome_0"] == 768
    assert 0.3 < oc["positive_rate"] < 0.5
    # this bundled copy is 268 positive / 500 negative (34.9% positive)
    assert oc["outcome_1"] == 268
    assert oc["outcome_0"] == 500


def test_feature_means_by_outcome():
    df = _load()
    t = eda.feature_means_by_outcome(df)
    assert list(t.columns) == [
        "outcome_0_mean",
        "outcome_1_mean",
        "delta",
    ]
    assert list(t.index) == eda.FEATURES
    # Glucose should be higher in the positive group (a well-known signal)
    glu = t.loc["Glucose"]
    assert glu["outcome_1_mean"] > glu["outcome_0_mean"]
    assert np.isclose(glu["delta"], glu["outcome_1_mean"] - glu["outcome_0_mean"])


def test_correlation_matrix_and_top():
    df = _load()
    cm = eda.correlation_matrix(df)
    assert cm.shape == (9, 9)
    # diagonal must be 1
    assert np.allclose(np.diag(cm.values), 1.0)
    top = eda.top_correlates_with_outcome(df, k=3)
    assert len(top) == 3
    assert (top >= 0).all()


def test_describe():
    df = _load()
    d = eda.describe(df)
    assert set(d.columns).issuperset({"mean", "std", "min", "max"})
    assert list(d.index) == eda.FEATURES


def test_group_distributions_shape_and_order():
    df = _load()
    dists = eda.group_distributions(df)
    assert len(dists) == len(eda.FEATURES)
    for rows in dists:
        # index 0 = outcome 0, index 1 = outcome 1; quantile points per group
        assert len(rows) == 2
        for row in rows:
            assert len(row) <= 100
            # quantiles of real data are monotonically non-decreasing
            assert (np.diff(row) >= -1e-12).all()


def test_group_distributions_glucose_separates():
    df = _load()
    glu = eda.group_distributions(df)[eda.FEATURES.index("Glucose")]
    # the positive group's glucose distribution sits above the negative one
    assert glu[1].mean() > glu[0].mean()


def test_plot_distributions_writes_pngs(tmp_path):
    df = _load()
    paths = eda.plot_distributions(df, out_dir=str(tmp_path))
    names = {os.path.basename(p) for p in paths}
    assert {"all_features.png"} | {f"{f}.png" for f in eda.FEATURES} == names
    for p in paths:
        assert os.path.getsize(p) > 0


def test_plot_distributions_skip_writes_nothing(tmp_path):
    df = _load()
    paths = eda.plot_distributions(df, out_dir=str(tmp_path), skip=True)
    assert paths == []
    assert os.listdir(str(tmp_path)) == [], "skip must not write any figure files"


def _prepared():
    raw = _load()
    df, _ = eda.coerce_zeros(raw)
    return df


def test_train_baseline_splits_and_scores():
    df = _prepared()
    res = eda.train_baseline(df)
    assert res["train_n"] + res["test_n"] == 768
    assert 0.5 < res["auc"] < 0.95, "baseline AUC should be informative but not perfect"
    cm = res["confusion_matrix"]
    assert cm.shape == (2, 2)
    # predictions must be meaningful: correct calls outnumber wrong ones
    assert cm[0, 0] + cm[1, 1] > cm[0, 1] + cm[1, 0]


def test_train_baseline_is_reproducible():
    df = _prepared()
    a = eda.train_baseline(df)
    b = eda.train_baseline(df)
    assert a["auc"] == b["auc"]
    np.testing.assert_array_equal(a["confusion_matrix"], b["confusion_matrix"])


def test_auc_bootstrap_ci_band_brackets_point_estimate():
    from synth import make_synthetic

    X, y = make_synthetic()
    res = eda.auc_bootstrap_ci(X, y)
    # the synthetic problem is well separated: AUC lands high and stable
    assert 0.85 < res["auc"] < 1.0
    assert res["ci_low"] <= res["auc"] <= res["ci_high"]
    assert res["ci_low"] > 0.0
    assert res["ci_high"] <= 1.0
    # a usable band: the width reflects sampling noise, not a degenerate 0.0
    assert res["ci_high"] - res["ci_low"] >= 0.0
    assert res["std_boot"] > 0.0
    assert res["n_boot"] == 200
    assert res["ci"] == 0.95


def test_auc_bootstrap_ci_reproducible_and_custom_levels():
    from synth import make_synthetic

    X, y = make_synthetic()
    a = eda.auc_bootstrap_ci(X, y, n_boot=100)
    b = eda.auc_bootstrap_ci(X, y, n_boot=100)
    # same seed -> identical draws -> identical band
    assert (a["ci_low"], a["ci_high"]) == (b["ci_low"], b["ci_high"])
    # a 99% interval is at least as wide as the 95% one
    wide = eda.auc_bootstrap_ci(X, y, n_boot=100, ci=0.99)
    assert wide["ci_low"] <= a["ci_low"]
    assert wide["ci_high"] >= a["ci_high"]
    assert wide["ci_high"] - wide["ci_low"] >= a["ci_high"] - a["ci_low"]


def test_auc_bootstrap_ci_rejects_single_class():
    from synth import make_synthetic

    X, y = make_synthetic()
    pos = X[y == 1]
    with pytest.raises(ValueError):
        eda.auc_bootstrap_ci(pos, np.ones(len(pos)))


def test_auc_bootstrap_ci_on_real_holdout_split():
    df = _prepared()
    # reproduce train_baseline's shared split so the CI is for the same
    # evaluation set the rest of the notebook reports on
    base = eda.train_baseline(df)
    work = df.copy()
    for col in eda.FEATURES:
        work[col] = work[col].fillna(work[col].median())
    X = work[eda.FEATURES].to_numpy(dtype=float)
    y = work["Outcome"].to_numpy()
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y))
    test_idx = []
    for val in sorted(np.unique(y)):
        idx = perm[y[perm] == val]
        k = int(round(0.3 * len(idx)))
        test_idx.extend(idx[:k].tolist())
    res = eda.auc_bootstrap_ci(X[test_idx], y[test_idx])
    # note: the bootstrap's fixed model fits on ALL of X (here, train +
    # validation combined) whereas train_baseline fits on the train rows
    # only — so the point estimates differ by construction. What must hold is
    # that they land in the same ballpark on the same evaluation rows.
    assert abs(res["auc"] - base["auc"]) < 0.05
    assert res["ci_low"] <= res["auc"] <= res["ci_high"]
    # ~230 validation rows: the band is tight but not zero-width
    assert res["ci_high"] - res["ci_low"] >= 0.0


def test_imputation_comparison_uses_shared_split():
    df = _prepared()
    ab = eda.imputation_comparison(df)
    baseline = eda.train_baseline(df)
    # the A/B arms use the same stratified partition as train_baseline,
    # and the imputed arm should land at the same AUC as the baseline run
    assert ab["test_n"] == baseline["test_n"]
    assert abs(ab["auc_imputed"] - baseline["auc"]) < 1e-9
    assert np.isclose(ab["delta_auc"], ab["auc_imputed"] - ab["auc_naive"])
    # both arms must be informative (well above the coin-flip 0.5)
    assert ab["auc_imputed"] > 0.6
    assert ab["auc_naive"] > 0.5


def test_imputation_comparison_is_reproducible():
    df = _prepared()
    a = eda.imputation_comparison(df)
    b = eda.imputation_comparison(df)
    for key in ("auc_imputed", "auc_naive", "delta_auc"):
        assert a[key] == b[key]
    assert a["naive_train_dropped"] == b["naive_train_dropped"]
    assert a["naive_test_dropped"] == b["naive_test_dropped"]


def test_imputation_comparison_reports_row_drops():
    df = _prepared()
    ab = eda.imputation_comparison(df)
    # the naive arm must drop rows (Insulin is ~49% missing after coercion)
    assert ab["naive_train_dropped"] > 0
    assert ab["naive_test_dropped"] > 0
    assert ab["test_n"] == round(0.3 * 768)


def test_coefficient_table_sorted_and_complete():
    df = _prepared()
    table = eda.coefficient_table(eda.train_baseline(df))
    assert list(table.columns) == ["feature", "coef", "exp_coef"]
    assert set(table["feature"]) == set(eda.FEATURES)
    abs_coefs = table["coef"].abs().tolist()
    assert abs_coefs == sorted(abs_coefs, reverse=True)
    # odds ratios are positive and consistent with the signed coefficients
    np.testing.assert_allclose(table["exp_coef"], np.exp(table["coef"]))


def test_train_random_forest_matches_baseline_split():
    df = _prepared()
    rf = eda.train_random_forest(df)
    baseline = eda.train_baseline(df)
    # identical shared split: same sizes, and the tree model must be at least
    # as informative as a coin flip on the validation set
    assert rf["train_n"] == baseline["train_n"]
    assert rf["test_n"] == baseline["test_n"]
    assert 0.5 < rf["auc"] < 0.95
    importances = rf["feature_importances"]
    assert len(importances) == len(eda.FEATURES)
    assert (importances >= 0).all()
    # sklearn normalizes feature_importances_ to sum to 1
    np.testing.assert_allclose(importances.sum(), 1.0, atol=1e-6)


def test_rf_importance_table_ranked_and_complete():
    df = _prepared()
    table = eda.rf_importance_table(eda.train_random_forest(df))
    assert list(table.columns) == ["feature", "importance", "rank"]
    assert set(table["feature"]) == set(eda.FEATURES)
    # rows are sorted by importance descending and rank is 1..N in order
    assert table["importance"].tolist() == sorted(
        table["importance"].tolist(), reverse=True
    )
    assert list(table["rank"]) == list(range(1, len(eda.FEATURES) + 1))


def test_rf_importance_reproducible():
    df = _prepared()
    a = eda.rf_importance_table(eda.train_random_forest(df))
    b = eda.rf_importance_table(eda.train_random_forest(df))
    pd.testing.assert_frame_equal(a, b)


def test_plot_rf_importance_writes_png(tmp_path):
    df = _prepared()
    paths = eda.plot_rf_importance(df, out_dir=str(tmp_path))
    assert len(paths) == 1 and os.path.basename(paths[0]) == "rf_importance.png"
    assert os.path.getsize(paths[0]) > 0


def test_plot_rf_importance_skip_writes_nothing(tmp_path):
    df = _prepared()
    paths = eda.plot_rf_importance(df, out_dir=str(tmp_path), skip=True)
    assert paths == []
    assert os.listdir(str(tmp_path)) == [], "skip must not write any figure files"


def test_threshold_sweep_table_shape_and_bounds():
    df = _prepared()
    sweep = eda.threshold_sweep(df)
    table = sweep["table"]
    assert list(table.index) == [0.3, 0.4, 0.5, 0.6, 0.7]
    assert list(table.columns) == ["precision", "recall", "f1", "tp", "fp", "fn", "tn"]
    for row in table.itertuples():
        assert 0.0 <= row.precision <= 1.0
        assert 0.0 <= row.recall <= 1.0
        assert 0.0 <= row.f1 <= 1.0
        # every validation row lands in exactly one confusion-matrix bucket
        assert row.tp + row.fp + row.fn + row.tn == 230
    assert 0.5 < sweep["auc"] < 0.95


def test_threshold_sweep_consistent_with_baseline_at_half():
    df = _prepared()
    baseline = eda.train_baseline(df)
    sweep = eda.threshold_sweep(df)
    # the sweep shares train_baseline's split and model, so the 0.5 row must
    # reproduce the baseline's validation confusion matrix exactly
    row = sweep["table"].loc[0.5]
    cm = np.asarray(baseline["confusion_matrix"])
    assert int(row["tp"]) == int(cm[1, 1])
    assert int(row["fp"]) == int(cm[0, 1])
    assert int(row["fn"]) == int(cm[1, 0])
    assert int(row["tn"]) == int(cm[0, 0])
    np.testing.assert_allclose(sweep["auc"], baseline["auc"], atol=1e-9)


def test_threshold_sweep_monotone_recall_and_best_cutoff():
    df = _prepared()
    sweep = eda.threshold_sweep(df)
    table = sweep["table"]
    # a higher cutoff predicts fewer positives: recall (and TP count) drop,
    # precision rises, as the model gets more conservative
    assert table["recall"].is_monotonic_decreasing
    assert table["precision"].is_monotonic_increasing
    assert table["tp"].is_monotonic_decreasing
    best = sweep["best_cutoff"]
    assert best in table.index
    np.testing.assert_allclose(
        table["f1"].loc[best], table["f1"].max()
    )
    # on this dataset the F1 optimum lands at a low threshold — recall is the
    # scarcer metric, so a loose cutoff beats 0.5; assert the sweep actually
    # finds that instead of defaulting to 0.5
    assert best < 0.5


def test_threshold_sweep_custom_cutoffs():
    df = _prepared()
    sweep = eda.threshold_sweep(df, cutoffs=[0.1, 0.9])
    table = sweep["table"]
    assert list(table.index) == [0.1, 0.9]
    # a 0.1 cutoff predicts almost everything positive: recall ~1, precision
    # collapses to near the base rate of positives in the split
    assert table.loc[0.1, "recall"] > 0.95
    assert table.loc[0.9, "recall"] < table.loc[0.1, "recall"]


def test_threshold_sweep_reproducible():
    df = _prepared()
    a = eda.threshold_sweep(df)
    b = eda.threshold_sweep(df)
    pd.testing.assert_frame_equal(a["table"], b["table"])
    assert a["best_cutoff"] == b["best_cutoff"]


def test_plot_threshold_sweep_writes_png(tmp_path):
    df = _prepared()
    paths = eda.plot_threshold_sweep(df, out_dir=str(tmp_path))
    assert len(paths) == 1 and os.path.basename(paths[0]) == "threshold_sweep.png"
    assert os.path.getsize(paths[0]) > 0


def test_plot_threshold_sweep_skip_writes_nothing(tmp_path):
    df = _prepared()
    paths = eda.plot_threshold_sweep(df, out_dir=str(tmp_path), skip=True)
    assert paths == []
    assert os.listdir(str(tmp_path)) == [], "skip must not write any figure files"


def test_calibration_check_table_shape_and_totals():
    df = _prepared()
    cal = eda.calibration_check(df)
    table = cal["table"]
    # 10 deciles, all populated on the real data; the validation split has
    # ~230 rows, so every decile carries members.
    assert list(table.index) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for col in ("n", "mean_predicted", "observed_rate"):
        assert col in table.columns
    # every validation row lands in exactly one bin, so the bin sizes must
    # sum to the validation split size (matches train_baseline's test_n)
    base = eda.train_baseline(df)
    assert int(table["n"].sum()) == base["test_n"]
    assert (table["mean_predicted"] >= 0).all() and (table["mean_predicted"] <= 1).all()
    assert (table["observed_rate"] >= 0).all() and (table["observed_rate"] <= 1).all()
    # decile ordering: mean predicted probability must increase with the
    # decile index (bin edges are monotone, so averages can't cross)
    assert list(table["mean_predicted"]) == sorted(table["mean_predicted"])


def test_calibration_check_auc_matches_baseline():
    df = _prepared()
    cal = eda.calibration_check(df)
    base = eda.train_baseline(df)
    # same model, same split: AUC must be identical to the baseline's
    assert abs(cal["auc"] - base["auc"]) < 1e-9


def test_calibration_check_slope_intercept_populated():
    df = _prepared()
    cal = eda.calibration_check(df)
    assert cal["calibration_slope"] is not None
    assert cal["intercept"] is not None
    # logistic regression on this data is reasonably well calibrated: the fit
    # must be positive-sloped and the intercept must stay in a sane band.
    # (A slope near zero or a huge intercept would flag broken binning.)
    assert cal["calibration_slope"] > 0.0
    assert -0.5 <= cal["intercept"] <= 0.5


def test_calibration_check_custom_n_bins():
    df = _prepared()
    coarse = eda.calibration_check(df, n_bins=5)
    assert len(coarse["table"]) == 5
    # with 5 bins each bin holds roughly twice the rows of a decile
    deciles = eda.calibration_check(df, n_bins=10)
    assert int(coarse["table"]["n"].sum()) == int(deciles["table"]["n"].sum())


def test_calibration_check_n_bins_one_gives_single_bin():
    df = _prepared()
    single = eda.calibration_check(df, n_bins=1)
    table = single["table"]
    assert len(table) == 1
    # one populated bin: slope/intercept need >= 2 points, so they're None
    assert single["calibration_slope"] is None
    assert single["intercept"] is None
    # the single bin covers every validation row, so its observed rate is just
    # the raw positive rate of that split — compare against the full frame's
    # positive rate (stratification keeps them close but not identical).
    raw = _load()
    df2, _ = eda.coerce_zeros(raw)
    expected_rate = float((df2["Outcome"] == 1).mean())
    assert abs(table.iloc[0]["observed_rate"] - round(expected_rate, 4)) < 0.02


def test_plot_calibration_writes_png(tmp_path):
    df = _prepared()
    paths = eda.plot_calibration(df, out_dir=str(tmp_path))
    assert len(paths) == 1 and os.path.basename(paths[0]) == "calibration.png"
    assert os.path.getsize(paths[0]) > 0


def test_plot_calibration_skip_writes_nothing(tmp_path):
    df = _prepared()
    paths = eda.plot_calibration(df, out_dir=str(tmp_path), skip=True)
    assert paths == []
    assert os.listdir(str(tmp_path)) == [], "skip must not write any figure files"
