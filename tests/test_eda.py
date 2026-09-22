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


def test_coefficient_table_sorted_and_complete():
    df = _prepared()
    table = eda.coefficient_table(eda.train_baseline(df))
    assert list(table.columns) == ["feature", "coef", "exp_coef"]
    assert set(table["feature"]) == set(eda.FEATURES)
    abs_coefs = table["coef"].abs().tolist()
    assert abs_coefs == sorted(abs_coefs, reverse=True)
    # odds ratios are positive and consistent with the signed coefficients
    np.testing.assert_allclose(table["exp_coef"], np.exp(table["coef"]))
