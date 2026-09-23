#!/usr/bin/env python3
"""Run the Pima Indians Diabetes EDA and print the key tables to stdout.

A headless stand-in for the notebook — useful in CI or on a machine without
Jupyter. The same analysis lives in ``eda.py``; this just wires the prints.
"""

from __future__ import annotations

import os

import eda


def main() -> None:
    raw = eda.load()
    df, coerced = eda.coerce_zeros(raw)

    print(f"Loaded {len(df)} rows, {df.shape[1]} columns.")
    print("\nZero-as-missing values coerced to NaN:")
    for col, n in sorted(coerced.items()):
        if n:
            print(f"  {col}: {n}")

    print("\nMissingness report (post-coercion):")
    print(eda.missing_report(df).to_string(index=False))

    oc = eda.outcome_counts(raw)
    print(
        f"\nOutcome: {oc['outcome_1']} positive / {oc['outcome_0']} negative "
        f"({oc['positive_rate']:.1%} positive)"
    )

    print("\nFeature means by outcome (delta = pos - neg):")
    print(eda.feature_means_by_outcome(df).to_string())

    print("\nTop features by |correlation| with Outcome:")
    top = eda.top_correlates_with_outcome(df, k=5)
    for feat, val in top.items():
        sign = "+" if eda.correlation_matrix(df).loc[feat, "Outcome"] >= 0 else "-"
        print(f"  {feat}: {sign}{val:.3f}")

    print("\nFull correlation matrix:")
    print(eda.correlation_matrix(df).to_string())

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    paths = eda.plot_distributions(df, out_dir=out_dir)
    print(f"\nWrote {len(paths)} boxplot comparison figures to {out_dir}/")

    result = eda.train_baseline(df)
    cm = result["confusion_matrix"]
    print(
        f"\nLogisticRegression baseline: ROC-AUC {result['auc']:.3f} "
        f"(train {result['train_n']} / validation {result['test_n']})"
    )
    print("Validation confusion matrix:")
    print(f"  TN {cm[0,0]:4d}  FP {cm[0,1]:4d}")
    print(f"  FN {cm[1,0]:4d}  TP {cm[1,1]:4d}")

    print("\nFeature coefficients (sorted by |coef|; exp_coef = odds ratio):")
    table = eda.coefficient_table(result)
    for row in table.itertuples(index=False):
        print(f"  {row.feature:28s} coef={row.coef:+.3f}  odds_ratio={row.exp_coef:.3f}")

    ab = eda.imputation_comparison(df)
    verdict = "imputation helps" if ab["delta_auc"] >= 0 else "imputation hurts"
    print(
        f"\nImputation A/B (median vs. no imputation, shared split of {ab['test_n']} validation rows): "
        f"imputed ROC-AUC {ab['auc_imputed']:.3f} vs. naive {ab['auc_naive']:.3f} "
        f"(delta {ab['delta_auc']:+.3f}; the naive arm drops "
        f"{ab['naive_train_dropped']} train / {ab['naive_test_dropped']} validation rows) — {verdict}"
    )


if __name__ == "__main__":
    main()
