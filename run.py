#!/usr/bin/env python3
"""Run the Pima Indians Diabetes EDA and print the key tables to stdout.

A headless stand-in for the notebook — useful in CI or on a machine without
Jupyter. The same analysis lives in ``eda.py``; this just wires the prints.
"""

from __future__ import annotations

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


if __name__ == "__main__":
    main()
