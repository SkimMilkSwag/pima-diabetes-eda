#!/usr/bin/env python3
"""Build a synthetic two-class dataset for bootstrap-CI unit tests.

The generator is deterministic (fixed RNG seed) and produces a problem the
baseline LogisticRegression solves *well* on purpose: the positive class
(mean 1.0, std 0.5) and the negative class (mean 0.0, std 0.5) are clearly
separable, so the holdout ROC-AUC lands comfortably above 0.5 — which is the
condition ``auc_bootstrap_ci`` needs in order to compute a non-degenerate
confidence band.
"""

import numpy as np


def make_synthetic(n_per_class: int = 400, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)`` for a synthetic two-class classification problem.

    ``n_per_class`` rows per class (equal-size by construction — no stratified
    split needed downstream). One informative feature per row plus one pure
    noise feature; labels are 0 (negative) / 1 (positive).
    """
    rng = np.random.default_rng(seed)
    info = np.concatenate([rng.normal(0.0, 0.5, n_per_class), rng.normal(1.0, 0.5, n_per_class)])
    noise = rng.normal(0.0, 1.0, 2 * n_per_class)
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return np.column_stack([info, noise]), y
