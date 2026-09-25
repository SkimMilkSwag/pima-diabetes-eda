# pima-diabetes-eda

Exploratory data analysis on the **Pima Indians Diabetes Dataset** (UCI, 768
rows, 8 features, binary outcome). The goal is not to build a great model — it's
to understand the data: what's genuinely missing vs. recorded as zero, how the
two outcome groups differ on each feature, and which features correlate most
with the label.

Two ways in, same code underneath:

- **`EDA.ipynb`** — the notebook walkthrough (descriptive stats, missingness,
  group comparisons, correlations).
- **`run.py`** — a headless script that prints the same tables to stdout, for
  CI or machines without Jupyter.

All analysis lives in **`eda.py`** as small, importable functions so the
notebook, the script, and the tests share one implementation.

## Layout

```
pima-diabetes-eda/
├── EDA.ipynb          # interactive analysis
├── eda.py             # the analysis functions (shared)
├── run.py             # headless: print all tables to stdout
├── data/diabetes.csv  # bundled UCI dataset copy
└── tests/test_eda.py  # unit tests for eda.py
```

## Setup

Requires `pandas` and `numpy`. (scikit-learn / scipy are optional — the core
analysis is stdlib + pandas only.)

```bash
python -m pip install pandas numpy scikit-learn matplotlib
# run headless analysis
PYTHONPATH=. python run.py
# run tests
PYTHONPATH=. python -m pytest tests/ -q
```

`run.py` detects a CI environment (the `CI` env var, set by GitHub Actions)
and skips the boxplot figure rendering there — figures are a plotting-only
concern and slow the run down with no analysis value in CI. Set `CI=1`
locally to preview that path.

## What the analysis covers

| Section | Function(s) |
|---|---|
| Load + zero-as-missing coercion | `load()`, `coerce_zeros()` |
| Missingness report (count / % / severity) | `missing_report()` |
| Outcome distribution & positive rate | `outcome_counts()` |
| Per-feature means, split by outcome + delta | `feature_means_by_outcome()` |
| Pearson correlation matrix (features + Outcome) | `correlation_matrix()` |
| Top-k features by \|corr\| with Outcome | `top_correlates_with_outcome()` |
| LogisticRegression baseline (ROC-AUC, confusion matrix) | `train_baseline()` |
| Median-imputation A/B on a shared split | `imputation_comparison()` |
| Coefficients / odds-ratio table | `coefficient_table()` |

## CI

GitHub Actions runs the test suite on Python 3.10/3.11/3.12 for every push
and pull request (`.github/workflows/ci.yml`).

A note on the data: the UCI export records several "not available" measurements
as `0` (a glucose of 0 is not physically plausible). `coerce_zeros()` turns
those into NaN and reports how many were coerced, so missingness stats reflect
real gaps rather than an artifact.

## Dataset provenance

Bundled copy sourced from [npradaschnor/Pima-Indians-Diabetes-Dataset](https://github.com/npradaschnor/Pima-Indians-Diabetes-Dataset)
(`diabetes.csv`), which mirrors the canonical UCI Machine Learning Repository
release. 768 rows; outcome is balanced ~43% / 57%.

## License

[MIT](LICENSE) — see that file for terms.
