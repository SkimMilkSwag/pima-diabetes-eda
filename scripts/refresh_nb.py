#!/usr/bin/env python3
"""Refresh EDA.ipynb with the imputation A/B comparison section.

Inserts one narrative markdown cell and one code cell (running
``eda.imputation_comparison(df)`` and printing its table) right after the
zero-as-missing narrative — so the notebook walks from missingness directly
into "what does imputation buy us?". The cells carry no outputs, matching the
rest of the notebook. Idempotent: if the marker is already present the file
is left untouched.

Run from anywhere:  python scripts/refresh_nb.py
"""
from __future__ import annotations

import json
import os
import sys

NB_PATH = os.path.join(os.path.dirname(__file__), "..", "EDA.ipynb")
MARKER = "## Imputation A/B"

NARRATIVE = (
    "## Imputation A/B\n"
    "\n"
    "The missing values above are not free: a model has to do *something* with them. "
    "`imputation_comparison()` A/B-tests the two obvious strategies on the **same** "
    "stratified train/validation split, so the delta isolates the imputation choice "
    "alone — the 'imputed' arm fills each feature's NaNs with the column median, the "
    "'naive' arm leaves the NaNs in and drops any row that carries one."
)

CODE = (
    "ab = eda.imputation_comparison(df)\n"
    "print(\n"
    '    f"ROC-AUC  imputed={ab[\'auc_imputed\']:.3f}  naive={ab[\'auc_naive\']:.3f}  "\n'
    '    f"delta={ab[\'delta_auc\']:+.3f}")\n'
    "print(\n"
    '    f"rows the naive arm dropped: train={ab[\'naive_train_dropped\']}  "\n'
    '    f"validation={ab[\'naive_test_dropped\']}"\n'
    ")"
)


def _md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def main() -> int:
    with open(NB_PATH) as f:
        nb = json.load(f)

    if any(MARKER in "".join(c.get("source", [])) for c in nb["cells"]):
        print("marker already present; notebook unchanged")
        return 0

    cells = nb["cells"]
    # Anchor: the markdown cell narrating the missingness report (~49% Insulin).
    anchor = None
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "markdown":
            continue
        src = "".join(cell.get("source", []))
        if "`Insulin` (~49%)" in src:
            anchor = i
            break
    if anchor is None:
        print("could not locate the missingness narrative cell; notebook unchanged")
        return 1

    cells[anchor + 1 : anchor + 1] = [_md_cell(NARRATIVE), _code_cell(CODE)]
    with open(NB_PATH, "w") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")
    print(f"inserted 2 cells after index {anchor} into {NB_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
