"""
Step 3k (follow-up, prompted by external review) — Seed sensitivity of the
headline whole-cohort result.

03_nested_cv.py's main result (the ~0.82 AUC reported everywhere) uses a
single outer/inner CV split, determined entirely by SEED=42. A reviewer
correctly noted that with n=189 and already-wide bootstrap CIs, a single
favorable train/test split could be doing some of the work, and that the
learning-curve check (which does use 5 seeds) doesn't cover the headline
number itself. This reruns the identical nested-CV pipeline (logistic
regression, full param grid, same structure as 03_nested_cv.py) with 5
different outer/inner CV random seeds, to see how much the headline AUC
moves under nothing more than a different random split of the same data.

Outputs:
  results/seed_sensitivity.csv
"""
import time

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DATA_PROCESSED, RESULTS, SEED, get_logger, set_global_seed
from feature_selection import mannwhitney_score_func

logger = get_logger("15_seed_sensitivity")
set_global_seed()

N_OUTER, N_INNER = 5, 5
N_SEEDS = 5

PIPELINE = Pipeline([
    ("select", SelectKBest(score_func=mannwhitney_score_func)),
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(solver="liblinear", max_iter=5000, class_weight="balanced", random_state=SEED)),
])
PARAM_GRID = {"select__k": [50, 100, 200], "clf__C": [0.01, 0.1, 1, 10], "clf__penalty": ["l1", "l2"]}


def nested_cv_auc(X, y, seed):
    outer_cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=seed)
    inner_cv = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=seed)
    oof_true, oof_proba = [], []
    for train_idx, test_idx in outer_cv.split(X, y):
        search = GridSearchCV(PIPELINE, PARAM_GRID, scoring="roc_auc", cv=inner_cv, n_jobs=-1)
        search.fit(X[train_idx], y[train_idx])
        proba = search.best_estimator_.predict_proba(X[test_idx])[:, 1]
        oof_true.extend(y[test_idx].tolist())
        oof_proba.extend(proba.tolist())
    return roc_auc_score(oof_true, oof_proba)


def main():
    expr = pd.read_csv(DATA_PROCESSED / "expression_log2_filtered.csv", index_col=0)
    meta = pd.read_csv(DATA_PROCESSED / "sample_metadata_qc.csv", index_col="sample_id")
    meta = meta.reindex(expr.index)
    X = expr.values
    y = (meta["label"].values == "FM").astype(int)

    seeds = [SEED + i for i in range(N_SEEDS)]  # SEED (=42, the headline result) + 4 others
    rows = []
    for s in seeds:
        t0 = time.time()
        auc = nested_cv_auc(X, y, seed=s)
        rows.append({"seed": s, "roc_auc": auc, "is_headline_seed": s == SEED})
        logger.info("seed=%d: AUC=%.4f%s (%.1fs)", s, auc, " [HEADLINE]" if s == SEED else "", time.time() - t0)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "seed_sensitivity.csv", index=False)

    aucs = df["roc_auc"].values
    logger.warning("Seed sensitivity of the headline logreg result across %d full-grid outer/inner "
                    "CV splits (full param grid, same as 03_nested_cv.py): mean=%.4f, std=%.4f, "
                    "range=[%.4f, %.4f]. Headline seed (42) AUC=%.4f. Spread across seeds is %s "
                    "the bootstrap CI half-width reported for the headline result (~0.062), "
                    "suggesting seed choice %s a major driver of the headline number.",
                    N_SEEDS, aucs.mean(), aucs.std(), aucs.min(), aucs.max(),
                    df.loc[df["is_headline_seed"], "roc_auc"].iloc[0],
                    "within" if (aucs.max() - aucs.min()) / 2 <= 0.062 else "larger than",
                    "is not" if (aucs.max() - aucs.min()) / 2 <= 0.062 else "may be")
    logger.info("Wrote results/seed_sensitivity.csv")


if __name__ == "__main__":
    main()
