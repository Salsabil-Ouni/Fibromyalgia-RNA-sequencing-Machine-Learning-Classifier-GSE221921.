"""
Step 3h (follow-up) — Learning curve: is performance sample-size-limited?

Answers a question the limitations section otherwise has to hand-wave: would
more samples likely help, or has this classifier roughly plateaued at n=189?
For a set of training-fraction values, we take the SAME outer stratified
5-fold split as the main analysis (test fold composition/size is always
held fixed) but subsample only the training portion of each fold down to a
given fraction (stratified), refit the identical Mann-Whitney-select + scale
+ logistic-regression pipeline with a small inner-CV grid, and evaluate on
the untouched test fold. Repeated over multiple random subsamples per
fraction (different outer-CV seeds) to get a spread, not just one lucky/
unlucky sample.

Logistic regression only (not xgboost) for speed and because it was already
the better-calibrated model (see 10_calibration_check.py).

Outputs:
  results/learning_curve.csv
  figures/learning_curve.png
"""
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DATA_PROCESSED, FIGURES, RESULTS, SEED, get_logger, set_global_seed
from feature_selection import mannwhitney_score_func

logger = get_logger("12_learning_curve")
set_global_seed()

FRACTIONS = [0.3, 0.5, 0.7, 0.85, 1.0]
N_REPEATS = 5
N_OUTER, N_INNER = 5, 3

PIPELINE = Pipeline([
    ("select", SelectKBest(score_func=mannwhitney_score_func)),
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(solver="liblinear", max_iter=5000, class_weight="balanced", random_state=SEED)),
])
PARAM_GRID = {"select__k": [50, 100, 200], "clf__C": [0.1, 1], "clf__penalty": ["l2"]}


def subsample_stratified(train_idx, y, fraction, rng):
    if fraction >= 1.0:
        return train_idx
    keep = []
    for cls in np.unique(y[train_idx]):
        cls_idx = train_idx[y[train_idx] == cls]
        n_keep = max(2, int(round(len(cls_idx) * fraction)))
        keep.extend(rng.choice(cls_idx, size=n_keep, replace=False))
    return np.array(keep)


def main():
    expr = pd.read_csv(DATA_PROCESSED / "expression_log2_filtered.csv", index_col=0)
    meta = pd.read_csv(DATA_PROCESSED / "sample_metadata_qc.csv", index_col="sample_id")
    meta = meta.reindex(expr.index)
    X = expr.values
    y = (meta["label"].values == "FM").astype(int)

    rows = []
    for fraction in FRACTIONS:
        fold_aucs = []
        t0 = time.time()
        for rep in range(N_REPEATS):
            seed = SEED + rep
            outer_cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=seed)
            inner_cv = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=seed)
            rng = np.random.RandomState(seed)
            for train_idx, test_idx in outer_cv.split(X, y):
                sub_train_idx = subsample_stratified(train_idx, y, fraction, rng)
                search = GridSearchCV(PIPELINE, PARAM_GRID, scoring="roc_auc", cv=inner_cv, n_jobs=-1)
                search.fit(X[sub_train_idx], y[sub_train_idx])
                proba = search.best_estimator_.predict_proba(X[test_idx])[:, 1]
                fold_aucs.append(roc_auc_score(y[test_idx], proba))
        fold_aucs = np.array(fold_aucs)
        n_train_approx = int(round(fraction * (len(y) - len(y) // N_OUTER)))
        rows.append({
            "train_fraction": fraction, "approx_n_train": n_train_approx,
            "n_fold_evals": len(fold_aucs), "auc_mean": fold_aucs.mean(),
            "auc_std": fold_aucs.std(), "auc_ci_lo": np.percentile(fold_aucs, 2.5),
            "auc_ci_hi": np.percentile(fold_aucs, 97.5),
        })
        logger.info("fraction=%.2f (~%d train samples): AUC=%.3f +/- %.3f over %d fold-evals (%.1fs)",
                     fraction, n_train_approx, fold_aucs.mean(), fold_aucs.std(), len(fold_aucs), time.time() - t0)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(RESULTS / "learning_curve.csv", index=False)
    logger.info("Wrote results/learning_curve.csv")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(out_df["approx_n_train"], out_df["auc_mean"], yerr=out_df["auc_std"],
                marker="o", capsize=4, color="steelblue")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance")
    ax.set_xlabel("Approx. number of training samples (per outer fold)")
    ax.set_ylabel("ROC-AUC on held-out fold")
    ax.set_title(f"Learning curve (logreg, {N_REPEATS} repeats x {N_OUTER}-fold CV per point)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "learning_curve.png", dpi=150)
    plt.close(fig)
    logger.info("Wrote figures/learning_curve.png")


if __name__ == "__main__":
    main()
