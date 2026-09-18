"""
Step 3b — Permutation-label null baseline.

Reruns the *entire* nested-CV pipeline (feature selection refit inside each
outer-training fold, inner CV hyperparameter tuning, everything) on FM/control
labels that have been randomly shuffled, repeated N_PERMUTATIONS times, for
the model family that performed best in 03_nested_cv.py. This is the single
most important sanity check for an n<<p genomics classifier: if the real
model's AUC does not clearly clear this null distribution, the "real" result
is not distinguishable from what you'd get by chance on a dataset this shape.

To keep runtime tractable the permutation runs use a smaller inner grid than
the main analysis (fewer k / hyperparameter combinations) — this is noted
explicitly since it means the null is, if anything, slightly conservative in
favor of the null being narrow (a fully-matched grid per permutation would be
prohibitively slow, but a smaller grid does not know the true label pattern
either, so it does not bias the *location* of the null).

Outputs:
  results/permutation_null_distribution.csv
  figures/permutation_null_distribution.png
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

logger = get_logger("04_permutation_test")
set_global_seed()

N_PERMUTATIONS = 100
N_OUTER = 5
N_INNER = 3  # reduced vs. main analysis, for tractability (see module docstring)

PIPELINE = Pipeline([
    ("select", SelectKBest(score_func=mannwhitney_score_func)),
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(solver="liblinear", max_iter=5000, class_weight="balanced", random_state=SEED)),
])
PARAM_GRID = {"select__k": [50, 200], "clf__C": [0.1, 1], "clf__penalty": ["l2"]}


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

    logger.info("Computing real (non-permuted) nested-CV AUC with the reduced permutation-test grid "
                "(this differs slightly from the full-grid AUC in performance_summary.csv; "
                "used only as the reference point for the null comparison).")
    real_auc = nested_cv_auc(X, y, seed=SEED)
    logger.info("Reduced-grid real AUC = %.4f", real_auc)

    rng = np.random.RandomState(SEED)
    null_aucs = []
    for i in range(N_PERMUTATIONS):
        t0 = time.time()
        y_perm = rng.permutation(y)
        auc = nested_cv_auc(X, y_perm, seed=SEED + i + 1)
        null_aucs.append(auc)
        logger.info("Permutation %d/%d: AUC=%.4f (%.1fs)", i + 1, N_PERMUTATIONS, auc, time.time() - t0)

    null_aucs = np.array(null_aucs)
    p_value = (np.sum(null_aucs >= real_auc) + 1) / (len(null_aucs) + 1)

    out_df = pd.DataFrame({"permutation": range(1, N_PERMUTATIONS + 1), "null_auc": null_aucs})
    out_df.attrs["real_auc"] = real_auc
    out_df.to_csv(RESULTS / "permutation_null_distribution.csv", index=False)
    with open(RESULTS / "permutation_null_distribution.csv", "a") as f:
        f.write(f"# real_auc,{real_auc}\n# empirical_p_value,{p_value}\n"
                f"# null_mean,{null_aucs.mean()}\n# null_std,{null_aucs.std()}\n")

    logger.info("Null AUC distribution: mean=%.4f std=%.4f min=%.4f max=%.4f",
                null_aucs.mean(), null_aucs.std(), null_aucs.min(), null_aucs.max())
    logger.info("Real AUC=%.4f vs. null. Empirical p-value = %.4f (fraction of permutations >= real)",
                real_auc, p_value)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(null_aucs, bins=15, color="0.7", edgecolor="black", label=f"Permuted-label null (n={N_PERMUTATIONS})")
    ax.axvline(real_auc, color="crimson", linewidth=2, label=f"Real model AUC = {real_auc:.3f}")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Chance (AUC=0.5)")
    ax.set_xlabel("ROC-AUC (nested CV, reduced grid)")
    ax.set_ylabel("Count")
    ax.set_title(f"Real model vs. permutation-label null\nempirical p = {p_value:.4f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "permutation_null_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Wrote results/permutation_null_distribution.csv and figures/permutation_null_distribution.png")


if __name__ == "__main__":
    main()
