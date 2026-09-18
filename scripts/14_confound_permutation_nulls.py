"""
Step 3j (follow-up, prompted by external review) — Permutation nulls for the
two sex-confound-check models.

04_permutation_test.py established a permutation-label null for the main
whole-cohort model, but the two models that are actually load-bearing for
this paper's central claim ("real signal persists after removing the sex
confound") — the female-only model (n=132) and the autosomal-only model
(X/Y genes excluded, n=189) — were only compared informally against the
whole-cohort number and the sex-only floor, with no null of their own. A
reviewer correctly pointed out that a smaller cohort (female-only, n=132)
could plausibly have a different, wider null AUC distribution than the full
cohort, so borrowing the n=189 null as an implicit reference would be a
mistake. This script gives each model its own dedicated null.

Uses the same reduced hyperparameter grid as 04_permutation_test.py, for the
same tractability reasons (logistic regression, inner 3-fold CV,
select__k in {50,200}, C in {0.1,1}, L2 only).

Outputs:
  results/permutation_null_female_only.csv
  results/permutation_null_autosomal_only.csv
  figures/permutation_null_confound_checks.png
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

logger = get_logger("14_confound_permutation_nulls")
set_global_seed()

N_PERMUTATIONS = 100
N_OUTER = 5
N_INNER = 3

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


def run_permutation_suite(X, y, real_auc, label, out_csv, n_perm=N_PERMUTATIONS):
    rng = np.random.RandomState(SEED)
    null_aucs = []
    for i in range(n_perm):
        t0 = time.time()
        y_perm = rng.permutation(y)
        auc = nested_cv_auc(X, y_perm, seed=SEED + i + 1)
        null_aucs.append(auc)
        logger.info("[%s] permutation %d/%d: AUC=%.4f (%.1fs)", label, i + 1, n_perm, auc, time.time() - t0)

    null_aucs = np.array(null_aucs)
    p_value = (np.sum(null_aucs >= real_auc) + 1) / (len(null_aucs) + 1)
    pd.DataFrame({"permutation": range(1, n_perm + 1), "null_auc": null_aucs}).to_csv(out_csv, index=False)
    with open(out_csv, "a") as f:
        f.write(f"# real_auc,{real_auc}\n# empirical_p_value,{p_value}\n"
                f"# null_mean,{null_aucs.mean()}\n# null_std,{null_aucs.std()}\n")
    logger.warning("[%s] null AUC: mean=%.4f std=%.4f min=%.4f max=%.4f | real AUC=%.4f | "
                    "empirical p=%.4f", label, null_aucs.mean(), null_aucs.std(), null_aucs.min(),
                    null_aucs.max(), real_auc, p_value)
    return null_aucs, p_value


def main():
    expr = pd.read_csv(DATA_PROCESSED / "expression_log2_filtered.csv", index_col=0)
    meta = pd.read_csv(DATA_PROCESSED / "sample_metadata_qc.csv", index_col="sample_id")
    meta = meta.reindex(expr.index)
    gene_meta = pd.read_csv(DATA_PROCESSED / "gene_metadata.csv")

    y_all = (meta["label"].values == "FM").astype(int)

    # --- Female-only null ---
    female_mask = (meta["sex"].values == "Female")
    X_f = expr.values[female_mask]
    y_f = y_all[female_mask]
    real_auc_f = nested_cv_auc(X_f, y_f, seed=SEED)
    logger.info("Female-only reduced-grid real AUC = %.4f (n=%d)", real_auc_f, len(y_f))
    null_f, p_f = run_permutation_suite(
        X_f, y_f, real_auc_f, "female-only", RESULTS / "permutation_null_female_only.csv"
    )

    # --- Autosomal-only (X/Y genes excluded) null, whole cohort ---
    chrom_map = gene_meta.set_index("Ensembl_GeneID.version")["Chromosome/scaffold name"].astype(str)
    gene_chrom = chrom_map.reindex(expr.columns)
    sex_chrom_mask = gene_chrom.isin(["X", "Y"])
    X_auto = expr.loc[:, ~sex_chrom_mask.values].values
    real_auc_auto = nested_cv_auc(X_auto, y_all, seed=SEED)
    logger.info("Autosomal-only reduced-grid real AUC = %.4f (n=%d, %d genes)",
                real_auc_auto, len(y_all), X_auto.shape[1])
    null_auto, p_auto = run_permutation_suite(
        X_auto, y_all, real_auc_auto, "autosomal-only", RESULTS / "permutation_null_autosomal_only.csv"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (null_aucs, real_auc, p_value, title) in zip(axes, [
        (null_f, real_auc_f, p_f, f"Female-only (n={len(y_f)})"),
        (null_auto, real_auc_auto, p_auto, f"Autosomal-only, whole cohort (n={len(y_all)})"),
    ]):
        ax.hist(null_aucs, bins=15, color="0.7", edgecolor="black", label=f"Permuted-label null (n={N_PERMUTATIONS})")
        ax.axvline(real_auc, color="crimson", linewidth=2, label=f"Real AUC = {real_auc:.3f}")
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Chance")
        ax.set_xlabel("ROC-AUC (reduced grid)")
        ax.set_ylabel("Count")
        ax.set_title(f"{title}\nempirical p = {p_value:.4f}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "permutation_null_confound_checks.png", dpi=150)
    plt.close(fig)
    logger.info("Wrote results/permutation_null_female_only.csv, "
                "results/permutation_null_autosomal_only.csv, "
                "figures/permutation_null_confound_checks.png")


if __name__ == "__main__":
    main()
