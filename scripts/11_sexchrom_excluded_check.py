"""
Step 3g (follow-up) — Sex-chromosome-excluded reanalysis.

Complements 09_sex_confound_check.py's female-only subset test with a second,
different way of probing the same sex confound: keep the full (sex-imbalanced)
cohort, but remove every X- and Y-linked gene from the candidate feature pool
before feature selection, so the model cannot use direct sex-chromosome
dosage differences at all. If AUC holds up close to the whole-cohort number,
that argues the signal is not simply "detecting sex" via sex-chromosome
genes. If it drops sharply, sex-chromosome genes were doing a lot of the
work. Note this does NOT remove autosomal genes that are differentially
expressed *because of* sex (hormonal/other indirect effects) — the
female-only subset in 09 is the stronger, structural control for that; this
script targets a narrower, complementary mechanism (direct sex-chromosome
dosage).

Outputs:
  results/performance_summary_autosomal_only.csv
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DATA_PROCESSED, RESULTS, SEED, get_logger, set_global_seed
from feature_selection import mannwhitney_score_func

logger = get_logger("11_sexchrom_excluded_check")
set_global_seed()

N_OUTER, N_INNER = 5, 5
PIPELINE = Pipeline([
    ("select", SelectKBest(score_func=mannwhitney_score_func)),
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(solver="liblinear", max_iter=5000, class_weight="balanced", random_state=SEED)),
])
PARAM_GRID = {"select__k": [50, 100, 200], "clf__C": [0.01, 0.1, 1, 10], "clf__penalty": ["l1", "l2"]}


def main():
    expr = pd.read_csv(DATA_PROCESSED / "expression_log2_filtered.csv", index_col=0)
    meta = pd.read_csv(DATA_PROCESSED / "sample_metadata_qc.csv", index_col="sample_id")
    meta = meta.reindex(expr.index)
    gene_meta = pd.read_csv(DATA_PROCESSED / "gene_metadata.csv")

    chrom_map = gene_meta.set_index("Ensembl_GeneID.version")["Chromosome/scaffold name"].astype(str)
    gene_chrom = chrom_map.reindex(expr.columns)
    sex_chrom_mask = gene_chrom.isin(["X", "Y"])
    n_sex_chrom = int(sex_chrom_mask.sum())
    logger.info("Excluding %d / %d genes on chromosomes X or Y from the candidate feature pool "
                "(%d remain, autosomal only).", n_sex_chrom, expr.shape[1], (~sex_chrom_mask).sum())

    expr_autosomal = expr.loc[:, ~sex_chrom_mask.values]
    X = expr_autosomal.values
    y = (meta["label"].values == "FM").astype(int)

    outer_cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)
    inner_cv = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=SEED)
    rows, oof_true, oof_proba = [], [], []
    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        search = GridSearchCV(PIPELINE, PARAM_GRID, scoring="roc_auc", cv=inner_cv, n_jobs=-1)
        search.fit(X[train_idx], y[train_idx])
        proba = search.best_estimator_.predict_proba(X[test_idx])[:, 1]
        auc = roc_auc_score(y[test_idx], proba)
        pr_auc = average_precision_score(y[test_idx], proba)
        rows.append({"outer_fold": fold_i, "roc_auc": auc, "pr_auc": pr_auc, "best_params": str(search.best_params_)})
        oof_true.extend(y[test_idx].tolist())
        oof_proba.extend(proba.tolist())
        logger.info("[autosomal-only] outer fold %d/%d: AUC=%.3f PR-AUC=%.3f best_params=%s",
                     fold_i + 1, N_OUTER, auc, pr_auc, search.best_params_)

    oof_true_arr, oof_proba_arr = np.array(oof_true), np.array(oof_proba)
    pooled_auc = roc_auc_score(oof_true_arr, oof_proba_arr)
    pooled_pr = average_precision_score(oof_true_arr, oof_proba_arr)

    rng = np.random.RandomState(SEED)
    boot_aucs = []
    for _ in range(2000):
        idx = rng.randint(0, len(oof_true_arr), len(oof_true_arr))
        if len(np.unique(oof_true_arr[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(oof_true_arr[idx], oof_proba_arr[idx]))
    ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])

    rows.append({"outer_fold": "AGGREGATE (pooled OOF, bootstrap 95% CI)", "roc_auc": pooled_auc,
                 "pr_auc": pooled_pr, "best_params": "", "roc_auc_ci_lo": ci_lo, "roc_auc_ci_hi": ci_hi})
    pd.DataFrame(rows).to_csv(RESULTS / "performance_summary_autosomal_only.csv", index=False)

    whole_cohort_auc = pd.read_csv(RESULTS / "performance_summary.csv").pipe(
        lambda df: df.loc[(df["model"] == "logreg") & (df["outer_fold"].astype(str).str.startswith("AGGREGATE")), "roc_auc"].iloc[0]
    )
    logger.warning("Autosomal-only (X/Y genes excluded) whole-cohort nested-CV AUC = %.3f [%.3f, %.3f] "
                    "vs. full-genome whole-cohort logreg AUC = %.3f. Difference = %.3f.",
                    pooled_auc, ci_lo, ci_hi, whole_cohort_auc, whole_cohort_auc - pooled_auc)
    logger.info("Wrote results/performance_summary_autosomal_only.csv")


if __name__ == "__main__":
    main()
