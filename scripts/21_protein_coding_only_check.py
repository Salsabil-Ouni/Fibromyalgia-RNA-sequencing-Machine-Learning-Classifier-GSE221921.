"""
Step 3h (follow-up, reviewer-requested) — Protein-coding-genes-only sensitivity check.

Several of the top-SHAP-ranked genes (main text, Section 3.6) are pseudogenes or
noncoding transcripts (e.g. MIR1281, TTC3P1, OR2L1P, SAGE4P, NAPSB), which are
prone to multi-mapping artifacts in RNA-seq quantification. This script tests
directly whether the nested-CV performance estimate depends on these
artifact-prone features, by restricting the candidate feature pool to genes
with an Ensembl biotype of "protein_coding" (per data/processed/gene_biotypes.csv,
fetched from the Ensembl REST API; see fetch_gene_biotypes.py) before feature
selection, and rerunning the identical nested-CV pipeline used for the
whole-cohort headline result (03_nested_cv.py) and the X/Y-exclusion check
(11_sexchrom_excluded_check.py).

Outputs:
  results/performance_summary_protein_coding_only.csv
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

logger = get_logger("21_protein_coding_only_check")
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
    biotypes = pd.read_csv(DATA_PROCESSED / "gene_biotypes.csv")

    biotype_map = biotypes.set_index("Ensembl_GeneID.version")["biotype"]
    gene_biotype = biotype_map.reindex(expr.columns)
    n_missing = int(gene_biotype.isna().sum())
    protein_coding_mask = (gene_biotype == "protein_coding").values
    n_protein_coding = int(protein_coding_mask.sum())
    logger.info("Biotype resolved for %d / %d genes (%d unresolved, e.g. retired Ensembl IDs); "
                "%d / %d genes are protein_coding and retained as the candidate pool.",
                expr.shape[1] - n_missing, expr.shape[1], n_missing, n_protein_coding, expr.shape[1])

    expr_pc = expr.loc[:, protein_coding_mask]
    X = expr_pc.values
    y = (meta["label"].values == "FM").astype(int)

    sample_ids = expr.index.to_numpy()
    outer_cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)
    inner_cv = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=SEED)
    rows, oof_true, oof_proba, oof_sample_id = [], [], [], []
    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        search = GridSearchCV(PIPELINE, PARAM_GRID, scoring="roc_auc", cv=inner_cv, n_jobs=-1)
        search.fit(X[train_idx], y[train_idx])
        proba = search.best_estimator_.predict_proba(X[test_idx])[:, 1]
        auc = roc_auc_score(y[test_idx], proba)
        pr_auc = average_precision_score(y[test_idx], proba)
        rows.append({"outer_fold": fold_i, "roc_auc": auc, "pr_auc": pr_auc, "best_params": str(search.best_params_)})
        oof_true.extend(y[test_idx].tolist())
        oof_proba.extend(proba.tolist())
        oof_sample_id.extend(sample_ids[test_idx].tolist())
        logger.info("[protein-coding-only] outer fold %d/%d: AUC=%.3f PR-AUC=%.3f best_params=%s",
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
    pd.DataFrame(rows).to_csv(RESULTS / "performance_summary_protein_coding_only.csv", index=False)

    whole_cohort = pd.read_csv(RESULTS / "performance_summary.csv")
    whole_cohort_auc = whole_cohort.pipe(
        lambda df: df.loc[(df["model"] == "logreg") & (df["outer_fold"].astype(str).str.startswith("AGGREGATE")), "roc_auc"].iloc[0]
    )

    # Paired bootstrap difference vs. the whole-cohort (all-genes) logistic-regression model,
    # aligned explicitly by sample_id (both runs share the same outer-CV scheme/seed, but
    # merging by ID rather than assuming row order is the safer, verifiable approach).
    full_oof = pd.read_csv(RESULTS / "outer_fold_predictions.csv")
    full_oof_logreg = full_oof[full_oof["model"] == "logreg"].set_index("sample_id")
    pc_oof = pd.DataFrame({"sample_id": oof_sample_id, "y_true": oof_true, "proba_pc": oof_proba}).set_index("sample_id")
    merged = pc_oof.join(full_oof_logreg[["y_pred_proba"]], how="inner").rename(columns={"y_pred_proba": "proba_full"})
    if len(merged) == len(pc_oof):
        y_true_m = merged["y_true"].to_numpy()
        proba_pc_m = merged["proba_pc"].to_numpy()
        proba_full_m = merged["proba_full"].to_numpy()
        diffs = []
        rng2 = np.random.RandomState(SEED)
        n = len(y_true_m)
        for _ in range(2000):
            idx = rng2.randint(0, n, n)
            if len(np.unique(y_true_m[idx])) < 2:
                continue
            auc_pc = roc_auc_score(y_true_m[idx], proba_pc_m[idx])
            auc_full = roc_auc_score(y_true_m[idx], proba_full_m[idx])
            diffs.append(auc_pc - auc_full)
        d_lo, d_hi = np.percentile(diffs, [2.5, 97.5])
        logger.info("Paired bootstrap AUC difference (protein-coding-only minus full-genome), "
                    "aligned by sample_id (n=%d) = %.3f (95%% CI %.3f to %.3f)",
                    n, pooled_auc - whole_cohort_auc, d_lo, d_hi)
    else:
        logger.warning("Could not align protein-coding-only OOF predictions with the full-genome "
                        "logreg OOF predictions by sample_id (%d vs %d samples); paired bootstrap "
                        "comparison skipped.", len(pc_oof), len(merged))

    logger.warning("Protein-coding-only whole-cohort nested-CV AUC = %.3f [%.3f, %.3f] "
                    "vs. full-genome whole-cohort logreg AUC = %.3f. Difference = %.3f.",
                    pooled_auc, ci_lo, ci_hi, whole_cohort_auc, pooled_auc - whole_cohort_auc)
    logger.info("Wrote results/performance_summary_protein_coding_only.csv")


if __name__ == "__main__":
    main()
