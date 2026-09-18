"""
Step 3e (follow-up, prompted by a discovered confound) — Sex-confound check.

Discovered late: FM/Control are severely sex-imbalanced in this cohort
(91/96 FM = female [95%] vs. 41/93 Control = female [44%], chi-square
p=1.0e-13). A sex-only logistic-regression classifier (no gene expression at
all) already achieves AUC=0.73 on this data. That is uncomfortably close to
the ~0.82 nested-CV AUC of the full gene-expression model, which means part
(possibly a large part) of the classifier's apparent performance could be
picking up sex-linked expression differences rather than fibromyalgia
biology. This is the single biggest threat to validity found in this
analysis and is treated here as a first-class result, not a footnote.

This script:
  1. Records the sex/label contingency table and its association test.
  2. Reports the standalone sex-only classifier AUC as the "floor" a
     gene-expression model must clear to add real information.
  3. Reruns the IDENTICAL nested-CV pipeline (Mann-Whitney selection inside
     each training fold, same inner/outer CV structure, logistic-regression
     model family) restricted to a FEMALE-ONLY subset (91 FM vs. 41 Control,
     n=132) — this removes the sex confound structurally rather than
     statistically, at the cost of a smaller and now class-imbalanced sample.
     If AUC holds up in this subset, that is evidence of real FM signal
     beyond sex composition. If it collapses toward 0.5, the whole-cohort
     result should be re-read as substantially confounded.

Outputs:
  results/sex_confound_summary.csv
  results/performance_summary_female_only.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                      cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DATA_PROCESSED, RESULTS, SEED, get_logger, set_global_seed
from feature_selection import mannwhitney_score_func

logger = get_logger("09_sex_confound_check")
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

    # --- 1. sex/label association ---
    ct = pd.crosstab(meta["label"], meta["sex"])
    chi2, chi2_p, dof, _ = chi2_contingency(ct)
    logger.warning("Sex/label contingency table:\n%s", ct.to_string())
    logger.warning("Sex is NOT balanced between FM and Control (chi2=%.4f, p=%.2e). This is a "
                    "genuine confound risk for any classifier trained on the whole cohort.", chi2, chi2_p)

    # --- 2. sex-only baseline classifier ---
    y_all = (meta["label"].values == "FM").astype(int)
    sex_feature = (meta["sex"].values == "Female").astype(int).reshape(-1, 1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    sex_only_proba = cross_val_predict(
        LogisticRegression(class_weight="balanced", random_state=SEED), sex_feature, y_all,
        cv=cv, method="predict_proba",
    )[:, 1]
    sex_only_auc = roc_auc_score(y_all, sex_only_proba)

    # Bootstrap CI for the sex-only baseline (previously reported as a bare point estimate,
    # which was not apples-to-apples with the nested-CV numbers it's being compared against).
    rng_sex = np.random.RandomState(SEED)
    n_all = len(y_all)
    sex_boot_aucs = []
    for _ in range(2000):
        idx = rng_sex.randint(0, n_all, n_all)
        if len(np.unique(y_all[idx])) < 2:
            continue
        sex_boot_aucs.append(roc_auc_score(y_all[idx], sex_only_proba[idx]))
    sex_ci_lo, sex_ci_hi = np.percentile(sex_boot_aucs, [2.5, 97.5])

    logger.warning("Sex-ONLY classifier (no gene expression) AUC = %.3f (95%% CI %.3f-%.3f, plain "
                    "5-fold CV, no inner tuning loop — simpler estimation than the nested-CV "
                    "numbers it's compared against, so treat this CI as approximate). The full "
                    "gene-expression model's nested-CV AUC (~0.82-0.86, see performance_summary.csv) "
                    "must be read relative to this floor, not relative to chance (0.5).",
                    sex_only_auc, sex_ci_lo, sex_ci_hi)

    # --- 3. female-only nested CV re-run ---
    female_mask = (meta["sex"].values == "Female")
    X_f = expr.values[female_mask]
    y_f = y_all[female_mask]
    sample_ids_f = expr.index.to_numpy()[female_mask]
    logger.info("Female-only subset: n=%d (FM=%d, Control=%d)", len(y_f), y_f.sum(), (1 - y_f).sum())

    outer_cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)
    inner_cv = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=SEED)
    rows, oof_true, oof_proba, oof_sample_id = [], [], [], []
    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X_f, y_f)):
        search = GridSearchCV(PIPELINE, PARAM_GRID, scoring="roc_auc", cv=inner_cv, n_jobs=-1)
        search.fit(X_f[train_idx], y_f[train_idx])
        proba = search.best_estimator_.predict_proba(X_f[test_idx])[:, 1]
        auc = roc_auc_score(y_f[test_idx], proba)
        pr_auc = average_precision_score(y_f[test_idx], proba)
        rows.append({"outer_fold": fold_i, "n_test": len(test_idx), "roc_auc": auc, "pr_auc": pr_auc,
                     "best_params": str(search.best_params_)})
        oof_true.extend(y_f[test_idx].tolist())
        oof_proba.extend(proba.tolist())
        oof_sample_id.extend(sample_ids_f[test_idx].tolist())
        logger.info("[female-only] outer fold %d/%d: AUC=%.3f PR-AUC=%.3f best_params=%s",
                     fold_i + 1, N_OUTER, auc, pr_auc, search.best_params_)

    pooled_auc = roc_auc_score(oof_true, oof_proba)
    pooled_pr = average_precision_score(oof_true, oof_proba)

    rng = np.random.RandomState(SEED)
    oof_true_arr, oof_proba_arr = np.array(oof_true), np.array(oof_proba)
    boot_aucs = []
    for _ in range(2000):
        idx = rng.randint(0, len(oof_true_arr), len(oof_true_arr))
        if len(np.unique(oof_true_arr[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(oof_true_arr[idx], oof_proba_arr[idx]))
    ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])

    rows.append({"outer_fold": "AGGREGATE (pooled OOF, bootstrap 95% CI)", "n_test": len(oof_true),
                 "roc_auc": pooled_auc, "pr_auc": pooled_pr, "best_params": "",
                 "roc_auc_ci_lo": ci_lo, "roc_auc_ci_hi": ci_hi})
    pd.DataFrame(rows).to_csv(RESULTS / "performance_summary_female_only.csv", index=False)
    pd.DataFrame({
        "sample_id": oof_sample_id, "y_true": oof_true, "y_pred_proba": oof_proba,
    }).to_csv(RESULTS / "female_only_oof_predictions.csv", index=False)

    logger.warning("Female-only nested-CV AUC = %.3f [%.3f, %.3f] (n=%d, FM=%d/Control=%d) "
                    "vs. whole-cohort AUC ~0.82 (confounded by sex) and sex-only-classifier floor "
                    "of %.3f.", pooled_auc, ci_lo, ci_hi, len(y_f), y_f.sum(), (1 - y_f).sum(), sex_only_auc)

    summary = {
        "n_total": len(meta), "n_fm": int(y_all.sum()), "n_control": int((1 - y_all).sum()),
        "pct_female_fm": float((meta.loc[meta["label"] == "FM", "sex"] == "Female").mean()),
        "pct_female_control": float((meta.loc[meta["label"] == "Control", "sex"] == "Female").mean()),
        "chi2_sex_vs_label_statistic": chi2,
        "chi2_sex_vs_label_pvalue": chi2_p,
        "sex_only_classifier_auc": sex_only_auc,
        "sex_only_classifier_auc_ci_lo": sex_ci_lo,
        "sex_only_classifier_auc_ci_hi": sex_ci_hi,
        "whole_cohort_logreg_auc_from_03": pd.read_csv(RESULTS / "performance_summary.csv").pipe(
            lambda df: df.loc[(df["model"] == "logreg") & (df["outer_fold"].astype(str).str.startswith("AGGREGATE")), "roc_auc"].iloc[0]
        ),
        "female_only_nested_cv_auc": pooled_auc,
        "female_only_nested_cv_auc_ci_lo": ci_lo,
        "female_only_nested_cv_auc_ci_hi": ci_hi,
        "n_female_only": int(len(y_f)), "n_female_only_fm": int(y_f.sum()), "n_female_only_control": int((1 - y_f).sum()),
    }
    pd.DataFrame([summary]).to_csv(RESULTS / "sex_confound_summary.csv", index=False)
    logger.info("Wrote results/sex_confound_summary.csv and results/performance_summary_female_only.csv")


if __name__ == "__main__":
    main()
