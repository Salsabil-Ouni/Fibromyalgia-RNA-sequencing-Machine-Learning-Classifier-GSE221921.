"""
Step 3d (follow-up) — Feature-selection method robustness check.

03_nested_cv.py uses a Mann-Whitney U univariate score to rank genes inside
each training fold. This reruns the identical nested-CV procedure with a
Welch's t-test score instead (feature_selection_ttest.py) for the logistic-
regression model family, to check whether the ~0.82 AUC is specific to the
choice of univariate test or holds up under a different (but comparably
simple) one. Only logreg is rerun here (not xgboost) since the point is to
isolate the effect of the feature-selection method, not to re-tune every
classifier choice again.

Outputs:
  results/performance_summary_ttest_robustness.csv
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
from feature_selection_ttest import welch_ttest_score_func

logger = get_logger("08_featsel_robustness")
set_global_seed()

N_OUTER, N_INNER = 5, 5

PIPELINE = Pipeline([
    ("select", SelectKBest(score_func=welch_ttest_score_func)),
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(solver="liblinear", max_iter=5000, class_weight="balanced", random_state=SEED)),
])
PARAM_GRID = {"select__k": [50, 100, 200], "clf__C": [0.01, 0.1, 1, 10], "clf__penalty": ["l1", "l2"]}


def main():
    expr = pd.read_csv(DATA_PROCESSED / "expression_log2_filtered.csv", index_col=0)
    meta = pd.read_csv(DATA_PROCESSED / "sample_metadata_qc.csv", index_col="sample_id")
    meta = meta.reindex(expr.index)
    X = expr.values
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
        logger.info("t-test-selection outer fold %d/%d: AUC=%.3f PR-AUC=%.3f best_params=%s",
                     fold_i + 1, N_OUTER, auc, pr_auc, search.best_params_)

    oof_true = np.array(oof_true)
    oof_proba = np.array(oof_proba)
    pooled_auc = roc_auc_score(oof_true, oof_proba)
    pooled_pr = average_precision_score(oof_true, oof_proba)

    # Bootstrap CI for the t-test-selection AUC itself (previously reported as a bare point
    # estimate, which is not apples-to-apples with every other CI in this project).
    rng = np.random.RandomState(SEED)
    n = len(oof_true)
    boot_aucs = []
    for _ in range(2000):
        idx = rng.randint(0, n, n)
        if len(np.unique(oof_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(oof_true[idx], oof_proba[idx]))
    ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])

    rows.append({"outer_fold": "AGGREGATE (pooled OOF, bootstrap 95% CI)", "roc_auc": pooled_auc,
                 "pr_auc": pooled_pr, "best_params": "", "roc_auc_ci_lo": ci_lo, "roc_auc_ci_hi": ci_hi})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(RESULTS / "performance_summary_ttest_robustness.csv", index=False)

    # Paired bootstrap difference test vs. Mann-Whitney U selection (03_nested_cv.py, logreg).
    # StratifiedKFold(y, n_splits, shuffle, random_state) depends only on y/n_splits/seed, not
    # on X — so the outer-fold sample order here is IDENTICAL, position-for-position, to the
    # Mann-Whitney run in 03_nested_cv.py (same y, same SEED, same N_OUTER, same shuffle). This
    # lets us do a genuinely paired bootstrap on the AUC difference, resampling the same
    # positions for both, rather than treating them as two independent unpaired estimates.
    mw_oof = pd.read_csv(RESULTS / "outer_fold_predictions.csv")
    mw_oof = mw_oof[mw_oof["model"] == "logreg"].reset_index(drop=True)
    assert len(mw_oof) == n, "Sample count mismatch between Mann-Whitney and t-test OOF runs"
    assert (mw_oof["y_true"].values == oof_true).all(), (
        "Fold order mismatch between Mann-Whitney (03_nested_cv.py) and t-test (08) OOF runs — "
        "paired bootstrap below would be invalid if this ever fails."
    )
    mw_proba = mw_oof["y_pred_proba"].values

    diffs = []
    for _ in range(2000):
        idx = rng.randint(0, n, n)
        if len(np.unique(oof_true[idx])) < 2:
            continue
        auc_tt = roc_auc_score(oof_true[idx], oof_proba[idx])
        auc_mw = roc_auc_score(oof_true[idx], mw_proba[idx])
        diffs.append(auc_tt - auc_mw)
    diff_mean, diff_lo, diff_hi = float(np.mean(diffs)), *np.percentile(diffs, [2.5, 97.5])
    mw_logreg_auc = roc_auc_score(oof_true, mw_proba)

    pd.DataFrame([{
        "ttest_auc": pooled_auc, "ttest_auc_ci_lo": ci_lo, "ttest_auc_ci_hi": ci_hi,
        "mannwhitney_auc": mw_logreg_auc,
        "paired_bootstrap_diff_mean": diff_mean, "paired_bootstrap_diff_ci_lo": diff_lo,
        "paired_bootstrap_diff_ci_hi": diff_hi,
        "diff_ci_excludes_zero": bool(diff_lo > 0 or diff_hi < 0),
    }]).to_csv(RESULTS / "featsel_ttest_vs_mannwhitney_diff_test.csv", index=False)

    logger.info("Robustness check: Welch t-test feature selection AUC=%.3f (95%% CI %.3f-%.3f) vs. "
                "Mann-Whitney U AUC=%.3f (same logreg model family, same outer/inner CV structure, "
                "identical fold assignments). Paired bootstrap difference (t-test - MW) = %.3f "
                "(95%% CI %.3f to %.3f) — %s.",
                pooled_auc, ci_lo, ci_hi, mw_logreg_auc, diff_mean, diff_lo, diff_hi,
                "CI excludes zero: the shift is a real, if modest, effect of the univariate test choice"
                if (diff_lo > 0 or diff_hi < 0) else
                "CI includes zero: the shift is not statistically distinguishable from sampling noise")
    logger.info("Wrote results/performance_summary_ttest_robustness.csv and "
                "results/featsel_ttest_vs_mannwhitney_diff_test.csv")


if __name__ == "__main__":
    main()
