"""
Step 3f (follow-up, closing a stated limitation) — Empirical-Bayes moderated
t-test robustness check.

08_featsel_robustness.py already reruns the nested-CV pipeline with a plain
Welch's t-test in place of the Mann-Whitney U score, but that check was
explicitly flagged as "not a full limma/DESeq2 empirical-Bayes moderated test"
because it omits variance shrinkage. This script closes that gap using
feature_selection_moderated_ttest.py, a from-scratch implementation of the
Smyth (2004) empirical-Bayes moderated t-test (the algorithm behind limma's
eBayes()), validated against synthetic data with known homogeneous and
heterogeneous variance structure before use here.

DESeq2 itself (a negative-binomial count model) is not used because only the
author-processed FPKM matrix, not raw integer counts, is available from GEO
for this series; a moderated t-test on log-transformed continuous expression
is the correct empirical-Bayes analogue for this kind of data (this is the
same model limma was originally designed for: microarray-style continuous
intensities, not count data), and both this check and 08 use pooled
(equal-variance) rather than Welch (unequal-variance) per-gene variance
estimates, so this is a genuinely different reference test, not merely a
shrinkage-only refinement of 08.

Outputs:
  results/performance_summary_moderated_ttest_robustness.csv
  results/featsel_moderated_ttest_vs_mannwhitney_diff_test.csv
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
from feature_selection_moderated_ttest import moderated_ttest_score_func

logger = get_logger("19_featsel_moderated_ttest_robustness")
set_global_seed()

N_OUTER, N_INNER = 5, 5

PIPELINE = Pipeline([
    ("select", SelectKBest(score_func=moderated_ttest_score_func)),
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
        logger.info("moderated-t-test-selection outer fold %d/%d: AUC=%.3f PR-AUC=%.3f best_params=%s",
                     fold_i + 1, N_OUTER, auc, pr_auc, search.best_params_)

    oof_true = np.array(oof_true)
    oof_proba = np.array(oof_proba)
    pooled_auc = roc_auc_score(oof_true, oof_proba)
    pooled_pr = average_precision_score(oof_true, oof_proba)

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
    out_df.to_csv(RESULTS / "performance_summary_moderated_ttest_robustness.csv", index=False)

    # Paired bootstrap difference test vs. Mann-Whitney U selection (03_nested_cv.py, logreg),
    # same identical-fold-order argument as in 08_featsel_robustness.py.
    mw_oof = pd.read_csv(RESULTS / "outer_fold_predictions.csv")
    mw_oof = mw_oof[mw_oof["model"] == "logreg"].reset_index(drop=True)
    assert len(mw_oof) == n, "Sample count mismatch between Mann-Whitney and moderated-t-test OOF runs"
    assert (mw_oof["y_true"].values == oof_true).all(), (
        "Fold order mismatch between Mann-Whitney (03_nested_cv.py) and moderated-t-test (19) OOF "
        "runs — paired bootstrap below would be invalid if this ever fails."
    )
    mw_proba = mw_oof["y_pred_proba"].values

    diffs = []
    for _ in range(2000):
        idx = rng.randint(0, n, n)
        if len(np.unique(oof_true[idx])) < 2:
            continue
        auc_mt = roc_auc_score(oof_true[idx], oof_proba[idx])
        auc_mw = roc_auc_score(oof_true[idx], mw_proba[idx])
        diffs.append(auc_mt - auc_mw)
    diff_mean, diff_lo, diff_hi = float(np.mean(diffs)), *np.percentile(diffs, [2.5, 97.5])
    mw_logreg_auc = roc_auc_score(oof_true, mw_proba)

    pd.DataFrame([{
        "moderated_ttest_auc": pooled_auc, "moderated_ttest_auc_ci_lo": ci_lo,
        "moderated_ttest_auc_ci_hi": ci_hi, "mannwhitney_auc": mw_logreg_auc,
        "paired_bootstrap_diff_mean": diff_mean, "paired_bootstrap_diff_ci_lo": diff_lo,
        "paired_bootstrap_diff_ci_hi": diff_hi,
        "diff_ci_excludes_zero": bool(diff_lo > 0 or diff_hi < 0),
    }]).to_csv(RESULTS / "featsel_moderated_ttest_vs_mannwhitney_diff_test.csv", index=False)

    logger.info("Robustness check: moderated t-test feature selection AUC=%.3f (95%% CI %.3f-%.3f) "
                "vs. Mann-Whitney U AUC=%.3f (same logreg model family, same outer/inner CV "
                "structure, identical fold assignments). Paired bootstrap difference "
                "(moderated-t - MW) = %.3f (95%% CI %.3f to %.3f) - %s.",
                pooled_auc, ci_lo, ci_hi, mw_logreg_auc, diff_mean, diff_lo, diff_hi,
                "CI excludes zero: the shift is a real, if modest, effect of the univariate test choice"
                if (diff_lo > 0 or diff_hi < 0) else
                "CI includes zero: the shift is not statistically distinguishable from sampling noise")
    logger.info("Wrote results/performance_summary_moderated_ttest_robustness.csv and "
                "results/featsel_moderated_ttest_vs_mannwhitney_diff_test.csv")


if __name__ == "__main__":
    main()
