"""
Step 3 — Nested cross-validation.

Outer 5-fold stratified CV gives an unbiased performance estimate; an inner
5-fold CV (via GridSearchCV) tunes both the classifier hyperparameters AND
the number of univariate-selected features (k), so k is *never* fixed in
advance and is chosen using training-fold information only.

Feature selection (Mann-Whitney univariate score) and standardization are
both wrapped inside an sklearn Pipeline, so both are refit from scratch on
each outer-training fold — the single biggest leakage risk for n<<p genomics
classifiers is avoided by construction, not by discipline.

Two model families are compared: L1/L2-regularized logistic regression and
XGBoost gradient-boosted trees.

Outputs:
  results/performance_summary.csv       per-fold + aggregate metrics (with bootstrap 95% CI)
  results/outer_fold_predictions.csv    pooled out-of-fold predicted probabilities (for reuse
                                         in the permutation test and for later reporting)
"""
import time

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                              confusion_matrix, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import DATA_PROCESSED, RESULTS, SEED, get_logger, log_versions, set_global_seed
from feature_selection import mannwhitney_score_func

logger = get_logger("03_nested_cv")
set_global_seed()
log_versions(logger)

N_OUTER = 5
N_INNER = 5

MODEL_GRIDS = {
    "logreg": {
        "pipeline": Pipeline([
            ("select", SelectKBest(score_func=mannwhitney_score_func)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(solver="liblinear", max_iter=5000, class_weight="balanced", random_state=SEED)),
        ]),
        "param_grid": {
            "select__k": [50, 100, 200],
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__penalty": ["l1", "l2"],
        },
    },
    "xgboost": {
        "pipeline": Pipeline([
            ("select", SelectKBest(score_func=mannwhitney_score_func)),
            ("scale", StandardScaler()),
            ("clf", XGBClassifier(
                objective="binary:logistic", eval_metric="logloss",
                random_state=SEED, n_jobs=1, verbosity=0,
            )),
        ]),
        "param_grid": {
            "select__k": [50, 100, 200],
            "clf__max_depth": [2, 3],
            "clf__learning_rate": [0.05, 0.1],
            "clf__n_estimators": [100, 200],
        },
    },
}


def bootstrap_ci(y_true, y_score, metric_func, n_boot=2000, seed=SEED, **kwargs):
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(metric_func(y_true[idx], y_score[idx], **kwargs))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.mean(vals)), float(lo), float(hi)


def sens_spec_acc(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    acc = accuracy_score(y_true, y_pred)
    return sens, spec, acc


def run_nested_cv(X, y, model_name, spec, logger, label="", sample_ids=None):
    outer_cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)
    inner_cv = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=SEED)

    fold_rows = []
    oof_true, oof_pred_proba, oof_fold, oof_sample_id = [], [], [], []

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        t0 = time.time()
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        search = GridSearchCV(
            spec["pipeline"], spec["param_grid"], scoring="roc_auc",
            cv=inner_cv, n_jobs=-1, refit=True,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_

        proba = best.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        auc = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)
        sens, spec_, acc = sens_spec_acc(y_test, pred)

        fold_rows.append({
            "model": model_name, "outer_fold": fold_i, "n_test": len(test_idx),
            "roc_auc": auc, "pr_auc": pr_auc, "sensitivity": sens, "specificity": spec_,
            "accuracy": acc, "best_params": str(search.best_params_),
        })
        oof_true.extend(y_test.tolist())
        oof_pred_proba.extend(proba.tolist())
        oof_fold.extend([fold_i] * len(test_idx))
        if sample_ids is not None:
            oof_sample_id.extend(np.asarray(sample_ids)[test_idx].tolist())

        logger.info("%s[%s] outer fold %d/%d: AUC=%.3f PR-AUC=%.3f sens=%.3f spec=%.3f acc=%.3f "
                     "best_params=%s (%.1fs)",
                     label, model_name, fold_i + 1, N_OUTER, auc, pr_auc, sens, spec_, acc,
                     search.best_params_, time.time() - t0)

    fold_df = pd.DataFrame(fold_rows)

    # aggregate: mean +/- 95% CI via bootstrap over pooled out-of-fold predictions
    oof_true = np.array(oof_true)
    oof_pred_proba = np.array(oof_pred_proba)
    oof_pred = (oof_pred_proba >= 0.5).astype(int)

    auc_mean, auc_lo, auc_hi = bootstrap_ci(oof_true, oof_pred_proba, roc_auc_score)
    pr_mean, pr_lo, pr_hi = bootstrap_ci(oof_true, oof_pred_proba, average_precision_score)

    def sens_metric(yt, yp):
        return sens_spec_acc(yt, (yp >= 0.5).astype(int))[0]

    def spec_metric(yt, yp):
        return sens_spec_acc(yt, (yp >= 0.5).astype(int))[1]

    def acc_metric(yt, yp):
        return sens_spec_acc(yt, (yp >= 0.5).astype(int))[2]

    sens_mean, sens_lo, sens_hi = bootstrap_ci(oof_true, oof_pred_proba, sens_metric)
    spec_mean, spec_lo, spec_hi = bootstrap_ci(oof_true, oof_pred_proba, spec_metric)
    acc_mean, acc_lo, acc_hi = bootstrap_ci(oof_true, oof_pred_proba, acc_metric)

    agg_row = {
        "model": model_name, "outer_fold": "AGGREGATE (pooled OOF, bootstrap 95% CI)",
        "n_test": len(oof_true),
        "roc_auc": auc_mean, "roc_auc_ci_lo": auc_lo, "roc_auc_ci_hi": auc_hi,
        "pr_auc": pr_mean, "pr_auc_ci_lo": pr_lo, "pr_auc_ci_hi": pr_hi,
        "sensitivity": sens_mean, "sensitivity_ci_lo": sens_lo, "sensitivity_ci_hi": sens_hi,
        "specificity": spec_mean, "specificity_ci_lo": spec_lo, "specificity_ci_hi": spec_hi,
        "accuracy": acc_mean, "accuracy_ci_lo": acc_lo, "accuracy_ci_hi": acc_hi,
        "best_params": "",
    }

    oof_data = {"model": model_name, "outer_fold": oof_fold, "y_true": oof_true, "y_pred_proba": oof_pred_proba}
    if oof_sample_id:
        oof_data["sample_id"] = oof_sample_id
    oof_df = pd.DataFrame(oof_data)
    return fold_df, agg_row, oof_df


def main():
    expr = pd.read_csv(DATA_PROCESSED / "expression_log2_filtered.csv", index_col=0)
    meta = pd.read_csv(DATA_PROCESSED / "sample_metadata_qc.csv", index_col="sample_id")
    meta = meta.reindex(expr.index)

    X = expr.values
    y = (meta["label"].values == "FM").astype(int)
    logger.info("Modeling matrix: X=%s, y positive (FM)=%d, negative (Control)=%d",
                X.shape, y.sum(), (1 - y).sum())

    sample_ids = expr.index.to_numpy()

    all_fold_rows, all_agg_rows, all_oof = [], [], []
    for model_name, spec in MODEL_GRIDS.items():
        logger.info("=== Nested CV: %s ===", model_name)
        fold_df, agg_row, oof_df = run_nested_cv(X, y, model_name, spec, logger, sample_ids=sample_ids)
        all_fold_rows.append(fold_df)
        all_agg_rows.append(agg_row)
        all_oof.append(oof_df)

    perf_df = pd.concat(all_fold_rows + [pd.DataFrame(all_agg_rows)], ignore_index=True)
    perf_df.to_csv(RESULTS / "performance_summary.csv", index=False)
    logger.info("Wrote results/performance_summary.csv")

    oof_all = pd.concat(all_oof, ignore_index=True)
    oof_all.to_csv(RESULTS / "outer_fold_predictions.csv", index=False)
    logger.info("Wrote results/outer_fold_predictions.csv")

    for row in all_agg_rows:
        logger.info("SUMMARY %s: AUC=%.3f [%.3f, %.3f]  PR-AUC=%.3f [%.3f, %.3f]",
                     row["model"], row["roc_auc"], row["roc_auc_ci_lo"], row["roc_auc_ci_hi"],
                     row["pr_auc"], row["pr_auc_ci_lo"], row["pr_auc_ci_hi"])


if __name__ == "__main__":
    main()
