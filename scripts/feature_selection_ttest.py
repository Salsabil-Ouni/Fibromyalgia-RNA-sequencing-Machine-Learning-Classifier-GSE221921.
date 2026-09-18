"""Alternative univariate scorer: Welch's t-test (unequal variance), vectorized
across genes via scipy.stats.ttest_ind(..., axis=0). Used as a robustness check
against the Mann-Whitney U scorer in feature_selection.py — if the nested-CV
AUC is similar under both, the result is not an artifact of picking one
particular univariate test.

Note: this is a plain per-gene Welch t-test, NOT the full limma/DESeq2 empirical-
Bayes variance-moderation procedure (which shrinks each gene's variance estimate
toward a fitted prior across all genes). Implementing true empirical Bayes
moderation was out of scope here; this is disclosed as a limitation rather than
mislabeled as "moderated t-test".
"""
import numpy as np
from scipy import stats


def welch_ttest_score_func(X, y):
    X = np.asarray(X)
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("welch_ttest_score_func expects a binary label vector")
    group0 = X[y == classes[0]]
    group1 = X[y == classes[1]]
    stat, pval = stats.ttest_ind(group0, group1, axis=0, equal_var=False)
    pval = np.nan_to_num(pval, nan=1.0)
    scores = -np.log10(pval + 1e-300)
    return scores, pval
