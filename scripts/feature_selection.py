"""Vectorized Mann-Whitney U univariate feature scorer, safe for use inside a
sklearn Pipeline (fit only ever sees the training fold passed to it)."""
import numpy as np
from scipy import stats


def mannwhitney_score_func(X, y):
    """Return (scores, pvalues) per column, higher score = more differential.

    Vectorized across all genes at once via scipy's axis-wise mannwhitneyu,
    which is why we can afford to do this fresh inside every CV fold instead
    of pre-selecting on the full dataset (that would be leakage).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("mannwhitney_score_func expects a binary label vector")
    group0 = X[y == classes[0]]
    group1 = X[y == classes[1]]
    stat, pval = stats.mannwhitneyu(group0, group1, axis=0, alternative="two-sided", method="asymptotic")
    pval = np.nan_to_num(pval, nan=1.0)
    scores = -np.log10(pval + 1e-300)
    return scores, pval
