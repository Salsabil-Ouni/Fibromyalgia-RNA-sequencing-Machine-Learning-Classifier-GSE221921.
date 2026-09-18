"""Alternative univariate scorer: an empirical-Bayes moderated t-test, implementing
the Smyth (2004) algorithm used by limma's eBayes()/squeezeVar() for microarray-style
continuous expression data (not the DESeq2 negative-binomial count model, which
requires raw integer counts we do not have for this dataset; only the author-
processed FPKM matrix is available from GEO for this series).

Model (Smyth 2004, "Linear Models and Empirical Bayes Methods for Assessing
Differential Expression in Microarray Experiments"):
  s_g^2 | sigma_g^2 ~ (sigma_g^2 / d_g) * ChiSq(d_g)          [per-gene pooled
                                                                 within-group variance]
  1 / sigma_g^2      ~ (1 / (d0 * s0^2)) * ChiSq(d0)          [prior across genes]
Posterior mean (moderated variance):
  s_tilde_g^2 = (d0*s0^2 + d_g*s_g^2) / (d0 + d_g)
Moderated t-statistic:
  t_g = mean_diff_g / sqrt(s_tilde_g^2 * (1/n1 + 1/n0))
with d0 + d_g degrees of freedom (d0 -> infinity recovers the ordinary pooled-
variance t-test with no shrinkage; d0 -> 0 means no useful information is
borrowed across genes).

d0 and s0^2 are estimated from the empirical distribution of the per-gene sample
variances s_g^2 via the moment-based method in Smyth (2004) Appendix, solving for
the prior degrees of freedom via the inverse trigamma function.
"""
import numpy as np
from scipy import stats
from scipy.optimize import brentq
from scipy.special import polygamma


def _trigamma_inverse(x):
    """Solve trigamma(y) = x for y > 0 (Smyth 2004's trigammaInverse), elementwise."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.empty_like(x)
    for i, xi in enumerate(x):
        if xi <= 0:
            out[i] = np.inf
            continue
        # trigamma(y) -> infinity as y -> 0, -> 0 as y -> infinity: monotonic decreasing.
        lo, hi = 1e-8, 1.0
        while polygamma(1, hi) > xi:
            hi *= 2
            if hi > 1e8:
                break
        out[i] = brentq(lambda y: polygamma(1, y) - xi, lo, hi, xtol=1e-10, rtol=1e-10)
    return out


def fit_prior_variance(s2, d_g):
    """Smyth (2004) moment estimator of the prior (d0, s0^2) from per-gene pooled
    variances s2 (array, one per gene), given a common residual df d_g (scalar,
    constant across genes for a fixed two-group design with no missing values)."""
    s2 = np.asarray(s2, dtype=float)
    s2 = s2[np.isfinite(s2) & (s2 > 0)]
    e = np.log(s2) - polygamma(0, d_g / 2.0) + np.log(d_g / 2.0)
    e_mean = e.mean()
    e_var = e.var(ddof=1) - polygamma(1, d_g / 2.0)
    if e_var > 0:
        d0 = 2.0 * _trigamma_inverse(e_var)[0]
        s0_sq = np.exp(e_mean + polygamma(0, d0 / 2.0) - np.log(d0 / 2.0))
    else:
        d0 = np.inf
        s0_sq = np.exp(e_mean)
    return d0, s0_sq


def moderated_ttest_score_func(X, y):
    """Return (scores, pvalues) per column: empirical-Bayes moderated t-test,
    vectorized across genes, safe for use inside a sklearn Pipeline (fit only
    ever sees the training fold passed to it)."""
    X = np.asarray(X)
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("moderated_ttest_score_func expects a binary label vector")
    g0 = X[y == classes[0]]
    g1 = X[y == classes[1]]
    n0, n1 = len(g0), len(g1)
    d_g = n0 + n1 - 2

    mean_diff = g1.mean(axis=0) - g0.mean(axis=0)
    ss0 = ((g0 - g0.mean(axis=0)) ** 2).sum(axis=0)
    ss1 = ((g1 - g1.mean(axis=0)) ** 2).sum(axis=0)
    s2 = (ss0 + ss1) / d_g  # per-gene pooled (equal-variance) sample variance

    d0, s0_sq = fit_prior_variance(s2, d_g)
    if np.isinf(d0):
        s2_tilde = s2
        df_total = d_g
    else:
        s2_tilde = (d0 * s0_sq + d_g * s2) / (d0 + d_g)
        df_total = d0 + d_g

    se = np.sqrt(s2_tilde * (1.0 / n0 + 1.0 / n1))
    se = np.where(se > 0, se, np.nan)
    t_stat = mean_diff / se
    pval = 2.0 * stats.t.sf(np.abs(t_stat), df=df_total)
    pval = np.nan_to_num(pval, nan=1.0)
    scores = -np.log10(pval + 1e-300)
    return scores, pval
