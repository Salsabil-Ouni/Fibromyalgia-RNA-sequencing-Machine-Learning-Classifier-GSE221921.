"""
Step 6b (follow-up, prompted by critical review) — Additional sensitivity checks.

1. Female-only preranked GSEA. The whole-cohort GSEA in 18_gsea_pathway_check.py
   compares FM against control while sex is 95% vs 44% female. Repeating the
   identical analysis in females only (91 FM vs 41 control) removes that
   confound structurally.
2. Composition-adjusted female-only GSEA. PBMC cell-type composition can drive
   pathway-level differences (lysosomal and interferon genes are largely
   myeloid). Each gene's expression is residualized on four marker-based
   composition proxy scores (monocyte, T cell, B cell, NK cell) before the
   same ranking and enrichment test. This is a coarse proxy adjustment, not a
   formal deconvolution, and it also removes any real disease signal that is
   correlated with composition, so it is a conservative test.
3. Paired bootstrap comparison of the two model families (logistic regression
   vs XGBoost) on the shared out-of-fold predictions, replacing the informal
   "overlapping confidence intervals" comparison.

Outputs:
  results/gsea_sensitivity_results.csv
  results/composition_proxy_comparison.csv
  results/model_paired_comparison.csv
"""
import importlib

import numpy as np
import pandas as pd
import gseapy as gp
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from config import DATA_PROCESSED, RESULTS, SEED, get_logger, set_global_seed
from feature_selection import mannwhitney_score_func

GENE_SETS = importlib.import_module("06_pathway_crosscheck").GENE_SETS
logger = get_logger("20_additional_sensitivity_checks")
set_global_seed()

MARKERS = {
    "monocyte": ["CD14", "LYZ", "CST3", "FCN1", "CSF1R", "VCAN"],
    "t_cell": ["CD3D", "CD3E", "CD2", "IL7R", "TRAC"],
    "b_cell": ["MS4A1", "CD79A", "CD79B", "CD19"],
    "nk_cell": ["NKG7", "GNLY", "KLRD1", "NCAM1", "KLRF1"],
}


def signed_scores(X, y):
    _, p = mannwhitney_score_func(X, y)
    diff = X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)
    return np.sign(diff) * -np.log10(p + 1e-300)


def run_gsea(scores, columns, symbol_map):
    s = pd.Series(scores, index=columns).to_frame("s").join(symbol_map, how="left")
    s = s.dropna(subset=["Hugo_Gene_Symbol"])
    s["a"] = s["s"].abs()
    s = s.sort_values("a", ascending=False).drop_duplicates("Hugo_Gene_Symbol", keep="first")
    rnk = s.set_index("Hugo_Gene_Symbol")["s"].sort_values(ascending=False).reset_index()
    rnk.columns = ["gene_name", "signed_score"]
    res = gp.prerank(rnk=rnk, gene_sets=GENE_SETS, min_size=1, max_size=1000,
                     permutation_num=1000, seed=SEED, outdir=None, no_plot=True)
    return res.res2d[["Term", "ES", "NES", "NOM p-val", "FDR q-val", "Tag %"]].copy(), len(rnk)


def main():
    expr = pd.read_csv(DATA_PROCESSED / "expression_log2_filtered.csv", index_col=0)
    meta = pd.read_csv(DATA_PROCESSED / "sample_metadata_qc.csv", index_col="sample_id").reindex(expr.index)
    gene_meta = pd.read_csv(DATA_PROCESSED / "gene_metadata.csv")
    symbol_map = gene_meta.set_index("Ensembl_GeneID.version")["Hugo_Gene_Symbol"]
    col_of_symbol = {g: c for c, g in symbol_map.items() if isinstance(g, str)}

    female = (meta["sex"].values == "Female")
    y_f = (meta["label"].values[female] == "FM").astype(int)
    X_f = expr.values[female]
    logger.info("Female-only subset: n=%d (FM=%d, Control=%d)", len(y_f), y_f.sum(), (1 - y_f).sum())

    # ---- 1. female-only GSEA ----
    res1, n_ranked = run_gsea(signed_scores(X_f, y_f), expr.columns, symbol_map)
    res1.insert(0, "analysis", "female_only")
    logger.info("Female-only GSEA ranked %d genes", n_ranked)

    # ---- 2. composition proxy scores and adjusted GSEA ----
    comp = {}
    comp_rows = []
    for name, genes in MARKERS.items():
        cols = [col_of_symbol[g] for g in genes if g in col_of_symbol]
        z = (expr[cols] - expr[cols].mean()) / expr[cols].std(ddof=0)
        comp[name] = z.mean(axis=1).values
        sc = comp[name][female]
        p = mannwhitneyu(sc[y_f == 1], sc[y_f == 0], alternative="two-sided").pvalue
        comp_rows.append({"score": name, "n_markers_present": len(cols),
                          "mean_FM": sc[y_f == 1].mean(), "mean_control": sc[y_f == 0].mean(),
                          "mannwhitney_p": p})
    pd.DataFrame(comp_rows).to_csv(RESULTS / "composition_proxy_comparison.csv", index=False)

    C = np.column_stack([np.ones(female.sum())] + [comp[k][female] for k in MARKERS])
    beta, *_ = np.linalg.lstsq(C, X_f, rcond=None)
    X_resid = X_f - C @ beta
    res2, _ = run_gsea(signed_scores(X_resid, y_f), expr.columns, symbol_map)
    res2.insert(0, "analysis", "female_only_composition_adjusted")

    pd.concat([res1, res2]).to_csv(RESULTS / "gsea_sensitivity_results.csv", index=False)
    for _, r in pd.concat([res1, res2]).iterrows():
        logger.info("%s | %s: NES=%.3f p=%s FDR q=%s leading_edge=%s", r["analysis"], r["Term"],
                    float(r["NES"]), r["NOM p-val"], r["FDR q-val"], r["Tag %"])

    # ---- 2b. composition-only baseline classifier (analogue of the sex-only floor) ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y_all = (meta["label"].values == "FM").astype(int)
    base_rows = []
    for subset_name, mask in (("whole_cohort", np.ones(len(y_all), bool)), ("female_only", female)):
        for feat_name, feats in (("monocyte_score_only", ["monocyte"]), ("four_composition_scores", list(MARKERS))):
            Xc = np.column_stack([comp[k][mask] for k in feats])
            yc = y_all[mask]
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            proba = cross_val_predict(
                make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", random_state=SEED)),
                Xc, yc, cv=cv, method="predict_proba")[:, 1]
            auc = roc_auc_score(yc, proba)
            rng_b = np.random.RandomState(SEED)
            boots = []
            for _ in range(2000):
                idx = rng_b.randint(0, len(yc), len(yc))
                if len(np.unique(yc[idx])) < 2:
                    continue
                boots.append(roc_auc_score(yc[idx], proba[idx]))
            blo, bhi = np.percentile(boots, [2.5, 97.5])
            base_rows.append({"subset": subset_name, "features": feat_name, "n": int(mask.sum()),
                              "cv_auc": auc, "ci_lo": blo, "ci_hi": bhi})
            logger.info("Composition-only baseline [%s, %s]: AUC=%.3f (95%% CI %.3f-%.3f)",
                        subset_name, feat_name, auc, blo, bhi)
    pd.DataFrame(base_rows).to_csv(RESULTS / "composition_only_baseline.csv", index=False)

    # Paired comparison: female-only expression classifier (09_sex_confound_check.py) versus the
    # four-score composition-only baseline, on the same 132 females.
    f_oof = pd.read_csv(RESULTS / "female_only_oof_predictions.csv").set_index("sample_id")
    f_ids = expr.index[female]
    Xc = np.column_stack([comp[k][female] for k in MARKERS])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    comp_proba = cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", random_state=SEED)),
        Xc, y_f, cv=cv, method="predict_proba")[:, 1]
    expr_proba = f_oof.loc[f_ids, "y_pred_proba"].values
    assert (f_oof.loc[f_ids, "y_true"].values == y_f).all()
    rng_p = np.random.RandomState(SEED)
    d_boot = []
    for _ in range(2000):
        idx = rng_p.randint(0, len(y_f), len(y_f))
        if len(np.unique(y_f[idx])) < 2:
            continue
        d_boot.append(roc_auc_score(y_f[idx], expr_proba[idx]) - roc_auc_score(y_f[idx], comp_proba[idx]))
    dlo, dhi = np.percentile(d_boot, [2.5, 97.5])
    pd.DataFrame([{
        "auc_female_only_expression_model": roc_auc_score(y_f, expr_proba),
        "auc_composition_only": roc_auc_score(y_f, comp_proba),
        "paired_diff_mean": float(np.mean(d_boot)), "paired_diff_ci_lo": dlo, "paired_diff_ci_hi": dhi,
        "ci_excludes_zero": bool(dlo > 0 or dhi < 0),
    }]).to_csv(RESULTS / "expression_vs_composition_paired.csv", index=False)
    logger.info("Female-only expression model minus composition-only baseline: paired AUC "
                "difference = %.3f (95%% CI %.3f to %.3f)", np.mean(d_boot), dlo, dhi)

    # ---- 3. paired bootstrap, logistic regression vs XGBoost ----
    oof = pd.read_csv(RESULTS / "outer_fold_predictions.csv")
    lr = oof[oof["model"] == "logreg"].set_index("sample_id")
    xg = oof[oof["model"] == "xgboost"].set_index("sample_id").loc[lr.index]
    y = lr["y_true"].values
    p_lr, p_xg = lr["y_pred_proba"].values, xg["y_pred_proba"].values
    rng = np.random.RandomState(SEED)
    diffs = []
    for _ in range(2000):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], p_xg[idx]) - roc_auc_score(y[idx], p_lr[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    pd.DataFrame([{
        "auc_logreg": roc_auc_score(y, p_lr), "auc_xgboost": roc_auc_score(y, p_xg),
        "paired_diff_mean": float(np.mean(diffs)), "paired_diff_ci_lo": lo, "paired_diff_ci_hi": hi,
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
    }]).to_csv(RESULTS / "model_paired_comparison.csv", index=False)
    logger.info("XGBoost - logreg paired AUC difference = %.3f (95%% CI %.3f to %.3f)",
                np.mean(diffs), lo, hi)


if __name__ == "__main__":
    main()
