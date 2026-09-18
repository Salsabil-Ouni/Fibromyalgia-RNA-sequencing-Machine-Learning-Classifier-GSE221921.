"""
Step 3c (follow-up) — Feature-selection stability.

The main nested-CV run (03_nested_cv.py) already refits feature selection
inside every outer training fold, which prevents leakage — but it says
nothing about whether the *same* genes get selected across folds/reruns. A
classifier can have a stable, real AUC while its univariate-selected gene set
is fold-to-fold noisy on n=189 samples; that would mean the specific "top
genes" reported in results/shap_feature_importance.csv are less trustworthy
than the AUC itself. This script checks that directly.

Method: repeat stratified 5-fold outer CV R times with different random
splits (different seeds). In each of the R*5 = 50 fold fits, refit the same
Mann-Whitney-select + scale + logistic-regression pipeline (logreg chosen for
speed; feature selection is the object under test, not the classifier), with
a small inner-CV grid over k and C. Record which genes were selected (i.e.
survived SelectKBest) in each fold.

Outputs:
  results/feature_selection_stability.csv   per-gene selection frequency across all R*5 folds
  results/feature_stability_summary.csv     scalar summary stats (mean pairwise Jaccard, etc.)
  figures/feature_stability.png             histogram of pairwise fold-to-fold Jaccard overlap
"""
import itertools
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DATA_PROCESSED, FIGURES, RESULTS, SEED, get_logger, set_global_seed
from feature_selection import mannwhitney_score_func

logger = get_logger("07_feature_stability")
set_global_seed()

N_REPEATS = 10
N_OUTER = 5
N_INNER = 3

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
    X = expr.values
    y = (meta["label"].values == "FM").astype(int)
    gene_ids = expr.columns.to_numpy()

    selected_sets = []  # list of sets of gene ids, one per fold across all repeats
    gene_selection_count = pd.Series(0, index=gene_ids, dtype=int)
    n_total_folds = N_REPEATS * N_OUTER

    for rep in range(N_REPEATS):
        outer_seed = SEED + rep
        outer_cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=outer_seed)
        inner_cv = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=outer_seed)
        t0 = time.time()
        for fold_i, (train_idx, _) in enumerate(outer_cv.split(X, y)):
            search = GridSearchCV(PIPELINE, PARAM_GRID, scoring="roc_auc", cv=inner_cv, n_jobs=-1)
            search.fit(X[train_idx], y[train_idx])
            selector = search.best_estimator_.named_steps["select"]
            genes = set(gene_ids[selector.get_support()])
            selected_sets.append(genes)
            gene_selection_count.loc[list(genes)] += 1
        logger.info("Repeat %d/%d done (%.1fs)", rep + 1, N_REPEATS, time.time() - t0)

    # pairwise Jaccard overlap between every pair of fold-selected gene sets
    jaccards = []
    for a, b in itertools.combinations(selected_sets, 2):
        union = len(a | b)
        inter = len(a & b)
        jaccards.append(inter / union if union else 0.0)
    jaccards = np.array(jaccards)

    freq_df = (gene_selection_count / n_total_folds).sort_values(ascending=False).reset_index()
    freq_df.columns = ["gene_ensembl_id", "selection_frequency"]
    gene_meta = pd.read_csv(DATA_PROCESSED / "gene_metadata.csv")
    freq_df = freq_df.merge(
        gene_meta[["Ensembl_GeneID.version", "Hugo_Gene_Symbol"]],
        left_on="gene_ensembl_id", right_on="Ensembl_GeneID.version", how="left",
    ).drop(columns=["Ensembl_GeneID.version"])
    freq_df = freq_df[freq_df["selection_frequency"] > 0][["gene_ensembl_id", "Hugo_Gene_Symbol", "selection_frequency"]]
    freq_df.to_csv(RESULTS / "feature_selection_stability.csv", index=False)

    # cross-reference against the final SHAP top-30 genes from 05_shap_explain.py
    shap_path = RESULTS / "shap_feature_importance.csv"
    overlap_note = "shap_feature_importance.csv not found; run 05_shap_explain.py first"
    if shap_path.exists():
        shap_df = pd.read_csv(shap_path)
        top30_shap = set(shap_df.head(30)["gene_ensembl_id"])
        top30_freq = freq_df[freq_df["gene_ensembl_id"].isin(top30_shap)]
        mean_freq_of_shap_top30 = top30_freq["selection_frequency"].mean() if len(top30_freq) else 0.0
        overall_mean_freq = freq_df["selection_frequency"].mean()
        overlap_note = (
            f"Mean selection frequency of the SHAP top-30 genes = {mean_freq_of_shap_top30:.3f} "
            f"(vs. {overall_mean_freq:.3f} mean over all genes ever selected); "
            f"{top30_freq['selection_frequency'].ge(0.5).sum()}/30 of the SHAP top genes were "
            f"selected in >=50% of the {n_total_folds} stability folds."
        )
        logger.info(overlap_note)

    summary = {
        "n_repeats": N_REPEATS, "n_outer_folds_per_repeat": N_OUTER, "n_total_folds": n_total_folds,
        "n_unique_genes_ever_selected": int((gene_selection_count > 0).sum()),
        "mean_pairwise_jaccard": float(jaccards.mean()),
        "median_pairwise_jaccard": float(np.median(jaccards)),
        "std_pairwise_jaccard": float(jaccards.std()),
        "shap_top30_crosscheck": overlap_note,
    }
    pd.DataFrame([summary]).to_csv(RESULTS / "feature_stability_summary.csv", index=False)
    logger.info("Stability summary: %s", summary)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(jaccards, bins=30, color="steelblue", edgecolor="black")
    ax.axvline(jaccards.mean(), color="crimson", linewidth=2,
               label=f"mean Jaccard = {jaccards.mean():.3f}")
    ax.set_xlabel("Pairwise Jaccard overlap between fold-selected gene sets")
    ax.set_ylabel("Count (fold pairs)")
    ax.set_title(f"Feature-selection stability across {n_total_folds} outer folds ({N_REPEATS} repeats x {N_OUTER}-fold CV)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_stability.png", dpi=150)
    plt.close(fig)
    logger.info("Wrote results/feature_selection_stability.csv, results/feature_stability_summary.csv, "
                "figures/feature_stability.png")


if __name__ == "__main__":
    main()
