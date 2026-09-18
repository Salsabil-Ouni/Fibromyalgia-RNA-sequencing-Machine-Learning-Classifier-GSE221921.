"""
Step 4 — Explainability.

IMPORTANT: the nested-CV performance estimate from 03_nested_cv.py is already
locked in before this script runs. The model refit here (on the FULL dataset,
with its own hyperparameters retuned by a single round of CV on the full
data) is for interpretation only — its SHAP values describe what the "best
effort" model looks at, but its own internal CV score must NOT be reported
as a performance estimate (that would be circular: the SHAP genes are chosen
using knowledge of the full dataset, including the test folds used in
03_nested_cv.py).

Uses whichever model family had the higher nested-CV AUC in
results/performance_summary.csv (chosen automatically, not hand-picked).

Outputs:
  figures/shap_summary.png
  figures/shap_dependence_top_genes.png
  results/shap_feature_importance.csv   (ranked genes by mean |SHAP|)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import DATA_PROCESSED, FIGURES, RESULTS, SEED, get_logger, set_global_seed
from feature_selection import mannwhitney_score_func

logger = get_logger("05_shap_explain")
set_global_seed()

MODEL_GRIDS = {
    "logreg": (
        Pipeline([
            ("select", SelectKBest(score_func=mannwhitney_score_func)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(solver="liblinear", max_iter=5000, class_weight="balanced", random_state=SEED)),
        ]),
        {"select__k": [50, 100, 200], "clf__C": [0.01, 0.1, 1, 10], "clf__penalty": ["l1", "l2"]},
    ),
    "xgboost": (
        Pipeline([
            ("select", SelectKBest(score_func=mannwhitney_score_func)),
            ("scale", StandardScaler()),
            ("clf", XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=SEED, n_jobs=1, verbosity=0)),
        ]),
        {"select__k": [50, 100, 200], "clf__max_depth": [2, 3], "clf__learning_rate": [0.05, 0.1], "clf__n_estimators": [100, 200]},
    ),
}


def main():
    expr = pd.read_csv(DATA_PROCESSED / "expression_log2_filtered.csv", index_col=0)
    meta = pd.read_csv(DATA_PROCESSED / "sample_metadata_qc.csv", index_col="sample_id")
    meta = meta.reindex(expr.index)
    X = expr.values
    y = (meta["label"].values == "FM").astype(int)
    gene_ids = expr.columns.to_numpy()

    perf = pd.read_csv(RESULTS / "performance_summary.csv")
    agg = perf[perf["outer_fold"].astype(str).str.startswith("AGGREGATE")]
    best_model = agg.loc[agg["roc_auc"].idxmax(), "model"]
    logger.info("Best model family by nested-CV AUC: %s (see performance_summary.csv for the "
                "locked-in CV estimate; this script's own internal CV score below is NOT that "
                "estimate and must not be reported as one).", best_model)

    pipeline, param_grid = MODEL_GRIDS[best_model]
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(pipeline, param_grid, scoring="roc_auc", cv=inner_cv, n_jobs=-1)
    search.fit(X, y)
    best_pipeline = search.best_estimator_
    logger.info("Refit %s on full dataset for interpretation. best_params=%s, full-data CV AUC=%.3f "
                "(interpretation-only number, not a held-out estimate)",
                best_model, search.best_params_, search.best_score_)

    selector = best_pipeline.named_steps["select"]
    scaler = best_pipeline.named_steps["scale"]
    clf = best_pipeline.named_steps["clf"]
    selected_mask = selector.get_support()
    selected_genes = gene_ids[selected_mask]
    X_selected_scaled = scaler.transform(selector.transform(X))

    logger.info("Computing SHAP values for %d selected features...", len(selected_genes))
    if best_model == "xgboost":
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_selected_scaled)
    else:
        explainer = shap.LinearExplainer(clf, X_selected_scaled)
        shap_values = explainer.shap_values(X_selected_scaled)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({"gene_ensembl_id": selected_genes, "mean_abs_shap": mean_abs_shap})
    importance_df = importance_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    gene_meta = pd.read_csv(DATA_PROCESSED / "gene_metadata.csv")
    importance_df = importance_df.merge(
        gene_meta[["Ensembl_GeneID.version", "Hugo_Gene_Symbol"]],
        left_on="gene_ensembl_id", right_on="Ensembl_GeneID.version", how="left",
    ).drop(columns=["Ensembl_GeneID.version"])
    importance_df = importance_df[["gene_ensembl_id", "Hugo_Gene_Symbol", "mean_abs_shap"]]
    importance_df.to_csv(RESULTS / "shap_feature_importance.csv", index=False)
    logger.info("Wrote results/shap_feature_importance.csv (%d genes, model=%s)", len(importance_df), best_model)

    # --- Global summary plot (top 20-30 genes) ---
    top_n = min(30, len(selected_genes))
    order = np.argsort(mean_abs_shap)[::-1][:top_n]
    symbols = importance_df["Hugo_Gene_Symbol"].fillna(importance_df["gene_ensembl_id"]).to_numpy()

    plt.figure(figsize=(9, 10))
    shap.summary_plot(
        shap_values[:, order], X_selected_scaled[:, order],
        feature_names=symbols[:top_n], show=False, plot_size=None,
    )
    plt.title(f"SHAP summary — top {top_n} genes ({best_model}, refit on full dataset)")
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_summary.png", dpi=150)
    plt.close()
    logger.info("Wrote figures/shap_summary.png")

    # --- Dependence plots for top 3-5 genes ---
    top_k_dep = min(5, len(selected_genes))
    fig, axes = plt.subplots(1, top_k_dep, figsize=(4.5 * top_k_dep, 4))
    if top_k_dep == 1:
        axes = [axes]
    for i in range(top_k_dep):
        idx = order[i]
        axes[i].scatter(X_selected_scaled[:, idx], shap_values[:, idx], c=y, cmap="coolwarm", s=18, alpha=0.8)
        axes[i].set_xlabel(f"{symbols[i]} (scaled log2 FPKM)")
        axes[i].set_ylabel("SHAP value")
        axes[i].set_title(symbols[i])
    fig.suptitle("SHAP dependence — top genes (color = FM [red] / Control [blue])")
    fig.tight_layout()
    fig.savefig(FIGURES / "shap_dependence_top_genes.png", dpi=150)
    plt.close(fig)
    logger.info("Wrote figures/shap_dependence_top_genes.png")


if __name__ == "__main__":
    main()
