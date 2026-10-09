"""
Step 4b (follow-up, reviewer-requested) — Logistic-regression SHAP comparison.

05_shap_explain.py picks whichever model family had the (nominally) higher
nested-CV AUC — XGBoost here — for the main explainability analysis. Since
the XGBoost/logistic-regression AUC difference is not statistically
significant (Section 3.1) and logistic regression is better-calibrated
(Section 3.4), this script computes the same SHAP analysis for logistic
regression directly, so the choice of explainability model can be audited
rather than taken on faith, and reports the top-gene overlap between the two.

Outputs:
  results/shap_feature_importance_logreg.csv
  results/shap_model_overlap.csv
"""
import numpy as np
import pandas as pd
import shap
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DATA_PROCESSED, RESULTS, SEED, get_logger, set_global_seed
from feature_selection import mannwhitney_score_func

logger = get_logger("22_logreg_shap_compare")
set_global_seed()

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

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(PIPELINE, PARAM_GRID, scoring="roc_auc", cv=inner_cv, n_jobs=-1)
    search.fit(X, y)
    best_pipeline = search.best_estimator_
    logger.info("Refit logreg on full dataset for interpretation. best_params=%s, full-data CV AUC=%.3f "
                "(interpretation-only number, not a held-out estimate)",
                search.best_params_, search.best_score_)

    selector = best_pipeline.named_steps["select"]
    scaler = best_pipeline.named_steps["scale"]
    clf = best_pipeline.named_steps["clf"]
    selected_mask = selector.get_support()
    selected_genes = gene_ids[selected_mask]
    X_selected_scaled = scaler.transform(selector.transform(X))

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
    importance_df.to_csv(RESULTS / "shap_feature_importance_logreg.csv", index=False)
    logger.info("Wrote results/shap_feature_importance_logreg.csv (%d genes)", len(importance_df))

    xgb_importance = pd.read_csv(RESULTS / "shap_feature_importance.csv")
    for top_n in (10, 20, 30, 50):
        xgb_top = set(xgb_importance.head(top_n)["gene_ensembl_id"])
        lr_top = set(importance_df.head(min(top_n, len(importance_df)))["gene_ensembl_id"])
        overlap = xgb_top & lr_top
        logger.info("Top-%d overlap between XGBoost and logistic-regression SHAP rankings: "
                    "%d / %d genes (%s)", top_n, len(overlap), top_n,
                    ", ".join(sorted(gene_meta.set_index("Ensembl_GeneID.version")["Hugo_Gene_Symbol"]
                                     .reindex(list(overlap)).dropna().tolist())) or "none")

    rows = []
    for top_n in (10, 20, 30, 50):
        xgb_top = set(xgb_importance.head(top_n)["gene_ensembl_id"])
        lr_top = set(importance_df.head(min(top_n, len(importance_df)))["gene_ensembl_id"])
        rows.append({"top_n": top_n, "overlap_count": len(xgb_top & lr_top)})
    pd.DataFrame(rows).to_csv(RESULTS / "shap_model_overlap.csv", index=False)
    logger.info("Wrote results/shap_model_overlap.csv")


if __name__ == "__main__":
    main()
