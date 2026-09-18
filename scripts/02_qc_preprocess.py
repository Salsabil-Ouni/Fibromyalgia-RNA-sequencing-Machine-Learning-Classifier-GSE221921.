"""
Step 2 — QC and preprocessing.

- confirm sample counts against the paper
- check missing values / duplicate sample IDs
- drop genes with (near-)zero expression across >90% of samples
- log2(FPKM + 1) transform
- PCA plot colored by FM/control and by sex (the only technical covariate available)

Writes data/processed/expression_log2_filtered.csv (samples x genes, QC'd + log-transformed)
and figures/pca_qc.png. This file is INPUT to modeling but all further feature selection /
scaling happens per-CV-fold inside the modeling scripts, never here on the full dataset.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from config import DATA_PROCESSED, FIGURES, SEED, get_logger, set_global_seed

logger = get_logger("02_qc_preprocess")
set_global_seed()


def main():
    expr = pd.read_csv(DATA_PROCESSED / "expression_fpkm.csv", index_col=0)
    meta = pd.read_csv(DATA_PROCESSED / "sample_metadata.csv", index_col="sample_id")

    n_fm = (meta["label"] == "FM").sum()
    n_ctrl = (meta["label"] == "Control").sum()
    logger.info("Loaded expression matrix: %d samples x %d genes", *expr.shape)
    logger.info("Sample counts: FM=%d, Control=%d (paper: 96 FM / 93 Control)", n_fm, n_ctrl)
    if (n_fm, n_ctrl) != (96, 93):
        logger.warning("Sample counts differ from the paper — see data/README.md for discussion.")

    # duplicate sample IDs
    dup_ids = expr.index[expr.index.duplicated()].tolist()
    if dup_ids:
        logger.warning("Found %d duplicate sample IDs: %s — keeping first occurrence.", len(dup_ids), dup_ids)
        expr = expr[~expr.index.duplicated(keep="first")]
    else:
        logger.info("No duplicate sample IDs.")

    # missing values
    n_missing = expr.isna().sum().sum()
    logger.info("Missing values in expression matrix: %d", n_missing)
    if n_missing:
        n_before = expr.shape[1]
        expr = expr.dropna(axis=1, how="any")
        logger.warning("Dropped %d genes containing any missing value (%d -> %d).",
                        n_before - expr.shape[1], n_before, expr.shape[1])

    # align meta to expr order
    meta = meta.reindex(expr.index)
    if meta["label"].isna().any():
        missing = meta.index[meta["label"].isna()].tolist()
        raise ValueError(f"{len(missing)} expression samples have no metadata label: {missing}")

    # drop near-zero-expressed genes: FPKM == 0 in > 90% of samples
    n_before = expr.shape[1]
    frac_zero = (expr == 0).mean(axis=0)
    keep = frac_zero <= 0.90
    expr = expr.loc[:, keep]
    logger.info("Dropped %d / %d genes with FPKM==0 in >90%% of samples (%d genes remain).",
                n_before - expr.shape[1], n_before, expr.shape[1])

    # log2(FPKM + 1)
    expr_log = np.log2(expr + 1.0)
    logger.info("Applied log2(FPKM + 1) transform.")

    expr_log.to_csv(DATA_PROCESSED / "expression_log2_filtered.csv")
    meta.to_csv(DATA_PROCESSED / "sample_metadata_qc.csv")
    logger.info("Wrote expression_log2_filtered.csv (%d x %d) and sample_metadata_qc.csv", *expr_log.shape)

    # --- PCA QC plot ---
    pca = PCA(n_components=2, random_state=SEED)
    pcs = pca.fit_transform(expr_log.values)
    pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"], index=expr_log.index)
    pca_df["label"] = meta["label"].values
    pca_df["sex"] = meta["sex"].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="label", ax=axes[0], palette="Set1", s=40)
    axes[0].set_title(f"PCA colored by FM/Control\n(PC1 {pca.explained_variance_ratio_[0]*100:.1f}%, "
                       f"PC2 {pca.explained_variance_ratio_[1]*100:.1f}%)")
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="sex", ax=axes[1], palette="Set2", s=40)
    axes[1].set_title("PCA colored by sex (only technical covariate available)")
    fig.tight_layout()
    fig.savefig(FIGURES / "pca_qc.png", dpi=150)
    plt.close(fig)
    logger.info("Saved figures/pca_qc.png")


if __name__ == "__main__":
    main()
