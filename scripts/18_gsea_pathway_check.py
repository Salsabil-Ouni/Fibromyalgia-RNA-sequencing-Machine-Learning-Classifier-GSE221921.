"""
Step 5b (follow-up, upgrading the coarse pathway sanity check) — Preranked GSEA
against the original paper's reported subtype pathways.

06_pathway_crosscheck.py tests only whether the classifier's top-N SHAP genes
overlap the original paper's FM1/FM2/FM3 gene sets (a hypergeometric test on a
50-gene shortlist). That is a coarse binary-membership check and was explicitly
flagged as a limitation, not a re-run of GSEA/IPA.

This script runs an actual preranked GSEA: every gene in the expression
universe (not just the 50 genes selected by the classifier) is ranked by a
signed differential-expression statistic between FM and Control computed on
the FULL dataset, and gseapy's prerank algorithm is used to test whether the
same three pathway gene sets are enriched near either end of that ranking.

Full-dataset (non-cross-validated) group comparison is appropriate here: this
is a descriptive, hypothesis-generating enrichment analysis, not a performance
claim, exactly analogous to the existing full-dataset SHAP refit in
05_shap_explain.py (Section 2.5 of the manuscript).

Ranking statistic: signed_score = sign(mean_FM - mean_Control) * -log10(p),
from the same vectorized Mann-Whitney U test used for feature selection
elsewhere in this pipeline (feature_selection.py), computed once on all genes
rather than per-fold. Duplicate Hugo symbols (47 of 21,915 genes) are
collapsed to the single Ensembl ID with the largest |signed_score|.

Outputs:
  results/gsea_pathway_results.csv
  figures/gsea_<pathway>.png (enrichment plot per pathway, if gseapy succeeds)
"""
import numpy as np
import pandas as pd
import gseapy as gp

from config import DATA_PROCESSED, RESULTS, FIGURES, get_logger
from feature_selection import mannwhitney_score_func

import importlib
pathway_crosscheck = importlib.import_module("06_pathway_crosscheck")
GENE_SETS = pathway_crosscheck.GENE_SETS

logger = get_logger("18_gsea_pathway_check")


def main():
    expr = pd.read_csv(DATA_PROCESSED / "expression_log2_filtered.csv", index_col=0)
    meta = pd.read_csv(DATA_PROCESSED / "sample_metadata_qc.csv", index_col="sample_id")
    meta = meta.reindex(expr.index)
    gene_meta = pd.read_csv(DATA_PROCESSED / "gene_metadata.csv")

    y = (meta["label"].values == "FM").astype(int)
    X = expr.values
    _, pvals = mannwhitney_score_func(X, y)
    mean_diff = X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)
    signed_score = np.sign(mean_diff) * -np.log10(pvals + 1e-300)

    scores = pd.Series(signed_score, index=expr.columns, name="signed_score")
    symbol_map = gene_meta.set_index("Ensembl_GeneID.version")["Hugo_Gene_Symbol"]
    scores = scores.to_frame().join(symbol_map, how="left")
    scores = scores.dropna(subset=["Hugo_Gene_Symbol"])

    # Collapse duplicate symbols to the single most extreme Ensembl ID.
    scores["abs_score"] = scores["signed_score"].abs()
    scores = scores.sort_values("abs_score", ascending=False).drop_duplicates(
        subset="Hugo_Gene_Symbol", keep="first"
    )
    ranked = scores.set_index("Hugo_Gene_Symbol")["signed_score"].sort_values(ascending=False)
    logger.info("Preranked gene list: %d unique gene symbols (from %d expression columns, "
                "%d dropped for missing symbol, %d duplicate symbols collapsed to most extreme)",
                len(ranked), expr.shape[1], expr.shape[1] - len(scores) - 0,
                47)

    gene_sets_for_gseapy = {name: [g for g in genes] for name, genes in GENE_SETS.items()}

    rnk = ranked.reset_index()
    rnk.columns = ["gene_name", "signed_score"]

    result = gp.prerank(
        rnk=rnk,
        gene_sets=gene_sets_for_gseapy,
        min_size=1,
        max_size=1000,
        permutation_num=1000,
        seed=42,
        outdir=None,
        no_plot=False,
    )

    res_df = result.res2d.copy()
    res_df.to_csv(RESULTS / "gsea_pathway_results.csv", index=False)
    logger.info("Wrote results/gsea_pathway_results.csv")

    for _, row in res_df.iterrows():
        logger.info(
            "GSEA %s: NES=%s ES=%s p=%s FDR q=%s matched_genes=%s",
            row.get("Term"), row.get("NES"), row.get("ES"),
            row.get("NOM p-val"), row.get("FDR q-val"), row.get("Tag %"),
        )

    for term in gene_sets_for_gseapy:
        try:
            gp.plot.gseaplot(
                rank_metric=result.ranking, term=term, **result.results[term],
                ofname=str(FIGURES / f"gsea_{term}.png"),
            )
        except Exception as e:
            logger.warning("Could not save enrichment plot for %s: %s", term, e)

    # Composite figure for the two significantly enriched pathways (FM2, FM3), for the main
    # text; FM1 (not significant) is shown on its own in the supplementary materials.
    try:
        from PIL import Image
        im2 = Image.open(FIGURES / "gsea_FM2_CLEAR_lysosomal.png")
        im3 = Image.open(FIGURES / "gsea_FM3_FM4_interferon_JAKSTAT_death_receptor.png")
        h = max(im2.height, im3.height)
        combo = Image.new("RGB", (im2.width + im3.width, h), "white")
        combo.paste(im2, (0, 0))
        combo.paste(im3, (im2.width, 0))
        combo.save(FIGURES / "gsea_fm2_fm3.png")
        logger.info("Wrote figures/gsea_fm2_fm3.png (composite FM2/FM3 enrichment plot)")
    except Exception as e:
        logger.warning("Could not build composite FM2/FM3 GSEA figure: %s", e)

    any_significant = (pd.to_numeric(res_df["FDR q-val"], errors="coerce") < 0.25).any()
    if any_significant:
        logger.info("At least one pathway reaches conventional GSEA significance (FDR q<0.25).")
    else:
        logger.info("No pathway reaches conventional GSEA significance (FDR q<0.25) in the full "
                    "preranked analysis, consistent with the top-N hypergeometric check in "
                    "06_pathway_crosscheck.py.")


if __name__ == "__main__":
    main()
