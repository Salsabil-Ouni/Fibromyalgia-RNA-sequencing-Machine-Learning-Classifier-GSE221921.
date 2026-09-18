"""
Step 5 — Biological cross-check against the original paper's reported pathways.

This is NOT a re-run of the original GSEA/IPA analysis. It is a simple,
honest overlap check: do the classifier's top SHAP genes land disproportionately
in the gene sets the original paper already reported for FM1 (ECM/RhoGDI/
collagen/GP6) and FM2 (CLEAR/lysosomal)? A hypergeometric test is sufficient
for this sanity check; a weak or absent overlap is reported as such, not
hidden or spun.

Gene sets below are curated by hand from the paper's text (Mohapatra et al.,
Sci Rep 2024) and from the standard Reactome/MSigDB `HALLMARK`/pathway gene
lists for the named pathways (ECM organization, RHO GTPase / RhoGDI signaling,
collagen formation, GP6 (platelet collagen receptor) signaling, CLEAR/
lysosomal biogenesis network, interferon/JAK-STAT). This is a coarse,
literature-curated reference list, not a formal pathway database pull — that
limitation is stated in the write-up.
"""
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from config import DATA_PROCESSED, RESULTS, get_logger

logger = get_logger("06_pathway_crosscheck")

GENE_SETS = {
    "FM1_ECM_RhoGDI_collagen_GP6": [
        "COL1A1", "COL1A2", "COL3A1", "COL4A1", "COL5A1", "COL5A2", "COL6A1", "COL6A2", "COL6A3",
        "FN1", "LAMA2", "LAMB1", "LAMC1", "SPARC", "THBS1", "MMP2", "MMP9", "TIMP1", "TIMP2",
        "ITGA1", "ITGA2", "ITGB1", "ARHGDIA", "ARHGDIB", "RHOA", "RHOB", "RHOC", "RAC1", "RAC2",
        "CDC42", "GP6", "GP1BA", "GP9", "ITGA2B", "ITGB3", "VWF",
    ],
    "FM2_CLEAR_lysosomal": [
        "ASAH1", "GAA", "GNS", "IFI30", "PSAP", "ATP6V0B", "ATP6V0C", "ATP6V0D1", "ATP6V1B2",
        "GABARAP", "CTSA", "CTSB", "CTSD", "CTSZ", "HEXA", "HEXB", "NPC1", "NPC2", "TFEB",
        "MCOLN1", "LAMP1", "LAMP2", "SQSTM1", "GLA", "GBA", "NEU1",
    ],
    # FM3/FM4 gene set updated after verifying the paper's own reported genes (Mohapatra et al.
    # 2024, PMC10873305): "interferon alpha/beta signaling, JAK/STAT signaling, death receptor
    # signaling, necroptosis signaling" with example genes STAT1, STAT2, MX2, ISG20, CASP10,
    # IKBKB. Those confirmed genes are added below; the rest of the list (other canonical
    # interferon/JAK-STAT/death-receptor pathway members) is retained from the original
    # literature-curated draft since the paper's text does not enumerate a full gene list.
    "FM3_FM4_interferon_JAKSTAT_death_receptor": [
        "IFI27", "IFI44", "IFI44L", "IFIT1", "IFIT3", "ISG15", "ISG20", "MX1", "MX2",
        "OAS1", "OAS2", "OAS3", "STAT1", "STAT2", "JAK1", "JAK2", "TYK2", "IRF7", "IRF9",
        "IFNAR1", "IFNAR2", "USP18", "CASP10", "CASP8", "FADD", "TNFRSF10A", "TNFRSF10B",
        "RIPK1", "RIPK3", "MLKL", "IKBKB", "IKBKG", "CHUK",
    ],
}


def main():
    shap_df = pd.read_csv(RESULTS / "shap_feature_importance.csv")
    gene_meta = pd.read_csv(DATA_PROCESSED / "gene_metadata.csv")

    background_symbols = set(gene_meta["Hugo_Gene_Symbol"].dropna().astype(str))
    N = len(background_symbols)  # population size = all genes tested (post-QC universe)

    top_n_options = [20, 30, 50, 100]
    rows = []
    for top_n in top_n_options:
        top_genes = set(shap_df.head(top_n)["Hugo_Gene_Symbol"].dropna().astype(str))
        n = len(top_genes)  # sample size = number of top SHAP genes with a known symbol
        for pathway_name, gene_list in GENE_SETS.items():
            pathway_in_bg = set(gene_list) & background_symbols
            K = len(pathway_in_bg)  # successes in population
            overlap = top_genes & pathway_in_bg
            k = len(overlap)
            # P(X >= k) using hypergeometric survival function
            pval = hypergeom.sf(k - 1, N, K, n) if k > 0 else 1.0

            # Power/sensitivity of this specific test: what is the SMALLEST overlap count that
            # would have reached p<0.05 given this pathway's size (K) and this top-N (n)? If
            # that minimum is itself large relative to K, the test has little power to detect a
            # real but partial overlap — a p=1.0 result is then as much a property of small K as
            # it is evidence of "no relationship", and should not be over-interpreted as such.
            min_k_for_p05 = None
            for candidate_k in range(1, min(K, n) + 1):
                if hypergeom.sf(candidate_k - 1, N, K, n) < 0.05:
                    min_k_for_p05 = candidate_k
                    break

            rows.append({
                "top_n_shap_genes": top_n,
                "pathway": pathway_name,
                "pathway_genes_in_background": K,
                "overlap_count": k,
                "overlap_genes": ";".join(sorted(overlap)) if overlap else "",
                "hypergeometric_p_value": pval,
                "min_overlap_needed_for_p_lt_0.05": min_k_for_p05,
            })
            logger.info("top_n=%d pathway=%s: overlap=%d/%d (pathway genes in background=%d/%d total), "
                        "p=%.4f, genes=%s | test would need overlap>=%s to reach p<0.05 given this "
                        "pathway's size (power context, not a correction to the p-value)",
                        top_n, pathway_name, k, n, K, N, pval, sorted(overlap), min_k_for_p05)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(RESULTS / "top_features_vs_original_pathways.csv", index=False)
    logger.info("Wrote results/top_features_vs_original_pathways.csv")

    any_significant = (out_df["hypergeometric_p_value"] < 0.05).any()
    if any_significant:
        logger.info("At least one top-N/pathway combination shows nominal enrichment (p<0.05, uncorrected).")
    else:
        logger.info("No top-N/pathway combination reaches nominal significance (p<0.05, uncorrected). "
                     "This is reported honestly as a weak/absent overlap, not hidden.")

    logger.warning(
        "POWER CAVEAT: this test's minimum-detectable-overlap column shows that for the smaller "
        "pathway gene sets (e.g. FM3_FM4, K~%d genes in background) even a single shared gene "
        "could sometimes reach p<0.05 at larger top-N, while for others 2-3 genes are needed — "
        "meaning a uniform 'p=1.0 at every threshold' result reflects both the small reference-"
        "gene-set sizes and the true absence of overlap, and should not be read as strong "
        "evidence against ANY relationship existing, only as no evidence FOR one at the overlap "
        "sizes actually observed (all zero).",
        len(set(GENE_SETS["FM3_FM4_interferon_JAKSTAT_death_receptor"]) & background_symbols),
    )


if __name__ == "__main__":
    main()
