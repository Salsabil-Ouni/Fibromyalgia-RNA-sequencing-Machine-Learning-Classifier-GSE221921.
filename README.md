# Fibromyalgia RNA-seq ML Classifier

A confound-aware, explainable machine-learning classifier for fibromyalgia (FM), built on the
public PBMC RNA-seq dataset from Mohapatra et al. (2024, *Scientific Reports*, GEO accession
[GSE221921](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE221921)). The original paper
did unsupervised subtyping and pathway enrichment; it did not build or validate a predictive
classifier. That's the gap this project fills, with a heavy emphasis on catching the two things
that usually go wrong in small-n, high-dimensional genomics classifiers: leakage and confounding.



## Headline results

| | |
|---|---|
| Nested cross-validated AUC | 0.821–0.825 (95% CI 0.756–0.891) |
| Permutation-label null (100 reruns) | mean 0.501 ± 0.059, empirical **p = 0.0099** |
| Sex-only baseline (no gene expression) | AUC 0.727 |
| Female-only replication (sex confound removed) | AUC 0.783 (95% CI 0.705–0.858) |
| X/Y-gene-exclusion check | AUC 0.834 — no drop vs. full-genome |

The cohort is severely sex-imbalanced (95% female among FM patients vs. 44% among controls),
consistent with well-documented referral bias in clinically-ascertained FM samples. Rather than
footnote this, the pipeline tests it directly from two independent angles and finds that the
disease signal is real and largely autosomal, not a sex-composition artifact — full detail in
the manuscript's Results §3.3.

## Repo structure

```
data/
  README.md              provenance: exact GEO accession, download date, checksums
  raw/                   never modified in place (gitignored — regenerate via script 01)
  processed/             tidy CSVs (gitignored — regenerate via scripts 01-02)
scripts/                 numbered to match the manuscript's methods/results order (01-16)
results/                 all numeric outputs + pipeline.log (full seed/version/param record)
figures/                 all plots
requirements.txt         pinned dependencies
```

### Script-to-analysis mapping

| Script | Produces |
|---|---|
| `config.py` | Global seed (42), logging, paths — imported by every other script |
| `feature_selection.py` | Vectorized Mann-Whitney U scorer, used *inside* every CV fold |
| `01_parse_data.py` | `data/processed/{expression_fpkm,gene_metadata,sample_metadata}.csv` |
| `02_qc_preprocess.py` | `data/processed/expression_log2_filtered.csv`, `figures/pca_qc.png` |
| `03_nested_cv.py` | `results/performance_summary.csv`, `results/outer_fold_predictions.csv` |
| `04_permutation_test.py` | `results/permutation_null_distribution.csv` + figure |
| `05_shap_explain.py` | `figures/shap_summary.png`, `shap_dependence_top_genes.png`, `results/shap_feature_importance.csv` |
| `06_pathway_crosscheck.py` | `results/top_features_vs_original_pathways.csv` |
| `07_feature_stability.py` | `results/feature_selection_stability.csv`, `feature_stability_summary.csv`, `figures/feature_stability.png` |
| `08_featsel_robustness.py` | `results/performance_summary_ttest_robustness.csv`, `featsel_ttest_vs_mannwhitney_diff_test.csv` |
| `09_sex_confound_check.py` | `results/sex_confound_summary.csv`, `performance_summary_female_only.csv` |
| `10_calibration_check.py` | `results/calibration_summary.csv`, `figures/calibration_curves.png` |
| `11_sexchrom_excluded_check.py` | `results/performance_summary_autosomal_only.csv` |
| `12_learning_curve.py` | `results/learning_curve.csv`, `figures/learning_curve.png` |
| `13_confound_paired_diff_test.py` | `results/sex_confound_paired_diff_test.csv` — paired bootstrap test of whether whole-cohort AUC is inflated relative to female-only AUC |
| `14_confound_permutation_nulls.py` | Dedicated permutation nulls for the female-only and autosomal-only models |
| `15_seed_sensitivity.py` | `results/seed_sensitivity.csv` — 5-seed rerun of the headline result |
| `16_auc_pair_decomposition.py` | `results/auc_pair_decomposition.csv` — decomposes pooled AUC into within-sex vs. cross-sex pairwise concordance |

Run in numeric order. Each script only reads from `data/processed/` or `results/` (never from
`data/raw/` directly, except script 01), and is idempotent.

## Reproducing this

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows; use .venv/bin/pip on Linux/macOS
```

Download the raw GEO files as documented in `data/README.md`, then run the scripts in numeric
order. A global seed of **42** is used everywhere (numpy, every sklearn splitter, XGBoost).

**Verified reproducibility:** a clean-room rerun (fresh venv, only `scripts/`,
`requirements.txt`, and the raw GEO download — no cached processed data or results) reproduced
every output file byte-for-byte identical to the original run, confirmed via SHA-256 checksum
on all CSVs and PNGs.

## Methodology at a glance

- **No leakage.** Feature selection and standardization are always inside the same
  `sklearn.Pipeline` as the classifier, refit from scratch on every training fold. The only
  place either touches the full dataset is the final SHAP refit (script 05), which is for
  interpretation only and never used to report a performance number.
- **Permutation-tested.** The headline AUC is checked against a 100-repetition permutation-label
  null, and the two sex-confound-check models each get their own dedicated null rather than
  borrowing the main one.
- **Confound-checked, not just confound-disclosed.** The severe sex imbalance in this cohort is
  tested from two independent angles (female-only subset; X/Y-gene exclusion), plus a formal
  paired bootstrap test of whether the pooled AUC is actually inflated by it (it isn't,
  significantly) and a first-principles decomposition of *why* the pooled metric is still
  shaped by cohort composition (cross-sex pairs in the AUC calculation are easier to rank).
- **Explainable.** SHAP values identify the genes driving predictions; their overlap with the
  original paper's reported pathway gene sets is tested directly (found to be minimal — reported
  as a real finding, not adjusted after the fact).
- **Honest about limitations.** Single cohort, correlational, hypothesis-generating — not a
  diagnostic tool. See the manuscript's Discussion/Limitations for the full list.

## Citation

If you use this code or refer to this analysis, please cite the preprint (details to be
added once preprinted) and the original dataset:

> Mohapatra G, Dachet F, Coleman LJ, Gillis B, Behm FG. Identification of unique genomic
> signatures in patients with fibromyalgia and chronic pain. *Sci Rep*. 2024;14:3949.
> doi:10.1038/s41598-024-53874-8.


## Contact

Salsabil Ouni — salsabil.ouni.i@gmail.com
