# Data provenance

## Source

GEO accession: **GSE221921** — "Identification of unique genomic signatures in patients with
fibromyalgia and chronic pain" (Mohapatra G, Dachet F, Coleman LJ, Gillis B, Behm FG.
*Scientific Reports* 2024;14:3949. doi:10.1038/s41598-024-53874-8. PMID 38366049).

- Series page: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE221921
- 189 samples total: 96 fibromyalgia (FM) patients, 93 matched controls, PBMC RNA-seq (FPKM,
  hybridization-capture RNA-seq, HG38 alignment).

## Files downloaded

Downloaded 2026-08-29 directly from NCBI GEO (processed supplementary file — no raw FASTQ
processing was needed, since GEO provides an author-processed FPKM matrix).

| File | Source URL | SHA-256 |
|---|---|---|
| `raw/GSE221921_FM_ProcessedData.xlsx` | `https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE221921&format=file&file=GSE221921_FM_ProcessedData.xlsx` | see `raw/checksums.sha256` |
| `raw/GSE221921_series_matrix.txt.gz` | `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE221nnn/GSE221921/matrix/GSE221921_series_matrix.txt.gz` | see `raw/checksums.sha256` |

Exact checksums are recorded in `data/raw/checksums.sha256` (generated with `sha256sum` at
download time). Files in `data/raw/` are never modified in place; all downstream files are
written to `data/processed/`.

## Contents of `GSE221921_FM_ProcessedData.xlsx`

Three sheets:

1. **Metadata (Genes)** — Ensembl gene ID (with version), HUGO symbol, description, chromosome,
   start/end coordinates, strand, karyotype band.
2. **Metadata (Samples)** — `Sample` (e.g. `Sample_278`), `Etiology` (`FM` / `Control`),
   `Gender` (`Male` / `Female`). No age or batch field is provided in this sheet.
3. **Values (FPKM)** — gene metadata columns followed by one FPKM column per sample
   (`Sample_XXX`), pre-normalized (FPKM) but **not** log-transformed.

## Metadata caveats

- The series matrix (`GSM` records) also encodes `Sex` and `disease state` per `GSM` accession,
  independently of the processed xlsx `Sample_XXX` naming; `scripts/01_parse_data.py`
  cross-checks the two and flags any mismatch instead of silently trusting one source.
- No age, batch, or sequencing-lane covariate is available from GEO for this series. The QC PCA
  (Step 2) can therefore only check for confounding by sex, not by age or batch — this is stated
  explicitly as a limitation in the write-up.
- The original paper additionally excluded 5 samples (3 controls, 2 FM patients) before defining
  their FM1 (n=43)/FM2 (n=30)/FM3 (n=17) subtypes on the remaining 94 FM / 90 control cases.
  That exclusion is specific to their *unsupervised subtyping* analysis; the processed sample
  sheet released on GEO retains all 189 samples (96 FM / 93 control, matching the FM+control
  total reported in the paper's abstract). We do not attempt to re-identify or drop those 5
  cases ourselves since GEO does not label which ones they were — this is noted as a limitation,
  not silently reproduced.
- The paper reports FM patients are 91/96 female (95%) with median age 48 (range 28–77) vs.
  controls at 43/93(*) female with median age 45 (range 20–69) in their 94/90 subtyping cohort
  (*the full 93-control GEO sheet used here breaks down as 41 F / 52 M — the difference from the
  paper's 43/90 figure comes from the 3 controls excluded from their subtyping analysis but
  retained in the GEO release). This severe sex imbalance between FM and Control is therefore a
  known feature of the source cohort, not an artifact introduced by our reanalysis — see the
  sex-confound analysis in `results/draft_results.md` §3 and `scripts/09_sex_confound_check.py`.
