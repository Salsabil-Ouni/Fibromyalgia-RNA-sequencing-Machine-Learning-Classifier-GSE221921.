"""
Step 1 — Data acquisition / parsing.

Reads the raw GEO supplementary files (never modified in place) and writes
tidy, analysis-ready tables to data/processed/:

  - expression_fpkm.csv       samples (rows) x genes (columns), raw FPKM
  - gene_metadata.csv         one row per gene (Ensembl ID, symbol, etc.)
  - sample_metadata.csv       one row per sample (label, sex), cross-checked
                               against the independent GEO series-matrix file
"""
import gzip
import re

import openpyxl
import pandas as pd

from config import DATA_RAW, DATA_PROCESSED, get_logger

logger = get_logger("01_parse_data")

XLSX_PATH = DATA_RAW / "GSE221921_FM_ProcessedData.xlsx"
SERIES_MATRIX_PATH = DATA_RAW / "GSE221921_series_matrix.txt.gz"


def parse_series_matrix(path):
    """Pull GSM accession, Sample_title, Sex, disease-state from the series matrix."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    fields = {}
    for line in lines:
        if not line.startswith("!Sample_"):
            continue
        parts = line.rstrip("\n").split("\t")
        key = parts[0].lstrip("!")
        vals = [v.strip('"') for v in parts[1:]]
        fields.setdefault(key, []).append(vals)

    titles = fields["Sample_title"][0]
    gsm = fields["Sample_geo_accession"][0]

    sex, disease = [None] * len(titles), [None] * len(titles)
    for row in fields.get("Sample_characteristics_ch1", []):
        for i, v in enumerate(row):
            if v.lower().startswith("sex:"):
                sex[i] = v.split(":", 1)[1].strip()
            elif v.lower().startswith("disease state:"):
                disease[i] = v.split(":", 1)[1].strip()

    df = pd.DataFrame({
        "sample_title": titles,
        "gsm_accession": gsm,
        "series_matrix_sex": sex,
        "series_matrix_disease_state": disease,
    })
    return df


def main():
    logger.info("Loading %s (this is large; may take a minute)...", XLSX_PATH.name)
    wb = openpyxl.load_workbook(XLSX_PATH, read_only=True, data_only=True)

    # --- Sample metadata sheet ---
    ws = wb["Metadata (Samples)"]
    rows = list(ws.iter_rows(values_only=True))
    sample_meta = pd.DataFrame(rows[1:], columns=rows[0])
    sample_meta.columns = [str(c).strip() for c in sample_meta.columns]
    sample_meta = sample_meta.rename(columns={"Sample": "sample_id", "Etiology": "label", "Gender": "sex"})
    sample_meta["label"] = sample_meta["label"].map(
        lambda x: "FM" if str(x).strip().lower().startswith("fibrom") else str(x).strip()
    )
    logger.info("Parsed 'Metadata (Samples)': %d samples", len(sample_meta))

    # --- Gene metadata + FPKM values sheet ---
    ws = wb["Values (FPKM)"]
    row_iter = ws.iter_rows(values_only=True)
    header = next(row_iter)
    header = [str(h).strip() for h in header]
    gene_meta_cols = header[:8]
    sample_cols = header[8:]

    data = [r for r in row_iter if r[0] is not None]
    n_raw_rows = ws.max_row - 1
    full_df = pd.DataFrame(data, columns=header)
    logger.info(
        "Parsed 'Values (FPKM)': %d genes x %d samples (sheet reported %d data rows; "
        "%d trailing blank rows past the real data were discarded as an Excel sheet-"
        "dimension artifact, not missing biological data)",
        full_df.shape[0], len(sample_cols), n_raw_rows, n_raw_rows - full_df.shape[0],
    )

    gene_meta = full_df[gene_meta_cols].copy()
    expr = full_df[sample_cols].apply(pd.to_numeric, errors="coerce")
    expr.index = gene_meta["Ensembl_GeneID.version"]
    expr = expr.T  # samples x genes
    expr.index.name = "sample_id"

    # --- Cross-check sample metadata against the independent series-matrix file ---
    smx = parse_series_matrix(SERIES_MATRIX_PATH)

    def norm_disease(x):
        if x is None:
            return None
        x = x.lower()
        if "control" in x or "healthy" in x:
            return "Control"
        if "fibrom" in x or "fma" in x:
            return "FM"
        return x

    smx["label_from_series_matrix"] = smx["series_matrix_disease_state"].map(norm_disease)
    smx["sex_from_series_matrix"] = smx["series_matrix_sex"]

    n_fm_xlsx = (sample_meta["label"] == "FM").sum()
    n_ctrl_xlsx = (sample_meta["label"] == "Control").sum()
    n_fm_smx = (smx["label_from_series_matrix"] == "FM").sum()
    n_ctrl_smx = (smx["label_from_series_matrix"] == "Control").sum()
    logger.info("xlsx sheet label counts: FM=%d Control=%d", n_fm_xlsx, n_ctrl_xlsx)
    logger.info("series-matrix label counts: FM=%d Control=%d", n_fm_smx, n_ctrl_smx)

    if {n_fm_xlsx, n_fm_smx} != {96} or {n_ctrl_xlsx, n_ctrl_smx} != {93}:
        logger.warning(
            "DISCREPANCY vs paper (96 FM / 93 control) or between the two GEO metadata "
            "sources. xlsx: FM=%d Control=%d | series-matrix: FM=%d Control=%d. "
            "Proceeding with the xlsx 'Metadata (Samples)' sheet as the label source of "
            "truth (it is keyed by the same Sample_XXX ids as the expression matrix); "
            "this discrepancy is recorded here and must be repeated in the write-up.",
            n_fm_xlsx, n_ctrl_xlsx, n_fm_smx, n_ctrl_smx,
        )
    else:
        logger.info("Sample counts match the paper (96 FM / 93 control) and agree between sources.")

    # sex cross-check where sample naming allows a join; series matrix uses GSM/Sample_title,
    # xlsx uses Sample_XXX directly matching Sample_title, so join on that.
    merged = sample_meta.merge(smx, left_on="sample_id", right_on="sample_title", how="left")
    sex_mismatch = merged[
        merged["sex"].notna() & merged["sex_from_series_matrix"].notna()
        & (merged["sex"].str.lower() != merged["sex_from_series_matrix"].str.lower())
    ]
    if len(sex_mismatch):
        logger.warning("Sex mismatch between xlsx and series-matrix for %d samples: %s",
                        len(sex_mismatch), sex_mismatch["sample_id"].tolist())
    else:
        logger.info("Sex fields agree between xlsx and series-matrix for all joined samples.")

    out_sample_meta = sample_meta.merge(
        smx[["sample_title", "gsm_accession"]], left_on="sample_id", right_on="sample_title", how="left"
    ).drop(columns=["sample_title"])

    expr.to_csv(DATA_PROCESSED / "expression_fpkm.csv")
    gene_meta.to_csv(DATA_PROCESSED / "gene_metadata.csv", index=False)
    out_sample_meta.to_csv(DATA_PROCESSED / "sample_metadata.csv", index=False)
    logger.info("Wrote expression_fpkm.csv, gene_metadata.csv, sample_metadata.csv to data/processed/")


if __name__ == "__main__":
    main()
