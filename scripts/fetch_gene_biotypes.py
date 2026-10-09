"""
One-off helper (not part of the numbered pipeline): fetches Ensembl gene
biotypes for every gene in data/processed/gene_metadata.csv via the Ensembl
REST API (POST /lookup/id, batched up to 1000 IDs per request), and caches
the result to data/processed/gene_biotypes.csv.

Run once; 21_protein_coding_only_check.py reads the cached file.
"""
import json
import time
import urllib.request

import pandas as pd

from config import DATA_PROCESSED, get_logger

logger = get_logger("fetch_gene_biotypes")

ENSEMBL_LOOKUP_URL = "https://rest.ensembl.org/lookup/id"
BATCH_SIZE = 1000


def fetch_batch(ensembl_ids):
    payload = json.dumps({"ids": ensembl_ids}).encode("utf-8")
    req = urllib.request.Request(
        ENSEMBL_LOOKUP_URL,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    gene_meta = pd.read_csv(DATA_PROCESSED / "gene_metadata.csv")
    versioned_ids = gene_meta["Ensembl_GeneID.version"].tolist()
    base_ids = [vid.split(".")[0] for vid in versioned_ids]

    out_path = DATA_PROCESSED / "gene_biotypes.csv"
    rows = []
    n = len(base_ids)
    for start in range(0, n, BATCH_SIZE):
        batch = base_ids[start:start + BATCH_SIZE]
        attempt = 0
        while True:
            try:
                result = fetch_batch(batch)
                break
            except Exception as e:
                attempt += 1
                if attempt > 3:
                    logger.error("Batch starting at %d failed after 3 attempts: %s", start, e)
                    result = {}
                    break
                logger.warning("Batch starting at %d failed (attempt %d): %s; retrying", start, attempt, e)
                time.sleep(2)
        for gid in batch:
            info = result.get(gid)
            biotype = info.get("biotype") if info else None
            rows.append({"Ensembl_GeneID_base": gid, "biotype": biotype})
        logger.info("Fetched biotypes for genes %d-%d / %d", start, start + len(batch), n)
        time.sleep(0.2)

    df = pd.DataFrame(rows)
    df["Ensembl_GeneID.version"] = versioned_ids
    df.to_csv(out_path, index=False)
    logger.info("Wrote %s (%d rows); biotype value counts:\n%s", out_path, len(df), df["biotype"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
