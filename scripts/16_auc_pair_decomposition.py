"""
Step 3l (follow-up, prompted by external review) — Decompose the pooled AUC
into within-sex and cross-sex pairwise contributions.

The draft claimed, by elimination rather than direct measurement, that the
whole-cohort model's higher POOLED AUC (0.821) relative to its own AUC
restricted to females (0.776) is explained by a known property of AUC as a
rank statistic: pooled AUC counts every (FM, Control) pair in the full
sample, including cross-sex pairs (e.g. a male control vs. a female FM
patient), and because sex correlates strongly with label, those cross-sex
pairs may be disproportionately easy to rank correctly. A reviewer correctly
pointed out this was asserted by ruling out alternatives, not directly
demonstrated — and that it needed to be reconciled with the X/Y-gene-
exclusion result (AUC goes UP slightly when sex-chromosome genes are
removed, which would seem to cut against "sex-linked signal" if that phrase
is read narrowly as chromosome dosage).

ROC-AUC = (# concordant pairs + 0.5 * # tied pairs) / (n_pos * n_neg), where
a pair is one FM sample and one Control sample, and "concordant" means the FM
sample got a higher predicted probability. This script directly computes,
for the whole-cohort model's pooled out-of-fold predictions, the concordance
rate separately for:
  - FF pairs   (FM female vs. Control female)
  - MM pairs   (FM male   vs. Control male)
  - cross pairs (FM of one sex vs. Control of the other sex, both directions)
and reports each subgroup's pair count and concordance rate, plus each
subgroup's own within-group AUC (equivalent to the FF/MM concordance rates)
for direct comparison to the female-only (0.783) and male-restricted (0.592)
numbers already reported.

This turns "the more likely mechanism" into a directly measured decomposition.

Outputs:
  results/auc_pair_decomposition.csv
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import DATA_PROCESSED, RESULTS, get_logger

logger = get_logger("16_auc_pair_decomposition")


def concordance_rate(fm_proba, control_proba):
    """Fraction of (FM, Control) pairs where FM's predicted probability > Control's,
    with ties counted as 0.5 (matches the standard ROC-AUC / Mann-Whitney U convention)."""
    fm_proba = np.asarray(fm_proba)
    control_proba = np.asarray(control_proba)
    # Broadcast comparison: for every FM sample vs every Control sample.
    diff = fm_proba[:, None] - control_proba[None, :]
    concordant = (diff > 0).sum()
    tied = (diff == 0).sum()
    n_pairs = fm_proba.size * control_proba.size
    rate = (concordant + 0.5 * tied) / n_pairs
    return rate, n_pairs


def main():
    meta = pd.read_csv(DATA_PROCESSED / "sample_metadata_qc.csv", index_col="sample_id")
    oof = pd.read_csv(RESULTS / "outer_fold_predictions.csv")
    oof = oof[oof["model"] == "logreg"].set_index("sample_id")

    df = meta.join(oof[["y_true", "y_pred_proba"]], how="inner")
    assert (df["y_true"].values == (df["label"].values == "FM").astype(int)).all()

    fm = df[df["label"] == "FM"]
    ctrl = df[df["label"] == "Control"]

    fm_f, fm_m = fm.loc[fm["sex"] == "Female", "y_pred_proba"], fm.loc[fm["sex"] == "Male", "y_pred_proba"]
    ctrl_f, ctrl_m = ctrl.loc[ctrl["sex"] == "Female", "y_pred_proba"], ctrl.loc[ctrl["sex"] == "Male", "y_pred_proba"]

    logger.info("Group sizes: FM female=%d, FM male=%d, Control female=%d, Control male=%d",
                len(fm_f), len(fm_m), len(ctrl_f), len(ctrl_m))

    rows = []
    for name, fm_group, ctrl_group in [
        ("FF (FM-female vs Control-female)", fm_f, ctrl_f),
        ("MM (FM-male vs Control-male)", fm_m, ctrl_m),
        ("cross: FM-female vs Control-male", fm_f, ctrl_m),
        ("cross: FM-male vs Control-female", fm_m, ctrl_f),
    ]:
        if len(fm_group) == 0 or len(ctrl_group) == 0:
            continue
        rate, n_pairs = concordance_rate(fm_group.values, ctrl_group.values)
        rows.append({"pair_type": name, "n_fm": len(fm_group), "n_control": len(ctrl_group),
                     "n_pairs": n_pairs, "concordance_rate": rate})
        logger.info("%s: n_pairs=%d, concordance_rate=%.4f", name, n_pairs, rate)

    # Combined cross-sex concordance (both directions pooled) and combined within-sex.
    cross_fm = pd.concat([fm_f, fm_m.reindex(fm_m.index)])  # not used directly; compute pairwise below
    # Recompute pooled cross vs within directly from the pair-level counts for a clean summary.
    ff_rate, ff_n = concordance_rate(fm_f.values, ctrl_f.values)
    mm_rate, mm_n = concordance_rate(fm_m.values, ctrl_m.values)
    cross1_rate, cross1_n = concordance_rate(fm_f.values, ctrl_m.values)
    cross2_rate, cross2_n = concordance_rate(fm_m.values, ctrl_f.values)

    within_concordant = ff_rate * ff_n + mm_rate * mm_n
    within_n = ff_n + mm_n
    within_rate = within_concordant / within_n if within_n else float("nan")

    cross_concordant = cross1_rate * cross1_n + cross2_rate * cross2_n
    cross_n = cross1_n + cross2_n
    cross_rate = cross_concordant / cross_n if cross_n else float("nan")

    overall_concordant = within_concordant + cross_concordant
    overall_n = within_n + cross_n
    overall_rate = overall_concordant / overall_n
    overall_auc_check = roc_auc_score(df["y_true"], df["y_pred_proba"])

    rows.append({"pair_type": "ALL within-sex (FF + MM combined)", "n_fm": None, "n_control": None,
                 "n_pairs": within_n, "concordance_rate": within_rate})
    rows.append({"pair_type": "ALL cross-sex (both directions combined)", "n_fm": None, "n_control": None,
                 "n_pairs": cross_n, "concordance_rate": cross_rate})
    rows.append({"pair_type": "ALL pairs (should equal pooled AUC)", "n_fm": None, "n_control": None,
                 "n_pairs": overall_n, "concordance_rate": overall_rate})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(RESULTS / "auc_pair_decomposition.csv", index=False)

    logger.warning(
        "DECOMPOSITION: within-sex pair concordance = %.4f (n_pairs=%d) vs. cross-sex pair "
        "concordance = %.4f (n_pairs=%d). Reconstructed overall concordance from the pair "
        "decomposition = %.4f, vs. sklearn roc_auc_score on the same data = %.4f (sanity check, "
        "should match). %s",
        within_rate, within_n, cross_rate, cross_n, overall_rate, overall_auc_check,
        "Cross-sex pairs ARE more concordant than within-sex pairs, directly confirming the "
        "proposed mechanism." if cross_rate > within_rate else
        "Cross-sex pairs are NOT more concordant than within-sex pairs — the proposed mechanism "
        "is NOT supported by direct measurement and the manuscript's explanation needs revision."
    )
    logger.info("Wrote results/auc_pair_decomposition.csv")


if __name__ == "__main__":
    main()
