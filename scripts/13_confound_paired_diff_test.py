"""
Step 3i (follow-up, prompted by external review) — Formal paired test of the
"sex confound inflates the whole-cohort AUC" claim.

The draft previously stated that the whole-cohort AUC (0.821) "should be read
as modestly inflated" relative to the female-only AUC (0.783), based only on
eyeballing overlapping confidence intervals. That is not a formal test, and
with this much CI overlap the two numbers alone cannot distinguish "real
inflation" from "same effect, sampling noise" — a reviewer correctly flagged
this as a claim stated with more confidence than the evidence supported.

This script instead runs the correct, directly-relevant comparison: for the
132 female subjects, compare (a) the prediction each one received from the
WHOLE-COHORT model, restricted to her outer test fold from 03_nested_cv.py's
logreg run, against (b) the prediction each one received from the FEMALE-ONLY
model (09_sex_confound_check.py), in her outer test fold from that separate
run. Both are genuine out-of-fold predictions for the same 132 people, so
this supports a PAIRED bootstrap (resample subjects, not independent
resamples of two different-sized unpaired samples), which is the correct way
to test whether training on the full mixed-sex cohort systematically inflates
performance relative to training on females only, on the same evaluation
subjects.

Outputs:
  results/sex_confound_paired_diff_test.csv
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import DATA_PROCESSED, RESULTS, SEED, get_logger

logger = get_logger("13_confound_paired_diff_test")


def main():
    meta = pd.read_csv(DATA_PROCESSED / "sample_metadata_qc.csv", index_col="sample_id")

    whole_oof = pd.read_csv(RESULTS / "outer_fold_predictions.csv")
    if "sample_id" not in whole_oof.columns:
        raise RuntimeError(
            "results/outer_fold_predictions.csv has no sample_id column — rerun "
            "03_nested_cv.py (updated to save sample IDs) before this script."
        )
    whole_oof = whole_oof[whole_oof["model"] == "logreg"].set_index("sample_id")

    female_oof = pd.read_csv(RESULTS / "female_only_oof_predictions.csv").set_index("sample_id")

    female_ids = meta.index[meta["sex"] == "Female"]
    joined = pd.DataFrame({
        "y_true": female_oof.loc[female_ids, "y_true"].values,
        "whole_cohort_proba": whole_oof.loc[female_ids, "y_pred_proba"].values,
        "female_only_proba": female_oof.loc[female_ids, "y_pred_proba"].values,
    }, index=female_ids)

    assert (whole_oof.loc[female_ids, "y_true"].values == joined["y_true"].values).all(), (
        "Label mismatch between the two OOF sources for the same sample_id — something is wrong "
        "with the join."
    )

    n = len(joined)
    y_true = joined["y_true"].values
    whole_proba = joined["whole_cohort_proba"].values
    female_proba = joined["female_only_proba"].values

    auc_whole_on_females = roc_auc_score(y_true, whole_proba)
    auc_female_only = roc_auc_score(y_true, female_proba)
    logger.info("On the same %d female subjects: whole-cohort-trained model AUC=%.4f, "
                "female-only-trained model AUC=%.4f (raw point estimates).",
                n, auc_whole_on_females, auc_female_only)

    rng = np.random.RandomState(SEED)
    diffs = []
    for _ in range(5000):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y_true[idx], whole_proba[idx]) - roc_auc_score(y_true[idx], female_proba[idx]))
    diffs = np.array(diffs)
    diff_mean = float(diffs.mean())
    diff_lo, diff_hi = np.percentile(diffs, [2.5, 97.5])
    excludes_zero = bool(diff_lo > 0 or diff_hi < 0)
    p_two_sided = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))

    pd.DataFrame([{
        "n_female_subjects": n,
        "auc_whole_cohort_model_on_females": auc_whole_on_females,
        "auc_female_only_model_on_females": auc_female_only,
        "paired_bootstrap_diff_mean": diff_mean,
        "paired_bootstrap_diff_ci_lo": diff_lo,
        "paired_bootstrap_diff_ci_hi": diff_hi,
        "diff_ci_excludes_zero": excludes_zero,
        "approx_two_sided_p": p_two_sided,
    }]).to_csv(RESULTS / "sex_confound_paired_diff_test.csv", index=False)

    verdict = (
        "the whole-cohort model IS measurably better on these same female subjects than the "
        "female-only model, i.e. there is direct paired evidence of inflation from training on "
        "the mixed-sex cohort"
        if excludes_zero and diff_mean > 0 else
        "the CI includes zero: this paired comparison does NOT provide statistically significant "
        "evidence that the whole-cohort model outperforms the female-only model on the same "
        "female subjects — the earlier 'modestly inflated' language was not fully supported and "
        "should be softened accordingly"
    )
    logger.warning("Paired bootstrap difference (whole-cohort - female-only), evaluated on the "
                    "same %d female subjects: mean=%.4f, 95%% CI [%.4f, %.4f], approx p=%.4f. "
                    "Verdict: %s.", n, diff_mean, diff_lo, diff_hi, p_two_sided, verdict)

    # Follow-up: if the whole-cohort model isn't better on females, where does its higher
    # pooled AUC (0.821) come from relative to the female-only estimate (0.783)? Check its
    # performance on the male subset, which is heavily skewed toward controls (52/57 = 91%),
    # making male classification comparatively easy regardless of expression signal.
    male_ids = meta.index[meta["sex"] == "Male"]
    male_joined = whole_oof.loc[whole_oof.index.intersection(male_ids)]
    if male_joined["y_true"].nunique() > 1:
        auc_whole_on_males = roc_auc_score(male_joined["y_true"], male_joined["y_pred_proba"])
    else:
        auc_whole_on_males = float("nan")
    n_male_control = int((male_joined["y_true"] == 0).sum())
    n_male_fm = int((male_joined["y_true"] == 1).sum())
    logger.warning("Whole-cohort model's AUC restricted to the %d male subjects (%d control / %d FM, "
                    "i.e. %.0f%% control) = %s. This subset is heavily skewed toward controls, so "
                    "any pooled-AUC advantage of the whole-cohort model (0.821) over the female-only "
                    "estimate (0.783) more plausibly comes from comparatively easy male "
                    "classification than from sex-linked inflation on female predictions specifically.",
                    len(male_joined), n_male_control, n_male_fm,
                    100 * n_male_control / len(male_joined) if len(male_joined) else float("nan"),
                    f"{auc_whole_on_males:.4f}" if not np.isnan(auc_whole_on_males) else "undefined (too few positives)")

    logger.info("Wrote results/sex_confound_paired_diff_test.csv")


if __name__ == "__main__":
    main()
