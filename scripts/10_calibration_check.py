"""
Step 3f (follow-up) — Probability calibration check.

ROC-AUC only measures ranking quality; it says nothing about whether a
predicted probability of e.g. 0.8 actually corresponds to an 80% empirical
FM rate. Reuses the pooled out-of-fold predictions already produced by
03_nested_cv.py (no CV rerun needed) to compute Brier scores and reliability
(calibration) curves for both model families.

Outputs:
  results/calibration_summary.csv
  figures/calibration_curves.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from config import FIGURES, RESULTS, get_logger

logger = get_logger("10_calibration_check")


def main():
    oof = pd.read_csv(RESULTS / "outer_fold_predictions.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    rows = []
    for ax, model_name in zip(axes, sorted(oof["model"].unique())):
        sub = oof[oof["model"] == model_name]
        y_true, y_proba = sub["y_true"].values, sub["y_pred_proba"].values

        brier = brier_score_loss(y_true, y_proba)
        # brier score of the "always predict base rate" baseline, for context
        base_rate = y_true.mean()
        brier_baseline = brier_score_loss(y_true, np.full_like(y_proba, base_rate, dtype=float))

        frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=8, strategy="quantile")

        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.plot(mean_pred, frac_pos, "o-", color="crimson", label=f"{model_name} (Brier={brier:.3f})")
        ax.set_xlabel("Mean predicted probability (per bin)")
        ax.set_ylabel("Observed FM fraction (per bin)")
        ax.set_title(f"Calibration — {model_name}")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        rows.append({
            "model": model_name, "n": len(y_true), "brier_score": brier,
            "brier_score_baserate_baseline": brier_baseline,
            "base_rate_fm": base_rate,
        })
        logger.info("%s: Brier=%.4f (base-rate-only baseline Brier=%.4f, lower is better, "
                    "n=%d, base_rate_FM=%.3f)", model_name, brier, brier_baseline, len(y_true), base_rate)

    fig.tight_layout()
    fig.savefig(FIGURES / "calibration_curves.png", dpi=150)
    plt.close(fig)

    pd.DataFrame(rows).to_csv(RESULTS / "calibration_summary.csv", index=False)
    logger.info("Wrote results/calibration_summary.csv and figures/calibration_curves.png")


if __name__ == "__main__":
    main()
