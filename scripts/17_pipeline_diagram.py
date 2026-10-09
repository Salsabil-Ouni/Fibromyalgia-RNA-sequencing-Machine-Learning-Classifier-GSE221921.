"""
Step 6 (follow-up, prompted by reviewer request) — Pipeline overview figure.

Produces a single schematic diagram of the full analysis pipeline (data to
results), for use as Figure 1 in the manuscript. Purely illustrative — no
numeric outputs, just a visual map of how the numbered scripts fit together.

Design intent: each box carries a short, formal title plus at most one line
of plain-language detail. Cross-references to other figures/tables and the
full anti-leakage rationale belong in the caption and body text, not crammed
into the diagram itself.

Outputs:
  figures/pipeline_diagram.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

from config import FIGURES, get_logger

logger = get_logger("17_pipeline_diagram")

# Color palette (muted, colorblind-safe-ish)
C_DATA = "#E8EEF7"
C_CORE = "#DCEEE1"
C_MODEL = "#FFF3D6"
C_VALID = "#FBE3E3"
C_EXPLAIN = "#EDE3F7"
C_OUT = "#DDEEF2"
EDGE = "#4A4A4A"


def box(ax, x, y, w, h, title, subtitle, facecolor, title_size=10.5, sub_size=8.7):
    b = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.1, edgecolor=EDGE, facecolor=facecolor, zorder=2,
    )
    ax.add_patch(b)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
                 fontsize=title_size, weight="bold", zorder=3)
        ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center",
                 fontsize=sub_size, style="italic", color="#333333", zorder=3,
                 linespacing=1.3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                 fontsize=title_size, weight="bold", zorder=3)
    return (x + w / 2, y), (x + w / 2, y + h), (x, y + h / 2), (x + w, y + h / 2)


def arrow(ax, p1, p2, lw=1.3, color=EDGE):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13,
                         linewidth=lw, color=color, zorder=1,
                         connectionstyle="arc3,rad=0.0")
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(10, 12.7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.7, 15.1)
    ax.axis("off")

    # 1. Data & preprocessing (merged)
    box(ax, 2.6, 13.85, 4.8, 1.15, "Data & Preprocessing",
        "GSE221921 (GEO) · 96 FM, 93 control\nBlank-row filtering, log₂(FPKM+1) transform",
        C_DATA)

    # 2. Nested CV core
    box(ax, 1.9, 12.15, 6.2, 1.15, "Nested Cross-Validation",
        "5-fold outer / 5-fold inner; feature selection and\nscaling refit within each training fold",
        C_CORE)
    arrow(ax, (5.0, 13.85), (5.0, 13.3))

    # 3. Two model families
    box(ax, 0.7, 10.65, 3.1, 0.95, "Logistic Regression", "L1/L2-regularized", C_MODEL)
    box(ax, 6.2, 10.65, 3.1, 0.95, "XGBoost", "Gradient-boosted trees", C_MODEL)
    arrow(ax, (3.4, 12.15), (2.5, 11.6))
    arrow(ax, (6.6, 12.15), (7.5, 11.6))

    # 4. Performance evaluation
    box(ax, 2.6, 9.15, 4.8, 0.95, "Performance Evaluation",
        "ROC-AUC, PR-AUC, sensitivity, specificity", C_CORE)
    arrow(ax, (2.5, 10.65), (4.4, 10.1))
    arrow(ax, (7.5, 10.65), (5.6, 10.1))

    # 5. Validity + confound checks (side by side)
    box(ax, 0.3, 7.5, 4.2, 1.15, "Permutation & Robustness Testing",
        "Label permutation, seed sensitivity,\nlearning curve", C_VALID)
    box(ax, 5.5, 7.5, 4.2, 1.15, "Sex-Confound Analysis",
        "Female-only subset, X/Y-gene exclusion,\npaired difference test", C_VALID)
    arrow(ax, (4.0, 9.15), (2.4, 8.65))
    arrow(ax, (6.0, 9.15), (7.6, 8.65))

    # 6. Explainability
    box(ax, 2.6, 5.95, 4.8, 1.0, "Model Explainability",
        "SHAP value attribution on the refit model", C_EXPLAIN)
    arrow(ax, (2.4, 7.5), (4.4, 6.95))
    arrow(ax, (7.6, 7.5), (5.6, 6.95))

    # 7. Pathway cross-check + cell-composition confound check (side by side)
    box(ax, 0.3, 3.8, 4.4, 1.15, "Pathway Cross-Check",
        "Preranked GSEA and hypergeometric\noverlap with reported subtype pathways", C_EXPLAIN)
    box(ax, 5.3, 3.8, 4.4, 1.15, "Cell-Composition Confound Check",
        "Marker-based proxy scores; composition-\nadjusted GSEA and paired model comparison",
        C_VALID)
    arrow(ax, (4.0, 5.95), (2.5, 4.95))
    arrow(ax, (6.0, 5.95), (7.5, 4.95))

    # 8. Output
    box(ax, 3.0, 2.15, 4.0, 1.0, "Results", None, C_OUT)
    arrow(ax, (2.5, 3.8), (4.4, 3.15))
    arrow(ax, (7.5, 3.8), (5.6, 3.15))

    legend_elems = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_DATA, markersize=13, label="Data & QC"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_CORE, markersize=13, label="Core modeling"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_MODEL, markersize=13, label="Model families"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_VALID, markersize=13, label="Validity & confound checks"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_EXPLAIN, markersize=13, label="Explainability"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_OUT, markersize=13, label="Output"),
    ]
    ax.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, -0.04),
              ncol=3, fontsize=8.5, frameon=False)

    fig.tight_layout()
    fig.savefig(FIGURES / "pipeline_diagram.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote figures/pipeline_diagram.png")


if __name__ == "__main__":
    main()
