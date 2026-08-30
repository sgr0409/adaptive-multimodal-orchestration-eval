"""Generates every figure used in the paper from the results files already
written by run_benchmark.py, sweep_fusion_power.py, robustness_multiseed.py,
and run_planner_benchmark.py. Writes PDF (for LaTeX) and PNG (for quick
viewing) to experiments/figures/.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "experiments" / "results"
FIG_DIR = ROOT / "experiments" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"confidence_weighted": "#1f77b4", "equal_weight": "#ff7f0e",
          "clean": "#2ca02c", "degraded": "#d62728"}


def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {name}.pdf / .png")


def fig_confidence_reliability(results):
    check = results["modality_confidence_reliability_check"]
    modalities = ["text", "image", "telemetry"]
    clean = [check[m]["mean_confidence_when_not_degraded"] for m in modalities]
    degraded = [check[m]["mean_confidence_when_degraded"] for m in modalities]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(modalities))
    w = 0.35
    ax.bar(x - w / 2, clean, w, label="Clean (not degraded)", color=COLORS["clean"])
    ax.bar(x + w / 2, degraded, w, label="Degraded", color=COLORS["degraded"])
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in modalities])
    ax.set_ylabel("Mean confidence")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-modality confidence is genuinely lower on degraded inputs")
    ax.legend()
    save(fig, "confidence_reliability")


def fig_degraded_count_breakdown(results):
    subset = results["subset_breakdown"]
    counts = ["0", "1", "2"]
    cw = [subset[f"{c}_degraded_modalities"]["confidence_weighted_accuracy"] for c in counts]
    ew = [subset[f"{c}_degraded_modalities"]["equal_weight_accuracy"] for c in counts]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(counts))
    w = 0.35
    ax.bar(x - w / 2, cw, w, label="Confidence-weighted fusion", color=COLORS["confidence_weighted"])
    ax.bar(x + w / 2, ew, w, label="Equal-weight fusion", color=COLORS["equal_weight"])
    ax.set_xticks(x)
    ax.set_xticklabels(["0 degraded", "1 degraded", "2 degraded"])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.85, 1.01)
    ax.set_title("Fusion strategies only differ when 2 modalities are degraded")
    ax.legend(loc="lower left")
    save(fig, "degraded_count_breakdown")


def fig_fusion_power_sweep(sweep):
    powers = sorted(int(k) for k in sweep["sweep"].keys())
    acc = [sweep["sweep"][str(p)]["accuracy"] for p in powers]
    pvals = [sweep["sweep"][str(p)]["mcnemar_vs_equal_weight"]["p_value"] for p in powers]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(powers, acc, "o-", color=COLORS["confidence_weighted"], label="Accuracy (hard subset)")
    ax1.axhline(sweep["equal_weight_accuracy_on_hard_subset"], color=COLORS["equal_weight"],
                linestyle="--", label="Equal-weight baseline")
    ax1.set_xlabel("Confidence weighting power")
    ax1.set_ylabel("Accuracy")
    ax1.set_xticks(powers)

    ax2 = ax1.twinx()
    ax2.plot(powers, pvals, "s--", color="gray", alpha=0.7, label="McNemar p-value")
    ax2.axhline(0.05, color="black", linestyle=":", linewidth=1, alpha=0.6)
    ax2.set_ylabel("McNemar p-value vs. equal-weight")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)
    ax1.set_title("Weighting sharpness sweep: plateaus at power=3")
    save(fig, "fusion_power_sweep")


def fig_multiseed_robustness(robustness):
    per_seed = robustness["per_seed"]
    cw = [r["cw_acc_hard"] for r in per_seed]
    ew = [r["ew_acc_hard"] for r in per_seed]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(ew, cw, color=COLORS["confidence_weighted"], alpha=0.8)
    lims = [min(ew + cw) - 0.01, max(ew + cw) + 0.01]
    ax.plot(lims, lims, "k--", linewidth=1, label="Equal performance")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Equal-weight fusion accuracy (hard subset)")
    ax.set_ylabel("Confidence-weighted fusion accuracy (hard subset)")
    ax.set_title(f"Confidence-weighted fusion wins in {robustness['n_seeds_favoring_confidence_weighted']}"
                 f"/{robustness['n_seeds']} overlapping partitions")
    ax.legend()
    ax.set_aspect("equal")
    save(fig, "multiseed_robustness")


def fig_confusion_matrix(results):
    cm = np.array(results["confidence_weighted_fusion"]["confusion_matrix"])
    labels = results["confidence_weighted_fusion"]["confusion_matrix_labels"]

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confidence-weighted fusion (test split)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
    save(fig, "confusion_matrix")


def fig_planner_strategy_distribution(planner_results):
    counts = planner_results["strategy_selection_counts"]
    strategies = list(counts.keys())
    values = [counts[s] for s in strategies]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(strategies, values, color=["#1f77b4", "#ff7f0e", "#d62728"])
    ax.set_ylabel("Number of test examples")
    ax.set_title("Dynamic planner: strategy selection distribution")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 5, str(v), ha="center")
    save(fig, "planner_strategy_distribution")


def main():
    results = json.load(open(RESULTS_DIR / "results.json"))
    sweep = json.load(open(RESULTS_DIR / "fusion_power_sweep.json"))
    robustness = json.load(open(RESULTS_DIR / "robustness_multiseed.json"))
    planner_results = json.load(open(RESULTS_DIR / "planner_selection_results.json"))

    fig_confidence_reliability(results)
    fig_degraded_count_breakdown(results)
    fig_fusion_power_sweep(sweep)
    fig_multiseed_robustness(robustness)
    fig_confusion_matrix(results)
    fig_planner_strategy_distribution(planner_results)

    print(f"\nAll figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
