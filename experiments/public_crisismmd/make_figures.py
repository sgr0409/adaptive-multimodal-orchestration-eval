"""Figures for the public-CrisisMMD validation domain."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = ROOT / "experiments" / "public_crisismmd" / "results"
FIG_DIR = ROOT / "experiments" / "public_crisismmd" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"clean": "#2ca02c", "degraded": "#d62728", "cw": "#1f77b4"}


def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {name}.pdf / .png")


def fig_confidence_reliability(results):
    check = results["modality_confidence_reliability_check"]
    modalities = ["text", "image"]
    clean = [check[m]["mean_confidence_when_not_degraded"] for m in modalities]
    degraded = [check[m]["mean_confidence_when_degraded"] for m in modalities]

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(modalities))
    w = 0.35
    ax.bar(x - w / 2, clean, w, label="Clean (not degraded)", color=COLORS["clean"])
    ax.bar(x + w / 2, degraded, w, label="Degraded", color=COLORS["degraded"])
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in modalities])
    ax.set_ylabel("Mean confidence")
    ax.set_ylim(0, 1.05)
    ax.set_title("Public CrisisMMD domain: confidence gap is weak (image)\nor reversed (text) on real data")
    ax.legend()
    save(fig, "confidence_reliability")


def fig_multiseed_robustness(robustness):
    per_seed = robustness["per_seed"]
    cw = [r["cw_acc"] for r in per_seed]
    ew = [r["ew_acc"] for r in per_seed]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(ew, cw, color=COLORS["cw"], alpha=0.8)
    lims = [min(ew + cw) - 0.01, max(ew + cw) + 0.01]
    ax.plot(lims, lims, "k--", linewidth=1, label="Equal performance")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Equal-weight fusion accuracy")
    ax.set_ylabel("Confidence-weighted fusion accuracy")
    ax.set_title(f"Public CrisisMMD domain: equal-weight wins in "
                 f"{robustness['n_seeds_favoring_equal_weight']}/{robustness['n_seeds']} splits\n"
                 f"-- opposite of the synthetic domains' result")
    ax.legend()
    ax.set_aspect("equal")
    save(fig, "multiseed_robustness")


def main():
    results = json.load(open(RESULTS_DIR / "results.json"))
    fig_confidence_reliability(results)
    robustness_path = RESULTS_DIR / "robustness_multiseed.json"
    if robustness_path.exists():
        fig_multiseed_robustness(json.load(open(robustness_path)))
    print(f"\nFigures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
