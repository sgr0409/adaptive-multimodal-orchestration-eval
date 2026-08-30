"""Exact conflict--activation--value decomposition for late fusion.

For equal-weight (EW) and confidence-weighted (CW) fusion, define

    C = P(modality argmax predictions conflict),
    A = P(CW changes EW's decision | conflict), and
    V = P(CW correct | decision changed) - P(EW correct | decision changed).

Then the paired accuracy difference obeys the exact finite-sample identity

    Acc(CW) - Acc(EW) = C * A * V.

The script also reports confidence-ranking lift on "opportunity" examples:
examples having at least one correct and one incorrect modality prediction.
Ranking lift is the probability that a highest-confidence modality is correct
(ties receive fractional credit) minus the probability that a uniformly
selected modality is correct.

No model is fit and no test label enters either fusion rule. Labels are used
only after prediction to evaluate the paired decisions. Percentile intervals
use paired, example-level bootstrap resampling.

Usage:
    python experiments/reliability_complementarity_decomposition.py \
        --root /path/to/adaptive-multimodal-orchestration-eval
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path


SOURCES = {
    "Domain 1": "experiments/results/per_example.json",
    "Domain 2": "experiments/it_incidents/results/per_example.json",
    "CrisisMMD": "experiments/public_crisismmd/results/per_example.json",
    "MM-IMDb": "experiments/public_mmimdb/results/per_example.json",
}


def fused_label(row: dict, key: str) -> str:
    value = row[key]
    return value["label"] if isinstance(value, dict) else value


def modalities(row: dict) -> list[str]:
    return [
        name
        for name in ("text", "image", "telemetry")
        if isinstance(row.get(name), dict) and "probs" in row[name]
    ]


def highest_confidence_credit(row: dict, names: list[str]) -> float:
    highest = max(row[name]["confidence"] for name in names)
    tied = [
        name
        for name in names
        if math.isclose(row[name]["confidence"], highest, rel_tol=0.0, abs_tol=1e-15)
    ]
    return sum(row[name]["label"] == row["true_label"] for name in tied) / len(tied)


def encode(rows: list[dict]) -> list[dict]:
    encoded = []
    for row in rows:
        names = modalities(row)
        ew_key = "equal_weight_fusion" if "equal_weight_fusion" in row else "equal_weight"
        cw_key = (
            "confidence_weighted_fusion"
            if "confidence_weighted_fusion" in row
            else "confidence_weighted"
        )
        ew = fused_label(row, ew_key)
        cw = fused_label(row, cw_key)
        true = row["true_label"]
        modality_correct = [row[name]["label"] == true for name in names]
        conflict = len({row[name]["label"] for name in names}) > 1
        switch = ew != cw
        if switch and not conflict:
            raise AssertionError("A positive-weight fusion decision changed without argmax conflict")
        opportunity = any(modality_correct) and not all(modality_correct)
        encoded.append(
            {
                "conflict": int(conflict),
                "switch": int(switch),
                "paired_gain": int(cw == true) - int(ew == true),
                "opportunity": int(opportunity),
                "top_correct": highest_confidence_credit(row, names) if opportunity else 0.0,
                "random_correct": (
                    sum(modality_correct) / len(modality_correct) if opportunity else 0.0
                ),
            }
        )
    return encoded


def estimates(encoded: list[dict], indices: list[int] | None = None) -> dict:
    sample = encoded if indices is None else [encoded[index] for index in indices]
    n = len(sample)
    conflicts = sum(item["conflict"] for item in sample)
    switches = sum(item["switch"] for item in sample)
    gain = sum(item["paired_gain"] for item in sample)
    opportunities = sum(item["opportunity"] for item in sample)
    conflict_rate = conflicts / n
    # Use the natural zero convention in degenerate bootstrap samples: if no
    # decision changes, activation and the realized value of switching are 0.
    activation = switches / conflicts if conflicts else 0.0
    switch_value = gain / switches if switches else 0.0
    delta = gain / n
    ranking_accuracy = (
        sum(item["top_correct"] for item in sample) / opportunities
        if opportunities
        else 0.0
    )
    random_accuracy = (
        sum(item["random_correct"] for item in sample) / opportunities
        if opportunities
        else 0.0
    )
    ranking_lift = ranking_accuracy - random_accuracy
    product = conflict_rate * activation * switch_value
    if not math.isclose(delta, product, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"Decomposition failed: {delta=} != {product=}")
    return {
        "n": n,
        "conflicts": conflicts,
        "switches": switches,
        "opportunities": opportunities,
        "conflict_rate_C": conflict_rate,
        "activation_A": activation,
        "switch_value_V": switch_value,
        "accuracy_difference": delta,
        "decomposition_product": product,
        "ranking_accuracy": ranking_accuracy,
        "uniform_modality_accuracy": random_accuracy,
        "ranking_lift": ranking_lift,
    }


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return float("nan")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap(encoded: list[dict], resamples: int, seed: int) -> dict:
    rng = random.Random(seed)
    tracked = (
        "conflict_rate_C",
        "activation_A",
        "switch_value_V",
        "accuracy_difference",
        "ranking_lift",
    )
    draws = {metric: [] for metric in tracked}
    n = len(encoded)
    for _ in range(resamples):
        estimate = estimates(encoded, [rng.randrange(n) for _ in range(n)])
        for metric in tracked:
            draws[metric].append(estimate[metric])
    return {
        metric: [percentile(values, 0.025), percentile(values, 0.975)]
        for metric, values in draws.items()
    }


def permutation_dependence(rows: list[dict], permutations: int, seed: int) -> dict:
    """Moon et al.-style reliability-alignment permutation diagnostic.

    Per-modality confidence values are independently permuted across examples
    while probability vectors, inputs, the trained models, and labels remain
    fixed. This tests whether the instance-wise score/evidence alignment helps
    the frozen confidence-weighted rule; it does not compare that rule with EW.
    """
    rng = random.Random(seed)
    names = modalities(rows[0])
    true_labels = [row["true_label"] for row in rows]
    confidence = {name: [row[name]["confidence"] for row in rows] for name in names}
    cw_key = (
        "confidence_weighted_fusion"
        if "confidence_weighted_fusion" in rows[0]
        else "confidence_weighted"
    )
    observed_labels = [fused_label(row, cw_key) for row in rows]
    observed_accuracy = sum(
        predicted == true for predicted, true in zip(observed_labels, true_labels)
    ) / len(rows)

    null_accuracies = []
    for _ in range(permutations):
        permuted = {
            name: rng.sample(confidence[name], len(rows)) for name in names
        }
        correct = 0
        for index, row in enumerate(rows):
            labels = row[names[0]]["probs"].keys()
            scores = {
                label: sum(
                    permuted[name][index] ** 3 * row[name]["probs"][label]
                    for name in names
                )
                for label in labels
            }
            correct += max(scores, key=scores.get) == true_labels[index]
        null_accuracies.append(correct / len(rows))

    null_mean = sum(null_accuracies) / len(null_accuracies)
    return {
        "permutations": permutations,
        "observed_confidence_weighted_accuracy": observed_accuracy,
        "permuted_accuracy_mean": null_mean,
        "observed_minus_permuted_mean": observed_accuracy - null_mean,
        "permuted_accuracy_95_percentile_interval": [
            percentile(null_accuracies, 0.025),
            percentile(null_accuracies, 0.975),
        ],
        "one_sided_p_permuted_at_least_observed": (
            1 + sum(value >= observed_accuracy for value in null_accuracies)
        )
        / (permutations + 1),
    }


def analyze(rows: list[dict], resamples: int, permutations: int, seed: int) -> dict:
    max_degraded = max(len(row["degraded_modalities"]) for row in rows)
    subsets = {
        "all": rows,
        "hard": [row for row in rows if len(row["degraded_modalities"]) == max_degraded],
    }
    output = {"hard_subset_degraded_count": max_degraded}
    for offset, (name, subset) in enumerate(subsets.items()):
        encoded = encode(subset)
        point = estimates(encoded)
        point["bootstrap_95_percentile_ci"] = bootstrap(
            encoded, resamples, seed + offset
        )
        point["reliability_alignment_permutation"] = permutation_dependence(
            subset, permutations, seed + 10 + offset
        )
        output[name] = point
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = {
        "method": "paired example-level bootstrap percentile intervals",
        "bootstrap_resamples": args.resamples,
        "reliability_alignment_permutations": args.permutations,
        "bootstrap_seed": args.seed,
        "identity": "Acc(CW)-Acc(EW) = C * A * V",
        "domains": {},
        "source_files": {},
    }
    for domain_index, (domain, relative) in enumerate(SOURCES.items()):
        source = args.root / relative
        rows = json.loads(source.read_text())
        results["domains"][domain] = analyze(
            rows, args.resamples, args.permutations, args.seed + 100 * domain_index
        )
        results["source_files"][relative] = {"sha256": sha256(source), "n": len(rows)}

    rendered = json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
