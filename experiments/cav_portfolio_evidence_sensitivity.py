"""Replay strict CAV-Portfolio evidence settings without retraining candidates.

The independent-calibration artifact stores paired calibration benefits and
harms for every frozen candidate plus untouched test metrics.  This script
replays prespecified familywise error rates and minimum decisive-count rules;
it never refits candidates or uses test labels to select a path.
"""

import argparse
import json
from pathlib import Path

from scipy.stats import beta


DEFAULT_SETTINGS = (
    (0.01, 5),
    (0.05, 3),
    (0.05, 5),
    (0.05, 10),
    (0.10, 5),
)


def lower_bound(successes, failures, corrected_alpha):
    if successes == 0:
        return 0.0
    return float(beta.ppf(corrected_alpha, successes, failures + 1))


def select_path(record, familywise_error_rate, minimum_decisive_switches):
    stats = record["portfolio_stats"]
    candidates = tuple(stats["candidate_names"])
    corrected_alpha = familywise_error_rate / (2 * len(candidates))
    admitted = []
    for index, name in enumerate(candidates):
        evidence = stats["candidate_evidence"][name]
        decisive = evidence["decisive_switches"]
        lower_decisive = lower_bound(
            decisive,
            evidence["n_examples"] - decisive,
            corrected_alpha,
        )
        lower_benefit = lower_bound(
            evidence["benefits"], evidence["harms"], corrected_alpha
        )
        gain = lower_decisive * (2.0 * lower_benefit - 1.0)
        if decisive >= minimum_decisive_switches and gain > 0.0:
            admitted.append((gain, -index, name))
    return max(admitted)[2] if admitted else stats["reference_name"]


def compare(left, right):
    if left > right:
        return "better"
    if left < right:
        return "worse"
    return "tied"


def replay(artifact, familywise_error_rate, minimum_decisive_switches):
    domains = {}
    totals = {"better": 0, "tied": 0, "worse": 0}
    oracle_matches = 0
    for domain, block in artifact["domains"].items():
        selected_counts = {}
        accuracies = []
        domain_comparison = {"better": 0, "tied": 0, "worse": 0}
        domain_oracle_matches = 0
        for record in block["sensitivity_partitions"]:
            selected = select_path(
                record,
                familywise_error_rate,
                minimum_decisive_switches,
            )
            selected_counts[selected] = selected_counts.get(selected, 0) + 1
            metrics = record["metrics"]["target_subset"]
            accuracy = metrics[selected]["accuracy"]
            accuracies.append(accuracy)
            outcome = compare(
                accuracy, metrics["confidence_weighted"]["accuracy"]
            )
            domain_comparison[outcome] += 1
            totals[outcome] += 1
            matched = accuracy == record["oracle_test_target_accuracy"]
            domain_oracle_matches += int(matched)
            oracle_matches += int(matched)
        domains[domain] = {
            "mean_target_accuracy": sum(accuracies) / len(accuracies),
            "selected_path_counts": selected_counts,
            "versus_confidence_weighted": domain_comparison,
            "test_oracle_accuracy_matches": domain_oracle_matches,
        }
    return {
        "familywise_error_rate": familywise_error_rate,
        "minimum_decisive_switches": minimum_decisive_switches,
        "domains": domains,
        "all_domains_versus_confidence_weighted": totals,
        "all_domains_test_oracle_accuracy_matches": oracle_matches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiments/results/cav_portfolio_independent_calibration.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/cav_portfolio_evidence_sensitivity.json"),
    )
    args = parser.parse_args()
    artifact = json.loads(args.input.read_text())
    result = {
        "method": "CAV-Portfolio strict-evidence sensitivity replay",
        "source_artifact": str(args.input),
        "selection_uses_test_labels": False,
        "settings": [
            replay(artifact, alpha, minimum)
            for alpha, minimum in DEFAULT_SETTINGS
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
