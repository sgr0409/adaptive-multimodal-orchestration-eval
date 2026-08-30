"""Audit the strict CAV gain certificate against a generic bounded-loss bound.

The input artifact contains calibration counts and untouched test metrics.
This audit uses no test label for selection.  It reports the selected CAV
lower gain and its total-variation shift radius, and asks whether a direct
Hoeffding lower bound on the same paired accuracy contrast would certify any
candidate under the same familywise error rate.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def hoeffding_lower(evidence, familywise_error_rate, n_candidates):
    observed_gain = (
        evidence["benefits"] - evidence["harms"]
    ) / evidence["n_examples"]
    return float(observed_gain - np.sqrt(
        2.0 * np.log(n_candidates / familywise_error_rate)
        / evidence["n_examples"]
    ))


def audit(artifact):
    familywise_error_rate = artifact["settings"]["familywise_error_rate"]
    output = {
        "method": "CAV-Portfolio strict gain-certificate audit",
        "selection_uses_test_labels": False,
        "familywise_error_rate": familywise_error_rate,
        "domains": {},
    }
    for domain_name, block in artifact["domains"].items():
        records = block["sensitivity_partitions"]
        selected_radii = []
        selected_gains = []
        nonreference = 0
        hoeffding_certified = 0
        for record in records:
            stats = record["portfolio_stats"]
            selected = stats["selected_path"]
            nonreference += int(selected != stats["reference_name"])
            selected_gains.append(stats["selected_lower_accuracy_gain"])
            selected_radii.append(stats["selected_total_variation_radius"])
            evidence = stats["candidate_evidence"]
            hoeffding_certified += int(any(
                hoeffding_lower(
                    item, familywise_error_rate, len(evidence)
                ) > 0.0
                for item in evidence.values()
            ))
        positive_radii = [value for value in selected_radii if value > 0.0]
        output["domains"][domain_name] = {
            "partitions": len(records),
            "cav_nonreference_certificates": nonreference,
            "generic_hoeffding_nonreference_certificates": hoeffding_certified,
            "mean_selected_gain_lower_bound": float(np.mean(selected_gains)),
            "mean_selected_total_variation_radius": float(
                np.mean(selected_radii)
            ),
            "minimum_positive_total_variation_radius": (
                float(min(positive_radii)) if positive_radii else 0.0
            ),
            "maximum_total_variation_radius": float(max(selected_radii)),
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path(
            "experiments/results/"
            "cav_portfolio_independent_calibration.json"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "experiments/results/cav_portfolio_certificate_audit.json"
        ),
    )
    args = parser.parse_args()
    result = audit(json.loads(args.input.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
