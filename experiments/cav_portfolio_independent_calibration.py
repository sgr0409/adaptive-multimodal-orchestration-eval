"""Independent 40/30/30 validation for CAV-Portfolio.

Candidate fusion models are fitted on out-of-fold predictions from the 40%
training split.  The three fixed candidates (confidence weighting, stacking,
and learned gating) are compared with equal weighting only on the disjoint 30%
calibration split.  CAV-Portfolio freezes its selected path before the final
30% test labels are opened.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments.cav_guard_independent_calibration import (
    CALIBRATION_FRACTION,
    TEST_FRACTION,
    TRAIN_FRACTION,
    stratified_three_way_split,
)
from experiments.cav_guard_multidomain import (
    DOMAIN_SPECS,
    FUSION_POWER,
    accuracy_and_f1,
    encode_domain,
    full_train_predictions,
    out_of_fold_predictions,
    subset_indices,
)
from experiments.cav_portfolio_multidomain import (
    BASE_PATHS,
    CANDIDATES,
    FAMILYWISE_ERROR_RATE,
    METHODS,
    MINIMUM_DECISIVE_SWITCHES,
    REFERENCE,
    fit_candidate_models,
    path_results,
    select_naive_path,
    selected_predictions,
)
from framework.cav_portfolio_guard import CAVPortfolioGuardFusion
from framework.fusion_diagnostics import conflict_activation_value


def evaluate_partition(spec, scenarios, arrays, seed):
    labels = [row["label"] for row in scenarios]
    train_idx, calibration_idx, test_idx = stratified_three_way_split(
        labels, seed
    )
    y_train = [labels[index] for index in train_idx]
    y_calibration = [labels[index] for index in calibration_idx]
    y_test = [labels[index] for index in test_idx]

    oof_modality_sets = out_of_fold_predictions(
        arrays, labels, train_idx, seed
    )
    stacking, learned_gating = fit_candidate_models(
        oof_modality_sets, y_train, seed
    )
    calibration_modality_sets = full_train_predictions(
        arrays, labels, train_idx, calibration_idx
    )
    test_modality_sets = full_train_predictions(
        arrays, labels, train_idx, test_idx
    )
    calibration_paths = path_results(
        calibration_modality_sets, stacking, learned_gating
    )
    test_paths = path_results(test_modality_sets, stacking, learned_gating)

    portfolio = CAVPortfolioGuardFusion(
        reference_name=REFERENCE,
        familywise_error_rate=FAMILYWISE_ERROR_RATE,
        minimum_decisive_switches=MINIMUM_DECISIVE_SWITCHES,
    ).fit(
        calibration_paths[REFERENCE],
        {name: calibration_paths[name] for name in CANDIDATES},
        y_calibration,
        calibration_mode="independent_exact",
    )
    naive_path, naive_scores = select_naive_path(
        calibration_paths, y_calibration
    )
    predictions = {
        name: selected_predictions(test_paths, name) for name in BASE_PATHS
    }
    predictions["naive_oof_selector"] = selected_predictions(
        test_paths, naive_path
    )
    predictions["cav_portfolio"] = selected_predictions(
        test_paths, portfolio.stats_.selected_path
    )

    all_indices = list(range(len(test_idx)))
    target_indices = subset_indices(spec, scenarios, test_idx)
    metrics = {
        "overall": {
            name: accuracy_and_f1(predictions[name], y_test, all_indices)
            for name in METHODS
        },
        "target_subset": {
            name: accuracy_and_f1(predictions[name], y_test, target_indices)
            for name in METHODS
        },
    }
    oracle_path = max(
        BASE_PATHS,
        key=lambda name: metrics["target_subset"][name]["accuracy"],
    )
    decomposition = conflict_activation_value(
        [test_modality_sets[index] for index in target_indices],
        [predictions[REFERENCE][index] for index in target_indices],
        [predictions["cav_portfolio"][index] for index in target_indices],
        [y_test[index] for index in target_indices],
    )
    return {
        "seed": seed,
        "split_indices": {
            "train": train_idx.tolist(),
            "calibration": calibration_idx.tolist(),
            "test": test_idx.tolist(),
        },
        "n_train": len(train_idx),
        "n_calibration": len(calibration_idx),
        "n_test": len(test_idx),
        "target_subset_definition": spec.target_subset,
        "naive_calibration_selected_path": naive_path,
        "naive_calibration_path_accuracy": naive_scores,
        "portfolio_stats": asdict(portfolio.stats_),
        "oracle_test_selected_path": oracle_path,
        "oracle_test_target_accuracy": metrics["target_subset"][oracle_path][
            "accuracy"
        ],
        "metrics": metrics,
        "target_subset_portfolio_vs_equal_cav": decomposition,
    }


def comparison(values, baseline):
    gaps = values - baseline
    return {
        "mean_gap": float(gaps.mean()),
        "std_gap": float(gaps.std()),
        "partitions_better": int(np.sum(gaps > 0)),
        "partitions_tied": int(np.sum(gaps == 0)),
        "partitions_worse": int(np.sum(gaps < 0)),
    }


def summarize(partitions):
    summary = {
        "n_partitions": len(partitions),
        "split_fractions": {
            "train": TRAIN_FRACTION,
            "independent_calibration": CALIBRATION_FRACTION,
            "test": TEST_FRACTION,
        },
        "note": (
            "Candidates are fixed before independent calibration labels are "
            "used; final test labels are opened only after path selection. "
            "Repeated partitions overlap and remain a sensitivity analysis."
        ),
        "portfolio_selected_path_counts": {
            name: sum(
                partition["portfolio_stats"]["selected_path"] == name
                for partition in partitions
            )
            for name in BASE_PATHS
        },
        "naive_selected_path_counts": {
            name: sum(
                partition["naive_calibration_selected_path"] == name
                for partition in partitions
            )
            for name in BASE_PATHS
        },
        "oracle_test_selected_path_counts": {
            name: sum(
                partition["oracle_test_selected_path"] == name
                for partition in partitions
            )
            for name in BASE_PATHS
        },
    }
    for scope in ("overall", "target_subset"):
        scope_summary = {"methods": {}, "portfolio_comparisons": {}}
        for method in METHODS:
            values = np.asarray([
                partition["metrics"][scope][method]["accuracy"]
                for partition in partitions
            ])
            scope_summary["methods"][method] = {
                "mean_accuracy": float(values.mean()),
                "std_accuracy": float(values.std()),
                "min_accuracy": float(values.min()),
                "max_accuracy": float(values.max()),
            }
        portfolio_values = np.asarray([
            partition["metrics"][scope]["cav_portfolio"]["accuracy"]
            for partition in partitions
        ])
        for baseline in METHODS[:-1]:
            baseline_values = np.asarray([
                partition["metrics"][scope][baseline]["accuracy"]
                for partition in partitions
            ])
            scope_summary["portfolio_comparisons"][baseline] = comparison(
                portfolio_values, baseline_values
            )
        if scope == "target_subset":
            oracle_values = np.asarray([
                partition["oracle_test_target_accuracy"]
                for partition in partitions
            ])
            scope_summary["portfolio_vs_oracle_test_selector"] = comparison(
                portfolio_values, oracle_values
            )
        summary[scope] = scope_summary
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domains", nargs="+", choices=DOMAIN_SPECS,
        default=list(DOMAIN_SPECS),
    )
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--fixed-seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path,
        default=(
            ROOT / "experiments/results/"
            "cav_portfolio_independent_calibration.json"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    artifact = {
        "method": "CAV-Portfolio independent-calibration validation",
        "settings": {
            "reference": REFERENCE,
            "candidates": list(CANDIDATES),
            "train_fraction": TRAIN_FRACTION,
            "independent_calibration_fraction": CALIBRATION_FRACTION,
            "test_fraction": TEST_FRACTION,
            "familywise_error_rate": FAMILYWISE_ERROR_RATE,
            "minimum_decisive_switches": MINIMUM_DECISIVE_SWITCHES,
            "confidence_power": FUSION_POWER,
            "partition_seeds": list(range(args.seeds)),
            "fixed_split_seed": args.fixed_seed,
        },
        "domains": {},
    }
    for name in args.domains:
        spec = DOMAIN_SPECS[name]
        scenarios, arrays = encode_domain(spec)
        partitions = []
        for seed in range(args.seeds):
            result = evaluate_partition(spec, scenarios, arrays, seed)
            partitions.append(result)
            target = result["metrics"]["target_subset"]
            print(
                f"[{name}] seed={seed:2d} "
                f"selected={result['portfolio_stats']['selected_path']} "
                f"EW={target['equal_weight']['accuracy']:.4f} "
                f"CW={target['confidence_weighted']['accuracy']:.4f} "
                f"P={target['cav_portfolio']['accuracy']:.4f}",
                flush=True,
            )
        fixed = evaluate_partition(spec, scenarios, arrays, args.fixed_seed)
        artifact["domains"][name] = {
            "sensitivity_partitions": partitions,
            "sensitivity_summary": summarize(partitions),
            "fixed_split": fixed,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as stream:
            json.dump(artifact, stream, indent=2, allow_nan=False)
        print(f"[{name}] checkpointed {args.output}", flush=True)
    print(json.dumps({
        name: artifact["domains"][name]["sensitivity_summary"]
        for name in artifact["domains"]
    }, indent=2, allow_nan=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
