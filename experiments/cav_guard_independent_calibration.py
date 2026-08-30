"""Independent-calibration validation for H-CAV-Guard.

Each partition is stratified into 40% model training, 30% independent
calibration, and 30% final testing.  Base scorers, the global CAV gate, and
the local benefit model are fitted without calibration or test labels.  The
calibration labels are used only to choose the multiplicity-corrected local
score threshold.  The test labels are used only after the complete policy is
frozen.

Repeated partitions overlap and therefore remain a sensitivity analysis.
Within a partition, however, the calibration and test samples are disjoint,
activating the independent-calibration mode implemented by H-CAV-Guard.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cav_guard_multidomain import (
    CREDIBILITY_THRESHOLD,
    DOMAIN_SPECS,
    FUSION_POWER,
    ROOT,
    accuracy_and_f1,
    encode_domain,
    full_train_predictions,
    out_of_fold_predictions,
    subset_indices,
)
from framework.cav_guard_fusion import CAVGuardFusion, HierarchicalCAVGuardFusion
from framework.fusion import confidence_weighted_fusion, equal_weight_fusion
from framework.fusion_diagnostics import conflict_activation_value


METHODS = (
    "equal_weight",
    "confidence_weighted",
    "global_cav_guard",
    "hierarchical_cav_guard",
)
TRAIN_FRACTION = 0.40
CALIBRATION_FRACTION = 0.30
TEST_FRACTION = 0.30
LOCAL_SCORE_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)


def stratified_three_way_split(labels, seed):
    indices = np.arange(len(labels))
    train_idx, remainder_idx = train_test_split(
        indices,
        train_size=TRAIN_FRACTION,
        random_state=seed,
        stratify=np.asarray(labels),
    )
    calibration_idx, test_idx = train_test_split(
        remainder_idx,
        test_size=TEST_FRACTION / (CALIBRATION_FRACTION + TEST_FRACTION),
        random_state=seed,
        stratify=np.asarray(labels)[remainder_idx],
    )
    return (
        np.asarray(train_idx),
        np.asarray(calibration_idx),
        np.asarray(test_idx),
    )


def path_predictions(modality_results_list, guard, hierarchical_guard):
    predictions = {method: [] for method in METHODS}
    selected_paths = []
    for modality_results in modality_results_list:
        predictions["equal_weight"].append(
            equal_weight_fusion(modality_results)["label"]
        )
        predictions["confidence_weighted"].append(
            confidence_weighted_fusion(
                modality_results, power=FUSION_POWER
            )["label"]
        )
        predictions["global_cav_guard"].append(
            guard.predict(modality_results)["label"]
        )
        result = hierarchical_guard.predict(modality_results)
        predictions["hierarchical_cav_guard"].append(result["label"])
        selected_paths.append(result["selected_path"])
    return predictions, selected_paths


def evaluate_partition(spec, scenarios, arrays, seed):
    labels = [row["label"] for row in scenarios]
    train_idx, calibration_idx, test_idx = stratified_three_way_split(
        labels, seed
    )
    y_train = [labels[index] for index in train_idx]
    y_calibration = [labels[index] for index in calibration_idx]
    y_test = [labels[index] for index in test_idx]

    oof_results = out_of_fold_predictions(arrays, labels, train_idx, seed)
    calibration_results = full_train_predictions(
        arrays, labels, train_idx, calibration_idx
    )
    test_results = full_train_predictions(arrays, labels, train_idx, test_idx)

    global_guard = CAVGuardFusion(
        power=FUSION_POWER,
        credibility_threshold=CREDIBILITY_THRESHOLD,
    ).fit(oof_results, y_train)
    hierarchical_guard = HierarchicalCAVGuardFusion(
        power=FUSION_POWER,
        credibility_threshold=CREDIBILITY_THRESHOLD,
        familywise_error_rate=0.05,
        seed=seed,
        local_score_thresholds=LOCAL_SCORE_THRESHOLDS,
        require_local_certification=True,
    ).fit(
        oof_results,
        y_train,
        calibration_modality_results_list=calibration_results,
        calibration_y=y_calibration,
    )

    predictions, selected_paths = path_predictions(
        test_results, global_guard, hierarchical_guard
    )
    all_indices = list(range(len(test_idx)))
    target_indices = subset_indices(spec, scenarios, test_idx)
    metrics = {
        "overall": {
            method: accuracy_and_f1(predictions[method], y_test, all_indices)
            for method in METHODS
        },
        "target_subset": {
            method: accuracy_and_f1(
                predictions[method], y_test, target_indices
            )
            for method in METHODS
        },
    }

    target_modality_sets = [test_results[index] for index in target_indices]
    target_labels = [y_test[index] for index in target_indices]
    decomposition = conflict_activation_value(
        target_modality_sets,
        [predictions["equal_weight"][index] for index in target_indices],
        [predictions["hierarchical_cav_guard"][index]
         for index in target_indices],
        target_labels,
    )
    hierarchical_stats = asdict(hierarchical_guard.stats_)
    hierarchical_stats.update({
        "test_candidate_decisions": sum(
            path == "confidence_weighted" for path in selected_paths
        ),
        "test_candidate_rate": float(np.mean([
            path == "confidence_weighted" for path in selected_paths
        ])),
        "target_candidate_decisions": sum(
            selected_paths[index] == "confidence_weighted"
            for index in target_indices
        ),
        "target_candidate_rate": float(np.mean([
            selected_paths[index] == "confidence_weighted"
            for index in target_indices
        ])),
    })
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
        "global_guard_training_stats": asdict(global_guard.stats_),
        "hierarchical_guard_training_stats": hierarchical_stats,
        "metrics": metrics,
        "target_subset_hierarchical_guard_vs_equal_cav": decomposition,
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
            "Calibration and test samples are disjoint within each partition; "
            "partitions overlap and their aggregate remains descriptive."
        ),
        "calibration_modes": {},
    }
    modes = sorted({
        partition["hierarchical_guard_training_stats"]["calibration_mode"]
        for partition in partitions
    })
    summary["calibration_modes"] = {
        mode: sum(
            partition["hierarchical_guard_training_stats"]["calibration_mode"]
            == mode
            for partition in partitions
        )
        for mode in modes
    }
    summary["global_candidate_enabled"] = sum(
        partition["global_guard_training_stats"]["candidate_enabled"]
        for partition in partitions
    )
    summary["local_threshold_certified"] = sum(
        partition["hierarchical_guard_training_stats"]["local_model_fitted"]
        for partition in partitions
    )
    summary["mean_target_candidate_rate"] = float(np.mean([
        partition["hierarchical_guard_training_stats"]["target_candidate_rate"]
        for partition in partitions
    ]))
    summary["mean_target_switch_value"] = float(np.mean([
        partition["target_subset_hierarchical_guard_vs_equal_cav"]
        ["switch_value_V"]
        for partition in partitions
    ]))

    for scope in ("overall", "target_subset"):
        scope_summary = {"methods": {}, "comparisons": {}}
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
        hierarchical = np.asarray([
            partition["metrics"][scope]["hierarchical_cav_guard"]["accuracy"]
            for partition in partitions
        ])
        for baseline in METHODS[:-1]:
            baseline_values = np.asarray([
                partition["metrics"][scope][baseline]["accuracy"]
                for partition in partitions
            ])
            gaps = hierarchical - baseline_values
            scope_summary["comparisons"][baseline] = {
                "mean_gap": float(gaps.mean()),
                "std_gap": float(gaps.std()),
                "partitions_better": int(np.sum(gaps > 0)),
                "partitions_tied": int(np.sum(gaps == 0)),
                "partitions_worse": int(np.sum(gaps < 0)),
            }
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
        "--output",
        type=Path,
        default=(
            ROOT / "experiments/results/"
            "cav_guard_independent_calibration.json"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    artifact = {
        "method": "H-CAV-Guard independent-calibration validation",
        "settings": {
            "train_fraction": TRAIN_FRACTION,
            "independent_calibration_fraction": CALIBRATION_FRACTION,
            "test_fraction": TEST_FRACTION,
            "confidence_power": FUSION_POWER,
            "posterior_credibility_threshold": CREDIBILITY_THRESHOLD,
            "local_familywise_error_rate": 0.05,
            "local_minimum_decisive_switches": 5,
            "local_score_thresholds": list(LOCAL_SCORE_THRESHOLDS),
            "require_local_certification": True,
            "uncertified_action": "preserve_equal_weight_reference",
            "partition_seeds": list(range(args.seeds)),
            "fixed_split_seed": args.fixed_seed,
        },
        "domains": {},
    }
    for domain_name in args.domains:
        spec = DOMAIN_SPECS[domain_name]
        scenarios, arrays = encode_domain(spec)
        partitions = []
        for seed in range(args.seeds):
            result = evaluate_partition(spec, scenarios, arrays, seed)
            partitions.append(result)
            stats = result["hierarchical_guard_training_stats"]
            target = result["metrics"]["target_subset"]
            print(
                f"[{domain_name}] seed={seed:2d} "
                f"mode={stats['calibration_mode']} "
                f"local={stats['local_model_fitted']} "
                f"rate={stats['target_candidate_rate']:.3f} "
                f"EW={target['equal_weight']['accuracy']:.4f} "
                f"H={target['hierarchical_cav_guard']['accuracy']:.4f}",
                flush=True,
            )
        fixed = evaluate_partition(spec, scenarios, arrays, args.fixed_seed)
        artifact["domains"][domain_name] = {
            "partitions": partitions,
            "summary": summarize(partitions),
            "fixed_split": fixed,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as stream:
            json.dump(artifact, stream, indent=2, allow_nan=False)
        print(f"[{domain_name}] checkpointed {args.output}", flush=True)
    print(json.dumps({
        domain: artifact["domains"][domain]["summary"]
        for domain in artifact["domains"]
    }, indent=2, allow_nan=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
