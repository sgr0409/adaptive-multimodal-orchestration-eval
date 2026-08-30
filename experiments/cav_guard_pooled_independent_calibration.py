"""Pooled four-domain independent-calibration validation for H-CAV-Guard.

The four domains are split separately and stratified into 40% model training,
30% calibration, and 30% final testing.  Domain-specific base scorers are fit
only on their domain's training split.  One shared, label-free H-CAV policy is
then trained on the pooled out-of-fold switch examples and calibrated on the
pooled, domain-stratified calibration mixture.  Its fixed threshold grid is
chosen before calibration labels are examined.  If calibration certifies no
local region, the strict policy preserves the equal-weight reference.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cav_guard_independent_calibration import (
    CALIBRATION_FRACTION,
    LOCAL_SCORE_THRESHOLDS,
    METHODS,
    TEST_FRACTION,
    TRAIN_FRACTION,
    path_predictions,
    stratified_three_way_split,
)
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
from framework.fusion_diagnostics import conflict_activation_value


def prepare_domain_partition(spec, scenarios, arrays, seed):
    labels = [row["label"] for row in scenarios]
    train_idx, calibration_idx, test_idx = stratified_three_way_split(
        labels, seed
    )
    return {
        "split_indices": {
            "train": train_idx.tolist(),
            "calibration": calibration_idx.tolist(),
            "test": test_idx.tolist(),
        },
        "y_train": [labels[index] for index in train_idx],
        "y_calibration": [labels[index] for index in calibration_idx],
        "y_test": [labels[index] for index in test_idx],
        "oof_results": out_of_fold_predictions(
            arrays, labels, train_idx, seed
        ),
        "calibration_results": full_train_predictions(
            arrays, labels, train_idx, calibration_idx
        ),
        "test_results": full_train_predictions(
            arrays, labels, train_idx, test_idx
        ),
        "target_indices": subset_indices(spec, scenarios, test_idx),
    }


def evaluate_partition(encoded_domains, seed):
    prepared = {
        name: prepare_domain_partition(spec, scenarios, arrays, seed)
        for name, (spec, scenarios, arrays) in encoded_domains.items()
    }
    pooled_oof = []
    pooled_train_y = []
    pooled_calibration = []
    pooled_calibration_y = []
    for domain in prepared.values():
        pooled_oof.extend(domain["oof_results"])
        pooled_train_y.extend(domain["y_train"])
        pooled_calibration.extend(domain["calibration_results"])
        pooled_calibration_y.extend(domain["y_calibration"])

    global_guard = CAVGuardFusion(
        power=FUSION_POWER,
        credibility_threshold=CREDIBILITY_THRESHOLD,
    ).fit(pooled_oof, pooled_train_y)
    hierarchical_guard = HierarchicalCAVGuardFusion(
        power=FUSION_POWER,
        credibility_threshold=CREDIBILITY_THRESHOLD,
        familywise_error_rate=0.05,
        seed=seed,
        local_score_thresholds=LOCAL_SCORE_THRESHOLDS,
        require_local_certification=True,
    ).fit(
        pooled_oof,
        pooled_train_y,
        calibration_modality_results_list=pooled_calibration,
        calibration_y=pooled_calibration_y,
    )

    domain_results = {}
    all_test_modality_sets = []
    all_test_labels = []
    all_reference_predictions = []
    all_hierarchical_predictions = []
    all_selected_paths = []
    for name, domain in prepared.items():
        predictions, selected_paths = path_predictions(
            domain["test_results"], global_guard, hierarchical_guard
        )
        all_indices = list(range(len(domain["y_test"])))
        target_indices = domain["target_indices"]
        metrics = {
            "overall": {
                method: accuracy_and_f1(
                    predictions[method], domain["y_test"], all_indices
                )
                for method in METHODS
            },
            "target_subset": {
                method: accuracy_and_f1(
                    predictions[method], domain["y_test"], target_indices
                )
                for method in METHODS
            },
        }
        target_decomposition = conflict_activation_value(
            [domain["test_results"][index] for index in target_indices],
            [predictions["equal_weight"][index] for index in target_indices],
            [predictions["hierarchical_cav_guard"][index]
             for index in target_indices],
            [domain["y_test"][index] for index in target_indices],
        )
        domain_results[name] = {
            "split_indices": domain["split_indices"],
            "n_train": len(domain["y_train"]),
            "n_calibration": len(domain["y_calibration"]),
            "n_test": len(domain["y_test"]),
            "target_subset_definition": encoded_domains[name][0].target_subset,
            "metrics": metrics,
            "target_candidate_decisions": sum(
                selected_paths[index] == "confidence_weighted"
                for index in target_indices
            ),
            "target_candidate_rate": float(np.mean([
                selected_paths[index] == "confidence_weighted"
                for index in target_indices
            ])),
            "target_subset_hierarchical_guard_vs_equal_cav": (
                target_decomposition
            ),
        }
        all_test_modality_sets.extend(domain["test_results"])
        all_test_labels.extend(domain["y_test"])
        all_reference_predictions.extend(predictions["equal_weight"])
        all_hierarchical_predictions.extend(
            predictions["hierarchical_cav_guard"]
        )
        all_selected_paths.extend(selected_paths)

    pooled_decomposition = conflict_activation_value(
        all_test_modality_sets,
        all_reference_predictions,
        all_hierarchical_predictions,
        all_test_labels,
    )
    hierarchical_stats = asdict(hierarchical_guard.stats_)
    hierarchical_stats.update({
        "test_candidate_decisions": sum(
            path == "confidence_weighted" for path in all_selected_paths
        ),
        "test_candidate_rate": float(np.mean([
            path == "confidence_weighted" for path in all_selected_paths
        ])),
    })
    return {
        "seed": seed,
        "global_guard_training_stats": asdict(global_guard.stats_),
        "hierarchical_guard_training_stats": hierarchical_stats,
        "pooled_test_hierarchical_guard_vs_equal_cav": pooled_decomposition,
        "domains": domain_results,
    }


def summarize(partitions, domain_names):
    summary = {
        "n_partitions": len(partitions),
        "note": (
            "Calibration and test sets are disjoint within every partition. "
            "The 30 repeated partitions overlap and are a sensitivity "
            "analysis rather than independent replications."
        ),
        "calibration_modes": {},
        "global_candidate_enabled": sum(
            p["global_guard_training_stats"]["candidate_enabled"]
            for p in partitions
        ),
        "local_threshold_certified": sum(
            p["hierarchical_guard_training_stats"]["local_model_fitted"]
            for p in partitions
        ),
        "mean_pooled_test_candidate_rate": float(np.mean([
            p["hierarchical_guard_training_stats"]["test_candidate_rate"]
            for p in partitions
        ])),
        "mean_pooled_test_switch_value": float(np.mean([
            p["pooled_test_hierarchical_guard_vs_equal_cav"]["switch_value_V"]
            for p in partitions
        ])),
        "domains": {},
    }
    modes = sorted({
        p["hierarchical_guard_training_stats"]["calibration_mode"]
        for p in partitions
    })
    summary["calibration_modes"] = {
        mode: sum(
            p["hierarchical_guard_training_stats"]["calibration_mode"] == mode
            for p in partitions
        )
        for mode in modes
    }
    for name in domain_names:
        domain_summary = {
            "mean_target_candidate_rate": float(np.mean([
                p["domains"][name]["target_candidate_rate"]
                for p in partitions
            ])),
            "mean_target_switch_value": float(np.mean([
                p["domains"][name]
                ["target_subset_hierarchical_guard_vs_equal_cav"]
                ["switch_value_V"]
                for p in partitions
            ])),
        }
        for scope in ("overall", "target_subset"):
            scope_summary = {"methods": {}, "comparisons": {}}
            for method in METHODS:
                values = np.asarray([
                    p["domains"][name]["metrics"][scope][method]["accuracy"]
                    for p in partitions
                ])
                scope_summary["methods"][method] = {
                    "mean_accuracy": float(values.mean()),
                    "std_accuracy": float(values.std()),
                    "min_accuracy": float(values.min()),
                    "max_accuracy": float(values.max()),
                }
            hierarchical = np.asarray([
                p["domains"][name]["metrics"][scope]
                ["hierarchical_cav_guard"]["accuracy"]
                for p in partitions
            ])
            for baseline in METHODS[:-1]:
                baseline_values = np.asarray([
                    p["domains"][name]["metrics"][scope][baseline]["accuracy"]
                    for p in partitions
                ])
                gaps = hierarchical - baseline_values
                scope_summary["comparisons"][baseline] = {
                    "mean_gap": float(gaps.mean()),
                    "std_gap": float(gaps.std()),
                    "partitions_better": int(np.sum(gaps > 0)),
                    "partitions_tied": int(np.sum(gaps == 0)),
                    "partitions_worse": int(np.sum(gaps < 0)),
                }
            domain_summary[scope] = scope_summary
        summary["domains"][name] = domain_summary
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
            "cav_guard_pooled_independent_calibration.json"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    encoded_domains = {
        name: (DOMAIN_SPECS[name], *encode_domain(DOMAIN_SPECS[name]))
        for name in args.domains
    }
    partitions = []
    for seed in range(args.seeds):
        result = evaluate_partition(encoded_domains, seed)
        partitions.append(result)
        stats = result["hierarchical_guard_training_stats"]
        domain_text = " ".join(
            f"{name}:EW={result['domains'][name]['metrics']['target_subset']['equal_weight']['accuracy']:.4f},"
            f"H={result['domains'][name]['metrics']['target_subset']['hierarchical_cav_guard']['accuracy']:.4f}"
            for name in args.domains
        )
        print(
            f"[pooled] seed={seed:2d} mode={stats['calibration_mode']} "
            f"local={stats['local_model_fitted']} "
            f"rate={stats['test_candidate_rate']:.3f} {domain_text}",
            flush=True,
        )
    fixed = evaluate_partition(encoded_domains, args.fixed_seed)
    artifact = {
        "method": "H-CAV-Guard pooled independent-calibration validation",
        "settings": {
            "domains": args.domains,
            "train_fraction_per_domain": TRAIN_FRACTION,
            "independent_calibration_fraction_per_domain": CALIBRATION_FRACTION,
            "test_fraction_per_domain": TEST_FRACTION,
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
        "partitions": partitions,
        "summary": summarize(partitions, args.domains),
        "fixed_split": fixed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as stream:
        json.dump(artifact, stream, indent=2, allow_nan=False)
    print(json.dumps(artifact["summary"], indent=2, allow_nan=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
