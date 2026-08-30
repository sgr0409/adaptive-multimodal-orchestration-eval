"""Leave-one-domain-out transfer validation for strict H-CAV-Guard.

For each held-out domain, the global gate, local benefit model, and fixed-grid
calibration threshold use only the other three domains.  The held-out domain
supplies training labels solely to fit its task-specific base scorers; none of
its labels or predictions enter H-CAV policy fitting or calibration.  The
frozen source-domain guard is evaluated once on the held-out test split.

Each domain is stratified into the same 40% base-model training, 30%
calibration, and 30% test split used by the pooled independent-calibration
study.  The held-out calibration split remains unused by the guard so the
artifact can audit both sample separation and domain exclusion.
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
)
from experiments.cav_guard_multidomain import (
    CREDIBILITY_THRESHOLD,
    DOMAIN_SPECS,
    FUSION_POWER,
    ROOT,
    accuracy_and_f1,
    encode_domain,
)
from experiments.cav_guard_pooled_independent_calibration import (
    prepare_domain_partition,
)
from framework.cav_guard_fusion import CAVGuardFusion, HierarchicalCAVGuardFusion
from framework.fusion_diagnostics import conflict_activation_value


def source_domains_for(domain_names, held_out_domain):
    if held_out_domain not in domain_names:
        raise ValueError(f"Unknown held-out domain: {held_out_domain}")
    return tuple(name for name in domain_names if name != held_out_domain)


def fit_source_guard(prepared, source_domains, seed):
    pooled_oof = []
    pooled_train_y = []
    pooled_calibration = []
    pooled_calibration_y = []
    for name in source_domains:
        domain = prepared[name]
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
    return global_guard, hierarchical_guard


def evaluate_held_out(prepared, held_out_domain, source_domains, seed):
    global_guard, hierarchical_guard = fit_source_guard(
        prepared, source_domains, seed
    )
    target = prepared[held_out_domain]
    predictions, selected_paths = path_predictions(
        target["test_results"], global_guard, hierarchical_guard
    )
    all_indices = list(range(len(target["y_test"])))
    target_indices = target["target_indices"]
    metrics = {
        "overall": {
            method: accuracy_and_f1(
                predictions[method], target["y_test"], all_indices
            )
            for method in METHODS
        },
        "target_subset": {
            method: accuracy_and_f1(
                predictions[method], target["y_test"], target_indices
            )
            for method in METHODS
        },
    }
    decomposition = conflict_activation_value(
        [target["test_results"][index] for index in target_indices],
        [predictions["equal_weight"][index] for index in target_indices],
        [predictions["hierarchical_cav_guard"][index]
         for index in target_indices],
        [target["y_test"][index] for index in target_indices],
    )
    hierarchical_stats = asdict(hierarchical_guard.stats_)
    hierarchical_stats.update({
        "held_out_test_candidate_decisions": sum(
            path == "confidence_weighted" for path in selected_paths
        ),
        "held_out_test_candidate_rate": float(np.mean([
            path == "confidence_weighted" for path in selected_paths
        ])),
        "held_out_target_candidate_decisions": sum(
            selected_paths[index] == "confidence_weighted"
            for index in target_indices
        ),
        "held_out_target_candidate_rate": float(np.mean([
            selected_paths[index] == "confidence_weighted"
            for index in target_indices
        ])),
    })
    return {
        "held_out_domain": held_out_domain,
        "guard_source_domains": list(source_domains),
        "held_out_guard_training_examples": 0,
        "held_out_guard_calibration_examples": 0,
        "source_guard_training_examples": sum(
            len(prepared[name]["y_train"]) for name in source_domains
        ),
        "source_guard_calibration_examples": sum(
            len(prepared[name]["y_calibration"]) for name in source_domains
        ),
        "global_guard_training_stats": asdict(global_guard.stats_),
        "hierarchical_guard_training_stats": hierarchical_stats,
        "metrics": metrics,
        "target_subset_hierarchical_guard_vs_equal_cav": decomposition,
    }


def evaluate_partition(encoded_domains, seed):
    prepared = {
        name: prepare_domain_partition(spec, scenarios, arrays, seed)
        for name, (spec, scenarios, arrays) in encoded_domains.items()
    }
    domain_names = tuple(encoded_domains)
    held_out = {}
    for name in domain_names:
        sources = source_domains_for(domain_names, name)
        held_out[name] = evaluate_held_out(
            prepared, name, sources, seed
        )
    return {
        "seed": seed,
        "split_indices": {
            name: prepared[name]["split_indices"] for name in domain_names
        },
        "held_out_evaluations": held_out,
    }


def comparison(values, baseline_values):
    gaps = np.asarray(values) - np.asarray(baseline_values)
    return {
        "mean_gap": float(gaps.mean()),
        "std_gap": float(gaps.std()),
        "partitions_better": int(np.sum(gaps > 0)),
        "partitions_tied": int(np.sum(gaps == 0)),
        "partitions_worse": int(np.sum(gaps < 0)),
    }


def summarize(partitions, domain_names):
    summary = {
        "n_partitions": len(partitions),
        "note": (
            "Within each held-out evaluation, no held-out-domain example "
            "enters global/local guard fitting or calibration. Repeated "
            "partitions overlap and remain a sensitivity analysis."
        ),
        "held_out_domains": {},
    }
    total_wins = total_ties = total_losses = 0
    for name in domain_names:
        evaluations = [
            partition["held_out_evaluations"][name]
            for partition in partitions
        ]
        domain_summary = {
            "guard_source_domains": evaluations[0]["guard_source_domains"],
            "global_candidate_enabled": sum(
                evaluation["global_guard_training_stats"]["candidate_enabled"]
                for evaluation in evaluations
            ),
            "local_threshold_certified": sum(
                evaluation["hierarchical_guard_training_stats"]
                ["local_model_fitted"]
                for evaluation in evaluations
            ),
            "calibration_modes": {},
            "mean_target_candidate_rate": float(np.mean([
                evaluation["hierarchical_guard_training_stats"]
                ["held_out_target_candidate_rate"]
                for evaluation in evaluations
            ])),
            "mean_target_switch_value": float(np.mean([
                evaluation["target_subset_hierarchical_guard_vs_equal_cav"]
                ["switch_value_V"]
                for evaluation in evaluations
            ])),
        }
        modes = sorted({
            evaluation["hierarchical_guard_training_stats"]["calibration_mode"]
            for evaluation in evaluations
        })
        domain_summary["calibration_modes"] = {
            mode: sum(
                evaluation["hierarchical_guard_training_stats"]
                ["calibration_mode"] == mode
                for evaluation in evaluations
            )
            for mode in modes
        }
        for scope in ("overall", "target_subset"):
            scope_summary = {"methods": {}, "comparisons": {}}
            for method in METHODS:
                values = np.asarray([
                    evaluation["metrics"][scope][method]["accuracy"]
                    for evaluation in evaluations
                ])
                scope_summary["methods"][method] = {
                    "mean_accuracy": float(values.mean()),
                    "std_accuracy": float(values.std()),
                    "min_accuracy": float(values.min()),
                    "max_accuracy": float(values.max()),
                }
            hierarchical = [
                evaluation["metrics"][scope]
                ["hierarchical_cav_guard"]["accuracy"]
                for evaluation in evaluations
            ]
            for baseline in METHODS[:-1]:
                baseline_values = [
                    evaluation["metrics"][scope][baseline]["accuracy"]
                    for evaluation in evaluations
                ]
                scope_summary["comparisons"][baseline] = comparison(
                    hierarchical, baseline_values
                )
            domain_summary[scope] = scope_summary
        counts = domain_summary["target_subset"]["comparisons"]["equal_weight"]
        total_wins += counts["partitions_better"]
        total_ties += counts["partitions_tied"]
        total_losses += counts["partitions_worse"]
        summary["held_out_domains"][name] = domain_summary
    summary["all_held_out_domain_partition_outcomes_vs_equal_weight"] = {
        "better": total_wins,
        "tied": total_ties,
        "worse": total_losses,
    }
    summary["held_out_suite"] = {}
    for scope in ("overall", "target_subset"):
        reference_accuracies = []
        guard_accuracies = []
        candidate_rates = []
        for partition in partitions:
            reference_correct = 0.0
            guard_correct = 0.0
            candidate_decisions = 0
            n_examples = 0
            for name in domain_names:
                evaluation = partition["held_out_evaluations"][name]
                metrics = evaluation["metrics"][scope]
                n = metrics["equal_weight"]["n"]
                n_examples += n
                reference_correct += n * metrics["equal_weight"]["accuracy"]
                guard_correct += (
                    n * metrics["hierarchical_cav_guard"]["accuracy"]
                )
                stats = evaluation["hierarchical_guard_training_stats"]
                candidate_decisions += stats[
                    "held_out_test_candidate_decisions"
                    if scope == "overall"
                    else "held_out_target_candidate_decisions"
                ]
            reference_accuracies.append(reference_correct / n_examples)
            guard_accuracies.append(guard_correct / n_examples)
            candidate_rates.append(candidate_decisions / n_examples)
        summary["held_out_suite"][scope] = {
            "mean_equal_weight_accuracy": float(np.mean(reference_accuracies)),
            "mean_hierarchical_cav_guard_accuracy": float(
                np.mean(guard_accuracies)
            ),
            "mean_candidate_rate": float(np.mean(candidate_rates)),
            "comparison_to_equal_weight": comparison(
                guard_accuracies, reference_accuracies
            ),
        }
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
            "cav_guard_leave_one_domain_out.json"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.domains) < 2:
        raise ValueError("Leave-one-domain-out evaluation requires at least two domains")
    encoded_domains = {
        name: (DOMAIN_SPECS[name], *encode_domain(DOMAIN_SPECS[name]))
        for name in args.domains
    }
    partitions = []
    for seed in range(args.seeds):
        result = evaluate_partition(encoded_domains, seed)
        partitions.append(result)
        fields = []
        for name in args.domains:
            evaluation = result["held_out_evaluations"][name]
            stats = evaluation["hierarchical_guard_training_stats"]
            target = evaluation["metrics"]["target_subset"]
            fields.append(
                f"{name}:mode={stats['calibration_mode']},"
                f"rate={stats['held_out_target_candidate_rate']:.3f},"
                f"EW={target['equal_weight']['accuracy']:.4f},"
                f"H={target['hierarchical_cav_guard']['accuracy']:.4f}"
            )
        print(f"[LODO] seed={seed:2d} " + " ".join(fields), flush=True)
    fixed = evaluate_partition(encoded_domains, args.fixed_seed)
    artifact = {
        "method": "H-CAV-Guard leave-one-domain-out transfer validation",
        "settings": {
            "domains": args.domains,
            "base_model_train_fraction_per_domain": TRAIN_FRACTION,
            "source_guard_calibration_fraction_per_domain": CALIBRATION_FRACTION,
            "held_out_test_fraction_per_domain": TEST_FRACTION,
            "held_out_domain_calibration_usage": "none",
            "held_out_domain_guard_training_usage": "none",
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
