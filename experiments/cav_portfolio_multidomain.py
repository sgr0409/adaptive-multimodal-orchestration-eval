"""Evaluate a leakage-safe CAV fusion-policy portfolio on four domains.

Equal weighting is the protected reference.  The fixed candidate portfolio is
confidence weighting, stacking, and learned gating.  Base scorers first
produce five-fold OOF predictions.  Stacking and learned gating then receive a
second, stratified cross-fit so the portfolio guard never scores either
meta-learner on an example used to fit that meta-learner.  The corrected CAV
selector is descriptive in this 50/50 sensitivity study because its paired
evidence is cross-fitted rather than independently calibrated.
"""

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments.cav_guard_multidomain import (
    DOMAIN_SPECS,
    FUSION_POWER,
    accuracy_and_f1,
    encode_domain,
    full_train_predictions,
    out_of_fold_predictions,
    subset_indices,
)
from framework.cav_portfolio_guard import CAVPortfolioGuardFusion
from framework.fusion import confidence_weighted_fusion, equal_weight_fusion
from framework.fusion_diagnostics import conflict_activation_value
from framework.learned_fusion import LearnedGatingFusion, StackingFusion


REFERENCE = "equal_weight"
CANDIDATES = ("confidence_weighted", "stacking", "learned_gating")
BASE_PATHS = (REFERENCE,) + CANDIDATES
METHODS = BASE_PATHS + ("naive_oof_selector", "cav_portfolio")
N_META_FOLDS = 5
FAMILYWISE_ERROR_RATE = 0.05
MINIMUM_DECISIVE_SWITCHES = 5


def fit_candidate_models(oof_modality_sets, labels, seed):
    stacking = StackingFusion()
    stacking.fit(oof_modality_sets, labels)
    learned_gating = LearnedGatingFusion(seed=seed)
    learned_gating.fit(oof_modality_sets, labels)
    return stacking, learned_gating


def path_results(modality_sets, stacking, learned_gating):
    paths = {name: [] for name in BASE_PATHS}
    for modalities in modality_sets:
        paths[REFERENCE].append(equal_weight_fusion(modalities))
        paths["confidence_weighted"].append(
            confidence_weighted_fusion(modalities, power=FUSION_POWER)
        )
        paths["stacking"].append(stacking.predict(modalities))
        paths["learned_gating"].append(learned_gating.predict(modalities))
    return paths


def meta_cross_fitted_paths(oof_modality_sets, labels, seed):
    """Produce leakage-safe OOF predictions for every portfolio path."""
    n_examples = len(labels)
    paths = {name: [None] * n_examples for name in BASE_PATHS}
    labels_array = np.asarray(labels)
    folds = StratifiedKFold(
        n_splits=N_META_FOLDS, shuffle=True, random_state=seed
    )
    for meta_train, meta_holdout in folds.split(
            np.arange(n_examples), labels_array):
        fit_sets = [oof_modality_sets[index] for index in meta_train]
        fit_labels = [labels[index] for index in meta_train]
        stacking, learned_gating = fit_candidate_models(
            fit_sets, fit_labels, seed
        )
        for index in meta_holdout:
            modalities = oof_modality_sets[index]
            paths[REFERENCE][index] = equal_weight_fusion(modalities)
            paths["confidence_weighted"][index] = (
                confidence_weighted_fusion(
                    modalities, power=FUSION_POWER
                )
            )
            paths["stacking"][index] = stacking.predict(modalities)
            paths["learned_gating"][index] = learned_gating.predict(modalities)
    if any(result is None for values in paths.values() for result in values):
        raise RuntimeError("meta cross-fitting left an example without prediction")
    return paths


def accuracy(results, labels):
    return float(np.mean([
        result["label"] == label for result, label in zip(results, labels)
    ]))


def select_naive_path(paths, labels):
    """Raw OOF-accuracy selection with simpler-path tie breaking."""
    scores = {name: accuracy(paths[name], labels) for name in BASE_PATHS}
    selected = max(BASE_PATHS, key=lambda name: scores[name])
    return selected, scores


def selected_predictions(test_paths, selected_name):
    return [result["label"] for result in test_paths[selected_name]]


def evaluate_partition(spec, scenarios, arrays, seed):
    labels = [row["label"] for row in scenarios]
    shuffled = list(range(len(labels)))
    random.Random(seed).shuffle(shuffled)
    split = len(shuffled) // 2
    train_idx = np.asarray(shuffled[:split])
    test_idx = np.asarray(shuffled[split:])
    y_train = [labels[index] for index in train_idx]
    y_test = [labels[index] for index in test_idx]

    oof_modality_sets = out_of_fold_predictions(
        arrays, labels, train_idx, seed
    )
    cross_fitted_paths = meta_cross_fitted_paths(
        oof_modality_sets, y_train, seed
    )
    portfolio = CAVPortfolioGuardFusion(
        reference_name=REFERENCE,
        familywise_error_rate=FAMILYWISE_ERROR_RATE,
        minimum_decisive_switches=MINIMUM_DECISIVE_SWITCHES,
    ).fit(
        cross_fitted_paths[REFERENCE],
        {name: cross_fitted_paths[name] for name in CANDIDATES},
        y_train,
        calibration_mode="nested_cross_fitted_descriptive",
    )
    naive_path, naive_scores = select_naive_path(cross_fitted_paths, y_train)

    stacking, learned_gating = fit_candidate_models(
        oof_modality_sets, y_train, seed
    )
    test_modality_sets = full_train_predictions(
        arrays, labels, train_idx, test_idx
    )
    test_paths = path_results(test_modality_sets, stacking, learned_gating)
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
    target_decomposition = conflict_activation_value(
        [test_modality_sets[index] for index in target_indices],
        [predictions[REFERENCE][index] for index in target_indices],
        [predictions["cav_portfolio"][index] for index in target_indices],
        [y_test[index] for index in target_indices],
    )
    return {
        "seed": seed,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "target_subset_definition": spec.target_subset,
        "naive_oof_selected_path": naive_path,
        "naive_oof_path_accuracy": naive_scores,
        "portfolio_stats": asdict(portfolio.stats_),
        "oracle_test_selected_path": oracle_path,
        "oracle_test_target_accuracy": metrics["target_subset"][oracle_path][
            "accuracy"
        ],
        "metrics": metrics,
        "target_subset_portfolio_vs_equal_cav": target_decomposition,
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
        "note": (
            "Nested cross-fitted evidence is leakage-safe but descriptive; "
            "partitions overlap and are not independent replications."
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
                partition["naive_oof_selected_path"] == name
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
        oracle_values = np.asarray([
            partition["oracle_test_target_accuracy"] for partition in partitions
        ])
        if scope == "target_subset":
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
        default=ROOT / "experiments/results/cav_portfolio_multidomain.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    artifact = {
        "method": "CAV-Portfolio reference-preserving fusion-policy selection",
        "settings": {
            "reference": REFERENCE,
            "candidates": list(CANDIDATES),
            "base_oof_folds": 5,
            "meta_oof_folds": N_META_FOLDS,
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
