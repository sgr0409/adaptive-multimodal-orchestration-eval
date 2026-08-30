"""Evaluate CAV-Guard against fixed and learned late-fusion baselines.

CAV-Guard is fit only on five-fold out-of-fold predictions from each
partition's training half.  It never receives test labels or degradation
annotations.  The same pre-specified settings are used in all four domains:
confidence power 3, a uniform Beta(1, 1) prior, and a 0.95 posterior
credibility threshold for enabling the confidence-weighted candidate path.

The 30 random partitions overlap and are reported as a sensitivity analysis,
not independent replications.  Seed 42 is also evaluated separately to match
the paper's fixed-split tables.
"""

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from framework.cav_guard_fusion import (
    CAVGuardFusion,
    HierarchicalCAVGuardFusion,
)
from framework.fusion import confidence_weighted_fusion, equal_weight_fusion
from framework.fusion_diagnostics import conflict_activation_value
from framework.image_scorer import ImageScorer
from framework.learned_fusion import LearnedGatingFusion, StackingFusion
from framework.telemetry_scorer import extract_features as telemetry_features
from framework.text_scorer import TextScorer


@dataclass(frozen=True)
class DomainSpec:
    name: str
    data_dir: Path
    has_telemetry: bool
    target_subset: str


DOMAIN_SPECS = {
    "domain1": DomainSpec("domain1", ROOT / "data", True, "two_degraded"),
    "domain2": DomainSpec(
        "domain2", ROOT / "data_it_incidents", True, "two_degraded"
    ),
    "crisismmd": DomainSpec(
        "crisismmd", ROOT / "data_public_crisismmd", False, "all"
    ),
    "mmimdb": DomainSpec(
        "mmimdb", ROOT / "data_public_mmimdb", False, "all"
    ),
}

METHODS = (
    "equal_weight",
    "confidence_weighted",
    "stacking",
    "learned_gating",
    "oof_accuracy_selector",
    "cav_guard",
    "hierarchical_cav_guard",
)
N_FOLDS = 5
FUSION_POWER = 3
CREDIBILITY_THRESHOLD = 0.95


def fit_predict(x_train, y_train, x_eval, modality_name):
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42),
    )
    classifier.fit(x_train, y_train)
    probabilities = classifier.predict_proba(x_eval)
    classes = list(classifier.classes_)
    results = []
    for probability in probabilities:
        best = int(np.argmax(probability))
        results.append({
            "modality": modality_name,
            "label": classes[best],
            "confidence": float(probability[best]),
            "probs": {
                label: float(value)
                for label, value in zip(classes, probability)
            },
        })
    return results


def encode_domain(spec):
    scenarios = [
        json.loads(line) for line in open(spec.data_dir / "scenarios.jsonl")
    ]
    print(f"[{spec.name}] encoding {len(scenarios)} examples", flush=True)
    text_embeddings = TextScorer().encoder.encode(
        [row["text"] for row in scenarios], show_progress_bar=False
    )
    image_embeddings = ImageScorer().embed([
        str(spec.data_dir / row["image_path"]) for row in scenarios
    ])
    arrays = [("text", np.asarray(text_embeddings)),
              ("image", np.asarray(image_embeddings))]
    if spec.has_telemetry:
        arrays.append((
            "telemetry",
            np.asarray([
                telemetry_features(row["telemetry"]) for row in scenarios
            ]),
        ))
    return scenarios, arrays


def out_of_fold_predictions(arrays, labels, train_idx, seed):
    predictions = [[None] * len(train_idx) for _ in arrays]
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fit_local, hold_local in folds.split(np.arange(len(train_idx))):
        fit_global = train_idx[fit_local]
        hold_global = train_idx[hold_local]
        y_fit = [labels[index] for index in fit_global]
        for modality_index, (name, values) in enumerate(arrays):
            fold_predictions = fit_predict(
                values[fit_global], y_fit, values[hold_global], name
            )
            for prediction_index, local_index in enumerate(hold_local):
                predictions[modality_index][local_index] = (
                    fold_predictions[prediction_index]
                )
    return [
        [modality_predictions[index]
         for modality_predictions in predictions]
        for index in range(len(train_idx))
    ]


def full_train_predictions(arrays, labels, train_idx, test_idx):
    y_train = [labels[index] for index in train_idx]
    predictions = [
        fit_predict(values[train_idx], y_train, values[test_idx], name)
        for name, values in arrays
    ]
    return [
        [modality_predictions[index]
         for modality_predictions in predictions]
        for index in range(len(test_idx))
    ]


def subset_indices(spec, scenarios, test_idx):
    if spec.target_subset == "all":
        return list(range(len(test_idx)))
    return [
        local_index for local_index, global_index in enumerate(test_idx)
        if len(scenarios[global_index]["degraded_modalities"]) == 2
    ]


def accuracy_and_f1(predictions, labels, indices):
    selected_predictions = [predictions[index] for index in indices]
    selected_labels = [labels[index] for index in indices]
    return {
        "n": len(indices),
        "accuracy": float(np.mean(
            np.asarray(selected_predictions) == np.asarray(selected_labels)
        )),
        "macro_f1": float(f1_score(
            selected_labels, selected_predictions, average="macro"
        )),
    }


def evaluate_partition(spec, scenarios, arrays, seed):
    labels = [row["label"] for row in scenarios]
    shuffled = list(range(len(scenarios)))
    random.Random(seed).shuffle(shuffled)
    split = len(shuffled) // 2
    train_idx = np.asarray(shuffled[:split])
    test_idx = np.asarray(shuffled[split:])
    y_train = [labels[index] for index in train_idx]
    y_test = [labels[index] for index in test_idx]

    oof_results = out_of_fold_predictions(arrays, labels, train_idx, seed)
    test_results = full_train_predictions(
        arrays, labels, train_idx, test_idx
    )

    stacking = StackingFusion()
    stacking.fit(oof_results, y_train)
    learned_gating = LearnedGatingFusion(seed=seed)
    learned_gating.fit(oof_results, y_train)
    cav_guard = CAVGuardFusion(
        power=FUSION_POWER,
        credibility_threshold=CREDIBILITY_THRESHOLD,
    )
    cav_guard.fit(oof_results, y_train)
    oof_accuracy_candidate_enabled = (
        cav_guard.stats_.benefits > cav_guard.stats_.harms
    )
    hierarchical_cav_guard = HierarchicalCAVGuardFusion(
        power=FUSION_POWER,
        credibility_threshold=CREDIBILITY_THRESHOLD,
        familywise_error_rate=0.05,
        seed=seed,
    )
    hierarchical_cav_guard.fit(oof_results, y_train)

    predictions = {method: [] for method in METHODS}
    modality_sets = []
    hierarchical_selected_paths = []
    for modality_results in test_results:
        modality_sets.append(modality_results)
        predictions["equal_weight"].append(
            equal_weight_fusion(modality_results)["label"]
        )
        predictions["confidence_weighted"].append(
            confidence_weighted_fusion(
                modality_results, power=FUSION_POWER
            )["label"]
        )
        predictions["stacking"].append(
            stacking.predict(modality_results)["label"]
        )
        predictions["learned_gating"].append(
            learned_gating.predict(modality_results)["label"]
        )
        predictions["oof_accuracy_selector"].append(
            confidence_weighted_fusion(
                modality_results, power=FUSION_POWER
            )["label"]
            if oof_accuracy_candidate_enabled
            else equal_weight_fusion(modality_results)["label"]
        )
        predictions["cav_guard"].append(
            cav_guard.predict(modality_results)["label"]
        )
        hierarchical_result = hierarchical_cav_guard.predict(modality_results)
        predictions["hierarchical_cav_guard"].append(
            hierarchical_result["label"]
        )
        hierarchical_selected_paths.append(
            hierarchical_result["selected_path"]
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
    target_modality_sets = [modality_sets[index] for index in target_indices]
    target_labels = [y_test[index] for index in target_indices]
    guard_decomposition = conflict_activation_value(
        target_modality_sets,
        [predictions["equal_weight"][index] for index in target_indices],
        [predictions["cav_guard"][index] for index in target_indices],
        target_labels,
    )
    hierarchical_guard_decomposition = conflict_activation_value(
        target_modality_sets,
        [predictions["equal_weight"][index] for index in target_indices],
        [predictions["hierarchical_cav_guard"][index]
         for index in target_indices],
        target_labels,
    )
    stats = asdict(cav_guard.stats_)
    stats["switch_value"] = cav_guard.stats_.switch_value
    hierarchical_stats = asdict(hierarchical_cav_guard.stats_)
    hierarchical_stats["test_candidate_decisions"] = sum(
        path == "confidence_weighted"
        for path in hierarchical_selected_paths
    )
    hierarchical_stats["test_candidate_rate"] = (
        hierarchical_stats["test_candidate_decisions"] / len(test_idx)
    )
    hierarchical_stats["target_candidate_decisions"] = sum(
        hierarchical_selected_paths[index] == "confidence_weighted"
        for index in target_indices
    )
    hierarchical_stats["target_candidate_rate"] = (
        hierarchical_stats["target_candidate_decisions"] / len(target_indices)
    )
    return {
        "seed": seed,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "target_subset_definition": spec.target_subset,
        "guard_training_stats": stats,
        "oof_accuracy_selector_candidate_enabled": (
            oof_accuracy_candidate_enabled
        ),
        "hierarchical_guard_training_stats": hierarchical_stats,
        "metrics": metrics,
        "target_subset_guard_vs_equal_cav": guard_decomposition,
        "target_subset_hierarchical_guard_vs_equal_cav": (
            hierarchical_guard_decomposition
        ),
    }


def summarize(partitions):
    summary = {
        "n_partitions": len(partitions),
        "note": (
            "Partitions overlap and summaries are descriptive sensitivity "
            "analyses, not independent-replication inference."
        ),
        "selected_path_counts": {
            "confidence_weighted": sum(
                partition["guard_training_stats"]["candidate_enabled"]
                for partition in partitions
            ),
            "equal_weight": sum(
                not partition["guard_training_stats"]["candidate_enabled"]
                for partition in partitions
            ),
        },
        "oof_accuracy_selector": {
            "confidence_weighted": sum(
                partition["oof_accuracy_selector_candidate_enabled"]
                for partition in partitions
            ),
            "equal_weight": sum(
                not partition["oof_accuracy_selector_candidate_enabled"]
                for partition in partitions
            ),
        },
        "hierarchical_guard": {
            "global_candidate_enabled": sum(
                partition["hierarchical_guard_training_stats"]
                ["global_guard"]["candidate_enabled"]
                for partition in partitions
            ),
            "local_model_fitted": sum(
                partition["hierarchical_guard_training_stats"]
                ["local_model_fitted"]
                for partition in partitions
            ),
            "mean_test_candidate_rate": float(np.mean([
                partition["hierarchical_guard_training_stats"]
                ["test_candidate_rate"]
                for partition in partitions
            ])),
            "mean_target_candidate_rate": float(np.mean([
                partition["hierarchical_guard_training_stats"]
                ["target_candidate_rate"]
                for partition in partitions
            ])),
            "mean_target_switch_rate": float(np.mean([
                partition["target_subset_hierarchical_guard_vs_equal_cav"]
                ["switches"]
                / partition["target_subset_hierarchical_guard_vs_equal_cav"]
                ["n"]
                for partition in partitions
            ])),
            "mean_target_switch_value": float(np.mean([
                partition["target_subset_hierarchical_guard_vs_equal_cav"]
                ["switch_value_V"]
                for partition in partitions
            ])),
            "calibration_modes": {
                mode: sum(
                    partition["hierarchical_guard_training_stats"]
                    ["calibration_mode"] == mode
                    for partition in partitions
                )
                for mode in sorted({
                    partition["hierarchical_guard_training_stats"]
                    ["calibration_mode"]
                    for partition in partitions
                })
            },
        },
    }
    for scope in ("overall", "target_subset"):
        scope_summary = {
            "methods": {},
            "hierarchical_cav_guard_comparisons": {},
        }
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
        guard_values = np.asarray([
            partition["metrics"][scope]["hierarchical_cav_guard"]["accuracy"]
            for partition in partitions
        ])
        for baseline in METHODS[:-1]:
            baseline_values = np.asarray([
                partition["metrics"][scope][baseline]["accuracy"]
                for partition in partitions
            ])
            gaps = guard_values - baseline_values
            scope_summary["hierarchical_cav_guard_comparisons"][baseline] = {
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
        default=ROOT / "experiments/results/cav_guard_multidomain.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    artifact = {
        "method": (
            "Hierarchical CAV-Guard with global permission and "
            "instance-selective switch-value calibration"
        ),
        "settings": {
            "n_folds": N_FOLDS,
            "confidence_power": FUSION_POWER,
            "beta_prior": [1, 1],
            "posterior_credibility_threshold": CREDIBILITY_THRESHOLD,
            "local_familywise_error_rate": 0.05,
            "sensitivity_partition_seeds": list(range(args.seeds)),
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
            guard = result["guard_training_stats"]
            hierarchical = result["hierarchical_guard_training_stats"]
            target = result["metrics"]["target_subset"]
            print(
                f"[{domain_name}] seed={seed:2d} "
                f"B/H={guard['benefits']}/{guard['harms']} "
                f"P(V>0)={guard['posterior_probability_positive']:.3f} "
                f"path={'CW' if guard['candidate_enabled'] else 'EW'} "
                f"H-local={hierarchical['local_model_fitted']} "
                f"H-rate={hierarchical['test_candidate_rate']:.3f} "
                f"target H-guard={target['hierarchical_cav_guard']['accuracy']:.4f}",
                flush=True,
            )
        fixed = evaluate_partition(spec, scenarios, arrays, args.fixed_seed)
        artifact["domains"][domain_name] = {
            "sensitivity_partitions": partitions,
            "sensitivity_summary": summarize(partitions),
            "fixed_split": fixed,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as stream:
            json.dump(artifact, stream, indent=2)
        print(f"[{domain_name}] checkpointed {args.output}", flush=True)
    print(json.dumps({
        domain: artifact["domains"][domain]["sensitivity_summary"]
        for domain in artifact["domains"]
    }, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
