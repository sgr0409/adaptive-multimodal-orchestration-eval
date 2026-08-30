"""Diagnostics for paired equal- and confidence-weighted fusion outputs."""

import math


def conflict_activation_value(modality_sets, equal_labels, confidence_labels, true_labels):
    """Return the exact C*A*V decomposition and confidence-ranking lift.

    ``modality_sets`` is a sequence of per-example modality-result lists.
    The diagnostic is post-hoc: true labels evaluate already-produced
    decisions but never enter either fusion rule.
    """
    n = len(true_labels)
    if not (len(modality_sets) == len(equal_labels) == len(confidence_labels) == n):
        raise ValueError("All diagnostic inputs must have identical lengths")

    conflicts = switches = paired_gain = opportunities = 0
    top_correct_sum = random_correct_sum = 0.0
    for mods, ew, cw, true in zip(
        modality_sets, equal_labels, confidence_labels, true_labels
    ):
        conflict = len({modality["label"] for modality in mods}) > 1
        switch = ew != cw
        if switch and not conflict:
            raise AssertionError("Fusion changed decision without modality argmax conflict")
        conflicts += conflict
        switches += switch
        paired_gain += int(cw == true) - int(ew == true)

        correct = [modality["label"] == true for modality in mods]
        opportunity = any(correct) and not all(correct)
        if opportunity:
            opportunities += 1
            highest = max(modality["confidence"] for modality in mods)
            tied = [
                modality
                for modality in mods
                if math.isclose(
                    modality["confidence"], highest, rel_tol=0.0, abs_tol=1e-15
                )
            ]
            top_correct_sum += sum(modality["label"] == true for modality in tied) / len(tied)
            random_correct_sum += sum(correct) / len(correct)

    c_value = conflicts / n
    a_value = switches / conflicts if conflicts else 0.0
    v_value = paired_gain / switches if switches else 0.0
    accuracy_difference = paired_gain / n
    ranking_accuracy = top_correct_sum / opportunities if opportunities else 0.0
    uniform_accuracy = random_correct_sum / opportunities if opportunities else 0.0
    product = c_value * a_value * v_value
    if not math.isclose(accuracy_difference, product, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("Conflict--activation--value identity did not hold")
    return {
        "n": n,
        "conflicts": conflicts,
        "switches": switches,
        "opportunities": opportunities,
        "conflict_rate_C": c_value,
        "activation_A": a_value,
        "switch_value_V": v_value,
        "accuracy_difference": accuracy_difference,
        "decomposition_product": product,
        "ranking_accuracy": ranking_accuracy,
        "uniform_modality_accuracy": uniform_accuracy,
        "ranking_lift": ranking_accuracy - uniform_accuracy,
    }


def partition_summary(per_partition):
    """Descriptive summary only; overlapping partitions are not independent."""
    metrics = (
        "conflict_rate_C",
        "activation_A",
        "switch_value_V",
        "accuracy_difference",
        "ranking_lift",
    )
    summary = {}
    for metric in metrics:
        values = [partition[metric] for partition in per_partition]
        summary[metric] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    summary["partitions_positive_switch_value"] = sum(
        partition["switch_value_V"] > 0 for partition in per_partition
    )
    summary["partitions_negative_switch_value"] = sum(
        partition["switch_value_V"] < 0 for partition in per_partition
    )
    summary["partitions_zero_switch_value"] = sum(
        partition["switch_value_V"] == 0 for partition in per_partition
    )
    return summary
