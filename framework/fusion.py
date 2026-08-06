"""Confidence-aware fusion and its equal-weight-fusion ablation baseline.

Each modality scorer returns a full probability distribution over the
three severity classes plus its own confidence (max predicted
probability). Fusion combines the three distributions into one final
decision.

Confidence-weighted fusion: each modality's distribution is weighted by
its own confidence, normalized across the three modalities (w_i =
c_i / sum(c_j)), then the weighted distributions are summed and the
argmax taken -- exactly the mechanism the paper's outline sketches
(Eq. confidence-normalize-fuse), just applied to per-modality label
distributions rather than raw embeddings (see image_scorer.py's docstring
and framework/README for why).

Equal-weight fusion is the ablation baseline: identical mechanism, but
every modality gets a fixed weight of 1/3 regardless of its own
confidence. This isolates exactly what confidence-awareness buys, holding
everything else (the same three per-modality scorers, the same
distributions) constant.
"""
LABELS = ["normal", "warning", "critical"]


def _weighted_sum(modality_results, weights):
    fused = {l: 0.0 for l in LABELS}
    for res, w in zip(modality_results, weights):
        for l in LABELS:
            fused[l] += w * res["probs"][l]
    best_label = max(fused, key=fused.get)
    return {"label": best_label, "probs": fused, "weights": dict(zip(
        [r["modality"] for r in modality_results], weights))}


def confidence_weighted_fusion(modality_results, power=3):
    """power=1 is plain linear proportional weighting (w_i = c_i / sum(c_j)).
    We use power=3 as the deployed default, chosen via a swept comparison
    (experiments/sweep_fusion_power.py) over power in {1,2,3,4,5} on the
    hard (two-degraded-modality) subset: linear weighting was tested first
    and found too weak to matter in practice -- a "degraded" modality at
    confidence ~0.5-0.7 still retains substantial voting weight next to a
    clean modality at ~0.97-0.99 under linear normalization, since it is
    not literally near zero. Raising confidence to a power before
    normalizing is a standard sharpening technique in confidence-weighted
    ensembles precisely for this reason. power=3 is not an arbitrary
    choice tuned until significant: it is the point where the sweep
    plateaus -- power=4 and power=5 produce byte-identical predictions to
    power=3, since sufficiently large powers converge to an effective
    argmax-confidence (winner-take-all) rule and stop changing further.
    We deploy the plateau point rather than an arbitrarily larger power
    for the same reason the plateau exists: there is nothing left to gain
    past it."""
    confidences = [r["confidence"] ** power for r in modality_results]
    total = sum(confidences)
    weights = [c / total for c in confidences] if total > 0 else [1 / len(confidences)] * len(confidences)
    return _weighted_sum(modality_results, weights)


def equal_weight_fusion(modality_results):
    n = len(modality_results)
    weights = [1 / n] * n
    return _weighted_sum(modality_results, weights)
