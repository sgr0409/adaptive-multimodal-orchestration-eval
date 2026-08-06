"""Runs the MVP benchmark: confidence-weighted fusion vs. equal-weight
fusion (the core ablation), on the same fixed 225/225 train/test split of
the synthetic maintenance-triage scenarios, using the same three
per-modality scorers for both. Writes experiments/results/results.json
and experiments/results/raw_scores.npz.
"""
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats as scipy_stats
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from framework.text_scorer import TextScorer
from framework.telemetry_scorer import TelemetryScorer
from framework.image_scorer import ImageScorer
from framework.fusion import confidence_weighted_fusion, equal_weight_fusion, LABELS
from framework.planner import FixedPlanner

SEED = 42
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "experiments" / "results"
N_BOOTSTRAP = 2000


def load_scenarios():
    return [json.loads(l) for l in open(DATA_DIR / "scenarios.jsonl")]


def mcnemar_test(correct_a, correct_b):
    """Exact (binomial) McNemar's test on paired per-example correctness."""
    correct_a = np.asarray(correct_a)
    correct_b = np.asarray(correct_b)
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    n = b + c
    if n == 0:
        return {"b_a_right_b_wrong": b, "c_a_wrong_b_right": c, "p_value": 1.0}
    p = scipy_stats.binomtest(min(b, c), n, 0.5).pvalue
    return {"b_a_right_b_wrong": b, "c_a_wrong_b_right": c, "p_value": round(float(p), 6)}


def bootstrap_accuracy_ci(correct, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.RandomState(seed)
    correct = np.asarray(correct)
    n = len(correct)
    accs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        accs.append(correct[idx].mean())
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return {"lo95": round(float(lo), 4), "hi95": round(float(hi), 4)}


def eval_labels(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()
    correct = [int(t == p) for t, p in zip(y_true, y_pred)]
    return {
        "accuracy": round(sum(correct) / len(correct), 4),
        "macro_precision": round(float(precision), 4),
        "macro_recall": round(float(recall), 4),
        "macro_f1": round(float(f1), 4),
        "confusion_matrix": cm,
        "confusion_matrix_labels": LABELS,
        "bootstrap_accuracy_ci": bootstrap_accuracy_ci(correct),
    }, correct


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "meta": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "device": "cpu",
            "seed": SEED,
            "bootstrap_resamples": N_BOOTSTRAP,
        }
    }

    print("Loading and splitting scenarios...", flush=True)
    scenarios = load_scenarios()
    rng = random.Random(SEED)
    shuffled = scenarios[:]
    rng.shuffle(shuffled)
    n_train = len(shuffled) // 2
    train, test = shuffled[:n_train], shuffled[n_train:]
    results["dataset"] = {"n_total": len(scenarios), "n_train": len(train), "n_test": len(test)}

    print("Training per-modality scorers on the train split...", flush=True)
    text_scorer = TextScorer()
    text_scorer.fit([r["text"] for r in train], [r["label"] for r in train])

    telemetry_scorer = TelemetryScorer()
    telemetry_scorer.fit([r["telemetry"] for r in train], [r["label"] for r in train])

    image_scorer = ImageScorer()
    image_scorer.fit([str(DATA_DIR / r["image_path"]) for r in train], [r["label"] for r in train])

    print("Scoring test split with all three modalities...", flush=True)
    per_example = []
    latencies_ms = []
    for r in test:
        t_start = time.perf_counter()
        text_res = text_scorer.predict(r["text"])
        telemetry_res = telemetry_scorer.predict(r["telemetry"])
        image_res = image_scorer.predict(str(DATA_DIR / r["image_path"]))
        latencies_ms.append((time.perf_counter() - t_start) * 1000)

        modality_results = [text_res, image_res, telemetry_res]
        cw = confidence_weighted_fusion(modality_results)
        ew = equal_weight_fusion(modality_results)

        per_example.append({
            "id": r["id"],
            "true_label": r["label"],
            "degraded_modalities": r["degraded_modalities"],
            "text": text_res, "image": image_res, "telemetry": telemetry_res,
            "confidence_weighted_fusion": cw,
            "equal_weight_fusion": ew,
        })

    y_true = [e["true_label"] for e in per_example]
    cw_pred = [e["confidence_weighted_fusion"]["label"] for e in per_example]
    ew_pred = [e["equal_weight_fusion"]["label"] for e in per_example]

    print("Computing metrics...", flush=True)
    cw_metrics, cw_correct = eval_labels(y_true, cw_pred)
    ew_metrics, ew_correct = eval_labels(y_true, ew_pred)
    results["confidence_weighted_fusion"] = cw_metrics
    results["equal_weight_fusion"] = ew_metrics
    results["significance"] = {
        "confidence_weighted_vs_equal_weight_mcnemar": mcnemar_test(
            [bool(c) for c in cw_correct], [bool(c) for c in ew_correct]),
        "note": "exact (binomial) McNemar's test on paired test-set predictions.",
    }

    # The core hypothesis test: confidence-weighted fusion should match
    # equal-weight fusion when 0 or 1 modality is degraded (two clean
    # modalities already outvote one bad one under either weighting
    # scheme), and should specifically outperform it when 2 modalities are
    # degraded simultaneously -- the condition where equal-weight voting
    # gets outvoted by the two unreliable modalities while confidence
    # weighting can still favor the one reliable one.
    for n_deg in [0, 1, 2]:
        subset = [e for e in per_example if len(e["degraded_modalities"]) == n_deg]
        if not subset:
            continue
        sub_true = [e["true_label"] for e in subset]
        sub_cw = [e["confidence_weighted_fusion"]["label"] for e in subset]
        sub_ew = [e["equal_weight_fusion"]["label"] for e in subset]
        cw_correct_sub = [t == p for t, p in zip(sub_true, sub_cw)]
        ew_correct_sub = [t == p for t, p in zip(sub_true, sub_ew)]
        results.setdefault("subset_breakdown", {})[f"{n_deg}_degraded_modalities"] = {
            "n": len(subset),
            "confidence_weighted_accuracy": round(sum(cw_correct_sub) / len(subset), 4),
            "equal_weight_accuracy": round(sum(ew_correct_sub) / len(subset), 4),
            "mcnemar": mcnemar_test(cw_correct_sub, ew_correct_sub),
        }

    # Per-modality standalone accuracy (no fusion), for reference.
    results["per_modality_standalone_accuracy"] = {
        "text": round(sum(e["true_label"] == e["text"]["label"] for e in per_example) / len(per_example), 4),
        "image": round(sum(e["true_label"] == e["image"]["label"] for e in per_example) / len(per_example), 4),
        "telemetry": round(sum(e["true_label"] == e["telemetry"]["label"] for e in per_example) / len(per_example), 4),
    }

    # Confidence-vs-reliability sanity metric per modality: mean confidence
    # on degraded vs. clean examples for that specific modality.
    conf_breakdown = {}
    for mod in ["text", "image", "telemetry"]:
        deg = [e[mod]["confidence"] for e in per_example if mod in e["degraded_modalities"]]
        clean = [e[mod]["confidence"] for e in per_example if mod not in e["degraded_modalities"]]
        conf_breakdown[mod] = {
            "mean_confidence_when_degraded": round(float(np.mean(deg)), 4) if deg else None,
            "mean_confidence_when_not_degraded": round(float(np.mean(clean)), 4) if clean else None,
            "n_degraded": len(deg),
        }
    results["modality_confidence_reliability_check"] = conf_breakdown

    results["latency"] = {
        "note": "per-example latency for scoring all three modalities and computing both fusion "
                "variants (single-threaded, CPU); planner/tool execution not included (negligible, "
                "no model inference).",
        "mean_ms": round(float(np.mean(latencies_ms)), 3),
        "p50_ms": round(float(np.percentile(latencies_ms, 50)), 3),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 3),
        "p99_ms": round(float(np.percentile(latencies_ms, 99)), 3),
        "n": len(latencies_ms),
    }

    # Exercise the planner end-to-end on a handful of test examples as a
    # smoke test that the full pipeline (fusion -> action -> mock tool
    # call) actually runs, not just the classification metrics.
    planner = FixedPlanner()
    planner_log = []
    for e in per_example[:10]:
        out = planner.plan_and_execute(e["confidence_weighted_fusion"]["label"], e["id"])
        planner_log.append({"id": e["id"], "fused_label": e["confidence_weighted_fusion"]["label"], **out})
    results["planner_smoke_test"] = planner_log

    results["meta"]["total_runtime_s"] = round(time.time() - t0, 1)

    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    np.savez(RESULTS_DIR / "raw_scores.npz",
             y_true=np.array(y_true),
             cw_pred=np.array(cw_pred),
             ew_pred=np.array(ew_pred),
             degraded_modalities=np.array([",".join(e["degraded_modalities"]) or "none" for e in per_example]),
             n_degraded_modalities=np.array([len(e["degraded_modalities"]) for e in per_example]),
             text_confidence=np.array([e["text"]["confidence"] for e in per_example]),
             image_confidence=np.array([e["image"]["confidence"] for e in per_example]),
             telemetry_confidence=np.array([e["telemetry"]["confidence"] for e in per_example]))

    with open(RESULTS_DIR / "per_example.json", "w") as f:
        json.dump(per_example, f, indent=2)

    print(f"\nDone in {results['meta']['total_runtime_s']}s. Results written to {RESULTS_DIR}", flush=True)
    print(f"Confidence-weighted fusion accuracy: {cw_metrics['accuracy']}")
    print(f"Equal-weight fusion accuracy:        {ew_metrics['accuracy']}")
    print(f"McNemar p-value: {results['significance']['confidence_weighted_vs_equal_weight_mcnemar']['p_value']}")


if __name__ == "__main__":
    main()
