"""Domain-2 (IT incidents) copy of experiments/sweep_fusion_power.py.
Same methodology; see that file's docstring for rationale."""
import json
import random
import sys
from pathlib import Path

from scipy import stats as scipy_stats

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from framework.text_scorer import TextScorer
from framework.telemetry_scorer import TelemetryScorer
from framework.image_scorer import ImageScorer
from framework.fusion import confidence_weighted_fusion, equal_weight_fusion

SEED = 42
DATA_DIR = ROOT / "data_it_incidents"
RESULTS_DIR = ROOT / "experiments" / "it_incidents" / "results"


def mcnemar(correct_a, correct_b):
    b = sum(1 for a, bb in zip(correct_a, correct_b) if a and not bb)
    c = sum(1 for a, bb in zip(correct_a, correct_b) if not a and bb)
    n = b + c
    p = scipy_stats.binomtest(min(b, c), n, 0.5).pvalue if n > 0 else 1.0
    return {"b": b, "c": c, "p_value": round(float(p), 6)}


def main():
    scenarios = [json.loads(l) for l in open(DATA_DIR / "scenarios.jsonl")]
    rng = random.Random(SEED)
    shuffled = scenarios[:]
    rng.shuffle(shuffled)
    n_train = len(shuffled) // 2
    train, test = shuffled[:n_train], shuffled[n_train:]

    print("Training scorers...", flush=True)
    text_scorer = TextScorer()
    text_scorer.fit([r["text"] for r in train], [r["label"] for r in train])
    telemetry_scorer = TelemetryScorer()
    telemetry_scorer.fit([r["telemetry"] for r in train], [r["label"] for r in train])
    image_scorer = ImageScorer()
    image_scorer.fit([str(DATA_DIR / r["image_path"]) for r in train], [r["label"] for r in train])

    print("Scoring test split...", flush=True)
    per_example = []
    for r in test:
        results = [text_scorer.predict(r["text"]),
                   image_scorer.predict(str(DATA_DIR / r["image_path"])),
                   telemetry_scorer.predict(r["telemetry"])]
        per_example.append({"true_label": r["label"], "degraded_modalities": r["degraded_modalities"],
                             "results": results})

    hard = [e for e in per_example if len(e["degraded_modalities"]) == 2]
    ew_pred = [equal_weight_fusion(e["results"])["label"] for e in hard]
    ew_correct = [t == p for t, p in zip([e["true_label"] for e in hard], ew_pred)]
    ew_acc = sum(ew_correct) / len(hard)

    sweep = {}
    for power in [1, 2, 3, 4, 5]:
        cw_pred = [confidence_weighted_fusion(e["results"], power=power)["label"] for e in hard]
        cw_correct = [t == p for t, p in zip([e["true_label"] for e in hard], cw_pred)]
        sweep[str(power)] = {
            "accuracy": round(sum(cw_correct) / len(hard), 4),
            "mcnemar_vs_equal_weight": mcnemar(cw_correct, ew_correct),
        }
        print(f"power={power}: acc={sweep[str(power)]['accuracy']}  "
              f"mcnemar={sweep[str(power)]['mcnemar_vs_equal_weight']}")

    out = {
        "n_hard_subset": len(hard),
        "equal_weight_accuracy_on_hard_subset": round(ew_acc, 4),
        "sweep": sweep,
        "deployed_power": 3,
        "note": "power=3 is kept as the deployed default for consistency with domain 1's chosen "
                "value; this sweep checks whether that choice still makes sense on a second, "
                "independently-generated domain rather than re-tuning per domain.",
    }
    with open(RESULTS_DIR / "fusion_power_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {RESULTS_DIR / 'fusion_power_sweep.json'}")


if __name__ == "__main__":
    main()
