"""Domain-2 (IT incidents) copy of experiments/run_planner_benchmark.py.
Same methodology; see that file's docstring for rationale. Planners are
constructed with domain-2's action/tool mapping and escalation label
(outage) instead of the equipment-triage domain's defaults."""
import json
import sys
from pathlib import Path

from scipy import stats as scipy_stats

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from framework.planner import FixedPlanner, DynamicPlanner

ACTION_OF = {"minor": "monitor", "degraded": "investigate", "outage": "page_oncall"}
TOOL_OF_ACTION = {"monitor": "log_and_monitor", "investigate": "open_incident_ticket",
                   "page_oncall": "page_oncall_engineer"}
RESULTS_DIR = ROOT / "experiments" / "it_incidents" / "results"


def mcnemar(correct_a, correct_b):
    b = sum(1 for a, bb in zip(correct_a, correct_b) if a and not bb)
    c = sum(1 for a, bb in zip(correct_a, correct_b) if not a and bb)
    n = b + c
    p = scipy_stats.binomtest(min(b, c), n, 0.5).pvalue if n > 0 else 1.0
    return {"b_fixed_right_dynamic_wrong": b, "c_fixed_wrong_dynamic_right": c, "p_value": round(float(p), 6)}


def main():
    per_example = json.load(open(RESULTS_DIR / "per_example.json"))

    fixed = FixedPlanner(action_of=ACTION_OF, tool_of_action=TOOL_OF_ACTION)
    dynamic = DynamicPlanner(action_of=ACTION_OF, tool_of_action=TOOL_OF_ACTION,
                              escalation_label="outage", verification_tool="verify_reading",
                              escalation_tool="notify_oncall_lead")

    fixed_correct, dynamic_correct = [], []
    strategy_counts = {"direct": 0, "verification": 0, "escalation": 0}
    overrides = []
    fusion_was_wrong_recovered = 0
    fusion_was_wrong_still_wrong = 0
    fusion_was_right_broken = 0

    for e in per_example:
        true_action = ACTION_OF[e["true_label"]]
        fused = e["confidence_weighted_fusion"]
        modality_results = [e["text"], e["image"], e["telemetry"]]

        fixed_out = fixed.plan_and_execute(fused["label"], e["id"])
        dynamic_out = dynamic.plan_and_execute(
            fused["label"], e["id"],
            fused_confidence=fused["probs"][fused["label"]],
            modality_results=modality_results,
        )

        strategy_counts[dynamic_out["strategy"]] += 1
        fixed_ok = fixed_out["action"] == true_action
        dynamic_ok = dynamic_out["action"] == true_action
        fixed_correct.append(fixed_ok)
        dynamic_correct.append(dynamic_ok)

        fusion_was_wrong = fused["label"] != e["true_label"]
        if dynamic_out["overridden"]:
            overrides.append({
                "id": e["id"], "true_label": e["true_label"], "fused_label": fused["label"],
                "final_label": dynamic_out["final_label"], "fusion_was_wrong": fusion_was_wrong,
                "override_was_correct": dynamic_out["final_label"] == e["true_label"],
            })
            if fusion_was_wrong and dynamic_out["final_label"] == e["true_label"]:
                fusion_was_wrong_recovered += 1
            elif fusion_was_wrong:
                fusion_was_wrong_still_wrong += 1
            elif dynamic_out["final_label"] != e["true_label"]:
                fusion_was_right_broken += 1

    n = len(per_example)
    fixed_acc = sum(fixed_correct) / n
    dynamic_acc = sum(dynamic_correct) / n

    summary = {
        "n_test": n,
        "fixed_planner_accuracy": round(fixed_acc, 4),
        "dynamic_planner_accuracy": round(dynamic_acc, 4),
        "mcnemar": mcnemar(fixed_correct, dynamic_correct),
        "strategy_selection_counts": strategy_counts,
        "n_overrides": len(overrides),
        "overrides_that_recovered_a_fusion_error": fusion_was_wrong_recovered,
        "overrides_on_already_wrong_fusion_that_stayed_wrong": fusion_was_wrong_still_wrong,
        "overrides_that_broke_a_correct_fusion_decision": fusion_was_right_broken,
        "override_examples": overrides,
    }

    out_path = RESULTS_DIR / "planner_selection_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Fixed planner accuracy:   {fixed_acc:.4f}")
    print(f"Dynamic planner accuracy: {dynamic_acc:.4f}")
    print(f"McNemar: {summary['mcnemar']}")
    print(f"Strategy selection: {strategy_counts}")
    print(f"Overrides: {len(overrides)} total -- "
          f"{fusion_was_wrong_recovered} recovered a fusion error, "
          f"{fusion_was_wrong_still_wrong} left a fusion error still wrong (different wrong answer), "
          f"{fusion_was_right_broken} broke a previously-correct fusion decision")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
