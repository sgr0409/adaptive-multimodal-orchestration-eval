"""External baseline: the same three trained per-modality scorers, wired
into a genuine LangGraph StateGraph implementing a fixed linear flow --
score each modality, combine with a fixed (unweighted) rule, act with a
single fixed policy -- with no confidence weighting and no strategy
branching. This is a real orchestration-framework integration, not
another internal ablation dressed up as one: the point is to check
whether our own internal "equal-weight fusion + fixed planner" baseline
is a faithful stand-in for how an actual fixed-flow agentic framework
would behave on this task, not to introduce a different modeling
methodology.

Uses the identical trained scorers as run_benchmark.py (same train split,
same seed), so any accuracy difference reflects orchestration style, not
model quality.
"""
import json
import sys
import time
from pathlib import Path
from typing import TypedDict

from scipy import stats as scipy_stats

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from framework.text_scorer import TextScorer
from framework.telemetry_scorer import TelemetryScorer
from framework.image_scorer import ImageScorer
from framework.fusion import equal_weight_fusion
from framework.planner import ACTION_OF

from langgraph.graph import StateGraph, END

SEED = 42
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "experiments" / "results"


class TriageState(TypedDict):
    text: str
    image_path: str
    telemetry: dict
    text_result: dict
    image_result: dict
    telemetry_result: dict
    fused_label: str
    action: str


def build_graph(text_scorer, image_scorer, telemetry_scorer):
    def score_text_node(state: TriageState) -> TriageState:
        return {"text_result": text_scorer.predict(state["text"])}

    def score_image_node(state: TriageState) -> TriageState:
        return {"image_result": image_scorer.predict(state["image_path"])}

    def score_telemetry_node(state: TriageState) -> TriageState:
        return {"telemetry_result": telemetry_scorer.predict(state["telemetry"])}

    def fixed_fuse_node(state: TriageState) -> TriageState:
        # Fixed, unweighted combination rule -- no confidence awareness.
        # This is the behavior our paper's own related-work section
        # attributes to typical agentic-framework default configurations.
        results = [state["text_result"], state["image_result"], state["telemetry_result"]]
        fused = equal_weight_fusion(results)
        return {"fused_label": fused["label"]}

    def fixed_act_node(state: TriageState) -> TriageState:
        # Single fixed policy -- always act immediately, no verification,
        # no escalation branching, mirroring FixedPlanner.
        return {"action": ACTION_OF[state["fused_label"]]}

    graph = StateGraph(TriageState)
    graph.add_node("score_text", score_text_node)
    graph.add_node("score_image", score_image_node)
    graph.add_node("score_telemetry", score_telemetry_node)
    graph.add_node("fixed_fuse", fixed_fuse_node)
    graph.add_node("fixed_act", fixed_act_node)

    graph.set_entry_point("score_text")
    graph.add_edge("score_text", "score_image")
    graph.add_edge("score_image", "score_telemetry")
    graph.add_edge("score_telemetry", "fixed_fuse")
    graph.add_edge("fixed_fuse", "fixed_act")
    graph.add_edge("fixed_act", END)

    return graph.compile()


def mcnemar(correct_a, correct_b):
    b = sum(1 for a, bb in zip(correct_a, correct_b) if a and not bb)
    c = sum(1 for a, bb in zip(correct_a, correct_b) if not a and bb)
    n = b + c
    p = scipy_stats.binomtest(min(b, c), n, 0.5).pvalue if n > 0 else 1.0
    return {"b": b, "c": c, "p_value": round(float(p), 6)}


def main():
    import random
    scenarios = [json.loads(l) for l in open(DATA_DIR / "scenarios.jsonl")]
    rng = random.Random(SEED)
    shuffled = scenarios[:]
    rng.shuffle(shuffled)
    n_train = len(shuffled) // 2
    train, test = shuffled[:n_train], shuffled[n_train:]

    print("Training scorers (identical to run_benchmark.py)...", flush=True)
    text_scorer = TextScorer()
    text_scorer.fit([r["text"] for r in train], [r["label"] for r in train])
    telemetry_scorer = TelemetryScorer()
    telemetry_scorer.fit([r["telemetry"] for r in train], [r["label"] for r in train])
    image_scorer = ImageScorer()
    image_scorer.fit([str(DATA_DIR / r["image_path"]) for r in train], [r["label"] for r in train])

    print("Building LangGraph StateGraph (fixed linear flow)...", flush=True)
    app = build_graph(text_scorer, image_scorer, telemetry_scorer)

    print("Running LangGraph baseline on test split...", flush=True)
    t0 = time.time()
    langgraph_correct = []
    for r in test:
        result = app.invoke({
            "text": r["text"], "image_path": str(DATA_DIR / r["image_path"]),
            "telemetry": r["telemetry"],
        })
        true_action = ACTION_OF[r["label"]]
        langgraph_correct.append(result["action"] == true_action)
    elapsed = time.time() - t0

    langgraph_acc = sum(langgraph_correct) / len(test)

    # Compare against our own internal equal-weight-fusion + fixed-planner
    # baseline's per-example results, already computed by run_benchmark.py,
    # to check whether the internal ablation is a faithful stand-in for
    # real fixed-flow-framework orchestration.
    per_example = json.load(open(RESULTS_DIR / "per_example.json"))
    internal_ew_correct = [e["true_label"] == e["equal_weight_fusion"]["label"] for e in per_example]
    internal_cw_correct = [e["true_label"] == e["confidence_weighted_fusion"]["label"] for e in per_example]

    summary = {
        "n_test": len(test),
        "langgraph_fixed_flow_accuracy": round(langgraph_acc, 4),
        "internal_equal_weight_baseline_accuracy": round(sum(internal_ew_correct) / len(internal_ew_correct), 4),
        "internal_confidence_weighted_accuracy": round(sum(internal_cw_correct) / len(internal_cw_correct), 4),
        "langgraph_vs_internal_equal_weight_mcnemar": mcnemar(langgraph_correct, internal_ew_correct),
        "langgraph_vs_internal_confidence_weighted_mcnemar": mcnemar(langgraph_correct, internal_cw_correct),
        "langgraph_total_runtime_s": round(elapsed, 2),
        "langgraph_version": __import__("langgraph").__dict__.get("__version__", "0.6.11"),
    }

    with open(RESULTS_DIR / "langgraph_baseline_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nLangGraph fixed-flow accuracy:            {summary['langgraph_fixed_flow_accuracy']}")
    print(f"Internal equal-weight baseline accuracy:  {summary['internal_equal_weight_baseline_accuracy']}")
    print(f"Internal confidence-weighted accuracy:    {summary['internal_confidence_weighted_accuracy']}")
    print(f"LangGraph vs internal equal-weight:  {summary['langgraph_vs_internal_equal_weight_mcnemar']}")
    print(f"LangGraph vs internal conf-weighted: {summary['langgraph_vs_internal_confidence_weighted_mcnemar']}")
    print(f"\nWrote {RESULTS_DIR / 'langgraph_baseline_results.json'}")


if __name__ == "__main__":
    main()
