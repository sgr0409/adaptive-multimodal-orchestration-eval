"""Two planners, compared as an ablation:

FixedPlanner is the MVP planner from phase 1: a single fixed policy that
always acts immediately on the fused decision, no matter how confident or
contested it was. This is the "fixed reasoning strategy" baseline the
paper's own related-work section identifies as the gap in existing
agentic frameworks (ReAct, Plan-and-Execute, etc: "mostly fixed planning,
little runtime adaptation").

DynamicPlanner selects among three strategies based on two signals that
are already computed for free by fusion -- the fused decision's own
confidence, and whether the three raw per-modality predictions agree with
it -- rather than always taking the same action:

  - direct: high confidence and full agreement -> act immediately
    (identical to FixedPlanner's behavior for these cases).
  - verification: lower confidence or partial disagreement, and the
    decision is not "critical" -> drop the single least-confident modality
    (the one most likely to be the degraded one) and re-run
    confidence-weighted fusion over the remaining two; override the
    original fused decision if this changes the label.

    An earlier version of this step instead collapsed straight to the
    single *most*-confident modality's own raw prediction, discarding the
    other two entirely. Tested on the same data, it made things slightly
    worse (97.60% vs. 98.00% for the fixed planner) -- every override it
    made broke a previously-correct fusion decision, and it recovered zero
    fusion errors. That is not surprising in hindsight: confidence-weighted
    fusion is already good at combining all three modalities' information
    properly (that is the whole point of Section VI's core result), so
    discarding two-thirds of that information to defer to one modality's
    raw opinion is a step backwards, not a safety check. Re-fusing over
    the two modalities that remain after dropping the suspected-bad one
    keeps the same validated fusion mechanism in play rather than
    abandoning it, and is the version deployed here.
  - escalation: the fused decision is "critical", regardless of
    confidence -> add an explicit safety-officer notification step before
    shutdown, since high-stakes actions warrant an extra step independent
    of how confident the system is.

Selection thresholds (confidence >= 0.8 and full 3-way agreement for
"direct") are stated explicitly here, not tuned per test outcome, so the
mechanism is reproducible per the paper outline's own requirement
("describe the selection criteria clearly so the mechanism is
reproducible").
"""
from framework.fusion import confidence_weighted_fusion

ACTION_OF = {
    "normal": "monitor",
    "warning": "schedule_maintenance",
    "critical": "immediate_shutdown",
}

TOOL_OF_ACTION = {
    "monitor": "log_and_monitor",
    "schedule_maintenance": "open_maintenance_ticket",
    "immediate_shutdown": "escalate_and_shutdown",
}

DIRECT_CONFIDENCE_THRESHOLD = 0.8


class MockToolRegistry:
    """Mock stubs standing in for real enterprise tool/API calls. Each
    just records that it was invoked with what arguments; none of them
    talk to a real system."""

    def __init__(self):
        self.call_log = []

    def log_and_monitor(self, scenario_id):
        self.call_log.append(("log_and_monitor", scenario_id))
        return {"tool": "log_and_monitor", "status": "ok"}

    def open_maintenance_ticket(self, scenario_id):
        self.call_log.append(("open_maintenance_ticket", scenario_id))
        return {"tool": "open_maintenance_ticket", "status": "ok"}

    def escalate_and_shutdown(self, scenario_id):
        self.call_log.append(("escalate_and_shutdown", scenario_id))
        return {"tool": "escalate_and_shutdown", "status": "ok"}

    def verify_reading(self, scenario_id):
        self.call_log.append(("verify_reading", scenario_id))
        return {"tool": "verify_reading", "status": "ok"}

    def notify_safety_officer(self, scenario_id):
        self.call_log.append(("notify_safety_officer", scenario_id))
        return {"tool": "notify_safety_officer", "status": "ok"}


class FixedPlanner:
    """Phase-1 MVP planner: always acts immediately on the fused label,
    one tool call, no verification, no strategy selection."""

    def __init__(self, tools=None):
        self.tools = tools or MockToolRegistry()

    def plan_and_execute(self, fused_label, scenario_id, **kwargs):
        action = ACTION_OF[fused_label]
        tool_fn = getattr(self.tools, TOOL_OF_ACTION[action])
        result = tool_fn(scenario_id)
        return {"strategy": "direct", "action": action, "tool_calls": [result]}


class DynamicPlanner:
    def __init__(self, tools=None):
        self.tools = tools or MockToolRegistry()

    def select_strategy(self, fused_label, fused_confidence, modality_results):
        agreement = sum(1 for r in modality_results if r["label"] == fused_label)
        full_agreement = agreement == len(modality_results)
        if fused_label == "critical":
            return "escalation"
        if fused_confidence >= DIRECT_CONFIDENCE_THRESHOLD and full_agreement:
            return "direct"
        return "verification"

    def plan_and_execute(self, fused_label, scenario_id, fused_confidence, modality_results):
        strategy = self.select_strategy(fused_label, fused_confidence, modality_results)
        tool_calls = []

        if strategy == "direct":
            final_label = fused_label

        elif strategy == "verification":
            check = self.tools.verify_reading(scenario_id)
            tool_calls.append(check)
            remaining = sorted(modality_results, key=lambda r: r["confidence"], reverse=True)[:-1]
            re_fused = confidence_weighted_fusion(remaining)
            final_label = re_fused["label"]

        else:  # escalation
            notify = self.tools.notify_safety_officer(scenario_id)
            tool_calls.append(notify)
            final_label = fused_label

        action = ACTION_OF[final_label]
        tool_calls.append(getattr(self.tools, TOOL_OF_ACTION[action])(scenario_id))
        return {"strategy": strategy, "action": action, "final_label": final_label,
                "overridden": final_label != fused_label, "tool_calls": tool_calls}
