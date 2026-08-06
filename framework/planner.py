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
    decision is not the escalation label -> drop the single
    least-confident modality (the one most likely to be the degraded one)
    and re-run confidence-weighted fusion over the remaining two; override
    the original fused decision if this changes the label.

    An earlier version of this step instead collapsed straight to the
    single *most*-confident modality's own raw prediction, discarding the
    other two entirely. Tested on the equipment-maintenance-triage domain,
    it made things slightly worse (97.60% vs. 98.00% for the fixed
    planner) -- every override it made broke a previously-correct fusion
    decision, and it recovered zero fusion errors. That is not surprising
    in hindsight: confidence-weighted fusion is already good at combining
    all three modalities' information properly, so discarding two-thirds
    of that information to defer to one modality's raw opinion is a step
    backwards, not a safety check. Re-fusing over the two modalities that
    remain after dropping the suspected-bad one keeps the same validated
    fusion mechanism in play rather than abandoning it, and is the version
    deployed here.
  - escalation: the fused decision equals `escalation_label`, regardless
    of confidence -> an explicit extra notification tool call before
    acting, since high-stakes actions warrant an extra step independent
    of how confident the system is.

Selection thresholds (confidence >= 0.8 and full 3-way agreement for
"direct") are stated explicitly here, not tuned per test outcome, so the
mechanism is reproducible per the paper outline's own requirement
("describe the selection criteria clearly so the mechanism is
reproducible").

Domain parameterization: action_of, tool_of_action, escalation_label,
and the verification/escalation tool names are constructor arguments,
defaulting to the equipment-maintenance-triage domain's original values,
so the same planner logic works unchanged for a second task domain
(e.g. IT-incident triage's minor/degraded/outage labels) rather than being
hardcoded to one domain's label set.
"""
from framework.fusion import confidence_weighted_fusion

DEFAULT_ACTION_OF = {
    "normal": "monitor",
    "warning": "schedule_maintenance",
    "critical": "immediate_shutdown",
}

DEFAULT_TOOL_OF_ACTION = {
    "monitor": "log_and_monitor",
    "schedule_maintenance": "open_maintenance_ticket",
    "immediate_shutdown": "escalate_and_shutdown",
}

# Kept for backward-compat imports elsewhere in the codebase.
ACTION_OF = DEFAULT_ACTION_OF
TOOL_OF_ACTION = DEFAULT_TOOL_OF_ACTION

DIRECT_CONFIDENCE_THRESHOLD = 0.8


class MockToolRegistry:
    """Mock stub standing in for real enterprise tool/API calls. Fully
    generic: logs and acknowledges any tool name, since none of these
    talk to a real system regardless of domain."""

    def __init__(self):
        self.call_log = []

    def invoke(self, tool_name, scenario_id):
        self.call_log.append((tool_name, scenario_id))
        return {"tool": tool_name, "status": "ok"}


class FixedPlanner:
    """Phase-1 MVP planner: always acts immediately on the fused label,
    one tool call, no verification, no strategy selection."""

    def __init__(self, tools=None, action_of=None, tool_of_action=None):
        self.tools = tools or MockToolRegistry()
        self.action_of = action_of or DEFAULT_ACTION_OF
        self.tool_of_action = tool_of_action or DEFAULT_TOOL_OF_ACTION

    def plan_and_execute(self, fused_label, scenario_id, **kwargs):
        action = self.action_of[fused_label]
        result = self.tools.invoke(self.tool_of_action[action], scenario_id)
        return {"strategy": "direct", "action": action, "tool_calls": [result]}


class DynamicPlanner:
    def __init__(self, tools=None, action_of=None, tool_of_action=None,
                 escalation_label="critical", verification_tool="verify_reading",
                 escalation_tool="notify_safety_officer"):
        self.tools = tools or MockToolRegistry()
        self.action_of = action_of or DEFAULT_ACTION_OF
        self.tool_of_action = tool_of_action or DEFAULT_TOOL_OF_ACTION
        self.escalation_label = escalation_label
        self.verification_tool = verification_tool
        self.escalation_tool = escalation_tool

    def select_strategy(self, fused_label, fused_confidence, modality_results):
        agreement = sum(1 for r in modality_results if r["label"] == fused_label)
        full_agreement = agreement == len(modality_results)
        if fused_label == self.escalation_label:
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
            tool_calls.append(self.tools.invoke(self.verification_tool, scenario_id))
            remaining = sorted(modality_results, key=lambda r: r["confidence"], reverse=True)[:-1]
            re_fused = confidence_weighted_fusion(remaining)
            final_label = re_fused["label"]

        else:  # escalation
            tool_calls.append(self.tools.invoke(self.escalation_tool, scenario_id))
            final_label = fused_label

        action = self.action_of[final_label]
        tool_calls.append(self.tools.invoke(self.tool_of_action[action], scenario_id))
        return {"strategy": strategy, "action": action, "final_label": final_label,
                "overridden": final_label != fused_label, "tool_calls": tool_calls}
