"""Baseline planner: CoT + SOP, one-shot action selection.

Merged with cohort + state prediction: a single call returns the cohort label, the predicted
user_state, and the chosen action (`call_site=cohort_state_baseline`)."""

from __future__ import annotations
from typing import Optional
from ..schemas import TaskDefinition, PlannerTrace, CandidateAction, RetrievedPrecedent
from ..config import settings
from ..logger import ExperimentLogger, CandidateEntry
from ..llm.client import chat_json
from ..llm.prompts import (
    COHORT_STATE_BASELINE_SYSTEM, cohort_state_baseline_user_prompt,
    format_precedents_block,
)
from .sop_graph import SOPGraph
from .state_predictor import predict_user_state


async def select_action_baseline(
    task: TaskDefinition,
    history: list[dict[str, str]],
    state_log: list[str],
    trace: PlannerTrace,
    *,
    precedents: list[RetrievedPrecedent] | None = None,
    use_precedents_expand: bool = True,
    logger: Optional[ExperimentLogger] = None,
) -> tuple[str, str, str, str]:
    """Returns (chosen_action, cohort, predicted_user_state, state_rationale)."""
    precedents = precedents or []
    precedents_block = format_precedents_block(precedents) if (use_precedents_expand and precedents) else ""
    trace.precedents_used_expand = bool(precedents_block)
    g = SOPGraph(task)
    visited = g.visited_from_history(history, state_log)
    allowed = g.allowed_actions(visited)

    # |allowed|==1 short-circuit: no choice to make. Still need a state prediction for the trace.
    if len(allowed) == 1:
        only_action = allowed[0]
        state, state_rat, _ = await predict_user_state(task, history, logger=logger)
        trace.chosen_action = only_action
        trace.candidates = [CandidateAction(
            action=only_action, q_value=1.0, visits=0,
            rationale="Only SOP-legal action; baseline selection skipped.",
        )]
        trace.mode = "baseline"
        if logger is not None:
            logger.record_candidate(CandidateEntry(
                action=only_action, q_value=1.0, visits=0,
                rationale="Only SOP-legal action; baseline selection skipped.",
                was_chosen=True,
            ))
        return only_action, "", state, state_rat

    parsed, res = await chat_json(
        model=settings.MODEL_SELECTION,
        system=COHORT_STATE_BASELINE_SYSTEM,
        user=cohort_state_baseline_user_prompt(task, history, allowed, precedents_block=precedents_block),
        temperature=0.3,
        max_tokens=500,
        logger=logger,
        call_site="cohort_state_baseline",
    )
    action = parsed.get("action", "") or ""
    if action not in allowed:
        action = allowed[0] if allowed else (next(iter(g.action_names)) if g.action_names else "")
    rationale = parsed.get("action_rationale", "") or parsed.get("rationale", "")

    cohort = (parsed.get("cohort") or "").strip() or "unknown"
    if task.cohorts:
        vocab = {c.name for c in task.cohorts}
        if cohort not in vocab:
            cohort = next(iter(vocab))
    state = (parsed.get("user_state") or "").strip()
    valid_states = {s.name for s in task.user_states}
    if state not in valid_states and valid_states:
        state = next(iter(valid_states))
    state_rat = parsed.get("state_rationale", "") or ""
    trace.cohort = cohort

    trace.tokens_in += res.tokens_in
    trace.tokens_out += res.tokens_out
    trace.chosen_action = action
    trace.candidates = [
        CandidateAction(action=a, q_value=(1.0 if a == action else 0.0), visits=0,
                        rationale=rationale if a == action else "")
        for a in allowed
    ]
    trace.mode = "baseline"

    if logger is not None:
        for a in allowed:
            logger.record_candidate(CandidateEntry(
                action=a,
                q_value=1.0 if a == action else 0.0,
                visits=0,
                rationale=rationale if a == action else "",
                was_chosen=(a == action),
            ))
    return action, cohort, state, state_rat
