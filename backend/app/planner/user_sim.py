from __future__ import annotations
from typing import Optional
from ..schemas import TaskDefinition
from ..config import settings
from ..logger import ExperimentLogger
from ..llm.client import chat, chat_json, LLMResult
from ..llm.prompts import (
    USER_SIM_SYSTEM,
    user_sim_user_prompt,
    ROLLOUT_STATE_SYSTEM,
    rollout_state_user_prompt,
    USER_SIM_WITH_STATE_SYSTEM,
    user_sim_with_state_user_prompt,
    USER_SIM_END_ROLLOUT_SYSTEM,
    user_sim_end_rollout_user_prompt,
)


async def simulate_user_utterance(
    task: TaskDefinition,
    history: list[dict[str, str]],
    *,
    logger: Optional[ExperimentLogger] = None,
    call_site: str = "user_sim",
) -> tuple[str, LLMResult]:
    res = await chat(
        model=settings.MODEL_USER_SIM,
        system=USER_SIM_SYSTEM,
        user=user_sim_user_prompt(task, history),
        temperature=0.9,
        max_tokens=200,
        logger=logger,
        call_site=call_site,
    )
    return res.text.strip(), res


async def simulate_user_end_rollout(
    task: TaskDefinition,
    history: list[dict[str, str]],
    planned_actions: list[str],
    *,
    precedents_block: str = "",
    mood_name: str | None = None,
    mood_description: str = "",
    logger: Optional[ExperimentLogger] = None,
) -> tuple[str, str, float, LLMResult]:
    """LAST step of a rollout: returns (reply, state, rationality). Merges what used to be
    a user_sim_with_state + rationality pair into a single LLM call.

    `mood_name` + `mood_description` come from per-cohort mood sampling at rollout start.
    When provided, the simulator is steered into that disposition for this rollout's last
    step. Different parallel rollouts get different moods → state-prediction diversity.
    """
    parsed, res = await chat_json(
        model=settings.MODEL_USER_SIM,
        system=USER_SIM_END_ROLLOUT_SYSTEM,
        user=user_sim_end_rollout_user_prompt(
            task, history, planned_actions, precedents_block=precedents_block,
            mood_name=mood_name, mood_description=mood_description,
        ),
        # Bumped 0.7 → 1.05 to surface sampling diversity across parallel rollouts.
        # Task #113 ablation: does higher temperature alone produce non-degenerate state
        # distributions? Paired with the mood-diversity intervention from task #111.
        temperature=1.05,
        max_tokens=300,
        logger=logger,
        call_site="user_sim_end_rollout",
    )
    reply = (parsed.get("reply") or "").strip()
    state = (parsed.get("state") or "").strip()
    try:
        rationality = float(parsed.get("rationality", 0.0))
    except (TypeError, ValueError):
        rationality = 0.0
    rationality = max(0.0, min(1.0, rationality))
    valid = {s.name for s in task.user_states}
    if state not in valid and valid:
        state = next(iter(valid))
    return reply, state, rationality, res


async def simulate_user_with_state(
    task: TaskDefinition,
    history: list[dict[str, str]],
    *,
    mood_name: str | None = None,
    mood_description: str = "",
    logger: Optional[ExperimentLogger] = None,
    call_site: str = "user_sim_with_state",
) -> tuple[str, str, LLMResult]:
    """Single JSON call returning (reply_text, user_state). Saves one round-trip per rollout step.

    `mood_name`/`mood_description` come from per-cohort sampling at rollout start. Same
    mood is reused for every step inside one rollout (frozen-per-rollout). Different
    parallel rollouts get different moods so state predictions across rollouts diverge.
    """
    parsed, res = await chat_json(
        model=settings.MODEL_USER_SIM,
        system=USER_SIM_WITH_STATE_SYSTEM,
        user=user_sim_with_state_user_prompt(
            task, history,
            mood_name=mood_name, mood_description=mood_description,
        ),
        # Bumped 0.8 → 1.05 (task #113 ablation). See simulate_user_end_rollout above.
        temperature=1.05,
        max_tokens=250,
        logger=logger,
        call_site=call_site,
    )
    reply = (parsed.get("reply") or "").strip()
    state = (parsed.get("state") or "").strip()
    valid = {s.name for s in task.user_states}
    if state not in valid and valid:
        state = next(iter(valid))
    return reply, state, res


async def classify_user_state(
    task: TaskDefinition,
    history: list[dict[str, str]],
    *,
    logger: Optional[ExperimentLogger] = None,
    call_site: str = "rollout_state_classify",
) -> tuple[str, LLMResult]:
    parsed, res = await chat_json(
        model=settings.MODEL_USER_SIM,
        system=ROLLOUT_STATE_SYSTEM,
        user=rollout_state_user_prompt(task, history),
        temperature=0.1,
        max_tokens=200,
        logger=logger,
        call_site=call_site,
    )
    state = parsed.get("user_state", "") or ""
    valid = {s.name for s in task.user_states}
    if state not in valid and valid:
        state = next(iter(valid))
    return state, res
