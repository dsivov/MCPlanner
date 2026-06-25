"""Paper's reward: r = 0.5 * (LLM_rationality + 0.5 * task_progress)."""

from __future__ import annotations
from typing import Optional
from ..schemas import TaskDefinition
from ..config import settings
from ..logger import ExperimentLogger
from ..llm.client import chat_json, LLMResult
from ..llm.prompts import RATIONALITY_SYSTEM, rationality_user_prompt


async def llm_rationality(
    task: TaskDefinition,
    history: list[dict[str, str]],
    planned_actions: list[str],
    *,
    logger: Optional[ExperimentLogger] = None,
) -> tuple[float, LLMResult]:
    parsed, res = await chat_json(
        model=settings.MODEL_ROLLOUT,
        system=RATIONALITY_SYSTEM,
        user=rationality_user_prompt(task, history, planned_actions),
        temperature=0.2,
        max_tokens=200,
        logger=logger,
        call_site="rationality",
    )
    try:
        score = float(parsed.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    return max(0.0, min(1.0, score)), res


def task_progress_bonus(
    task: TaskDefinition,
    states_visited: list[str],
) -> float:
    markers = set(task.conversation_profile.success_markers)
    return 1.0 if any(s in markers for s in states_visited) else 0.0


def combine_reward(rationality: float, progress: float) -> float:
    return 0.5 * (rationality + 0.5 * progress)
