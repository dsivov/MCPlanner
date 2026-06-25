from __future__ import annotations
from typing import Optional
from ..schemas import TaskDefinition
from ..config import settings
from ..logger import ExperimentLogger
from ..llm.client import chat_json, LLMResult
from ..llm.prompts import STATE_PREDICTION_SYSTEM, state_prediction_user_prompt


async def predict_user_state(
    task: TaskDefinition,
    history: list[dict[str, str]],
    *,
    logger: Optional[ExperimentLogger] = None,
) -> tuple[str, str, LLMResult]:
    if not task.user_states:
        return "", "", LLMResult(text="")
    parsed, res = await chat_json(
        model=settings.MODEL_STATE,
        system=STATE_PREDICTION_SYSTEM,
        user=state_prediction_user_prompt(task, history),
        temperature=0.2,
        logger=logger,
        call_site="state_predictor",
    )
    state = parsed.get("user_state", "") or ""
    valid = {s.name for s in task.user_states}
    if state not in valid:
        state = next(iter(valid)) if valid else ""
    return state, parsed.get("rationale", ""), res
