from __future__ import annotations
from typing import Optional
from ..schemas import TaskDefinition, RetrievedPrecedent
from ..config import settings
from ..logger import ExperimentLogger
from ..llm.client import chat, LLMResult
from ..llm.prompts import RESPONSE_GEN_SYSTEM, response_gen_user_prompt, format_precedents_block


async def generate_response(
    task: TaskDefinition,
    history: list[dict[str, str]],
    chosen_action: str,
    *,
    precedents: list[RetrievedPrecedent] | None = None,
    use_precedents: bool = False,
    prefetched_context: list[str] | None = None,
    logger: Optional[ExperimentLogger] = None,
) -> tuple[str, LLMResult]:
    must_say: list[str] = []
    must_not_say: list[str] = []
    for a in task.agent_actions:
        if a.name == chosen_action:
            must_say = list(a.must_say or [])
            must_not_say = list(a.must_not_say or [])
            break

    block = ""
    if use_precedents and precedents:
        # Keep only same-action precedents for the style block, top 2.
        same_action = [p for p in precedents if p.action == chosen_action][:2]
        block = format_precedents_block(same_action)

    res = await chat(
        model=settings.MODEL_SELECTION,
        system=RESPONSE_GEN_SYSTEM,
        user=response_gen_user_prompt(
            task, history, chosen_action,
            must_say=must_say, must_not_say=must_not_say, precedents_block=block,
            prefetched_context=prefetched_context,
        ),
        temperature=0.7,
        max_tokens=300,
        logger=logger,
        call_site="response_gen",
    )
    return res.text.strip(), res
