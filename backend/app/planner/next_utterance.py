"""Cheap next-utterance predictor — the generative {user_text} slot for query-aware
prefetch, replacing MCTS rollouts as that slot's source.

Empirical counting predicts the next *action* (which data source); it cannot produce the
*free-text* query a RAG/KG dependency needs, because that depends on the novel sentence the
user will say next. MCTS rollouts used to supply this (Q6b), at ~25K tokens / ~10s per
pondering run. A single small-model call does the same job for ~hundreds of tokens. The
pool's misprediction tolerance means the prediction need only be approximately right.

Marked speculative (background): runs on slack via the PASTE scheduler.
"""
from __future__ import annotations
from typing import Optional

from ..schemas import TaskDefinition
from ..logger import ExperimentLogger
from ..llm.client import chat_json

_SYS = (
    "You predict, in ONE short sentence, what the USER will most likely say on their next "
    "turn of an SOP-guided service call. Be specific and concrete (mention the concern or "
    "request), as if quoting the user. Output JSON only: {\"utterance\": \"...\"}."
)


async def predict_next_utterance(
    task: TaskDefinition,
    history: list[dict],
    *,
    predicted_action: str,
    cohort: str = "",
    state: str = "",
    mood: str = "",
    model: str = "gpt-4o-mini",
    logger: Optional[ExperimentLogger] = None,
) -> str:
    """Return a single predicted next user utterance (or "" on failure)."""
    convo = "\n".join(
        f"{h.get('role','?').upper()}: {(h.get('content') or '')[:160]}"
        for h in history[-5:]
    )
    user = (
        f"COHORT: {cohort or '?'} | STATE: {state or '?'} | MOOD: {mood or '?'}\n"
        f"The agent is about to: {predicted_action}.\n\n"
        f"CONVERSATION SO FAR:\n{convo}\n\n"
        "Predict the user's next utterance. JSON only."
    )
    try:
        parsed, _res = await chat_json(
            model=model, system=_SYS, user=user,
            temperature=0.3, max_tokens=60, logger=logger, call_site="next_utterance",
        )
        return (parsed.get("utterance") or "").strip()
    except Exception:
        return ""
