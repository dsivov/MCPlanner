"""Avatar live-testing routes.

A human tester talks to a GPT-Realtime voice avatar (in the browser). The avatar
is the "weak voice agent": fast, real-time, generates its own speech. Our
supervisor runs server-side here, steering it under FULL SOP-action control:

  1. tester speaks -> browser transcribes -> POST /api/avatar/{sid}/plan-turn
  2. supervisor classifies cohort/state/mood, picks the SOP action, consumes/curates
     prefetched data from the blackboard, and schedules the NEXT turn's prefetch
  3. endpoint returns the chosen action + its constraints + the prefetched data context
  4. the browser turns that into a session.update(instructions=...) for GPT-Realtime
     and triggers response.create — the avatar speaks the SOP-constrained reply
  5. on the next call the browser passes back what the avatar actually said
     (avatar_prev_response) so history/persistence stay correct

This reuses the locked no-tier-3 supervisor path. We do NOT call response_gen — the
realtime model produces the speech. We measure the same benchmark signals (classify
latency, action, prefetch hit-rate, pool rerank latency, blackboard contents).
"""
from __future__ import annotations
import time
from datetime import datetime
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..db import Experiment, TurnRecord, get_session
from ..schemas import MCTSConfig, PlannerTrace, TaskDefinition, RetrievedPrecedent
from ..logger import ExperimentLogger
from ..planner.precedents import (
    retrieve_precedents, embed_text, finalize_experiment, fill_previous_immediate_outcome,
    situation_text_from_history,
)
from ..planner.mcts import _cohort_state_propose
from ..planner.sop_graph import SOPGraph
from ..planner.pool_rerank import rerank_pool_for_turn
from ..planner.data_prefetch import manager as data_prefetch_manager
from ..planner.trajectory_predictor import (
    EmpiricalTrajectoryPredictor, build_prefetch_plan_from_predictions,
)
from ..routes.chat import _load_experiment, _state_log

router = APIRouter(prefix="/api/avatar", tags=["avatar"])


# In-memory per-session pending state: what the supervisor planned, awaiting the
# avatar's spoken response (supplied on the next call). FastAPI is single-process.
_PENDING: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Phase 2: GPT-Realtime ephemeral token minting (manual turn control)
# ---------------------------------------------------------------------------

class RealtimeSessionRequest(BaseModel):
    session_id: Optional[str] = None    # if given, seed initial instructions from the SOP


@router.post("/realtime-session")
async def realtime_session(
    req: RealtimeSessionRequest, db: AsyncSession = Depends(get_session),
) -> dict:
    """Mint a short-lived ephemeral token for the browser's WebRTC session.

    The real OPENAI_API_KEY never leaves the server. Crucially we configure MANUAL turn
    control (turn_detection.create_response = false) so the model transcribes the user's
    speech but does NOT auto-respond — that gives our supervisor the window to run
    plan-turn and steer the reply via session.update + response.create. Input audio
    transcription is enabled so the browser receives the user's words to send to plan-turn.
    """
    if not settings.OPENAI_API_KEY:
        raise HTTPException(500, "OPENAI_API_KEY not configured on the server")

    # Seed initial instructions: a DIRECTIVE, goal-driving persona (not a passive
    # responder). The supervisor replaces these per turn via session.update, but this
    # governs the very first turn and sets the agent's stance.
    instructions = (
        "You are a goal-driven outbound voice agent, not a passive assistant. You LEAD "
        "the conversation toward a specific objective: take initiative, introduce the next "
        "step proactively, and steer back on track when the user digresses—do not merely "
        "answer questions and wait. Speak naturally and briefly (1-2 sentences). On each "
        "turn you receive the specific step to take and any relevant data; perform that "
        "step and move the conversation forward. Use provided data verbatim; never invent facts."
    )
    if req.session_id:
        try:
            exp, task = await _load_experiment(req.session_id, db)
            cp = task.conversation_profile
            instructions = (
                f"You are {cp.agent_role}. Your objective for this call: {cp.goal} "
                f"You LEAD the conversation toward that objective—proactive, not reactive: "
                f"take initiative, introduce the next step yourself, and steer back on track "
                f"when the user digresses, rather than only answering and waiting. "
                f"KNOWLEDGE: {cp.knowledge} Speak naturally and briefly (1-2 sentences). On "
                f"each turn you are told the specific step to take and given any relevant "
                f"data; perform that step to advance toward the objective. Use the data "
                f"verbatim where it fits; never invent facts beyond what you are given."
            )
        except Exception:
            pass

    session_cfg = {
        "type": "realtime",
        "model": settings.REALTIME_MODEL,
        "instructions": instructions,
        "audio": {
            "input": {
                "transcription": {"model": "whisper-1"},
                "turn_detection": {"type": "server_vad", "create_response": False},
            },
            "output": {"voice": settings.REALTIME_VOICE},
        },
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(
                "https://api.openai.com/v1/realtime/client_secrets",
                headers={
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"session": session_cfg},
            )
        data = r.json()
        if r.status_code >= 400:
            raise HTTPException(r.status_code, f"realtime session error: {data}")
        value = data.get("value") or (data.get("client_secret") or {}).get("value")
        return {"value": value, "model": settings.REALTIME_MODEL, "voice": settings.REALTIME_VOICE}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"failed to mint realtime token: {e}")


class PlanTurnRequest(BaseModel):
    user_message: str
    avatar_prev_response: Optional[str] = None   # what the avatar actually said last turn


class DataContextItem(BaseModel):
    dependency_name: str
    source_action: str
    summary: str


class PlanTurnResponse(BaseModel):
    # Supervisor's plan for this turn (the realtime model speaks under these constraints).
    chosen_action: str
    action_description: str
    must_say: list[str]
    must_not_say: list[str]
    data_context: list[DataContextItem]
    # Goal anchoring (so the per-turn instruction can frame the action as goal progress).
    # session.update replaces the realtime instructions each turn, so the directive
    # persona must be re-stated here rather than relying on the mint-time persona.
    agent_role: str = ""
    goal: str = ""
    # Classification
    cohort: str
    user_state: str
    mood: str
    state_rationale: str
    # Blackboard / metrics
    pool_size: int
    pool_rerank_ms: int
    classify_ms: int
    prefetch_consumed: int
    prefetch_live: int
    prefetch_latency_hidden_ms: int
    next_prefetch_scheduled: int
    turn_index: int
    terminal_outcome: Optional[str] = None
    # Full trace for the panels (candidates, tier, etc.)
    trace: PlannerTrace


async def _record_committed_turn(
    db: AsyncSession, *, exp: Experiment, task: TaskDefinition,
    user_text: str, assistant_text: str, pending: dict,
) -> None:
    """Persist the previous turn now that we know what the avatar said. Schedules the
    after-turn prefetch for the turn that just completed, and finalizes on terminal."""
    trace: PlannerTrace = pending["trace"]
    history = list(exp.history or [])
    history.append({"role": "user", "content": user_text, "turn": pending["turn_index"]})
    history.append({"role": "assistant", "content": assistant_text,
                    "action": pending["chosen_action"], "turn": pending["turn_index"]})
    exp.history = history

    turn = TurnRecord(
        experiment_id=exp.id,
        turn_index=pending["turn_index"],
        user_message=user_text,
        assistant_message=assistant_text,
        chosen_action=pending["chosen_action"],
        predicted_user_state=pending["user_state"],
        state_rationale=pending.get("state_rationale", ""),
        mood=pending.get("mood", ""),
        mode="avatar",
        duration_ms=pending.get("plan_ms", 0),
        trace=trace.model_dump(),
        started_at=pending.get("started_at") or datetime.utcnow(),
        ended_at=datetime.utcnow(),
    )
    db.add(turn)
    await db.flush()

    # Schedule next-turn prefetch off the critical path: empirical action prediction +
    # optional cheap-LLM {user_text} (MCTS dropped from retrieval). Backgrounded so the
    # avatar's plan-turn isn't slowed by the prediction work; the prefetch targets the
    # NEXT turn, so a small scheduling delay is harmless.
    mcts_cfg = MCTSConfig.model_validate(exp.mcts_config or {})
    if mcts_cfg.data_prefetch_enabled and task.data_dependencies and exp.terminal_outcome is None:
        import asyncio
        from ..planner.retrieval_prefetch import schedule_retrieval_prefetch_bg
        # role-formatted history for the next-utterance predictor
        hist_roles = list(exp.history or [])
        asyncio.create_task(schedule_retrieval_prefetch_bg(
            experiment_id=exp.id, sop_ref=exp.sop_ref, task=task, history=hist_roles,
            cohort=pending.get("cohort", "") or "", state=pending.get("user_state", "") or "",
            mood=pending.get("mood", "") or "", chosen_action=pending["chosen_action"],
            current_turn_index=pending["turn_index"], mcts_cfg=mcts_cfg,
        ))

    # Finalize on terminal state.
    succ = set(task.conversation_profile.success_markers)
    fail = set(task.conversation_profile.failure_markers)
    if pending["user_state"] in succ:
        await finalize_experiment(db, experiment_id=exp.id, outcome="success", reward=1.0)
        await data_prefetch_manager.finalize_session(exp.id)
    elif pending["user_state"] in fail:
        await finalize_experiment(db, experiment_id=exp.id, outcome="failure", reward=0.0)
        await data_prefetch_manager.finalize_session(exp.id)


@router.post("/{session_id}/plan-turn", response_model=PlanTurnResponse)
async def plan_turn(
    session_id: str, req: PlanTurnRequest, db: AsyncSession = Depends(get_session),
) -> PlanTurnResponse:
    try:
        return await _plan_turn_impl(session_id, req, db)
    except HTTPException:
        raise
    except Exception:
        import traceback, sys
        traceback.print_exc(file=sys.stderr); sys.stderr.flush()
        raise


async def _plan_turn_impl(
    session_id: str, req: PlanTurnRequest, db: AsyncSession,
) -> PlanTurnResponse:
    exp, task = await _load_experiment(session_id, db)
    if exp.terminal_outcome is not None:
        raise HTTPException(400, f"session already ended ({exp.terminal_outcome})")

    user_text = (req.user_message or "").strip()
    if not user_text:
        raise HTTPException(400, "user_message required")

    mcts_cfg = MCTSConfig.model_validate(exp.mcts_config or {})

    # Commit the previous turn first, now that we know what the avatar said.
    pending = _PENDING.pop(session_id, None)
    if pending is not None and req.avatar_prev_response is not None:
        await _record_committed_turn(
            db, exp=exp, task=task,
            user_text=pending["user_text"], assistant_text=req.avatar_prev_response.strip(),
            pending=pending,
        )
        if exp.terminal_outcome is not None:
            await db.commit()
            return _terminal_plan_response(exp, pending)

    history = list(exp.history or [])
    state_log = await _state_log(db, exp.id)
    logger = ExperimentLogger(experiment_id=exp.id)
    current_turn_index = await db.scalar(
        select(func.count()).select_from(TurnRecord).where(TurnRecord.experiment_id == exp.id)
    ) or 0

    # Build the working history for planning: prior committed history + this user turn.
    plan_history = history + [{"role": "user", "content": user_text}]

    started_at = datetime.utcnow()
    t0 = time.perf_counter()

    # 1) Retrieve precedents (SOP-filtered, cohort unknown pre-classification).
    situation_text = situation_text_from_history(plan_history, cohort="", user_state="", last_n=4)
    try:
        situation_embedding = await embed_text(situation_text, logger=logger)
    except Exception:
        situation_embedding = b""
    precedents: list[RetrievedPrecedent] = []
    if situation_embedding:
        precedents = await retrieve_precedents(
            db, sop_ref=exp.sop_ref, cohort="", query_text=situation_text,
            query_embedding=situation_embedding, k=mcts_cfg.top_k_precedents,
        )

    # 2) Classify + pick action. Use the SAME path chat.py's baseline tier uses:
    # _cohort_state_propose (which has the recent-actions / progression prompt fix AND
    # classifies mood), then take the top SOP-allowed candidate. This avoids the
    # Greeting-loop the pure baseline planner exhibited.
    trace = PlannerTrace(mode="baseline")  # type: ignore[arg-type]
    trace.precedents = precedents
    classify_t0 = time.perf_counter()
    sg = SOPGraph(task)
    visited = sg.visited_from_history(plan_history, state_log)
    allowed = sg.allowed_actions(visited)
    cohort, state, mood, state_rat, cands, _res = await _cohort_state_propose(
        task, plan_history, allowed, k=max(1, mcts_cfg.branching),
        precedents=precedents, use_precedents=mcts_cfg.use_precedents_expand, logger=logger,
    )
    classify_ms = int((time.perf_counter() - classify_t0) * 1000)
    chosen = cands[0][0] if cands else (allowed[0] if allowed else "")
    trace.mood = mood
    trace.predicted_user_state = state
    trace.cohort = cohort
    trace.chosen_action = chosen
    # Surface candidates in the trace for the panel (action + rationale).
    from ..schemas import CandidateAction
    trace.candidates = [
        CandidateAction(action=a, q_value=(1.0 if i == 0 else 0.0), visits=0, rationale=r)
        for i, (a, r) in enumerate(cands)
    ]

    await fill_previous_immediate_outcome(
        db, experiment_id=exp.id, new_user_state=state, immediate_reward=0.5,
    )

    # 3) Consume speculatively-prefetched data for the chosen action (blackboard hit).
    prefetch_stats = {"consumed": 0, "live": 0, "latency_hidden_ms": 0, "live_latency_ms": 0}
    if mcts_cfg.data_prefetch_enabled and task.data_dependencies:
        _payloads, prefetch_stats = await data_prefetch_manager.consume(
            experiment_id=exp.id, sop_ref=exp.sop_ref, task=task, action_name=chosen,
            current_turn_index=current_turn_index,
            await_in_flight_ms=mcts_cfg.data_prefetch_await_in_flight_ms, live_fallback=True,
        )

    # 4) Pool rerank — curate 0-3 blackboard data items for this turn's context.
    pool_size = 0
    pool_rerank_ms = 0
    data_context: list[DataContextItem] = []
    if mcts_cfg.data_prefetch_enabled and task.data_dependencies:
        live_pool = data_prefetch_manager.get_pool(exp.id)
        pool_size = len(live_pool)
        if live_pool:
            picks, _rationale, pool_rerank_ms, _ = await rerank_pool_for_turn(
                live_pool, live_user_message=user_text, classified_cohort=cohort or "",
                classified_mood=mood or "", classified_state=state or "", chosen_action=chosen,
                max_picks=3, logger=logger,
            )
            for p in picks:
                data_context.append(DataContextItem(
                    dependency_name=p.dependency_name, source_action=p.source_action,
                    summary=p.payload_summary,
                ))

    plan_ms = int((time.perf_counter() - t0) * 1000)

    # Resolve the chosen action's constraints from the SOP.
    action_obj = next((a for a in task.agent_actions if a.name == chosen), None)
    action_desc = action_obj.description if action_obj else ""
    must_say = list(action_obj.must_say or []) if action_obj else []
    must_not_say = list(action_obj.must_not_say or []) if action_obj else []

    # Stash pending state so the NEXT call can commit this turn with the avatar's words.
    _PENDING[session_id] = {
        "turn_index": current_turn_index,
        "user_text": user_text,
        "chosen_action": chosen,
        "cohort": cohort, "user_state": state, "mood": mood,
        "state_rationale": state_rat,
        "trace": trace, "plan_ms": plan_ms, "started_at": started_at,
    }
    await db.commit()

    return PlanTurnResponse(
        chosen_action=chosen, action_description=action_desc,
        must_say=must_say, must_not_say=must_not_say, data_context=data_context,
        agent_role=task.conversation_profile.agent_role,
        goal=task.conversation_profile.goal,
        cohort=cohort, user_state=state, mood=mood, state_rationale=state_rat,
        pool_size=pool_size, pool_rerank_ms=pool_rerank_ms, classify_ms=classify_ms,
        prefetch_consumed=int(prefetch_stats["consumed"]),
        prefetch_live=int(prefetch_stats["live"]),
        prefetch_latency_hidden_ms=int(prefetch_stats["latency_hidden_ms"]),
        next_prefetch_scheduled=0,
        turn_index=current_turn_index, terminal_outcome=None, trace=trace,
    )


def _terminal_plan_response(exp: Experiment, pending: dict) -> PlanTurnResponse:
    """Minimal response when the committed turn reached a terminal state."""
    return PlanTurnResponse(
        chosen_action="(session ended)", action_description="",
        must_say=[], must_not_say=[], data_context=[],
        cohort=pending.get("cohort", ""), user_state=pending.get("user_state", ""),
        mood=pending.get("mood", ""), state_rationale="",
        pool_size=0, pool_rerank_ms=0, classify_ms=0, prefetch_consumed=0, prefetch_live=0,
        prefetch_latency_hidden_ms=0, next_prefetch_scheduled=0,
        turn_index=pending.get("turn_index", 0),
        terminal_outcome=exp.terminal_outcome, trace=pending["trace"],
    )


@router.get("/{session_id}/blackboard")
async def get_blackboard(session_id: str, db: AsyncSession = Depends(get_session)) -> dict:
    """Live snapshot of the session's blackboard pool — for the panel display."""
    exp, _task = await _load_experiment(session_id, db)
    pool = data_prefetch_manager.get_pool(exp.id)
    return {
        "session_id": exp.id,
        "pool_size": len(pool),
        "items": [
            {
                "kind": getattr(p, "kind", "data"),
                "dependency_name": p.dependency_name,
                "source_action": p.source_action,
                "predicted_user_state": p.predicted_user_state,
                "summary": p.payload_summary,
                "confidence": p.confidence,
                "predictor_source": p.predictor_source,
            }
            for p in pool
        ],
    }
