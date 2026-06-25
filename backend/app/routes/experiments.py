"""Query + export endpoints for research analysis of past experiments.

A typical workflow:
  GET /api/experiments                                  -> list runs (paginated)
  GET /api/experiments/{id}                             -> full detail (turns + candidates + tokens summary)
  GET /api/experiments/{id}/llm-calls                   -> every LLM call (paginated)
  GET /api/experiments/{id}/export.jsonl                -> one JSONL file with everything, easy to load with pandas/duckdb
  PATCH /api/experiments/{id}                           -> attach notes
  DELETE /api/experiments/{id}                          -> remove a run
"""

from __future__ import annotations
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session, Experiment, TurnRecord, LLMCallRecord, MCTSCandidateRecord


router = APIRouter(prefix="/api/experiments", tags=["experiments"])


class ExperimentSummary(BaseModel):
    id: str
    sop_ref: str
    sop_name: str
    planner_mode: str
    chat_mode: str
    created_at: str
    updated_at: str
    notes: str = ""
    turn_count: int = 0
    tokens_in_total: int = 0
    tokens_out_total: int = 0
    llm_calls_total: int = 0
    duration_ms_total: int = 0


@router.get("", response_model=list[ExperimentSummary])
async def list_experiments(
    db: AsyncSession = Depends(get_session),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    planner_mode: str | None = None,
    sop_name: str | None = None,
) -> list[ExperimentSummary]:
    q = select(Experiment).order_by(Experiment.created_at.desc()).limit(limit).offset(offset)
    if planner_mode:
        q = q.where(Experiment.planner_mode == planner_mode)
    if sop_name:
        q = q.where(Experiment.sop_name == sop_name)
    rows = (await db.execute(q)).scalars().all()

    out: list[ExperimentSummary] = []
    for exp in rows:
        agg = (await db.execute(
            select(
                func.count(TurnRecord.id),
                func.coalesce(func.sum(TurnRecord.tokens_in), 0),
                func.coalesce(func.sum(TurnRecord.tokens_out), 0),
                func.coalesce(func.sum(TurnRecord.duration_ms), 0),
            ).where(TurnRecord.experiment_id == exp.id)
        )).one()
        llm_total = await db.scalar(
            select(func.count(LLMCallRecord.id)).where(LLMCallRecord.experiment_id == exp.id)
        )
        out.append(ExperimentSummary(
            id=exp.id,
            sop_ref=exp.sop_ref,
            sop_name=exp.sop_name,
            planner_mode=exp.planner_mode,
            chat_mode=exp.chat_mode,
            created_at=exp.created_at.isoformat(),
            updated_at=exp.updated_at.isoformat(),
            notes=exp.notes or "",
            turn_count=int(agg[0] or 0),
            tokens_in_total=int(agg[1] or 0),
            tokens_out_total=int(agg[2] or 0),
            llm_calls_total=int(llm_total or 0),
            duration_ms_total=int(agg[3] or 0),
        ))
    return out


class TurnDetail(BaseModel):
    id: str
    turn_index: int
    started_at: str
    duration_ms: int
    user_message: str
    assistant_message: str
    chosen_action: str
    predicted_user_state: str
    state_rationale: str
    mode: str
    tokens_in: int
    tokens_out: int
    mcts_iterations: int
    rollouts: int
    trace: dict
    candidates: list[dict]


class ExperimentDetail(BaseModel):
    id: str
    sop_ref: str
    sop_name: str
    sop_snapshot: dict
    planner_mode: str
    chat_mode: str
    mcts_config: dict
    models: dict
    notes: str
    history: list[dict]
    created_at: str
    turns: list[TurnDetail]


@router.get("/{exp_id}", response_model=ExperimentDetail)
async def get_experiment(exp_id: str, db: AsyncSession = Depends(get_session)) -> ExperimentDetail:
    exp = await db.get(Experiment, exp_id)
    if not exp:
        raise HTTPException(404, "experiment not found")
    turns = (await db.execute(
        select(TurnRecord).where(TurnRecord.experiment_id == exp_id).order_by(TurnRecord.turn_index)
    )).scalars().all()

    turn_details: list[TurnDetail] = []
    for t in turns:
        cands = (await db.execute(
            select(MCTSCandidateRecord).where(MCTSCandidateRecord.turn_id == t.id).order_by(MCTSCandidateRecord.rank)
        )).scalars().all()
        turn_details.append(TurnDetail(
            id=t.id,
            turn_index=t.turn_index,
            started_at=t.started_at.isoformat(),
            duration_ms=t.duration_ms or 0,
            user_message=t.user_message or "",
            assistant_message=t.assistant_message or "",
            chosen_action=t.chosen_action or "",
            predicted_user_state=t.predicted_user_state or "",
            state_rationale=t.state_rationale or "",
            mode=t.mode or "",
            tokens_in=t.tokens_in or 0,
            tokens_out=t.tokens_out or 0,
            mcts_iterations=t.mcts_iterations or 0,
            rollouts=t.rollouts or 0,
            trace=t.trace or {},
            candidates=[{
                "rank": c.rank, "action": c.action, "q_value": c.q_value,
                "visits": c.visits, "rationale": c.rationale, "was_chosen": c.was_chosen,
            } for c in cands],
        ))

    return ExperimentDetail(
        id=exp.id,
        sop_ref=exp.sop_ref,
        sop_name=exp.sop_name,
        sop_snapshot=exp.sop_snapshot,
        planner_mode=exp.planner_mode,
        chat_mode=exp.chat_mode,
        mcts_config=exp.mcts_config or {},
        models=exp.models or {},
        notes=exp.notes or "",
        history=exp.history or [],
        created_at=exp.created_at.isoformat(),
        turns=turn_details,
    )


class LLMCallDetail(BaseModel):
    id: str
    turn_id: str | None
    call_site: str
    model: str
    started_at: str
    duration_ms: int
    tokens_in: int
    tokens_out: int
    temperature: float | None
    max_tokens: int | None
    system_prompt: str
    user_prompt: str
    response_text: str
    response_json: dict | None
    is_json_mode: bool
    ok: bool
    error: str | None


@router.get("/{exp_id}/llm-calls", response_model=list[LLMCallDetail])
async def list_llm_calls(
    exp_id: str,
    db: AsyncSession = Depends(get_session),
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    call_site: str | None = None,
) -> list[LLMCallDetail]:
    q = select(LLMCallRecord).where(LLMCallRecord.experiment_id == exp_id).order_by(LLMCallRecord.started_at)
    if call_site:
        q = q.where(LLMCallRecord.call_site == call_site)
    q = q.limit(limit).offset(offset)
    rows = (await db.execute(q)).scalars().all()
    return [LLMCallDetail(
        id=r.id, turn_id=r.turn_id, call_site=r.call_site, model=r.model,
        started_at=r.started_at.isoformat(), duration_ms=r.duration_ms or 0,
        tokens_in=r.tokens_in or 0, tokens_out=r.tokens_out or 0,
        temperature=r.temperature, max_tokens=r.max_tokens,
        system_prompt=r.system_prompt or "", user_prompt=r.user_prompt or "",
        response_text=r.response_text or "", response_json=r.response_json,
        is_json_mode=bool(r.is_json_mode), ok=bool(r.ok), error=r.error,
    ) for r in rows]


@router.get("/{exp_id}/export.jsonl", response_class=PlainTextResponse)
async def export_jsonl(exp_id: str, db: AsyncSession = Depends(get_session)) -> str:
    """One JSONL stream: meta row, then one row per turn (with its candidates) and per LLM call.

    Each line is a dict with a 'kind' field: "experiment" | "turn" | "candidate" | "llm_call".
    Load with: `pd.read_json("export.jsonl", lines=True)`.
    """
    import json as _json
    exp = await db.get(Experiment, exp_id)
    if not exp:
        raise HTTPException(404, "experiment not found")
    lines: list[str] = []
    lines.append(_json.dumps({
        "kind": "experiment",
        "id": exp.id,
        "sop_ref": exp.sop_ref,
        "sop_name": exp.sop_name,
        "planner_mode": exp.planner_mode,
        "chat_mode": exp.chat_mode,
        "mcts_config": exp.mcts_config,
        "models": exp.models,
        "notes": exp.notes or "",
        "created_at": exp.created_at.isoformat(),
        "sop_snapshot": exp.sop_snapshot,
    }))
    turns = (await db.execute(
        select(TurnRecord).where(TurnRecord.experiment_id == exp_id).order_by(TurnRecord.turn_index)
    )).scalars().all()
    for t in turns:
        lines.append(_json.dumps({
            "kind": "turn",
            "experiment_id": exp.id,
            "id": t.id,
            "turn_index": t.turn_index,
            "started_at": t.started_at.isoformat(),
            "duration_ms": t.duration_ms or 0,
            "user_message": t.user_message,
            "assistant_message": t.assistant_message,
            "chosen_action": t.chosen_action,
            "predicted_user_state": t.predicted_user_state,
            "state_rationale": t.state_rationale,
            "mode": t.mode,
            "tokens_in": t.tokens_in,
            "tokens_out": t.tokens_out,
            "mcts_iterations": t.mcts_iterations,
            "rollouts": t.rollouts,
        }))
        cands = (await db.execute(
            select(MCTSCandidateRecord).where(MCTSCandidateRecord.turn_id == t.id).order_by(MCTSCandidateRecord.rank)
        )).scalars().all()
        for c in cands:
            lines.append(_json.dumps({
                "kind": "candidate", "experiment_id": exp.id, "turn_id": t.id,
                "rank": c.rank, "action": c.action, "q_value": c.q_value,
                "visits": c.visits, "rationale": c.rationale, "was_chosen": bool(c.was_chosen),
            }))
    calls = (await db.execute(
        select(LLMCallRecord).where(LLMCallRecord.experiment_id == exp_id).order_by(LLMCallRecord.started_at)
    )).scalars().all()
    for r in calls:
        lines.append(_json.dumps({
            "kind": "llm_call", "experiment_id": exp.id, "turn_id": r.turn_id,
            "id": r.id, "call_site": r.call_site, "model": r.model,
            "started_at": r.started_at.isoformat(), "duration_ms": r.duration_ms or 0,
            "tokens_in": r.tokens_in or 0, "tokens_out": r.tokens_out or 0,
            "temperature": r.temperature, "max_tokens": r.max_tokens,
            "system_prompt": r.system_prompt or "", "user_prompt": r.user_prompt or "",
            "response_text": r.response_text or "", "response_json": r.response_json,
            "is_json_mode": bool(r.is_json_mode), "ok": bool(r.ok), "error": r.error,
        }))
    return "\n".join(lines) + "\n"


class NotesPatch(BaseModel):
    notes: str


@router.patch("/{exp_id}")
async def patch_notes(exp_id: str, p: NotesPatch, db: AsyncSession = Depends(get_session)) -> dict:
    exp = await db.get(Experiment, exp_id)
    if not exp:
        raise HTTPException(404, "experiment not found")
    exp.notes = p.notes or ""
    await db.commit()
    return {"ok": True}


@router.delete("/{exp_id}")
async def delete_experiment(exp_id: str, db: AsyncSession = Depends(get_session)) -> dict:
    await db.execute(delete(Experiment).where(Experiment.id == exp_id))
    await db.commit()
    return {"ok": True}


@router.get("/{exp_id}/mcts-replay/{turn_index}")
async def mcts_replay(exp_id: str, turn_index: int, db: AsyncSession = Depends(get_session)) -> dict:
    """Return everything needed to reconstruct + replay one turn's MCTS search.

    Rollouts are returned in `rollout_index` order — that's the MCTS iteration order, so a
    UI can replay them sequentially to animate the search tree's evolution (Q updates,
    backprop, etc.).
    """
    from ..db import TurnRecord, MCTSCandidateRecord, RolloutRecord, Experiment as ExperimentTable
    exp = await db.get(ExperimentTable, exp_id)
    if not exp:
        raise HTTPException(404, "experiment not found")

    res = await db.execute(
        select(TurnRecord).where(
            TurnRecord.experiment_id == exp_id,
            TurnRecord.turn_index == turn_index,
        ).limit(1)
    )
    turn = res.scalar_one_or_none()
    if turn is None:
        raise HTTPException(404, "turn not found")

    cands = (await db.execute(
        select(MCTSCandidateRecord)
        .where(MCTSCandidateRecord.turn_id == turn.id)
        .order_by(MCTSCandidateRecord.rank)
    )).scalars().all()

    rollouts = (await db.execute(
        select(RolloutRecord)
        .where(RolloutRecord.turn_id == turn.id)
        .order_by(RolloutRecord.rollout_index)
    )).scalars().all()

    return {
        "experiment_id": exp_id,
        "turn_id": turn.id,
        "turn_index": turn.turn_index,
        "chosen_action": turn.chosen_action or "",
        "predicted_user_state": turn.predicted_user_state or "",
        "state_rationale": turn.state_rationale or "",
        "mode": turn.mode or "mcts",
        "config": exp.mcts_config or {},
        "trace": turn.trace or {},
        "duration_ms": turn.duration_ms or 0,
        "user_message": turn.user_message or "",
        "assistant_message": turn.assistant_message or "",
        "candidates": [{
            "rank": c.rank, "action": c.action, "q_value": c.q_value,
            "visits": c.visits, "rationale": c.rationale or "", "was_chosen": bool(c.was_chosen),
        } for c in cands],
        "rollouts": [{
            "rollout_index": r.rollout_index,
            "first_action": r.first_action or "",
            "planned_actions": r.planned_actions or [],
            "planned_states": r.planned_states or [],
            "final_state": r.final_state or "",
            "depth_completed": r.depth_completed or 0,
            "hit_failure": bool(r.hit_failure),
            "hit_success": bool(r.hit_success),
            "rationality": r.rationality,
            "progress_bonus": r.progress_bonus or 0.0,
            "reward": r.reward or 0.0,
            "rollout_mode": r.rollout_mode or "simulate",
            "duration_ms": r.duration_ms or 0,
        } for r in rollouts],
    }


@router.get("/{exp_id}/turn-indices")
async def list_turn_indices(exp_id: str, db: AsyncSession = Depends(get_session)) -> list[dict]:
    """Quick lookup for the MCTS-Replay UI to populate its turn picker."""
    from ..db import TurnRecord
    rows = (await db.execute(
        select(
            TurnRecord.turn_index, TurnRecord.chosen_action, TurnRecord.mode, TurnRecord.rollouts
        ).where(TurnRecord.experiment_id == exp_id).order_by(TurnRecord.turn_index)
    )).all()
    return [{
        "turn_index": r[0], "chosen_action": r[1] or "", "mode": r[2] or "",
        "rollouts": r[3] or 0,
    } for r in rows]


@router.get("/{exp_id}/data-fetches")
async def list_data_fetches(exp_id: str, db: AsyncSession = Depends(get_session)) -> list[dict]:
    from ..db import DataFetch
    rows = (await db.execute(
        select(DataFetch).where(DataFetch.experiment_id == exp_id).order_by(DataFetch.started_at)
    )).scalars().all()
    return [{
        "id": r.id,
        "cache_key": r.cache_key,
        "dependency_name": r.dependency_name,
        "action_name": r.action_name,
        "kind": r.kind,
        "issued_at_turn": r.issued_at_turn,
        "predicted_turn": r.predicted_turn,
        "consumed_at_turn": r.consumed_at_turn,
        "started_at": r.started_at.isoformat() if r.started_at else None,
        "completed_at": r.completed_at.isoformat() if r.completed_at else None,
        "fetch_duration_ms": r.fetch_duration_ms,
        "confidence": r.confidence,
        "consumed": bool(r.consumed),
        "wasted": bool(r.wasted),
        "speculative": bool(r.speculative),
        "fetch_error": r.fetch_error,
        "predictor_source": r.predictor_source or "mcts",
        "predicted_user_state": r.predicted_user_state,
    } for r in rows]
