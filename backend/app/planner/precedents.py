"""Precedent / context-graph retrieval + write helpers.

The Fast loop:
  - On every agent turn, after the response is produced, we embed the situation_text
    (cohort + predicted user_state + last-N dialogue lines) and write a PrecedentTrace row
    plus a parallel row in the vec0 virtual table.
  - At the START of the next turn (before action selection), we retrieve top-K precedents
    matching the current SOP + cohort. The retrieval is exposed to the planner via three
    independent injection points.

Outcome bookkeeping:
  - immediate_state of a PrecedentTrace is filled when the NEXT turn fires (we then know
    what user_state followed the action).
  - terminal_outcome is back-propagated when the session is finalized (success/failure
    marker hit or /api/chat/{id}/end called).
"""

from __future__ import annotations
import time
import struct
import json
from datetime import datetime
from typing import Optional

import numpy as np
from sqlalchemy import select, update, text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..db import EMBED_DIM, PrecedentTrace, PrecedentRetrieval, Experiment
from ..llm.client import client
from ..logger import ExperimentLogger, LLMCallEntry
from ..schemas import RetrievedPrecedent


# ---------- Embedding ----------

async def embed_text(
    text: str,
    *,
    logger: Optional[ExperimentLogger] = None,
    call_site: str = "embed_situation",
) -> bytes:
    """Returns float32 little-endian bytes of length EMBED_DIM*4."""
    t0 = time.perf_counter()
    started_at = datetime.utcnow()
    text = text or "(empty)"
    try:
        r = await client().embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        vec = np.array(r.data[0].embedding, dtype=np.float32)
        # Normalize to unit length so vec0's distance is comparable to cosine similarity.
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        duration_ms = int((time.perf_counter() - t0) * 1000)
        tokens = getattr(r.usage, "total_tokens", 0) or 0
        if logger is not None:
            logger.record_llm_call(LLMCallEntry(
                call_site=call_site,
                model="text-embedding-3-small",
                started_at=started_at,
                duration_ms=duration_ms,
                tokens_in=tokens,
                tokens_out=0,
                temperature=None,
                max_tokens=None,
                system_prompt="",
                user_prompt=text,
                response_text=f"<{EMBED_DIM}-dim float32 embedding>",
                response_json=None,
                is_json_mode=False,
                ok=True,
            ))
        return vec.tobytes()
    except Exception as e:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        if logger is not None:
            logger.record_llm_call(LLMCallEntry(
                call_site=call_site, model="text-embedding-3-small",
                started_at=started_at, duration_ms=duration_ms,
                tokens_in=0, tokens_out=0, temperature=None, max_tokens=None,
                system_prompt="", user_prompt=text, response_text="",
                response_json=None, is_json_mode=False, ok=False, error=str(e),
            ))
        raise


def situation_text_from_history(
    history: list[dict[str, str]],
    cohort: str,
    user_state: str,
    *,
    last_n: int = 4,
) -> str:
    """Compact, deterministic representation of the current situation for embedding."""
    tail = history[-last_n:] if last_n > 0 else history
    lines = [f"COHORT={cohort or 'unknown'}", f"USER_STATE={user_state or 'unknown'}", "RECENT:"]
    for m in tail:
        role = m.get("role", "?")[:1].upper()
        content = (m.get("content") or "").strip().replace("\n", " ")
        if len(content) > 200:
            content = content[:200] + "…"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _serialize_vec(emb_bytes: bytes) -> bytes:
    """sqlite-vec accepts raw little-endian float32 bytes for FLOAT[N] columns."""
    return emb_bytes


# ---------- Write ----------

async def write_precedent(
    db: AsyncSession,
    *,
    turn_id: str,
    experiment_id: str,
    sop_ref: str,
    cohort: str,
    situation_text: str,
    embedding_bytes: bytes,
    action: str,
    response_text: str,
    mood: str | None = None,
) -> str:
    rec = PrecedentTrace(
        turn_id=turn_id,
        experiment_id=experiment_id,
        sop_ref=sop_ref,
        cohort=cohort,
        mood=mood or None,
        situation_text=situation_text,
        situation_embedding=embedding_bytes,
        action=action,
        response_text=response_text,
        immediate_state=None,
        immediate_reward=0.0,
        terminal_outcome=None,
        terminal_reward=None,
    )
    db.add(rec)
    await db.flush()
    # Mirror into vec0 for nearest-neighbour search.
    try:
        await db.execute(sql_text(
            "INSERT INTO vec_precedents (trace_id, sop_ref, cohort, embedding) VALUES (:tid, :sop, :coh, :emb)"
        ), {"tid": rec.id, "sop": sop_ref, "coh": cohort, "emb": embedding_bytes})
    except Exception:
        # vec0 absent — retrieval will use ORDER BY rowid fallback.
        pass
    return rec.id


# ---------- Immediate-outcome back-fill ----------

async def fill_previous_immediate_outcome(
    db: AsyncSession,
    *,
    experiment_id: str,
    new_user_state: str,
    immediate_reward: float = 0.5,
) -> None:
    """When a new turn begins, the previous turn's PrecedentTrace gets its
    immediate_state filled in (the state we now observe). Called from the chat route."""
    if not new_user_state:
        return
    # Find the latest precedent for this experiment with no immediate_state yet.
    res = await db.execute(
        select(PrecedentTrace)
        .where(PrecedentTrace.experiment_id == experiment_id, PrecedentTrace.immediate_state.is_(None))
        .order_by(PrecedentTrace.created_at.desc())
        .limit(1)
    )
    prev = res.scalar_one_or_none()
    if prev is None:
        return
    prev.immediate_state = new_user_state
    prev.immediate_reward = float(immediate_reward)


# ---------- Terminal back-prop ----------

async def finalize_experiment(
    db: AsyncSession,
    *,
    experiment_id: str,
    outcome: str,
    reward: float,
) -> int:
    """Mark an experiment terminal and back-propagate the terminal outcome to every
    PrecedentTrace in the session. Returns the count of traces updated."""
    now = datetime.utcnow()
    exp = await db.get(Experiment, experiment_id)
    if not exp:
        return 0
    exp.terminal_outcome = outcome
    exp.terminal_reward = float(reward)
    exp.ended_at = now

    # Update all precedents. turn_distance_to_terminal counts back from the end.
    res = await db.execute(
        select(PrecedentTrace)
        .where(PrecedentTrace.experiment_id == experiment_id)
        .order_by(PrecedentTrace.created_at.asc())
    )
    rows = res.scalars().all()
    n = len(rows)
    for i, r in enumerate(rows):
        r.terminal_outcome = outcome
        r.terminal_reward = float(reward)
        r.turn_distance_to_terminal = n - 1 - i
    return n


# ---------- Retrieval ----------

async def retrieve_precedents(
    db: AsyncSession,
    *,
    sop_ref: str,
    cohort: str,
    query_text: str,
    query_embedding: bytes,
    k: int = 3,
    cohort_required: bool = False,
    mood_hint: str | None = None,
) -> list[RetrievedPrecedent]:
    """Vector top-K with same-SOP filter. If `cohort_required` is true, also restrict to
    the same cohort; otherwise prefer same cohort but allow cross-cohort matches.

    When `mood_hint` is given, the FIRST pass restricts to matching `(cohort, mood)`. If
    that pass yields < k unique-action results, we re-run without the mood filter and
    union the result lists (keeping the mood-matched ones first). This is Phase-2 of the
    disposition-diversity programme — sharpens response-style retrieval for mood-aware
    responses, with the staleness mitigation that the caller passes the PRIOR turn's
    classified mood (mood is medium-moving, so prior-turn is a reasonable proxy).

    Includes a simple MMR diversity pass: dedupe by action.
    """
    if k <= 0:
        return []

    async def _run_query(use_mood: bool) -> list[dict]:
        rows: list[dict] = []
        try:
            cohort_pred = "AND p.cohort = :cohort" if cohort_required and cohort else ""
            mood_pred = "AND p.mood = :mood" if (use_mood and mood_hint) else ""
            params = {"sop": sop_ref, "emb": query_embedding, "k": k * 3}
            if cohort_required and cohort:
                params["cohort"] = cohort
            if use_mood and mood_hint:
                params["mood"] = mood_hint
            q = sql_text(
                f"""
                SELECT v.trace_id, v.distance, p.cohort, p.action, p.immediate_state,
                       p.immediate_reward, p.terminal_outcome, p.terminal_reward, p.response_text,
                       p.mood
                FROM vec_precedents v
                JOIN precedent_traces p ON p.id = v.trace_id
                WHERE v.embedding MATCH :emb
                  AND k = :k
                  AND v.sop_ref = :sop
                  {cohort_pred}
                  {mood_pred}
                ORDER BY v.distance ASC
                """
            )
            res = await db.execute(q, params)
            for r in res.mappings():
                rows.append(dict(r))
        except Exception:
            # SQL fallback (no sqlite-vec): same shape, ORDER BY recency.
            q = select(PrecedentTrace).where(PrecedentTrace.sop_ref == sop_ref)
            if cohort_required and cohort:
                q = q.where(PrecedentTrace.cohort == cohort)
            if use_mood and mood_hint:
                q = q.where(PrecedentTrace.mood == mood_hint)
            q = q.order_by(PrecedentTrace.created_at.desc()).limit(k * 3)
            res = await db.execute(q)
            for p in res.scalars().all():
                rows.append({
                    "trace_id": p.id, "distance": 1.0, "cohort": p.cohort, "action": p.action,
                    "immediate_state": p.immediate_state, "immediate_reward": p.immediate_reward,
                    "terminal_outcome": p.terminal_outcome, "terminal_reward": p.terminal_reward,
                    "response_text": p.response_text, "mood": p.mood,
                })
        return rows

    # Two-pass retrieval: mood-conditional first, then unconditional to fill the rest.
    rows: list[dict]
    if mood_hint:
        primary = await _run_query(use_mood=True)
        # Dedup by action so we know how many UNIQUE-ACTION mood matches we have
        primary_actions = {r.get("action") for r in primary}
        if len(primary_actions) < k:
            fallback = await _run_query(use_mood=False)
            seen = {r["trace_id"] for r in primary}
            for r in fallback:
                if r["trace_id"] not in seen:
                    primary.append(r)
                    seen.add(r["trace_id"])
        rows = primary
    else:
        rows = await _run_query(use_mood=False)

    # MMR-ish diversification: keep highest-similarity per action until we hit k.
    seen_actions: set[str] = set()
    out: list[RetrievedPrecedent] = []
    for r in rows:
        a = r.get("action") or ""
        if a in seen_actions:
            continue
        seen_actions.add(a)
        dist = float(r.get("distance", 1.0))
        # vec0 returns cosine DISTANCE in [0, 2] for unit-norm vectors. Convert to similarity.
        sim = max(0.0, 1.0 - dist / 2.0) if dist > 1.0 else 1.0 - dist
        out.append(RetrievedPrecedent(
            id=r["trace_id"],
            cohort=r.get("cohort") or "",
            action=a,
            immediate_state=r.get("immediate_state") or "",
            terminal_outcome=r.get("terminal_outcome"),
            immediate_reward=float(r.get("immediate_reward") or 0.0),
            terminal_reward=(float(r["terminal_reward"]) if r.get("terminal_reward") is not None else None),
            similarity=round(sim, 4),
            response_text=(r.get("response_text") or "")[:400],
        ))
        if len(out) >= k:
            break
    return out


# ---------- Retrieval logging ----------

async def write_retrieval_log(
    db: AsyncSession,
    *,
    turn_id: str,
    experiment_id: str,
    sop_ref: str,
    cohort: str,
    query_text: str,
    top_k_requested: int,
    results: list[RetrievedPrecedent],
    used_expand: bool,
    used_score: bool,
    used_response: bool,
    duration_ms: int,
) -> None:
    db.add(PrecedentRetrieval(
        turn_id=turn_id,
        experiment_id=experiment_id,
        sop_ref=sop_ref,
        cohort=cohort,
        query_situation_text=query_text,
        top_k_requested=top_k_requested,
        results=[r.model_dump() for r in results],
        used_expand=used_expand,
        used_score=used_score,
        used_response=used_response,
        duration_ms=duration_ms,
    ))
