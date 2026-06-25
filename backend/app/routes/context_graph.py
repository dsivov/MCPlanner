"""Aggregated Context Graph endpoints for the third tab.

Two view modes:
  GET /api/context-graph?sop_ref=...   Structured graph: cohort -> action -> outcome nodes
                                       with edge weights (count) and lift (color).
  GET /api/context-graph/scatter?sop_ref=...  2D PCA projection of precedent embeddings.

Plus drill-down:
  GET /api/context-graph/traces?sop_ref=...&cohort=...&action=...&outcome=...
"""

from __future__ import annotations
from collections import defaultdict
from typing import Optional
import numpy as np
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session, PrecedentTrace, EMBED_DIM

router = APIRouter(prefix="/api/context-graph", tags=["context-graph"])


class GraphNode(BaseModel):
    id: str
    kind: str            # "cohort" | "action" | "outcome"
    label: str
    count: int = 0


class GraphEdge(BaseModel):
    src: str
    dst: str
    count: int = 0
    lift: float = 0.0    # for cohort->action edges; rate(action|cohort) - rate(cohort baseline)
    success_rate: float = 0.0


class GraphResponse(BaseModel):
    sop_ref: str
    n_precedents: int
    nodes: list[GraphNode]
    edges: list[GraphEdge]


OUTCOME_KEYS = ("success", "failure", "abandoned", "open")


@router.get("", response_model=GraphResponse)
async def get_graph(
    sop_ref: str = Query(...),
    cohort: Optional[str] = None,
    db: AsyncSession = Depends(get_session),
) -> GraphResponse:
    q = select(PrecedentTrace).where(PrecedentTrace.sop_ref == sop_ref)
    if cohort:
        q = q.where(PrecedentTrace.cohort == cohort)
    rows = list((await db.execute(q)).scalars().all())

    if not rows:
        return GraphResponse(sop_ref=sop_ref, n_precedents=0, nodes=[], edges=[])

    # Aggregate
    cohorts: dict[str, int] = defaultdict(int)
    actions: dict[str, int] = defaultdict(int)
    outcomes: dict[str, int] = defaultdict(int)
    by_cohort_action: dict[tuple[str, str], dict] = defaultdict(lambda: {"n": 0, "success": 0})
    by_action_outcome: dict[tuple[str, str], int] = defaultdict(int)
    by_cohort_total: dict[str, dict] = defaultdict(lambda: {"n": 0, "success": 0})

    for r in rows:
        c = r.cohort or "unknown"
        a = r.action or "(none)"
        o = r.terminal_outcome or "open"
        cohorts[c] += 1
        actions[a] += 1
        outcomes[o] += 1
        bucket = by_cohort_action[(c, a)]
        bucket["n"] += 1
        if o == "success":
            bucket["success"] += 1
            by_cohort_total[c]["success"] += 1
        by_cohort_total[c]["n"] += 1
        by_action_outcome[(a, o)] += 1

    nodes: list[GraphNode] = []
    for c, n in cohorts.items():
        nodes.append(GraphNode(id=f"cohort:{c}", kind="cohort", label=c, count=n))
    for a, n in actions.items():
        nodes.append(GraphNode(id=f"action:{a}", kind="action", label=a, count=n))
    for o in OUTCOME_KEYS:
        if outcomes.get(o, 0) > 0:
            nodes.append(GraphNode(id=f"outcome:{o}", kind="outcome", label=o, count=outcomes[o]))

    edges: list[GraphEdge] = []
    for (c, a), b in by_cohort_action.items():
        n = b["n"]
        s = b["success"]
        rate = s / n if n else 0.0
        baseline = (by_cohort_total[c]["success"] / by_cohort_total[c]["n"]) if by_cohort_total[c]["n"] else 0.0
        edges.append(GraphEdge(
            src=f"cohort:{c}", dst=f"action:{a}",
            count=n, lift=round(rate - baseline, 4), success_rate=round(rate, 4),
        ))
    for (a, o), n in by_action_outcome.items():
        edges.append(GraphEdge(src=f"action:{a}", dst=f"outcome:{o}", count=n))

    return GraphResponse(
        sop_ref=sop_ref, n_precedents=len(rows), nodes=nodes, edges=edges,
    )


class ScatterPoint(BaseModel):
    trace_id: str
    cohort: str
    action: str
    outcome: str
    x: float
    y: float


class ScatterResponse(BaseModel):
    sop_ref: str
    n_precedents: int
    points: list[ScatterPoint]


@router.get("/scatter", response_model=ScatterResponse)
async def get_scatter(
    sop_ref: str = Query(...),
    limit: int = Query(2000, ge=1, le=10000),
    db: AsyncSession = Depends(get_session),
) -> ScatterResponse:
    q = select(PrecedentTrace).where(PrecedentTrace.sop_ref == sop_ref).limit(limit)
    rows = list((await db.execute(q)).scalars().all())
    if not rows:
        return ScatterResponse(sop_ref=sop_ref, n_precedents=0, points=[])

    # Decode embeddings + PCA reduce. If only 1 point, return at origin.
    valid = [(r, r.situation_embedding) for r in rows if r.situation_embedding]
    if len(valid) < 2:
        points = []
        for r, _ in valid:
            points.append(ScatterPoint(
                trace_id=r.id, cohort=r.cohort or "unknown", action=r.action or "",
                outcome=r.terminal_outcome or "open", x=0.0, y=0.0,
            ))
        return ScatterResponse(sop_ref=sop_ref, n_precedents=len(valid), points=points)

    M = np.stack([np.frombuffer(b, dtype=np.float32) for _, b in valid])  # (N, D)
    # Mean-center then SVD-based PCA. For unit-norm embeddings the principal axes
    # capture the strongest semantic dimensions in the corpus — sufficient for inspection.
    centered = M - M.mean(axis=0, keepdims=True)
    # SVD over centered: thin, k=2
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pcs = centered @ Vt[:2].T  # (N, 2)
    # Normalize to a stable plotting range
    if pcs.shape[0] > 0:
        max_abs = np.max(np.abs(pcs)) or 1.0
        pcs = pcs / max_abs

    points: list[ScatterPoint] = []
    for (r, _), (x, y) in zip(valid, pcs):
        points.append(ScatterPoint(
            trace_id=r.id, cohort=r.cohort or "unknown", action=r.action or "",
            outcome=r.terminal_outcome or "open", x=float(x), y=float(y),
        ))
    return ScatterResponse(sop_ref=sop_ref, n_precedents=len(points), points=points)


class TraceDetail(BaseModel):
    id: str
    experiment_id: str
    sop_ref: str
    cohort: str
    action: str
    situation_text: str
    response_text: str
    immediate_state: str | None
    terminal_outcome: str | None
    immediate_reward: float
    terminal_reward: float | None
    created_at: str


@router.get("/traces", response_model=list[TraceDetail])
async def list_traces(
    sop_ref: str = Query(...),
    cohort: Optional[str] = None,
    action: Optional[str] = None,
    outcome: Optional[str] = None,
    limit: int = Query(200, ge=1, le=2000),
    db: AsyncSession = Depends(get_session),
) -> list[TraceDetail]:
    q = select(PrecedentTrace).where(PrecedentTrace.sop_ref == sop_ref)
    if cohort:
        q = q.where(PrecedentTrace.cohort == cohort)
    if action:
        q = q.where(PrecedentTrace.action == action)
    if outcome:
        if outcome == "open":
            q = q.where(PrecedentTrace.terminal_outcome.is_(None))
        else:
            q = q.where(PrecedentTrace.terminal_outcome == outcome)
    q = q.order_by(PrecedentTrace.created_at.desc()).limit(limit)
    rows = list((await db.execute(q)).scalars().all())
    return [TraceDetail(
        id=r.id,
        experiment_id=r.experiment_id,
        sop_ref=r.sop_ref,
        cohort=r.cohort,
        action=r.action,
        situation_text=r.situation_text,
        response_text=r.response_text or "",
        immediate_state=r.immediate_state,
        terminal_outcome=r.terminal_outcome,
        immediate_reward=r.immediate_reward or 0.0,
        terminal_reward=r.terminal_reward,
        created_at=r.created_at.isoformat(),
    ) for r in rows]
