"""Slow loop: lift analysis + phrase mining over accumulated PrecedentTrace rows.

Endpoints:
  POST /api/learn/mine?sop_ref=...   Run analysis; persist a LearningRun row; return proposals.
  POST /api/learn/apply              Apply selected proposals to a saved SOP (PUT-equivalent).
  GET  /api/learn/runs?sop_ref=...   List past runs.
  GET  /api/learn/runs/{run_id}      Full payload of one run.

The mining algorithm is deliberately simple — useful for hundreds of sessions but won't
generalize without statistical correction at larger scale. It's a research scaffold.
"""

from __future__ import annotations
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import (
    get_session, PrecedentTrace, Experiment, LearningRun, SOPRecord,
)
from ..schemas import TaskDefinition

router = APIRouter(prefix="/api/learn", tags=["learn"])


# ---------- Schemas ----------

class LiftRow(BaseModel):
    cohort: str
    action: str
    n_total: int
    n_success: int
    n_failure: int
    n_abandoned: int
    n_open: int
    success_rate: float
    cohort_baseline: float
    action_baseline: float
    lift_vs_cohort: float
    lift_vs_action: float


class Proposal(BaseModel):
    id: str
    cohort: str
    action: str
    must_say_add: list[str] = Field(default_factory=list)
    must_not_say_add: list[str] = Field(default_factory=list)
    citations_success: list[str] = Field(default_factory=list)
    citations_failure: list[str] = Field(default_factory=list)
    rationale: str = ""


class MineResponse(BaseModel):
    run_id: str
    sop_ref: str
    n_precedents: int
    n_sessions: int
    lift_table: list[LiftRow]
    proposals: list[Proposal]
    duration_ms: int


class ApplyRequest(BaseModel):
    run_id: str
    sop_id: str   # the saved SOP to update
    proposal_ids: list[str]


class ApplyResponse(BaseModel):
    sop_id: str
    accepted: list[str]
    actions_updated: int


class SaveAndApplyRequest(BaseModel):
    run_id: str
    source_sop_ref: str            # "seed:<file>" or a saved sop_id
    proposal_ids: list[str]
    new_name: str | None = None    # optional override; defaults to "<source name> (mined)"


class SaveAndApplyResponse(BaseModel):
    sop_id: str
    sop_name: str
    accepted: list[str]
    actions_updated: int


# ---------- Token / n-gram utilities ----------

# Conservative English stopwords. Add domain-specific ones if needed.
_STOP = set("""
a an the is are was were be been being am of in on at to for from with by as it its this that these those
i you he she we they him her us them my your his their our its mine yours hers ours theirs
do does did doing have has had having will would shall should may might must can could
and or but if then else not no nor so than too very also just only such own same other another
about above below between under over here there where when why how what who which whose whom
me very can will would just very well also even more most some any all
""".split())

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z'\-]+")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _is_useful_ngram(g: str) -> bool:
    parts = g.split()
    if not parts:
        return False
    # Require at least one content word
    if all(p in _STOP for p in parts):
        return False
    # Reject leading/trailing stopwords for bigrams/trigrams
    if len(parts) > 1 and (parts[0] in _STOP and parts[-1] in _STOP):
        return False
    if any(len(p) <= 2 for p in parts):
        return False
    return True


def _candidate_ngrams(text: str, *, ns: tuple[int, ...] = (2, 3, 4)) -> Counter:
    toks = _tokens(text)
    c: Counter = Counter()
    for n in ns:
        for g in _ngrams(toks, n):
            if _is_useful_ngram(g):
                c[g] += 1
    return c


# ---------- Lift + phrase mining ----------

def _build_lift_table(traces: list[PrecedentTrace]) -> list[LiftRow]:
    by_cohort_action: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"n": 0, "success": 0, "failure": 0, "abandoned": 0, "open": 0}
    )
    by_cohort: dict[str, dict] = defaultdict(lambda: {"n": 0, "success": 0})
    by_action: dict[str, dict] = defaultdict(lambda: {"n": 0, "success": 0})

    for t in traces:
        c, a = (t.cohort or "unknown"), (t.action or "")
        bucket = by_cohort_action[(c, a)]
        bucket["n"] += 1
        if t.terminal_outcome == "success":
            bucket["success"] += 1
            by_cohort[c]["success"] += 1
            by_action[a]["success"] += 1
        elif t.terminal_outcome == "failure":
            bucket["failure"] += 1
        elif t.terminal_outcome == "abandoned":
            bucket["abandoned"] += 1
        else:
            bucket["open"] += 1
        by_cohort[c]["n"] += 1
        by_action[a]["n"] += 1

    rows: list[LiftRow] = []
    for (c, a), b in by_cohort_action.items():
        n = b["n"]
        s = b["success"]
        success_rate = s / n if n else 0.0
        coh_base = (by_cohort[c]["success"] / by_cohort[c]["n"]) if by_cohort[c]["n"] else 0.0
        act_base = (by_action[a]["success"] / by_action[a]["n"]) if by_action[a]["n"] else 0.0
        rows.append(LiftRow(
            cohort=c, action=a,
            n_total=n, n_success=s, n_failure=b["failure"], n_abandoned=b["abandoned"], n_open=b["open"],
            success_rate=round(success_rate, 4),
            cohort_baseline=round(coh_base, 4),
            action_baseline=round(act_base, 4),
            lift_vs_cohort=round(success_rate - coh_base, 4),
            lift_vs_action=round(success_rate - act_base, 4),
        ))
    rows.sort(key=lambda r: (-r.lift_vs_cohort, -r.n_total))
    return rows


def _mine_phrases_for_pair(
    success_texts: list[tuple[str, str]],  # [(trace_id, response_text)]
    failure_texts: list[tuple[str, str]],
    *,
    top_n: int = 4,
    min_count: int = 2,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Returns (must_say_add, must_not_say_add, citations_success, citations_failure)."""
    succ_counter: Counter = Counter()
    fail_counter: Counter = Counter()
    succ_by_ngram: dict[str, list[str]] = defaultdict(list)
    fail_by_ngram: dict[str, list[str]] = defaultdict(list)

    for tid, txt in success_texts:
        for g, n in _candidate_ngrams(txt).items():
            succ_counter[g] += n
            succ_by_ngram[g].append(tid)
    for tid, txt in failure_texts:
        for g, n in _candidate_ngrams(txt).items():
            fail_counter[g] += n
            fail_by_ngram[g].append(tid)

    n_succ = max(1, len(success_texts))
    n_fail = max(1, len(failure_texts))

    # Lift score: rate-difference between success and failure populations
    lift_must_say: list[tuple[str, float]] = []
    lift_must_not_say: list[tuple[str, float]] = []
    all_ngrams = set(succ_counter.keys()) | set(fail_counter.keys())
    for g in all_ngrams:
        if succ_counter[g] < min_count and fail_counter[g] < min_count:
            continue
        succ_rate = succ_counter[g] / n_succ
        fail_rate = fail_counter[g] / n_fail
        diff = succ_rate - fail_rate
        if diff > 0:
            lift_must_say.append((g, diff))
        elif diff < 0:
            lift_must_not_say.append((g, -diff))

    lift_must_say.sort(key=lambda x: -x[1])
    lift_must_not_say.sort(key=lambda x: -x[1])

    must_say = [g for g, _ in lift_must_say[:top_n]]
    must_not_say = [g for g, _ in lift_must_not_say[:top_n]]

    citations_success: list[str] = []
    for g in must_say:
        citations_success.extend(succ_by_ngram.get(g, [])[:2])
    citations_failure: list[str] = []
    for g in must_not_say:
        citations_failure.extend(fail_by_ngram.get(g, [])[:2])
    # de-dup, preserving order
    citations_success = list(dict.fromkeys(citations_success))[:6]
    citations_failure = list(dict.fromkeys(citations_failure))[:6]
    return must_say, must_not_say, citations_success, citations_failure


# ---------- Routes ----------

@router.post("/mine", response_model=MineResponse)
async def mine(
    sop_ref: str = Query(..., description="The sop_ref to mine. Pass either 'seed:<file>' or a saved sop_id."),
    db: AsyncSession = Depends(get_session),
) -> MineResponse:
    started = time.perf_counter()
    res = await db.execute(
        select(PrecedentTrace).where(PrecedentTrace.sop_ref == sop_ref)
    )
    traces: list[PrecedentTrace] = list(res.scalars().all())

    if not traces:
        run = LearningRun(sop_ref=sop_ref, n_precedents=0, n_sessions=0, summary={}, proposals=[])
        db.add(run)
        await db.commit()
        await db.refresh(run)
        return MineResponse(
            run_id=run.id, sop_ref=sop_ref,
            n_precedents=0, n_sessions=0, lift_table=[], proposals=[], duration_ms=0,
        )

    # Aggregate session count
    sessions = {t.experiment_id for t in traces}

    # Lift table
    lift_rows = _build_lift_table(traces)

    # Phrase mining per (cohort, action). Only mine pairs with at least one finalized terminal trace.
    proposals: list[Proposal] = []
    by_ca: dict[tuple[str, str], list[PrecedentTrace]] = defaultdict(list)
    for t in traces:
        by_ca[(t.cohort or "unknown", t.action or "")].append(t)

    for (c, a), group in by_ca.items():
        succ = [(t.id, t.response_text or "") for t in group if t.terminal_outcome == "success"]
        fail = [(t.id, t.response_text or "") for t in group if t.terminal_outcome in ("failure", "abandoned")]
        if not succ and not fail:
            continue
        ms, mns, cs, cf = _mine_phrases_for_pair(succ, fail)
        if not (ms or mns):
            continue
        rationale = (
            f"Mined from {len(succ)} success and {len(fail)} negative outcomes "
            f"for (cohort={c}, action={a})."
        )
        proposals.append(Proposal(
            id=f"prop-{c}-{a}-{int(time.time()*1000)%1_000_000}",
            cohort=c, action=a,
            must_say_add=ms, must_not_say_add=mns,
            citations_success=cs, citations_failure=cf,
            rationale=rationale,
        ))

    duration_ms = int((time.perf_counter() - started) * 1000)
    run = LearningRun(
        sop_ref=sop_ref,
        duration_ms=duration_ms,
        n_precedents=len(traces),
        n_sessions=len(sessions),
        summary={"lift_table": [r.model_dump() for r in lift_rows]},
        proposals=[p.model_dump() for p in proposals],
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    return MineResponse(
        run_id=run.id,
        sop_ref=sop_ref,
        n_precedents=len(traces),
        n_sessions=len(sessions),
        lift_table=lift_rows,
        proposals=proposals,
        duration_ms=duration_ms,
    )


def _apply_proposals_to_task(task: TaskDefinition, run: LearningRun, chosen_ids: list[str]) -> tuple[TaskDefinition, set[str]]:
    """Mutates a TaskDefinition in-place to apply the chosen proposals. Returns (task, touched_action_names)."""
    proposals_by_id = {p["id"]: p for p in (run.proposals or [])}
    actions_by_name = {a.name: a for a in task.agent_actions}
    touched: set[str] = set()
    for pid in chosen_ids:
        p = proposals_by_id.get(pid)
        if not p:
            continue
        a = actions_by_name.get(p["action"])
        if a is None:
            continue
        existing_say = set(a.must_say or [])
        existing_nsay = set(a.must_not_say or [])
        a.must_say = list(existing_say | set(p.get("must_say_add") or []))
        a.must_not_say = list(existing_nsay | set(p.get("must_not_say_add") or []))
        touched.add(a.name)
    return task, touched


@router.post("/save-and-apply", response_model=SaveAndApplyResponse)
async def save_and_apply(req: SaveAndApplyRequest, db: AsyncSession = Depends(get_session)) -> SaveAndApplyResponse:
    """Load source (seed or saved), apply selected proposals, save as a NEW SOPRecord, return its id.

    Useful when the user mined on a seed file (read-only) and wants the learned advice
    materialized into a saved, editable copy.
    """
    from ..config import settings as _settings
    from pathlib import Path
    import json as _json

    run = await db.get(LearningRun, req.run_id)
    if not run:
        raise HTTPException(404, "learning run not found")

    chosen_ids = [pid for pid in req.proposal_ids if any(p["id"] == pid for p in (run.proposals or []))]
    if not chosen_ids:
        raise HTTPException(400, "no valid proposal_ids selected")

    # Load source TaskDefinition.
    if req.source_sop_ref.startswith("seed:"):
        p: Path = _settings.DATA_DIR / "sops" / req.source_sop_ref[len("seed:"):]
        if not p.exists():
            raise HTTPException(404, "source seed not found")
        task = TaskDefinition.model_validate(_json.loads(p.read_text()))
    else:
        rec = await db.get(SOPRecord, req.source_sop_ref)
        if not rec:
            raise HTTPException(404, "source sop not found")
        task = TaskDefinition.model_validate(rec.payload)

    task, touched = _apply_proposals_to_task(task, run, chosen_ids)
    # Keep the source name unless caller passed an override.
    if req.new_name:
        task.name = req.new_name

    new_rec = SOPRecord(name=task.name, description=task.description, payload=task.model_dump())
    db.add(new_rec)
    await db.flush()

    # Record acceptance on the source LearningRun.
    accepted = list(set(run.accepted_proposal_ids or []) | set(chosen_ids))
    run.accepted_proposal_ids = accepted
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(run, "accepted_proposal_ids")
    await db.commit()
    await db.refresh(new_rec)

    return SaveAndApplyResponse(
        sop_id=new_rec.id,
        sop_name=new_rec.name,
        accepted=chosen_ids,
        actions_updated=len(touched),
    )


@router.post("/apply", response_model=ApplyResponse)
async def apply(req: ApplyRequest, db: AsyncSession = Depends(get_session)) -> ApplyResponse:
    """Apply selected proposals from a learning run to a saved SOP. No-op for seed files
    (cannot mutate filesystem seeds). Logs acceptance into the LearningRun row."""
    run = await db.get(LearningRun, req.run_id)
    if not run:
        raise HTTPException(404, "learning run not found")
    rec = await db.get(SOPRecord, req.sop_id)
    if not rec:
        raise HTTPException(404, "sop not found (must be a saved SOP, not a seed)")

    proposals_by_id = {p["id"]: p for p in (run.proposals or [])}
    chosen_ids = [pid for pid in req.proposal_ids if pid in proposals_by_id]
    if not chosen_ids:
        return ApplyResponse(sop_id=req.sop_id, accepted=[], actions_updated=0)

    task = TaskDefinition.model_validate(rec.payload)
    task, touched = _apply_proposals_to_task(task, run, chosen_ids)

    rec.payload = task.model_dump()
    # Record acceptance on the LearningRun row.
    accepted = list(set(run.accepted_proposal_ids or []) | set(chosen_ids))
    run.accepted_proposal_ids = accepted

    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(rec, "payload")
    flag_modified(run, "accepted_proposal_ids")

    await db.commit()
    return ApplyResponse(sop_id=req.sop_id, accepted=chosen_ids, actions_updated=len(touched))


@router.get("/runs")
async def list_runs(
    sop_ref: Optional[str] = None,
    db: AsyncSession = Depends(get_session),
) -> list[dict]:
    q = select(LearningRun).order_by(LearningRun.started_at.desc()).limit(50)
    if sop_ref:
        q = q.where(LearningRun.sop_ref == sop_ref)
    rows = (await db.execute(q)).scalars().all()
    return [{
        "id": r.id, "sop_ref": r.sop_ref,
        "started_at": r.started_at.isoformat(),
        "duration_ms": r.duration_ms,
        "n_precedents": r.n_precedents,
        "n_sessions": r.n_sessions,
        "n_proposals": len(r.proposals or []),
        "accepted_proposal_ids": r.accepted_proposal_ids or [],
    } for r in rows]


@router.get("/runs/{run_id}")
async def get_run(run_id: str, db: AsyncSession = Depends(get_session)) -> dict:
    r = await db.get(LearningRun, run_id)
    if not r:
        raise HTTPException(404, "run not found")
    return {
        "id": r.id, "sop_ref": r.sop_ref,
        "started_at": r.started_at.isoformat(),
        "duration_ms": r.duration_ms,
        "n_precedents": r.n_precedents,
        "n_sessions": r.n_sessions,
        "summary": r.summary,
        "proposals": r.proposals,
        "accepted_proposal_ids": r.accepted_proposal_ids or [],
    }


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str, db: AsyncSession = Depends(get_session)) -> dict:
    await db.execute(delete(LearningRun).where(LearningRun.id == run_id))
    await db.commit()
    return {"ok": True}
