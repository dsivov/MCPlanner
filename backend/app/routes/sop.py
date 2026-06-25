from __future__ import annotations
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session, SOPRecord
from ..schemas import (
    TaskDefinition,
    BuildTurnRequest,
    BuildTurnResponse,
    SOPSaveRequest,
    SOPMeta,
)
from ..config import settings
from ..logger import ExperimentLogger
from ..llm.client import chat_json
from ..llm.prompts import SOP_BUILDER_SYSTEM, sop_builder_user_prompt

router = APIRouter(prefix="/api/sop", tags=["sop"])


def _to_named_item(x) -> dict | None:
    """Normalize an entry to {'name','description'}.
    The LLM frequently returns a bare string for agent_actions / user_states; treat that
    as a name with empty description rather than failing Pydantic validation silently.
    """
    if isinstance(x, str):
        name = x.strip()
        return {"name": name, "description": ""} if name else None
    if isinstance(x, dict):
        name = (x.get("name") or "").strip()
        if not name:
            return None
        return {"name": name, "description": x.get("description", "") or ""}
    return None


def _merge_named_items(current: list[dict], patch_list) -> list[dict]:
    """Merge by `name`. Existing items with the same name are updated in place; new names appended.
    Patch entries may be strings or dicts."""
    by_name: dict[str, dict] = {item["name"]: item for item in current if "name" in item}
    order: list[str] = [item["name"] for item in current if "name" in item]
    for raw in (patch_list or []):
        item = _to_named_item(raw)
        if not item:
            continue
        name = item["name"]
        if name in by_name:
            merged = {**by_name[name]}
            if item.get("description"):
                merged["description"] = item["description"]
            by_name[name] = merged
        else:
            by_name[name] = item
            order.append(name)
    return [by_name[n] for n in order]


def _merge_edges(current: list[dict], patch_list) -> list[dict]:
    """De-dup by (src, dst, direction). Existing edges are updated; new ones appended."""
    def key(e: dict) -> tuple[str, str, str]:
        return (e.get("src", ""), e.get("dst", ""), e.get("direction", "forward"))

    by_key: dict[tuple[str, str, str], dict] = {key(e): e for e in current}
    order: list[tuple[str, str, str]] = [key(e) for e in current]
    for e in (patch_list or []):
        if not isinstance(e, dict) or "src" not in e or "dst" not in e:
            continue
        direction = e.get("direction", "forward")
        if direction not in ("forward", "backward", "both"):
            direction = "forward"
        full = {
            "src": e["src"],
            "dst": e["dst"],
            "direction": direction,
            "note": e.get("note", ""),
        }
        k = key(full)
        if k in by_key:
            by_key[k] = {**by_key[k], **full}
        else:
            by_key[k] = full
            order.append(k)
    return [by_key[k] for k in order]


def _derive_sop_nodes(actions: list[dict], states: list[dict], edges: list[dict]) -> list[str]:
    """Union of action names, state names, and any node names referenced by edges."""
    nodes: dict[str, bool] = {}  # ordered set
    for a in actions:
        if a.get("name"):
            nodes[a["name"]] = True
    for s in states:
        if s.get("name"):
            nodes[s["name"]] = True
    for e in edges:
        for k in ("src", "dst"):
            if e.get(k):
                nodes[e[k]] = True
    return list(nodes.keys())


def _merge_patch(current: TaskDefinition, patch: dict) -> TaskDefinition:
    """Merge a partial dict over the current TaskDefinition.

    Semantics:
      - scalar fields (name, description): replaced
      - user_profile, conversation_profile (dicts): shallow merged
      - agent_actions, user_states: merged by name (string entries normalized to NamedItem)
      - sop.edges: de-duped by (src, dst, direction)
      - sop.nodes: re-derived from current actions + states + edges (LLM doesn't manage this)
    """
    base = current.model_dump()
    patch = patch or {}

    for k in ("name", "description"):
        if k in patch and patch[k]:
            base[k] = patch[k]

    for k in ("user_profile", "conversation_profile"):
        if k in patch and isinstance(patch[k], dict):
            base[k] = {**(base.get(k) or {}), **patch[k]}

    if "agent_actions" in patch:
        base["agent_actions"] = _merge_named_items(base.get("agent_actions") or [], patch["agent_actions"])
    if "user_states" in patch:
        base["user_states"] = _merge_named_items(base.get("user_states") or [], patch["user_states"])

    sop_patch = patch.get("sop")
    base_sop = base.get("sop") or {"nodes": [], "edges": []}
    if isinstance(sop_patch, dict):
        edges = _merge_edges(base_sop.get("edges") or [], sop_patch.get("edges") or [])
    else:
        edges = base_sop.get("edges") or []
    # nodes are always re-derived from the merged vocab + edges
    nodes = _derive_sop_nodes(base["agent_actions"], base["user_states"], edges)
    base["sop"] = {"nodes": nodes, "edges": edges}

    return TaskDefinition.model_validate(base)


@router.post("/build-turn", response_model=BuildTurnResponse)
async def build_turn(req: BuildTurnRequest, db: AsyncSession = Depends(get_session)) -> BuildTurnResponse:
    logger = ExperimentLogger(experiment_id=None)
    parsed, _ = await chat_json(
        model=settings.MODEL_BUILDER,
        system=SOP_BUILDER_SYSTEM,
        user=sop_builder_user_prompt(req.current_sop, req.history),
        temperature=0.4,
        max_tokens=1800,
        logger=logger,
        call_site="sop_builder",
    )
    await logger.flush(db, turn_id=None)
    await db.commit()
    message = parsed.get("assistant_message", "") or ""
    patch = parsed.get("sop_patch", {}) or {}
    is_complete = bool(parsed.get("is_complete", False))
    try:
        updated = _merge_patch(req.current_sop, patch)
    except Exception:
        updated = req.current_sop
    return BuildTurnResponse(assistant_message=message, updated_sop=updated, is_complete=is_complete)


@router.get("/seeds")
async def list_seeds() -> list[dict]:
    seeds_dir: Path = settings.DATA_DIR / "sops"
    out: list[dict] = []
    for p in sorted(seeds_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
            out.append({"file": p.name, "name": data.get("name", p.stem), "description": data.get("description", "")})
        except Exception:
            continue
    return out


@router.get("/seeds/{filename}")
async def get_seed(filename: str) -> TaskDefinition:
    p: Path = settings.DATA_DIR / "sops" / filename
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "seed not found")
    return TaskDefinition.model_validate(json.loads(p.read_text()))


@router.post("", response_model=SOPMeta)
async def save_sop(req: SOPSaveRequest, db: AsyncSession = Depends(get_session)) -> SOPMeta:
    rec = SOPRecord(name=req.sop.name, description=req.sop.description, payload=req.sop.model_dump())
    db.add(rec)
    await db.commit()
    await db.refresh(rec)
    return SOPMeta(id=rec.id, name=rec.name, description=rec.description or "", updated_at=rec.updated_at.isoformat())


@router.get("", response_model=list[SOPMeta])
async def list_sops(db: AsyncSession = Depends(get_session)) -> list[SOPMeta]:
    result = await db.execute(select(SOPRecord).order_by(SOPRecord.updated_at.desc()))
    rows = result.scalars().all()
    return [
        SOPMeta(id=r.id, name=r.name, description=r.description or "", updated_at=r.updated_at.isoformat())
        for r in rows
    ]


@router.get("/{sop_id}", response_model=TaskDefinition)
async def get_sop(sop_id: str, db: AsyncSession = Depends(get_session)) -> TaskDefinition:
    r = await db.get(SOPRecord, sop_id)
    if not r:
        raise HTTPException(404, "not found")
    return TaskDefinition.model_validate(r.payload)


@router.put("/{sop_id}", response_model=SOPMeta)
async def update_sop(sop_id: str, req: SOPSaveRequest, db: AsyncSession = Depends(get_session)) -> SOPMeta:
    rec = await db.get(SOPRecord, sop_id)
    if not rec:
        raise HTTPException(404, "not found")
    rec.name = req.sop.name
    rec.description = req.sop.description
    rec.payload = req.sop.model_dump()
    await db.commit()
    await db.refresh(rec)
    return SOPMeta(id=rec.id, name=rec.name, description=rec.description or "", updated_at=rec.updated_at.isoformat())


@router.delete("/{sop_id}")
async def delete_sop(sop_id: str, db: AsyncSession = Depends(get_session)) -> dict:
    await db.execute(delete(SOPRecord).where(SOPRecord.id == sop_id))
    await db.commit()
    return {"ok": True}
