"""Speculative data-prefetch pipeline.

After each agent turn, MCTS rollouts give us probability mass over future trajectories.
Each future action declares data dependencies. We:

  1. Aggregate (dependency, turn-offset) confidence over all rollouts.
  2. Schedule the top items into a per-session background queue (idempotent only).
  3. At each subsequent turn, before response_gen, consult the queue to serve a hit
     immediately or briefly await an in-flight fetch.

The point: hide external-I/O latency behind user think-time across MULTIPLE turns.
A 30-second fetch scheduled after turn 5, consumed at turn 7, is a two-turn pipeline
that the user never feels.

All fetches must be idempotent — mutations (booking, sending) are excluded by the
DataDependency.idempotent flag.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ..db import DataFetch, SessionLocal
from ..schemas import DataDependency, TaskDefinition


# ---------- Result + handle ----------


@dataclass
class FetchResult:
    """The materialized result of a fetcher call. Held in the per-session completed cache."""
    payload: Any
    payload_summary: str
    completed_at: datetime
    expires_at: datetime
    fetch_duration_ms: int
    started_at: datetime

    def expired(self, now: datetime | None = None) -> bool:
        now = now or datetime.utcnow()
        return now >= self.expires_at


@dataclass
class PoolItem:
    """One item in the per-session pool of prefetched material.

    The pool replaces the key-lookup consume model (see
    notes/2026-06-03-pool-based-cache-architecture.md). At consume time the
    supervisor's rerank step inspects every live PoolItem and picks 0-3 most
    useful for the next agent reply, instead of relying on exact `(action, query)`
    cache-key matches.
    """
    fetch_id: str                          # DataFetch.id ref for analysis
    fetched_at: datetime
    expires_at: datetime
    dependency_name: str
    source_action: str                     # action that triggered this prefetch
    payload: Any
    payload_summary: str                   # short, ≤ 200 chars for rerank prompt budget
    confidence: float                      # original schedule confidence
    predictor_source: str = "mcts"         # mcts / empirical / both / live
    source_query: str | None = None        # rendered query if Q-aware, else None
    predicted_user_state: str | None = None
    predicted_user_mood: str | None = None
    # Latency-fix v1: pre-cache the summary embedding at pool-insert time so the
    # rerank step doesn't pay for it on the critical path. Used to pre-filter the
    # pool to top-K most-similar to live user message before the LLM rerank call.
    summary_embedding: bytes = b""
    # Milestone B (instruction prefetch): "data" items go through the per-turn data
    # rerank as candidate context. "instruction" items hold a pre-generated agent
    # response text in `payload` — they bypass the data rerank and are looked up by
    # exact (source_action, predicted_user_state) match before the live response_gen
    # call. On hit, the live agent uses `payload` verbatim and skips response_gen.
    kind: str = "data"                     # "data" | "instruction"
    # Fix A telemetry: for instruction items, how many pool data items were baked into
    # the pre-generated text. Surfaced on hit as data-on-hit utilisation.
    instr_data_count: int = 0

    def expired(self, now: datetime | None = None) -> bool:
        now = now or datetime.utcnow()
        return now >= self.expires_at


@dataclass
class FetchHandle:
    """An in-flight fetch tracked in the per-session outstanding map."""
    key: str
    dependency_name: str
    action_name: str
    task: asyncio.Task                  # task whose result will be a FetchResult
    started_at: datetime
    confidence: float
    issued_at_turn: int
    predicted_turn: int                 # the turn we expected this to be consumed at
    kind: str
    ttl_s: int
    predictor_source: str = "mcts"      # which predictor scheduled this — surfaced in UI/CLI
    predicted_user_state: str | None = None  # state hint that conditioned this prediction
    rendered_query: str | None = None   # Q6b: the rendered RAG/KG query string, if any
    db_id: str = ""                     # persisted DataFetch.id (filled after insert)


# ---------- Fetcher interface ----------


class BaseFetcher:
    """Subclass per dependency kind. Each implementation must be idempotent / side-effect free."""

    async def fetch(
        self,
        dep: DataDependency,
        *,
        session_id: str,
        action_name: str,
        query: str | None = None,
    ) -> tuple[Any, str]:
        """Return (payload, short_summary). Subclasses override.

        `query` is the Q6b rendered query string (from `dep.query_template`). Fetchers
        for query-aware kinds (rag/kg) should use it as input to embedding/graph search.
        Static fetchers (mock/db) may ignore it."""
        raise NotImplementedError


class MockDataFetcher(BaseFetcher):
    """Configurable-latency stub. Sleeps for dep.expected_latency_ms then returns canned text.

    Config keys honoured:
        text:               canned payload string (default: '<mock {name}>').
        jitter_ms:          optional ± random jitter on the sleep time.
    """

    async def fetch(
        self, dep: DataDependency, *, session_id: str, action_name: str, query: str | None = None,
    ) -> tuple[Any, str]:
        import random
        cfg = dep.config or {}
        base = max(0, int(dep.expected_latency_ms or 0))
        jitter = int(cfg.get("jitter_ms", 0) or 0)
        sleep_ms = base + (random.randint(-jitter, jitter) if jitter else 0)
        await asyncio.sleep(max(0.0, sleep_ms / 1000.0))
        text = cfg.get("text") or f"<mock data for {dep.name} (session={session_id[:6]}, action={action_name})>"
        return (text, text[:140])


class RagFetcher(BaseFetcher):
    """Q6b: query-aware RAG fetcher. Embeds the rendered query and returns top-K docs
    from a per-SOP fixture corpus on disk. Falls back to canned text when no query is
    provided (back-compat with action-keyed fetches for older deps without a template).

    Config keys honoured (DataDependency.config):
        corpus:             path to the fixture corpus JSONL, relative to data/rag_corpus/
                            or absolute. Each line: {id, topic, tags, text}.
        top_k:              how many top documents to return (default 3).
        text:               fallback canned text when query is None.
    """

    # Per-corpus cache of (doc_id, doc_text, doc_embedding_bytes) tuples. Populated lazily
    # on first fetch per corpus; embedding all 25 docs once amortises across hundreds of
    # subsequent queries in a session. Memory is tiny (~150KB per corpus at 1536 dims).
    _corpus_cache: dict[str, list[tuple[str, str, bytes]]] = {}

    async def fetch(
        self, dep: DataDependency, *, session_id: str, action_name: str, query: str | None = None,
    ) -> tuple[Any, str]:
        cfg = dep.config or {}
        # Latency simulation: real embedding+search would take ~hundreds of ms, fixture is
        # ~ms. Sleep the declared budget so latency-hidden metrics are still meaningful.
        base = max(0, int(dep.expected_latency_ms or 0))
        await asyncio.sleep(max(0.0, base / 1000.0))

        if not query:
            # No query → fall back to canned text (back-compat for deps without template).
            text = cfg.get("text") or f"<rag fallback for {dep.name}>"
            return (text, text[:140])

        # Load + embed the corpus once per session lifetime, then re-use.
        corpus_path = cfg.get("corpus") or ""
        if not corpus_path:
            text = f"<rag: no corpus configured for {dep.name}>"
            return (text, text[:140])
        docs = await self._load_corpus(corpus_path)

        top_k = int(cfg.get("top_k") or 3)
        results = await self._semantic_search(query, docs, top_k=top_k)
        # Compose payload: top-K doc texts joined, plus a structured list for analysis.
        joined = "\n\n".join(f"[{doc_id}] {text}" for doc_id, text in results)
        summary = f"RAG over {dep.name}: {len(results)} docs for query '{query[:80]}...'"
        return ({"docs": results, "joined_text": joined, "query": query}, summary[:140])

    async def _load_corpus(self, corpus_path: str) -> list[tuple[str, str, bytes]]:
        if corpus_path in self._corpus_cache:
            return self._corpus_cache[corpus_path]
        import json, os
        # Resolve relative paths against data/rag_corpus/
        if not os.path.isabs(corpus_path):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            full = os.path.join(project_root, "data", "rag_corpus", corpus_path)
        else:
            full = corpus_path
        if not os.path.exists(full):
            return []
        docs: list[tuple[str, str, bytes]] = []
        from ..planner.precedents import embed_text
        for line in open(full):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            doc_id = d.get("id") or ""
            text = d.get("text") or ""
            if not (doc_id and text):
                continue
            try:
                emb = await embed_text(text)
            except Exception:
                emb = b""
            docs.append((doc_id, text, emb))
        self._corpus_cache[corpus_path] = docs
        return docs

    async def _semantic_search(
        self, query: str, docs: list[tuple[str, str, bytes]], *, top_k: int = 3,
    ) -> list[tuple[str, str]]:
        """Return list of (doc_id, doc_text) for top_k closest docs by cosine similarity."""
        if not docs:
            return []
        from ..planner.precedents import embed_text
        try:
            q_emb = await embed_text(query)
        except Exception:
            return [(d[0], d[1]) for d in docs[:top_k]]
        import struct
        # Convert byte embeddings back to float lists for cosine similarity.
        def _unpack(b: bytes) -> list[float]:
            return list(struct.unpack(f"<{len(b)//4}f", b)) if b else []
        q_v = _unpack(q_emb)
        if not q_v:
            return [(d[0], d[1]) for d in docs[:top_k]]
        q_norm = sum(x*x for x in q_v) ** 0.5 or 1.0
        scored: list[tuple[float, str, str]] = []
        for doc_id, text, emb in docs:
            v = _unpack(emb)
            if not v or len(v) != len(q_v):
                continue
            dot = sum(a*b for a, b in zip(q_v, v))
            n = (sum(x*x for x in v) ** 0.5) * q_norm
            sim = dot / n if n else 0.0
            scored.append((sim, doc_id, text))
        scored.sort(reverse=True)
        return [(doc_id, text) for _, doc_id, text in scored[:top_k]]


# Registry — RAG kind now uses the real fetcher; others remain mocks until needed.
_FETCHERS: dict[str, BaseFetcher] = {
    "mock": MockDataFetcher(),
    "rag": RagFetcher(),
    "kg": MockDataFetcher(),   # TODO: real KG fetcher
    "db": MockDataFetcher(),
    "api": MockDataFetcher(),
    "mcp": MockDataFetcher(),
}


def get_fetcher(kind: str) -> BaseFetcher:
    return _FETCHERS.get(kind) or _FETCHERS["mock"]


# ---------- Manager (per-process singleton) ----------


def cache_key(*, session_id: str, dep_name: str, action_name: str, extra: str = "") -> str:
    """Stable key. Same session + same (dep, action) + same extra args → same cache slot.

    For mock data we don't have real query args, so the key is deterministic in
    (session_id, dep_name, action_name). Real fetchers can pass `extra` to disambiguate
    (e.g., customer_id hash, or query embedding hash for Q6b)."""
    h = hashlib.sha1(f"{session_id}|{dep_name}|{action_name}|{extra}".encode("utf-8")).hexdigest()
    return h[:24]


# Q6b: query embedding hash for cache deduplication. We hash the rounded query embedding
# (rather than literal string) so near-duplicate paraphrases produce the same cache key.
# Decision Q-D. Falls back to literal hash on embed failure.
_QUERY_EMBED_CACHE: dict[str, str] = {}

def _query_embedding_hash(query: str) -> str:
    if not query:
        return ""
    cached = _QUERY_EMBED_CACHE.get(query)
    if cached is not None:
        return cached
    # Synchronous hash of literal-string SHA — embedding requires an async LLM call which
    # we can't do from this synchronous path. The literal-string hash is the cheap MVP
    # for Q-D; embedding-based dedup can be added later in an async pre-pass during
    # plan-build if cache hit rate becomes a bottleneck.
    h = hashlib.sha1(query.strip().lower().encode("utf-8")).hexdigest()[:12]
    _QUERY_EMBED_CACHE[query] = h
    return h


@dataclass
class PrefetchPlanItem:
    """One scheduled prefetch entry produced from a trajectory predictor."""
    dependency_name: str
    action_name: str
    confidence: float
    predicted_turn_offset: int
    predictor_source: str = "mcts"   # mcts / empirical / both / unknown — for post-hoc attribution
    # The user state the predictor expects at the offset turn (modal across rollouts when
    # available). Surfaced in UI / DB so we can audit state-conditional prefetch decisions.
    predicted_user_state: str | None = None
    # Full per-state distribution at the predicted offset {state: normalized share}. Empty
    # when no state info was available. Used by analysis tooling and (optionally) by
    # state-sensitive dependency branching. Not persisted to the DB.
    predicted_user_state_dist: dict[str, float] = field(default_factory=dict)
    # Q6b: rendered query string from the dep's query_template + prediction context.
    # When None, fetcher uses action-keyed canned behaviour (legacy). When set, fetcher
    # receives this as the query parameter and the cache key incorporates a hash of it.
    rendered_query: str | None = None


def derive_prefetch_plan(
    rollouts: list,                 # list of RolloutEntry (logger.rollouts)
    *,
    task: TaskDefinition,
    chosen_action_now: str,
    decay_lambda: float = 0.3,
) -> list[PrefetchPlanItem]:
    """Walk MCTS rollouts and produce a deduplicated, confidence-scored list of
    (dependency, predicted_turn_offset, action_at_that_offset) items.

    Only rollouts whose planned_actions[0] matches the just-chosen action count — the
    others explored alternate universes that are no longer reachable.
    """
    deps_by_action: dict[str, list[str]] = {a.name: list(a.data_dependencies or []) for a in task.agent_actions}
    if not any(deps_by_action.values()):
        return []
    scores: dict[tuple[str, int, str], float] = defaultdict(float)
    for r in rollouts:
        planned = getattr(r, "planned_actions", None) or []
        if not planned or planned[0] != chosen_action_now:
            continue
        reward = float(getattr(r, "reward", 0.0) or 0.0)
        # offset 1 = the agent's next turn (one turn after current)
        for offset, action_name in enumerate(planned[1:], start=1):
            deps = deps_by_action.get(action_name, [])
            if not deps:
                continue
            discount = math.exp(-decay_lambda * offset)
            score = reward * discount
            for dep_name in deps:
                key = (dep_name, offset, action_name)
                scores[key] += score
    plan: list[PrefetchPlanItem] = []
    for (dep_name, offset, action_name), score in scores.items():
        plan.append(PrefetchPlanItem(
            dependency_name=dep_name, action_name=action_name,
            confidence=round(score, 4), predicted_turn_offset=offset,
        ))
    plan.sort(key=lambda p: (-p.confidence, p.predicted_turn_offset))
    return plan


class DataPrefetchManager:
    """Per-process owner of all prefetch queues, keyed by experiment_id.

    Outstanding tasks live in `self.outstanding[experiment_id][key]`. Completed results
    (with TTLs) live in `self.completed[experiment_id][key]`. All access is async-safe
    via per-experiment locks.
    """

    # Pool-architecture v1 cap. Once a session has this many live items, evict
    # lowest-confidence on insert (preserves the high-confidence prefetches).
    POOL_MAX_PER_SESSION = 30

    def __init__(self, max_outstanding_per_session: int = 50) -> None:
        self.max_outstanding = max_outstanding_per_session
        self.outstanding: dict[str, dict[str, FetchHandle]] = defaultdict(dict)
        self.completed: dict[str, dict[str, FetchResult]] = defaultdict(dict)
        # Pool-architecture v1: per-session list of PoolItem the supervisor can rerank.
        # The completed map is still maintained for dedup at schedule time and for the
        # legacy key-lookup consume() path; the pool is the new consume substrate.
        self.pool: dict[str, list[PoolItem]] = defaultdict(list)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def _lock_for(self, experiment_id: str) -> asyncio.Lock:
        return self._locks[experiment_id]

    # ----- Scheduling -----

    async def schedule(
        self,
        *,
        experiment_id: str,
        sop_ref: str,
        task: TaskDefinition,
        plan: list[PrefetchPlanItem],
        current_turn_index: int,
        min_confidence: float = 0.05,
    ) -> list[FetchHandle]:
        """Schedule a batch of prefetches into the per-session queue, deduping by key
        and enforcing max-outstanding via LRU eviction of lowest-confidence entries.
        Returns the handles of newly-launched fetches (for diagnostics)."""
        if not plan:
            return []
        dep_by_name = {d.name: d for d in task.data_dependencies}
        launched: list[FetchHandle] = []
        async with self._lock_for(experiment_id):
            for item in plan:
                if item.confidence < min_confidence:
                    continue
                dep = dep_by_name.get(item.dependency_name)
                if dep is None or not dep.idempotent:
                    continue
                # Q6b: when the plan item carries a rendered_query, the cache key
                # incorporates a hash of the query so different predicted-question
                # variants of the same (dep, action) get separate cache slots.
                # Decision Q-D: hash the query embedding (rounded), not the literal
                # string, so near-duplicate paraphrases share a slot.
                query_hash = _query_embedding_hash(item.rendered_query) if item.rendered_query else ""
                key = cache_key(
                    session_id=experiment_id,
                    dep_name=item.dependency_name,
                    action_name=item.action_name,
                    extra=query_hash,
                )
                if key in self.outstanding[experiment_id]:
                    continue   # already in-flight
                cached = self.completed[experiment_id].get(key)
                if cached and not cached.expired():
                    continue   # already done, still valid
                # Cap enforcement
                if len(self.outstanding[experiment_id]) >= self.max_outstanding:
                    self._evict_one(experiment_id)
                handle = self._launch(
                    experiment_id=experiment_id,
                    sop_ref=sop_ref,
                    dep=dep,
                    key=key,
                    action_name=item.action_name,
                    confidence=item.confidence,
                    issued_at_turn=current_turn_index,
                    predicted_turn=current_turn_index + item.predicted_turn_offset,
                    predictor_source=item.predictor_source,
                    predicted_user_state=item.predicted_user_state,
                    rendered_query=item.rendered_query,
                    query_hash=query_hash,
                )
                launched.append(handle)
        return launched

    def _launch(
        self,
        *,
        experiment_id: str,
        sop_ref: str,
        dep: DataDependency,
        key: str,
        action_name: str,
        confidence: float,
        issued_at_turn: int,
        predicted_turn: int,
        predictor_source: str = "mcts",
        predicted_user_state: str | None = None,
        rendered_query: str | None = None,
        query_hash: str = "",
    ) -> FetchHandle:
        started_at = datetime.utcnow()
        task = asyncio.create_task(self._run_fetch(
            experiment_id=experiment_id, dep=dep, key=key,
            action_name=action_name, confidence=confidence,
            issued_at_turn=issued_at_turn, predicted_turn=predicted_turn, started_at=started_at,
            speculative=True, predictor_source=predictor_source,
            predicted_user_state=predicted_user_state,
            rendered_query=rendered_query,
            query_hash=query_hash,
        ))
        handle = FetchHandle(
            key=key, dependency_name=dep.name, action_name=action_name,
            task=task, started_at=started_at, confidence=confidence,
            issued_at_turn=issued_at_turn, predicted_turn=predicted_turn,
            kind=dep.kind, ttl_s=dep.cache_ttl_s,
            predictor_source=predictor_source,
            predicted_user_state=predicted_user_state,
            rendered_query=rendered_query,
        )
        self.outstanding[experiment_id][key] = handle
        return handle

    def _evict_one(self, experiment_id: str) -> None:
        """Drop the lowest-confidence outstanding handle. The task is left to complete
        (its result will simply be ignored on insertion since the handle is removed)."""
        items = self.outstanding[experiment_id]
        if not items:
            return
        victim_key = min(items.keys(), key=lambda k: items[k].confidence)
        items.pop(victim_key, None)

    async def _run_fetch(
        self,
        *,
        experiment_id: str,
        dep: DataDependency,
        key: str,
        action_name: str,
        confidence: float,
        issued_at_turn: int,
        predicted_turn: int,
        started_at: datetime,
        speculative: bool,
        predictor_source: str = "mcts",
        predicted_user_state: str | None = None,
        rendered_query: str | None = None,
        query_hash: str = "",
    ) -> FetchResult:
        """Owns one fetch lifecycle. Persists a DataFetch row when complete (or errors)."""
        fetcher = get_fetcher(dep.kind)
        t0 = time.perf_counter()
        payload: Any = None
        summary = ""
        err: Optional[str] = None
        try:
            payload, summary = await fetcher.fetch(
                dep, session_id=experiment_id, action_name=action_name,
                query=rendered_query,
            )
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        finished_at = datetime.utcnow()
        duration_ms = int((time.perf_counter() - t0) * 1000)
        result = FetchResult(
            payload=payload, payload_summary=summary or "",
            completed_at=finished_at, expires_at=finished_at + timedelta(seconds=dep.cache_ttl_s),
            fetch_duration_ms=duration_ms, started_at=started_at,
        )
        # Move from outstanding to completed (only if still tracked there; an eviction
        # may have removed the handle while we were running).
        if key in self.outstanding.get(experiment_id, {}):
            self.outstanding[experiment_id].pop(key, None)
            if err is None:
                self.completed[experiment_id][key] = result
        # Persist DataFetch row (fire-and-forget).
        fetch_id: str = ""
        try:
            async with SessionLocal() as db:
                row = DataFetch(
                    experiment_id=experiment_id,
                    cache_key=key,
                    dependency_name=dep.name,
                    action_name=action_name,
                    kind=dep.kind,
                    issued_at_turn=issued_at_turn,
                    predicted_turn=predicted_turn,
                    started_at=started_at,
                    completed_at=finished_at,
                    fetch_duration_ms=duration_ms,
                    confidence=confidence,
                    payload_summary=(summary or "")[:500],
                    consumed=False, wasted=False, evicted=False,
                    speculative=speculative,
                    fetch_error=err,
                    predictor_source=predictor_source,
                    predicted_user_state=predicted_user_state,
                    query_text=rendered_query,
                    query_hash=query_hash or None,
                )
                db.add(row)
                await db.commit()
                fetch_id = row.id
        except Exception:
            pass  # logging failures must never break a fetch
        # Pool insertion (v1): every successful fetch contributes to the session pool.
        # The rerank step at consume time picks 0-3 items per turn.
        if err is None and payload is not None:
            # Latency-fix v1: pre-embed the payload summary now so the rerank step can
            # cosine-prefilter the pool to top-K without paying for embeds on the
            # critical path. Fire-and-forget — pool still works if embed fails.
            summary_emb = b""
            short_summary = (summary or "")[:200]
            if short_summary:
                try:
                    from .precedents import embed_text
                    summary_emb = await embed_text(short_summary)
                except Exception:
                    summary_emb = b""
            self._pool_insert(
                experiment_id,
                PoolItem(
                    fetch_id=fetch_id,
                    fetched_at=finished_at,
                    expires_at=finished_at + timedelta(seconds=dep.cache_ttl_s),
                    dependency_name=dep.name,
                    source_action=action_name,
                    payload=payload,
                    payload_summary=short_summary,
                    confidence=confidence,
                    predictor_source=predictor_source,
                    source_query=rendered_query,
                    predicted_user_state=predicted_user_state,
                    predicted_user_mood=None,    # mood not yet plumbed to fetch level
                    summary_embedding=summary_emb,
                ),
            )
        if err is not None:
            raise RuntimeError(err)
        return result

    def _pool_insert(self, experiment_id: str, item: PoolItem) -> None:
        """Add item to pool. Drops expired items + evicts lowest-confidence on cap."""
        pool = self.pool[experiment_id]
        now = datetime.utcnow()
        pool[:] = [p for p in pool if not p.expired(now)]
        pool.append(item)
        if len(pool) > self.POOL_MAX_PER_SESSION:
            pool.sort(key=lambda p: p.confidence, reverse=True)
            del pool[self.POOL_MAX_PER_SESSION:]

    def get_pool(self, experiment_id: str) -> list[PoolItem]:
        """Return the live (non-expired) pool for a session, sorted by recency desc."""
        now = datetime.utcnow()
        live = [p for p in self.pool.get(experiment_id, []) if not p.expired(now)]
        live.sort(key=lambda p: p.fetched_at, reverse=True)
        return live

    # ----- Milestone B: Instruction prefetch -----

    def lookup_instruction(
        self,
        experiment_id: str,
        *,
        chosen_action: str,
        classified_state: str,
    ) -> Optional[PoolItem]:
        """Find a live kind="instruction" item matching the live turn's (action, state).

        Exact-match POC: source_action == chosen_action AND predicted_user_state ==
        classified_state. Returns the most-recent match (pool is sorted recency-desc),
        or None on miss. On a hit, the caller is expected to use the item's payload
        verbatim as the agent response and skip the live response_gen call.
        """
        if not chosen_action or not classified_state:
            return None
        for p in self.get_pool(experiment_id):
            if p.kind != "instruction":
                continue
            if p.source_action == chosen_action and p.predicted_user_state == classified_state:
                return p
        return None

    async def schedule_instruction_prefetch(
        self,
        *,
        experiment_id: str,
        task: "TaskDefinition",
        history: list[dict[str, str]],
        predicted_action: str,
        predicted_state: str,
        confidence: float = 0.5,
        ttl_s: int = 600,
    ) -> None:
        """Speculatively generate an agent response for a predicted (action, state) pair
        and insert it into the pool as kind="instruction". Fire-and-forget; the live turn
        consumer matches via lookup_instruction.

        The generated text is what the agent WOULD say if the predicted state holds. On a
        live-turn hit it is used verbatim, bypassing response_gen entirely (the speculative-
        context principle: the pool item's payload is the architecture's actual guarantee).
        """
        from .responder import generate_response
        from .precedents import embed_text
        from ..logger import ExperimentLogger
        from ..llm.scheduler import speculative_mode
        # Instruction pre-generation is background work — run its response_gen call on slack
        # only, under the speculative budget (PASTE-style; see llm/scheduler.py).
        speculative_mode.set(True)
        try:
            # Fix A (2026-06-07): make the pre-generated instruction aware of the pool's
            # prefetched data. Without this, an instruction-hit at the live turn skips
            # response_gen AND throws away the data the supervisor prefetched — the data
            # path is orphaned on hits. Here we attach the pool's data items relevant to
            # the predicted action (declared as its dependencies, or fetched under it) so
            # the instruction text already reflects that data. On hit, the agent's verbatim
            # response is data-informed; the data prefetch work is not wasted.
            action_obj = next((a for a in task.agent_actions if a.name == predicted_action), None)
            relevant_deps = set(action_obj.data_dependencies or []) if action_obj else set()
            data_context: list[str] = []
            for p in self.get_pool(experiment_id):
                if p.kind != "data":
                    continue
                if p.dependency_name in relevant_deps or p.source_action == predicted_action:
                    data_context.append(f"[{p.dependency_name} — {p.source_action}] {p.payload_summary}")
                if len(data_context) >= 3:
                    break

            logger = ExperimentLogger(experiment_id=experiment_id)
            text, _res = await generate_response(
                task, list(history), predicted_action,
                precedents=None,
                use_precedents=False,
                prefetched_context=data_context or None,
                logger=logger,
            )
            now = datetime.utcnow()
            short_summary = (text or "")[:200]
            try:
                summary_emb = await embed_text(short_summary) if short_summary else b""
            except Exception:
                summary_emb = b""
            # Synthetic fetch_id for analytics — we don't have a DataFetch row.
            fetch_id = f"instr-{experiment_id[:8]}-{predicted_action}-{predicted_state}-{int(now.timestamp())}"
            self._pool_insert(
                experiment_id,
                PoolItem(
                    fetch_id=fetch_id,
                    fetched_at=now,
                    expires_at=now + timedelta(seconds=ttl_s),
                    dependency_name=f"instruction:{predicted_action}",
                    source_action=predicted_action,
                    payload=text,
                    payload_summary=short_summary,
                    confidence=confidence,
                    predictor_source="empirical",
                    source_query=None,
                    predicted_user_state=predicted_state,
                    predicted_user_mood=None,
                    summary_embedding=summary_emb,
                    kind="instruction",
                    instr_data_count=len(data_context),
                ),
            )
        except Exception:
            # Best-effort. If speculation fails, the live response_gen fires as before.
            pass

    # ----- Consumption -----

    async def consume(
        self,
        *,
        experiment_id: str,
        sop_ref: str,
        task: TaskDefinition,
        action_name: str,
        current_turn_index: int,
        await_in_flight_ms: int = 2000,
        live_fallback: bool = True,
    ) -> tuple[dict[str, str], dict[str, int]]:
        """At consume time, look up all data deps required by `action_name`. For each:
          - completed in queue → use immediately
          - in-flight → await briefly; if still not done after timeout, optionally live-fetch
          - missing → live-fetch (counts toward live_count)
        Returns:
          (deps_payload, stats) where stats = {consumed, live, latency_hidden_ms, live_latency_ms}.
        """
        # Find the action and its declared deps
        action_obj = next((a for a in task.agent_actions if a.name == action_name), None)
        if action_obj is None or not action_obj.data_dependencies:
            return {}, {"consumed": 0, "live": 0, "latency_hidden_ms": 0, "live_latency_ms": 0}
        dep_by_name = {d.name: d for d in task.data_dependencies}

        consumed_count = 0
        live_count = 0
        latency_hidden_ms = 0
        live_latency_ms = 0
        payloads: dict[str, str] = {}

        for dep_name in action_obj.data_dependencies:
            dep = dep_by_name.get(dep_name)
            if dep is None:
                continue
            key = cache_key(session_id=experiment_id, dep_name=dep_name, action_name=action_name)
            now = datetime.utcnow()
            result: Optional[FetchResult] = None

            # 1) cached completed
            cached = self.completed.get(experiment_id, {}).get(key)
            if cached and not cached.expired(now):
                result = cached

            # 2) outstanding — await briefly
            if result is None and key in self.outstanding.get(experiment_id, {}):
                handle = self.outstanding[experiment_id][key]
                try:
                    awaited = await asyncio.wait_for(asyncio.shield(handle.task), timeout=await_in_flight_ms / 1000.0)
                    if isinstance(awaited, FetchResult):
                        result = awaited
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    result = None
                except Exception:
                    result = None

            if result is not None:
                # speculative hit ✓
                consumed_count += 1
                latency_hidden_ms += int(result.fetch_duration_ms or 0)
                payloads[dep_name] = result.payload_summary or str(result.payload)[:200]
                # Mark DataFetch row consumed
                await self._mark_consumed(experiment_id, key, current_turn_index)
                continue

            # 3) live fallback (blocking, counts as live latency)
            if live_fallback:
                t0 = time.perf_counter()
                try:
                    res = await self._run_fetch(
                        experiment_id=experiment_id, dep=dep, key=key,
                        action_name=action_name, confidence=0.0,
                        issued_at_turn=current_turn_index, predicted_turn=current_turn_index,
                        started_at=datetime.utcnow(), speculative=False,
                        predictor_source="live",
                    )
                    payloads[dep_name] = res.payload_summary or str(res.payload)[:200]
                except Exception:
                    pass
                live_count += 1
                live_latency_ms += int((time.perf_counter() - t0) * 1000)

        return payloads, {
            "consumed": consumed_count,
            "live": live_count,
            "latency_hidden_ms": latency_hidden_ms,
            "live_latency_ms": live_latency_ms,
        }

    async def _mark_consumed(self, experiment_id: str, cache_key_: str, turn_index: int) -> None:
        from sqlalchemy import update
        try:
            async with SessionLocal() as db:
                # Mark the most-recent un-consumed row with this key as consumed.
                await db.execute(
                    update(DataFetch)
                    .where(
                        DataFetch.experiment_id == experiment_id,
                        DataFetch.cache_key == cache_key_,
                        DataFetch.consumed.is_(False),
                        DataFetch.wasted.is_(False),
                    )
                    .values(consumed=True, consumed_at_turn=turn_index)
                )
                await db.commit()
        except Exception:
            pass

    # ----- Session lifecycle -----

    async def finalize_session(self, experiment_id: str) -> None:
        """On session end: mark all remaining outstanding/completed entries as wasted."""
        from sqlalchemy import update
        # Cancel AND await the outstanding fetch tasks before touching the DB. A bare
        # .cancel() returns before the task unwinds, so a fetch mid-write can hold its
        # SQLite transaction open and block the next session's chat-start (task #135).
        # Collect tasks under the lock, then gather them outside the lock.
        tasks = []
        async with self._lock_for(experiment_id):
            for handle in list(self.outstanding.get(experiment_id, {}).values()):
                if not handle.task.done():
                    handle.task.cancel()
                    tasks.append(handle.task)
            self.outstanding.pop(experiment_id, None)
            self.completed.pop(experiment_id, None)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        try:
            async with SessionLocal() as db:
                await db.execute(
                    update(DataFetch)
                    .where(
                        DataFetch.experiment_id == experiment_id,
                        DataFetch.consumed.is_(False),
                        DataFetch.wasted.is_(False),
                    )
                    .values(wasted=True)
                )
                await db.commit()
        except Exception:
            pass


# Process-wide singleton
manager = DataPrefetchManager()
