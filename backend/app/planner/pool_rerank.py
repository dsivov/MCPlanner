"""Pool rerank — supervisor's per-turn selection from the session knowledge pool.

The pool model (see notes/2026-06-03-pool-based-cache-architecture.md) replaces
the brittle (action, query_hash) cache key lookup with a per-turn selection that
picks 0-3 most-useful pool items for the current turn. The picks attach to the
weak agent's response_gen prompt as a PREFETCHED CONTEXT block.

v3 (current): embedding-only path, no LLM call on the critical path.
  - Embed live user message, cosine-rank pool by summary_embedding (pre-cached at
    pool-insert time), dedup near-duplicates, take top-`max_picks`.
  - ~50-150 ms total per turn vs ~1500-2200 ms for the previous LLM rerank.

The redesign came from a cosine-distribution analysis on the N=5 v2 run (see
notes/2026-06-04-session-comprehensive-summary.md §23+): 87% of the LLM's picks
were already in the embedding top-8, and the LLM's disagreements were mostly
about rank within top-N rather than about which items belonged in the shortlist.
The unique value the LLM was adding turned out to be dedup of near-duplicate
pool entries (e.g. two POLICY rows from different rollouts), which we now do
explicitly.

History: v1 was an input-token prefilter (top-K by cosine before LLM); v2 added
a confidence-gate fast path (skip LLM when K-th cosine + gap clear thresholds).
v2 fired 0% in N=5 because real pool cosines cluster tightly. See blog post
blog/2026-06-04-supervising-the-fast-mouth.html for the full story.
"""
from __future__ import annotations
import struct
import time
from typing import Optional

from ..llm.client import LLMResult
from ..logger import ExperimentLogger
from .data_prefetch import PoolItem
from .precedents import embed_text


# How many items to keep after cosine ranking, before dedup. 8 leaves headroom
# for dedup to discard a few and still find max_picks=3 distinct survivors.
MAX_PREFILTER_CANDIDATES = 8

# Near-duplicate detection. Two layers:
#   1) (dependency_name, payload_summary[:N]) key-equality — catches the common
#      case where the same fetch ran in multiple rollouts and produced identical
#      summaries (e.g. POLICY: #INS-882431 twice).
#   2) Pairwise summary-embedding cosine ≥ this threshold — catches near-dups
#      with slightly different summary phrasing (e.g. RAG queries that retrieved
#      overlapping doc sets).
DEDUP_PREFIX_LEN = 60
DEDUP_COSINE_THRESHOLD = 0.95


def _unpack_embedding(b: bytes) -> list[float]:
    if not b:
        return []
    return list(struct.unpack(f"<{len(b)//4}f", b))


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


async def _score_pool_by_cosine(
    pool: list[PoolItem],
    live_user_message: str,
) -> list[tuple[float, PoolItem, list[float]]]:
    """Returns [(cosine, item, item_embedding_unpacked), ...] sorted by cosine desc.

    Filters out `kind="instruction"` items: they're consumed via exact (action, state)
    match in chat.py BEFORE response_gen, not via per-turn cosine rerank. Data items
    are the only candidates that participate in the rerank.

    Items without a stored summary_embedding fall through with cosine=-1.0; they
    keep their original pool order relative to each other (recency desc — the
    natural order from get_pool). Embedding the live message costs one API call;
    pool items already have cached embeddings from pool-insert time.
    """
    pool = [p for p in pool if getattr(p, "kind", "data") == "data"]
    if not pool:
        return []
    if not live_user_message.strip():
        # No signal — return pool in original (recency-desc) order with sentinel cosine.
        return [(-1.0, p, _unpack_embedding(p.summary_embedding)) for p in pool]
    try:
        q_emb_bytes = await embed_text(live_user_message)
    except Exception:
        return [(-1.0, p, _unpack_embedding(p.summary_embedding)) for p in pool]
    q_v = _unpack_embedding(q_emb_bytes)
    if not q_v:
        return [(-1.0, p, _unpack_embedding(p.summary_embedding)) for p in pool]

    scored: list[tuple[float, PoolItem, list[float]]] = []
    unranked: list[tuple[float, PoolItem, list[float]]] = []
    for p in pool:
        v = _unpack_embedding(p.summary_embedding)
        if not v:
            unranked.append((-1.0, p, []))
            continue
        scored.append((_cosine(q_v, v), p, v))
    scored.sort(key=lambda x: -x[0])
    # Embedded items ranked by cosine first; un-embedded fall through last in pool order.
    return scored + unranked


def _dedup_take_top_k(
    scored: list[tuple[float, PoolItem, list[float]]],
    max_picks: int,
) -> list[tuple[float, PoolItem]]:
    """Walk scored items in cosine-desc order, skipping near-duplicates of items
    already picked. Returns up to max_picks (cosine, item) pairs."""
    picks: list[tuple[float, PoolItem]] = []
    pick_keys: set[tuple[str, str]] = set()
    pick_embs: list[list[float]] = []
    for cos, item, emb in scored:
        key = (item.dependency_name, (item.payload_summary or "")[:DEDUP_PREFIX_LEN].strip())
        if key in pick_keys:
            continue
        if emb and any(_cosine(emb, pv) >= DEDUP_COSINE_THRESHOLD for pv in pick_embs):
            continue
        picks.append((cos, item))
        pick_keys.add(key)
        if emb:
            pick_embs.append(emb)
        if len(picks) >= max_picks:
            break
    return picks


async def prefilter_pool(
    pool: list[PoolItem],
    live_user_message: str,
) -> tuple[list[PoolItem], list[float]]:
    """Kept for analysis/diagnostics (used by tests and ad-hoc measurement scripts).

    Returns top-`MAX_PREFILTER_CANDIDATES` pool items by cosine to live message.
    """
    scored = await _score_pool_by_cosine(pool, live_user_message)
    head = scored[:MAX_PREFILTER_CANDIDATES]
    return [item for _, item, _ in head], [cos for cos, _, _ in head]


async def rerank_pool_for_turn(
    pool: list[PoolItem],
    *,
    live_user_message: str,
    classified_cohort: str = "",   # noqa: ARG001 — kept for signature stability
    classified_mood: str = "",     # noqa: ARG001
    classified_state: str = "",    # noqa: ARG001
    chosen_action: str = "",       # noqa: ARG001
    max_picks: int = 3,
    logger: Optional[ExperimentLogger] = None,  # noqa: ARG001
) -> tuple[list[PoolItem], str, int, LLMResult | None]:
    """Pick 0-`max_picks` pool items most useful for the current turn (v3, no LLM).

    Returns (selected_items, rationale, duration_ms, llm_result=None). LLMResult is
    always None in v3 — kept in the return tuple so call sites don't need to change.
    """
    if not pool:
        return [], "", 0, None

    t0 = time.perf_counter()
    scored = await _score_pool_by_cosine(pool, live_user_message)
    # Cap the dedup search to the top candidates — going further down the cosine
    # tail rarely produces useful picks and risks letting low-signal items survive
    # dedup just because the higher-ranked candidates were all duplicates.
    picks_with_scores = _dedup_take_top_k(scored[:MAX_PREFILTER_CANDIDATES], max_picks)
    duration_ms = int((time.perf_counter() - t0) * 1000)

    if not picks_with_scores:
        return [], "", duration_ms, None

    picks = [item for _, item in picks_with_scores]
    cos_str = ", ".join(
        f"{c:.2f}" if c >= 0 else "n/a" for c, _ in picks_with_scores
    )
    rationale = f"cosine-rank+dedup: top-{len(picks)} cosines=[{cos_str}]"
    return picks, rationale, duration_ms, None
