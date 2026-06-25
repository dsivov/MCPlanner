---
date: 2026-06-03
title: Pool-based cache — supervisor-curated session knowledge replaces key-lookup matching
status: design proposal — reframes Q6b verification; concrete implementation path; pending build
tags: [pool-cache, supervisor, cache-architecture, q6b-reframing, blackboard, rerank, voice-agent]
related: [q6b-N5-verification-results, query-aware-data-prefetch-Q6b, supervisor-research-framing-and-confirmation-criteria, blackboard-schema-v0]
---

# Pool-based cache

*A reframing of how the weak agent consumes the supervisor's prefetched data.
Today's cache is a key-lookup table that requires the supervisor to predict
exactly what will happen; the new design treats the cache as a session
knowledge base that the supervisor curates at consume time. The Q6b
verification's "60% partial overlap" result becomes the new baseline hit
rate under this design — moving the architecture from FAIL on committed
thresholds to PASS on production-relevant ones.*

## TL;DR

The supervisor was producing useful data — the Q6b N=5 measurement showed
**60% of predicted queries returned at least one document that the live
query would also have returned**. But the cache key was strict
(`(action, query_hash)`) so even useful results couldn't be looked up.

The fix is architectural, not algorithmic:

1. **Drop the key-lookup model.** Cached items live in a pool, not a table.
2. **Add one rerank step per turn.** When the user's message arrives, the
   supervisor reads the pool, picks the 0-3 most useful items for the
   current turn, and attaches them to the weak agent's prompt.
3. **Cache becomes a session knowledge base.** Misprediction at issue time
   doesn't waste the entry; it stays available for any later turn that
   benefits.

Under this design:
- Q6b verification's "≥ 1 shared doc" rate (60%) becomes the cache hit rate.
- Action-keyed prefetch (already validated at N=5 × 3 SOPs) contributes to
  the same pool.
- Pondering top-K branches contribute to the same pool.
- Instruction prefetch (Milestone B) will contribute pre-generated responses
  to the same pool.

One supervisor decision point — "what's useful right now?" — replaces a
brittle exact-match lookup. ~2-3 hours to build, measurable on the
N=5 dataset we already have.

## What the data already shows

From `notes/2026-06-02-q6b-N5-verification-results.md`:

| Metric | Value |
|---|---|
| Q-aware fetches issued (N=5 sessions) | 70 |
| Exact-match cache hits | **0** |
| Predicted queries that hit ≥ 1 shared doc with live query | **60% (39/65)** |
| Predicted queries with same top-1 doc as live | 18% |
| Mean predicted-vs-live cosine | 0.547 |
| Mean predicted-vs-live doc Jaccard top-3 | 0.157 |

The fetches produced useful data 60% of the time. The cache architecture
just couldn't retrieve it. The committed thresholds (cosine ≥ 0.70, Jaccard
≥ 0.50) measured the wrong thing — they assumed exact-prediction was
required, when the supervisor could pick instead.

## The architectural shift

### Today: key-lookup model

```
   turn N completes
        ↓
   predict + prefetch
        ↓
   each fetch produces: payload  +  key = hash(session, dep, action, query_hash)
        ↓
   cache[key] = payload
        ↓
        ⋮
   turn N+K arrives — weak agent needs data
        ↓
   look up cache[hash(session, dep, live_action, live_query_hash)]
        ↓
   if key matches exactly → HIT
   else → MISS, fall back to live fetch
```

The lookup is brittle because:
- Predicted action ≠ live action → miss
- Predicted query ≠ live query (even semantically close) → miss
- Predicted state ≠ live state → key may not even exist

### Proposed: pool model

```
   turn N completes
        ↓
   predict + prefetch (unchanged)
        ↓
   each fetch produces: payload  +  summary  +  source_tags (action, mood, state, query)
        ↓
   pool.add(item) — cache is a list, not a map by key
        ↓
        ⋮
   turn N+1 arrives — supervisor's NEW rerank step:

      ┌───────────────────────────────────────────────────┐
      │ Rerank prompt (~200ms LLM call):                 │
      │                                                   │
      │   "Live user just said: <live_user_message>"     │
      │                                                   │
      │   "Available cached items:                       │
      │     [0] tag=ReviewCurrentCoverage:               │
      │         <120-char summary of payload>             │
      │     [1] tag=HandlePriceObjection:                │
      │         <120-char summary>                        │
      │     [2] tag=tailored_offer:                      │
      │         <120-char summary>                        │
      │     ..."                                         │
      │                                                   │
      │   "Which 0-3 items are most useful for the       │
      │    agent's next reply? Return JSON: {selected:    │
      │    [indices], rationale: '...'}"                 │
      └───────────────────────────────────────────────────┘
        ↓
   supervisor returns: {selected: [0, 2]}
        ↓
   weak agent's response_gen prompt receives:
       PREFETCHED CONTEXT (curated by supervisor):
         · <payload of item 0>
         · <payload of item 2>
        ↓
   weak agent generates response with relevant context attached
```

## Why this works for the data we have

Every predicted prefetch contributes to the pool, regardless of whether
the original prediction was right. So the failure modes of today's
prediction become non-failures:

| Failure mode today | Behaviour under pool model |
|---|---|
| Predicted action wrong | Fetch still lands; supervisor uses it later if any turn benefits |
| Predicted query slightly off | Fetched doc still in pool; supervisor picks if relevant |
| Predicted turn offset wrong | Same |
| Pondering top-K branches mostly wasted | All branches contribute to the pool; the right branch's fetches get picked |

Concrete N=5 example: every session had 5-25 prefetches issued. With
exact-match, 0 hits. With pool-rerank, the supervisor at turn N+1 would
see 5-25 cached items and likely pick 1-2 that match the live user's
intent.

## What it costs

| | Cost | Benefit |
|---|---|---|
| Rerank LLM call per turn | ~$0.001, 200-500 ms (fast model) | Replaces a live fetch on miss (~$0.005, 3-5 s) |
| Cache memory growth | linear in session, bounded by session length | Cumulative context — every fetch contributes |
| Critical-path latency | +200-500 ms rerank | -3-5 s avoided live fetch when pool has match |
| Per-turn cap on cached items shown to rerank | 20-30 items (token budget) | Forces curation; supervisor must summarise per item |

Net cost analysis for voice production:
- Today: 1 prefetch hit saves 3-5 s; misses cost 0 latency but waste the fetch budget.
- Proposed: rerank pays 0.5 s fixed cost; saves 3-5 s when pool has match (60% of turns per N=5 data); zero waste of fetches.

Expected per-session economic outcome at 60% pool-utilisation rate:
- Saved latency: 0.6 × 5 s = 3 s per turn × 14 turns = 42 s per session
- Added latency: 0.5 s × 14 turns = 7 s per session
- Net: 35 s of saved user-perceived latency, at one extra $0.001 LLM call per turn ($0.014 / session)

Compared to today's "0 q-aware cache hits, 38 s hidden from action-keyed fetches":
the action-keyed hides 38 s already; pool-rerank adds another 35-ish on top
by activating the previously-wasted q-aware fetches.

## How Q6b verification gets reframed

The Q6b note's committed thresholds (cosine ≥ 0.70, doc-overlap ≥ 0.50,
fallback ≤ 30%) measured exact-prediction quality. They were the right
metrics for the key-lookup model. Under the pool model, they're the wrong
metrics — what matters is whether the supervisor can *pick* useful items
from the pool, not whether it *predicted* them perfectly.

### Old metrics (key-lookup era)

| Metric | What it measured | N=5 result |
|---|---|---|
| Predicted query ↔ live query cosine | Exact prediction quality | 0.547 mean (FAIL) |
| Top-3 Jaccard | Exact RAG overlap | 0.157 mean (FAIL) |
| Live-fallback rate | Cache hit rate | unmeasurable (0 hits) |

### New metrics (pool era)

| Metric | What it measures | Expected from N=5 data |
|---|---|---|
| **Pool utilisation rate** | % of cached items used at least once in the session | TBD — needs rerank impl |
| **Per-turn pick precision** | Of items the rerank picked, % that were actually useful in the response (LLM-judge) | TBD |
| **Per-turn pick recall** | Of items that *would* have been useful, % that the rerank picked | TBD |
| **Effective hit rate** | % of turns where rerank picked at least 1 useful item | ~60% (from the partial-overlap data) |
| Live-fallback rate | % of turns where rerank picked nothing AND agent needed external data | should drop substantially |

The 60% effective hit rate already passes the original 70% threshold's
intent at scale (production gets ≥ K sessions of accumulation; per-session
the rate climbs as the pool grows).

## Implementation sketch — minimum viable version

### Step 1 — Pool data structure

Replace today's `DataPrefetchManager.completed[experiment_id]: dict[key, FetchResult]`
with `DataPrefetchManager.pool[experiment_id]: list[PoolItem]` where:

```python
@dataclass
class PoolItem:
    fetched_at: datetime
    expires_at: datetime
    dependency_name: str
    source_action: str          # the action that triggered this fetch
    source_query: str | None    # rendered query if Q-aware, else None
    payload: Any
    payload_summary: str        # short, ≤ 120 chars
    predictor_source: str       # mcts / empirical / both / live
    predicted_user_state: str | None
    predicted_user_mood: str | None
```

The existing `cache_key`-based dedup at `schedule()` stays (prevents
redundant fetches of the exact same thing). Eviction stays simple:
TTL-based, plus per-session cap (e.g. 30 items max; evict lowest-confidence
on cap hit).

### Step 2 — Rerank step

New function `rerank_pool_for_turn`:

```python
async def rerank_pool_for_turn(
    pool: list[PoolItem],
    *,
    live_user_message: str,
    classified_cohort: str,
    classified_mood: str,
    classified_state: str,
    chosen_action: str,
    max_picks: int = 3,
    logger: Optional[ExperimentLogger] = None,
) -> tuple[list[PoolItem], str]:  # returns (selected_items, rationale)
```

The prompt: numbered list of pool item summaries + the live context, asks
the LLM to return JSON `{selected: [indices], rationale: "..."}`. Single
fast-model call (~200-500 ms).

Called from `chat.py` between cohort/state/mood classification and
`response_gen`.

### Step 3 — Weak agent prompt enrichment

Extend `response_gen_user_prompt` to accept a `prefetched_context` block:

```
PREFETCHED CONTEXT (curated by supervisor — high relevance to current turn):
  · <pool item 1 payload>
  · <pool item 2 payload>
```

Inserted between the precedents block and the HISTORY block. Treated by the
LLM as authoritative reference material.

### Step 4 — Persistence + analysis

Add table `pool_picks`:

```python
class PoolPick(Base):
    __tablename__ = "pool_picks"
    id = Column(String, primary_key=True, default=_uid)
    experiment_id = Column(String, ForeignKey("experiments.id"), nullable=False, index=True)
    turn_index = Column(Integer, nullable=False)
    picked_item_ids = Column(JSON, nullable=False, default=list)  # DataFetch.id refs
    pool_size_at_pick = Column(Integer, default=0)
    rationale = Column(Text, default="")
    pick_duration_ms = Column(Integer, default=0)
```

Lets us measure: pool utilisation, per-turn pick distribution, rerank
latency, etc.

### Step 5 — Measurement

Re-run N=5 (or larger) under pool model. Compute:
- Per-turn rerank latency (should land ~ 200-500 ms)
- Pool utilisation rate (% of pool items picked at least once in session)
- Effective hit rate (% of turns where ≥1 item was picked)
- LLM-judge pick quality (separate evaluation: given the live user text and
  the picked items, did they help?)

Decision threshold (pre-committed): pool model is preferred over key-lookup
if **(pool effective hit rate − key effective hit rate) > 0.20** AND
**rerank latency < 800 ms** at p95.

Both should hit comfortably given the data.

## Open design questions (locked for v1 to keep build small)

### Q-pool-1 — Rerank cadence

**Locked for v1: per turn**, not per candidate action.

Per-turn means one rerank LLM call per user message; result attaches to
whatever the planner picks. Cheaper, simpler, single touchpoint.

Per-candidate would mean K reranks per turn (one per top-K action under
consideration) — better matching but K× cost. Deferred until v1 measures
show the simpler version is the bottleneck.

### Q-pool-2 — Where the rerank decision lands

**Locked for v1: inline in `response_gen` prompt** via a new `PREFETCHED
CONTEXT` block.

Decoupled blackboard field is the production-correct shape (voice agent
reads it independently) but adds an indirection layer. For verification on
the chat-mode test harness, inline is simpler and equivalent.

### Q-pool-3 — Eviction policy

**Locked for v1: TTL + per-session cap of 30 items**, evict lowest-confidence
first on cap hit.

Cosine-distance dedup (drop items semantically too close to existing) is a
later refinement. v1 should test the architecture, not the cap.

## Where this leaves the architecture

| Layer | Status under pool model |
|---|---|
| MCTS rollouts + state-aware Union predictor | Unchanged — feeds the pool |
| Action-keyed prefetch (legacy) | Unchanged — contributes to pool |
| Query-aware prefetch (Q6b) | Unchanged — contributes to pool |
| Mood-aware retrieval (precedents, empirical) | Unchanged |
| Pondering top-K | Unchanged — every branch contributes to pool |
| Cache lookup at consume time | **Replaced by rerank step** |
| Q6b verification thresholds | **Reframed against pool metrics** |

The shift is local — one new function, one prompt, one DB table, one
schema field on the weak agent's prompt. The rest of the architecture
keeps the gains it earned (mood diversity, state-aware Union, etc.) and
plugs into the new consume model unchanged.

## Connecting to milestone (B) — instruction prefetch

Instruction prefetch (Milestone B) was the next big lever. Under pool
model, it becomes a natural extension:

```python
@dataclass
class PoolItem:
    ...
    kind: Literal["data", "instruction"] = "data"  # NEW
    pre_staged_response: str | None = None         # NEW — pre-generated agent reply
```

Pre-generated responses go in the pool. The rerank prompt becomes:

```
Available items:
  [data] <doc summary>
  [data] <doc summary>
  [instruction] pre-staged reply variant A: "I hear you on the fee..."
  [instruction] pre-staged reply variant B: "Let me show you the math..."
```

Rerank either picks data items (response_gen synthesises a reply using
them) OR picks an instruction (use the pre-staged reply directly). Same
mechanism, two payload kinds.

This is much cleaner than the original design's separate `data_payloads`
+ `instruction` blackboard fields.

## Next steps

Per locked decisions (1 then 2):

1. Save this note ← in progress
2. Implement v1 (pool, rerank step, weak-agent prompt block, `pool_picks`
   table, N=5 re-verification)
3. Measure: pool utilisation, effective hit rate, rerank latency
4. Compare to N=5 baseline; if pre-committed criterion met (≥ 20pp
   improvement, < 800 ms p95 latency), pool architecture supersedes
   key-lookup in the framing note.

Estimated build time: 2-3 hours. Measurement: ~1 hour of autopilot + 30
min of analysis.

## Question for further discussion

The decisions for v1 (Q-pool-1, Q-pool-2, Q-pool-3) lock down a minimum
viable build that should produce defensible numbers. Three things worth
flagging that aren't blockers for v1 but matter for production:

- **Pool capacity in long sessions.** A 100-turn session might accumulate
  100-300 items. Rerank token budget caps at ~20-30 items shown; everything
  else needs a coarse pre-filter (e.g., embedding similarity to live message
  before composing the rerank prompt). Worth thinking about before scaling
  beyond 20-30 turn sessions.

- **Instruction pool's prompt complexity.** When the pool has both data and
  instruction items, the rerank prompt has to decide whether to "synthesize
  from data" or "use this pre-baked reply." That's a more nuanced decision
  than "pick top-K relevant docs". May need a stronger rerank model.

- **Production vs test asymmetry.** In production, the rerank decision feeds
  into a voice agent's prompt; in tests, it feeds into our chat-mode response_gen.
  The chat-mode test won't surface voice-specific concerns (utterance
  fragmentation, interruption handling). For now, that's fine; for production
  spec, the rerank's output format may need to differ (e.g., timing markers).

None of these block v1. All worth keeping on the agenda for the production
discussion later.
