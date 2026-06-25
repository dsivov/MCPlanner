---
date: 2026-05-24
title: Blackboard schema v0 — contract between supervisor and voice agent
status: design proposed v0 — pending review, no code yet
tags: [voice-agent, supervisor, blackboard, schema, contract, design, hit-rate]
related: [voice-agent-supervisor-kickoff, voice-agent-production-architecture, speculative-data-prefetch-pipeline]
---

# Blackboard schema v0 — contract between supervisor and voice agent

*First deliverable from the voice-agent supervisor kickoff thread*
*(`2026-05-24-voice-agent-supervisor-kickoff.md`). Concrete schema, read/write API,*
*and semantics for the shared store between the background supervisor and the*
*"voice agent" (which, per the evaluation methodology, is the POC's text-mode*
*planner reading from the blackboard rather than invoking MCTS on the critical*
*path). Each design decision is stated with the chosen position and the*
*alternative considered.*

## TL;DR

The blackboard is an in-process keyed store (POC) that the supervisor writes
pre-staged turn predictions to and the voice agent reads from at consume
time. Keyed by `(session_id, predicted_turn_index)`. Each entry carries one
or more **branches** (one per predicted user state, top-K hedge) plus a
mandatory **fallback**. The voice agent reads with its observed
`(cohort, user_state)` and gets back exactly one of: `HitExact`,
`HitFallback`, `Miss`, or `ExpiredMiss`. Every lookup is logged to a new
`blackboard_lookups` table — that table *is* the hit-rate SLI's raw material.

The schema is forward-compatible with production (RPC, distributed cache)
without semantic changes; only the storage backend and transport differ.

## Scope of this note

In scope:
- v0 schema: dataclasses + JSON shape for entries and lookup results.
- Read/write API surface.
- TTL, branch matching, fallback semantics.
- Storage model (in-memory + SQLite mirror for replay).
- SLI hooks at lookup time.
- Composition with existing `DataPrefetchManager` and planner.

Out of scope (deferred to v1+):
- Subscription / push notification on entry arrival.
- Partial-match branch ordering (cohort-only or state-only matching).
- Multi-tenant keying (tenant_id prefix).
- Payload-level TTL (separate from entry-level).
- RPC transport for production.

## The lookup model — what the voice agent does at consume time

At the start of every turn, the voice agent:

1. Runs the existing state classifier on the user's message to produce
   `(observed_cohort, observed_user_state)`. *(Already implemented; this is
   the existing per-turn cohort+state classification step.)*
2. Calls `blackboard.lookup(session_id, turn_index, observed_cohort,
   observed_user_state)`.
3. Gets back a `BlackboardLookupResult` (one of four kinds).
4. Acts on it:
   - `HitExact` → use the branch's instruction + data payloads directly.
   - `HitFallback` → use the entry's fallback (a safe tier-2 default).
   - `Miss` → fall back to the agent's own built-in policy (current
     baseline-mode behaviour).
   - `ExpiredMiss` → same as Miss for the agent, but recorded separately
     for SLI so we can tell stale-data misses apart from no-data misses.

The voice agent's existing state classifier is the *only* state inference
done at consume time. The supervisor does not re-classify state; it
predicts state ahead of time, and the agent's observed-state classification
is the ground truth that decides which branch (if any) matched.

**Why this lookup model:**
- It keeps the voice agent stateless w.r.t. the supervisor. The agent never
  reasons about what the supervisor knows; it just classifies and asks.
- It matches the existing autopilot harness exactly. No new components on
  the agent side; just one new call per turn.
- It makes branch matching deterministic: observed `(cohort, user_state)`
  is the key. No ambiguity about how the agent picks among branches.

**Alternative considered:** supervisor pre-classifies and stamps the entry
with the "expected" cohort+state, and the agent just reads the entry
without classifying. Rejected because (a) the supervisor's prediction is
exactly what we're trying to *measure* against ground truth, so the agent
must run its own classifier; (b) it would tightly couple the supervisor's
release cadence to the voice agent's turn cadence in a way that the queue
model is supposed to absorb.

## The schema

### Core types

```python
@dataclass
class BlackboardBranch:
    user_state: str                   # SOP user_state vocabulary; the branch's key
    action: str                       # SOP agent_action vocabulary
    instruction: str                  # NL prompt for the responder LLM
    data_payload_keys: dict[str, str] # dep_name → DataFetch.cache_key (deref via DataPrefetchManager)
    source: str                       # "mcts" | "empirical" | "both" | "cached_playbook"
    confidence: float                 # 0..1; supervisor's prediction confidence for this branch
    must_say: list[str] = field(default_factory=list)
    avoid:    list[str] = field(default_factory=list)

@dataclass
class BlackboardFallback:
    action: str
    instruction: str
    source: str                       # "tier_2" | "tier_1" | "built_in"

@dataclass
class BlackboardEntry:
    session_id:              str
    predicted_turn_index:    int
    cohort:                  str            # supervisor's best guess at the consume-time cohort
    predicted_user_states:   list[tuple[str, float]]  # ranked (state, probability), sums to 1
    branches:                list[BlackboardBranch]   # one per top-K predicted state, ordered by descending confidence
    fallback:                BlackboardFallback        # ALWAYS present
    supervisor_decided_at:   datetime
    expires_at:              datetime                  # entry-level TTL
    supervisor_tier:         str                       # "mcts" | "baseline" | "cached_playbook"
    version:                 int = 1                   # monotonic per (session_id, predicted_turn_index)
    schema_version:          int = 1                   # this schema version; bump on breaking changes
```

### Lookup result

```python
@dataclass
class BlackboardLookupResult:
    kind: Literal["HitExact", "HitFallback", "Miss", "ExpiredMiss"]
    entry:         BlackboardEntry | None = None    # populated for HitExact, HitFallback, ExpiredMiss
    branch:        BlackboardBranch | None = None   # populated for HitExact only
    fallback:      BlackboardFallback | None = None # populated for HitFallback
    matched_branch_index: int | None = None         # populated for HitExact
    latency_us:    int = 0                          # how long the lookup took
```

### JSON wire form (for inter-process / replay)

The dataclasses round-trip cleanly to JSON. `datetime` fields serialize as
ISO-8601 strings; everything else is plain JSON. `schema_version` is the
first thing a consumer checks — unknown values are treated as `Miss` and
logged for visibility.

## Read API

```python
def lookup(
    session_id: str,
    turn_index: int,
    observed_cohort: str | None = None,
    observed_user_state: str | None = None,
) -> BlackboardLookupResult: ...
```

Synchronous, in-process for the POC. Returns immediately. No blocking, no
timeout, no subscription.

**Resolution algorithm:**

```
1. Find all live entries for (session_id, turn_index). "Live" = highest version.
2. If none exist → return Miss.
3. Let E = the live entry.
4. If now() > E.expires_at → return ExpiredMiss(entry=E).
5. If observed_cohort != E.cohort:
       → return HitFallback(entry=E, fallback=E.fallback)
         (Cohort drift is a bigger miss than state mismatch; fallback is safer.)
6. For each branch B in E.branches:
       if B.user_state == observed_user_state:
           → return HitExact(entry=E, branch=B, matched_branch_index=B.index)
7. No branch matched observed state:
       → return HitFallback(entry=E, fallback=E.fallback)
```

**Why cohort-mismatch routes to fallback instead of best-effort branch
selection:** cohort drift means the supervisor's whole frame is off, not
just the next-state guess. Better to use the safe default than a branch
written for the wrong audience.

**Alternative considered:** soft matching — when no exact branch matches,
fall back to the branch whose `predicted_user_states` entry is closest by
probability. Rejected for v0 because (a) it complicates the SLI definition
of "hit" and (b) we want clean data on whether the top-K hedge is paying
off before we add fuzzy matching that would mask its failures.

## Write API

```python
def write(entry: BlackboardEntry) -> int:
    """Write a new entry; returns the assigned version.

    Versions are monotonic per (session_id, predicted_turn_index).
    The latest version wins on lookup; older versions are retained for audit.
    """
```

**Why versioned (vs last-writer-wins or append-only-no-versioning):**

- *Versioned* gives both a clean "current view" for fast lookup and a full
  audit history. Pondering may write multiple predictions for the same
  predicted turn as more upstream signal arrives (state classification at
  turn N may revise the prediction for N+2). We want to know which version
  the agent actually consumed and whether later revisions would have been
  better.
- *LWW* loses the revision history, which is exactly the signal we need to
  answer "does later-pondering improve hit rate."
- *Pure append-only* (no version field) requires consumers to do
  max-by-write-time at read time, which is fragile against clock skew.

Atomicity: writes are atomic in-memory; the SQLite mirror is best-effort
async. If the SQLite write fails, the entry is still live in memory.
Acceptable for the POC; production needs durable writes if the supervisor
crashes.

## TTL semantics

Each entry has its own `expires_at`. The default is computed at write time as:

```
expires_at = supervisor_decided_at + min(
    session_idle_timeout,             # default 60s
    min(payload.expected_freshness_s for payload in resolved_payloads),
)
```

i.e., the entry expires whenever the *shortest-lived* underlying signal
expires. Conservative: a stale balance is worse than no balance.

**Behavior on expired read:** the entry is returned as `ExpiredMiss` (not
silently dropped). The agent still falls back to its built-in policy, but
the SLI tracks `ExpiredMiss` separately from `Miss` so we can answer "are
TTLs too tight?" empirically.

**Default 60s** matches the production-note strawman. Worth tuning per SOP
once we have hit-rate data.

## Branch matching + fallback semantics

Already covered in the lookup algorithm above. Recapped:

| Observed cohort | Observed state matches a branch | Result |
|---|---|---|
| matches entry.cohort | yes | `HitExact` |
| matches entry.cohort | no | `HitFallback` |
| differs from entry.cohort | (irrelevant) | `HitFallback` |

The fallback is **always populated** — it's the supervisor's contractual
guarantee. The supervisor writes the fallback even when tier-3 produced a
high-confidence top-K hedge, because the agent might still observe an
unexpected `(cohort, user_state)` and the fallback is the safety net.

If the supervisor failed to produce *any* entry for the turn (cold start,
queue saturation, prediction failed), there is simply no entry → `Miss`.
That is distinct from "entry existed but had no matching branch"
(`HitFallback`). Both result in the agent using its own policy; the SLI
separation is what tells us whether the architecture is failing to predict
at all (Miss) vs predicting badly (HitFallback).

## Storage model

**In-memory**: dict keyed by `(session_id, predicted_turn_index)` → `list[BlackboardEntry]`,
ordered by version. Latest version is `list[-1]`. Process-local.

**SQLite mirror**: new `blackboard_entries` table:

```
id                       INTEGER PK
experiment_id            INTEGER FK → experiments.id
session_id               TEXT     -- same as experiments.session_id, denormalized for query speed
predicted_turn_index     INTEGER
version                  INTEGER
schema_version           INTEGER
cohort                   TEXT
supervisor_tier          TEXT
supervisor_decided_at    TIMESTAMP
expires_at               TIMESTAMP
entry_json               TEXT     -- full BlackboardEntry as JSON
UNIQUE (session_id, predicted_turn_index, version)
INDEX  (experiment_id, predicted_turn_index)
```

Mirror writes are **awaited/durable** (revised — see decision D4 in
`2026-05-28-blackboard-schema-design-review.md`; the original v0 proposal of
async best-effort was dropped). The in-memory store is the source of truth
for live reads; the awaited SQLite copy guarantees SLI data survives a
mid-run crash and enables replay, post-hoc SLI queries, and joining against
`turns` / `data_fetches`. Cost is invisible: the supervisor is off the
critical path and the eval is simulator-only, so a ~1ms write against
multi-second MCTS turns costs nothing measurable.

**Lifecycle.** On session end, all in-memory entries for the session are
flushed to SQLite (if not already mirrored) and dropped from memory.

**Migration.** Per project convention (see [[feedback-schema-migrations]]),
use Alembic `revision --autogenerate` for the new tables; do not drop
`planner.db`.

## SLI hooks

Every `lookup()` call writes one row to a new `blackboard_lookups` table:

```
id                       INTEGER PK
experiment_id            INTEGER FK → experiments.id
session_id               TEXT
turn_index               INTEGER
lookup_at                TIMESTAMP
observed_cohort          TEXT
observed_user_state      TEXT
result_kind              TEXT     -- 'HitExact' | 'HitFallback' | 'Miss' | 'ExpiredMiss'
entry_version            INTEGER  -- NULL on Miss
matched_branch_index     INTEGER  -- NULL except HitExact
matched_branch_source    TEXT     -- 'mcts' | 'empirical' | 'both' | 'cached_playbook' | NULL
matched_branch_confidence REAL
supervisor_tier          TEXT     -- NULL on Miss
fallback_source          TEXT     -- 'tier_2' | 'tier_1' | 'built_in' | NULL
latency_us               INTEGER
INDEX (experiment_id, turn_index)
INDEX (result_kind)
```

The single SQL that gives us the headline hit rate:

```sql
SELECT
  result_kind,
  COUNT(*) AS n,
  AVG(latency_us) AS avg_lookup_us
FROM blackboard_lookups
WHERE experiment_id = ?
GROUP BY result_kind;
```

Per-source attribution (which planner tier produced the hit) joins
`blackboard_lookups` to `blackboard_entries` via
`(session_id, turn_index, entry_version)`.

**Why log lookups, not just entries.** A `Miss` is a lookup with no entry
— there's nothing on the write side to log. The lookup table is the only
place that sees the full picture.

## Composition with existing systems

### Prefetch (`DataPrefetchManager`)

Each `BlackboardBranch.data_payload_keys` is a dict of `dep_name → cache_key`,
pointing at `DataFetch` rows owned by the prefetch manager. The voice agent
resolves these by calling the existing `DataPrefetchManager.consume(...)`
flow. On the consume path:

- Cache hit + valid TTL → instant return.
- In-flight + small wait → wait up to `data_prefetch_await_in_flight_ms`.
- Cache miss → live fetch (paying latency).

This preserves the existing prefetch metrics (consumed / wasted /
speculative). Blackboard hits where data payloads landed correctly will
show up as `consumed=true` on the prefetch side.

**Important:** the blackboard does NOT duplicate payload data. It carries
cache keys only. This keeps blackboard entries small and avoids two-tier
cache coherency problems.

### Planner

When the multi-tier router or the pondering scheduler produces a
prediction, it writes a `BlackboardEntry` via the new write API. The
existing rollout machinery and trajectory predictors are unchanged.

The "voice agent" in the POC is the existing planner+responder, but
configured to:
1. Call `blackboard.lookup(...)` at the top of each turn.
2. If `HitExact`, skip MCTS entirely and pass the branch's
   `instruction`, `must_say`, `avoid`, and resolved `data_payloads`
   straight to the responder.
3. If `HitFallback`, pass the fallback through the responder with
   minimal additional reasoning (effectively tier-2 routing for this turn).
4. If `Miss` or `ExpiredMiss`, run the existing planner pipeline
   (router → predictor → response_gen) as today.

This means **blackboard hits replace the planner pipeline** for those
turns — which is the whole point: that's the latency saving the production
architecture promises.

### Experiment logging

`PlannerTrace` gets one new field: `blackboard_lookup_result_kind: str`
(one of the four kinds). All other detail lives in `blackboard_lookups`.

## What's deferred to v1+

| Item | Why deferred |
|---|---|
| Payload-level TTL | Entry-level TTL is conservative; revisit once we see stale-data misses in SLI data. |
| Partial-match branches | Need v0 data to know whether soft matching would help. |
| Subscription / push | POC reads are synchronous in-process; subscription only matters for the RPC transport. |
| Multi-tenant keying | POC is single-tenant. |
| Cross-session priors in entries | That's the empirical predictor's job; the blackboard should stay session-scoped. |
| RPC transport | Out of scope for in-POC eval; the dataclass + JSON wire form is forward-compatible. |
| Confidence-weighted fallback selection | The fallback is currently always tier-2; could be tier-1 if a cached_playbook exists. v1. |

## Open questions to pin down before implementing

These are real design judgment calls the kickoff note flagged. The
positions above are *proposals* — push back if any of them are wrong.

1. **Lookup match strictness.** Should `HitExact` require exact cohort
   *and* state match (as proposed), or should we also count cohort-match +
   state-in-top-K-predicted as a hit? The latter is more permissive and
   probably what production wants; v0 strict makes the metric cleaner.
2. **Fallback always present.** Proposed: yes. Cost is one extra tier-2
   call per supervisor write. Alternative: omit fallback when tier-3
   confidence is very high. Saves cost but eliminates the safety net.
3. **TTL default of 60s.** Borrowed from the production-note strawman;
   should it be SOP-tunable from day one or hardcoded for v0?
4. **Async SQLite mirror.** Best-effort means a crash could lose recent
   entries. For research replay this matters; for live reads it doesn't.
   Acceptable for POC?
5. **Hit-rate definition includes `HitFallback`?** Proposed: no — only
   `HitExact` counts as a hit. `HitFallback` is "the supervisor predicted
   but missed the branch." But you could argue the agent *did* get useful
   pre-staged content, even if not the most-targeted one. Affects the
   headline number significantly.

## Next steps once the contract is approved

1. Land the Alembic migration for `blackboard_entries` + `blackboard_lookups`.
2. Implement `BlackboardManager` (in-memory dict + async SQLite mirror).
3. Wire `blackboard.lookup(...)` into the planner's turn-start path. Behind
   a config flag (`blackboard_enabled`) so the baseline pipeline still runs
   unchanged.
4. Wire `blackboard.write(...)` into the pondering scheduler's emit path
   (using existing trajectory predictions; no new prediction work).
5. Add a notebook / CLI that joins `blackboard_lookups` + `blackboard_entries`
   + `turns` + `data_fetches` and prints the per-experiment hit-rate
   breakdown — this is the SLI report.

Then we can run Experiment #1 from the kickoff note's sequence: baseline
supervisor run on `car_insurance_renewal`, 20-turn sessions × 10. Hit-rate
distribution becomes the v0 baseline.
