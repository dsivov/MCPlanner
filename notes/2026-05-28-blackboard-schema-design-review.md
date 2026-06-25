---
date: 2026-05-28
title: Blackboard schema v0 — design review
status: design review — all 5 decisions locked, ready for implementation
tags: [voice-agent, supervisor, blackboard, schema, contract, design, decision-record, hit-rate]
related: [blackboard-schema-v0, voice-agent-supervisor-kickoff, voice-agent-production-architecture]
---

# Blackboard schema v0 — design review

*Consolidated review artifact for the blackboard contract — deliverable 1 of*
*the voice-agent supervisor thread. Self-contained: a reviewer needs only*
*this document (plus the kickoff note for broader context). Captures the*
*design, the decisions made during the design walkthrough, the rationale for*
*each, and the one question that remains open.*

> **Canonical framing (supersedes parts of this report):** see
> `2026-05-31-supervisor-research-framing-and-confirmation-criteria.md`.
> That note locks the terminology used here ("voice agent" → **weak agent**)
> and identifies milestone **(B) instruction prefetch** — populating the
> blackboard's `instruction` slot — as the load-bearing missing piece, not
> the blackboard schema itself. The schema and decisions below are still the
> contract; what shifted is where the next research effort lands. See also
> `2026-05-31-query-aware-data-prefetch-Q6b.md` for an in-flight refinement
> to milestone (A) data prefetch that changes how `data_payloads` are keyed.

## Context

The voice-agent supervisor thread builds a **predict → queue →
consume-at-future-step** loop: a slow background planner predicts the next
1–3 turns, pre-stages an instruction (plus pre-fetched data) for each, and
writes them to a shared **blackboard**. A fast voice agent reads the
blackboard at consume time and uses whatever is there, falling back to its
own policy on a miss. The blackboard is the single seam between the two
halves — getting its contract right is what unblocks the rest of the thread.

Evaluation is **simulator-only**: no live voice agent. The "voice agent" is
the POC's text-mode planner reading from the blackboard; the user side is
the existing customer simulator. This is faithful for measuring prediction
quality, hit rate, latency-hidden, and tier distribution — not for prosody,
ASR latency, or real-user behaviour shape.

Full spec with field-level detail: `2026-05-24-blackboard-schema-v0.md`.
This report is the decision layer on top of it.

## The design at a glance

**Lookup model.** At each turn the voice agent classifies the user's message
into `(observed_cohort, observed_user_state)` (existing behaviour), then calls
`blackboard.lookup(session_id, turn_index, observed_cohort, observed_user_state)`
and gets back exactly one of: `HitExact`, `HitFallback`, `Miss`, `ExpiredMiss`.

**Entry shape.** Keyed by `(session_id, predicted_turn_index)`. Each entry
carries a predicted cohort, a ranked list of predicted user states, one or
more pre-staged **branches** (one per top-K predicted state), and a mandatory
**fallback**. Data payloads are stored as cache keys into the existing
`DataPrefetchManager`, never duplicated inline.

**What a hit does.** A `HitExact` lets the agent skip the planner pipeline
entirely for that turn — it passes the pre-staged instruction, must-say/avoid
lists, and resolved data payloads straight to the responder. That skip *is*
the latency saving the architecture promises.

**Storage + SLI.** In-memory dict is the source of truth for live reads; a
SQLite mirror (`blackboard_entries`) enables replay. Every lookup is logged
to `blackboard_lookups` — that table is the raw material for the hit-rate
SLI; a single SQL query yields the headline number with per-source
attribution.

## Decision log

### D1 — Match strictness for `HitExact` → **DECIDED: strict**

*Question:* should `HitExact` require exact cohort **and** state match, or
should a state match alone be enough even when cohort drifted?

*Decision:* strict. `HitExact` requires `observed_cohort == entry.cohort`
**and** `observed_user_state` matching a branch. On cohort drift the lookup
returns `HitFallback`, not a branch.

*Rationale:* cohort drift means the supervisor's whole frame was off, and a
branch written for the wrong audience risks an off-tone reply. It is easier
to loosen this in v1 (if data shows drift is common and low-risk) than to
walk back a hit-rate number a production team has already seen. Keeping it
strict also preserves the cohort/state-match gap as a *signal* about
cohort-classifier quality rather than masking it.

### D2 — Fallback presence → **DECIDED: always present, cheapest source**

*Question:* must every entry carry a fallback, and does that cost a tier-2
LLM call each time?

*Decision:* every entry always carries a fallback, but it is sourced from
the cheapest tier available — `tier_1` cached_playbook if one exists, else a
declared `built_in` default, else a fresh `tier_2` call as last resort.

*Rationale:* the original framing ("pay a tier-2 call per write") was a false
dichotomy — the fallback's cost depends on its source. Always-present keeps
the contract trivially simple for both producer and consumer; cheapest-source
keeps the average cost low and composes cleanly with the existing multi-tier
router. Pure-tier-2 was rejected as wasteful; confidence-conditional fallback
was rejected because it makes "no fallback" mean different things on
different turns and muddies the SLI.

### D3 — TTL default → **DECIDED: hardcoded 60s outer cap for v0**

*Question:* hardcode the 60s expiry cap, or make it SOP-tunable from day one?

*Decision:* hardcode 60s for v0. Note that data-payload freshness already
drives the effective TTL automatically via the `min(...)` in the expiry
formula; 60s is only the outer safety cap.

*Rationale:* we have zero measurements of how often the cap actually bites.
The SLI tracks `ExpiredMiss` per SOP — that count is the signal that tells us
when (and for which SOP) to tune. Graduating to SOP-tunable in v1 is a cheap,
non-breaking change once we have evidence. Encoding per-SOP guesses now would
enshrine intuition as config.

### D4 — Write durability → **DECIDED: awaited/durable writes (revised)**

*Question:* is an async best-effort SQLite mirror acceptable, given a crash
could lose recent entries from disk?

*Decision:* no — write to memory **and** await the SQLite write before
returning, for both `blackboard_entries` and `blackboard_lookups`. This
revises the original v0 proposal, which specified async best-effort.

*Rationale:* the async proposal was protecting a latency budget that does not
exist in the POC. The supervisor is off the critical path, and in the
simulator-only eval nothing is real-time, so awaiting a ~1ms SQLite write
costs nothing measurable against multi-second MCTS turns. Meanwhile the SLI
data *is* the deliverable of this thread, so it must survive a mid-run crash.
Durable writes also match how the app already persists `turns` and
`llm_calls`. The production hot-path concern (logging every lookup
synchronously would eat a real TTFB budget) is real but explicitly deferred
with RPC transport — flagged for the productionization step, out of scope
here.

### D5 — Hit-rate definition → **DECIDED: HitExact only**

*Question:* does the headline hit-rate number count only `HitExact`, or also
`HitFallback`? `HitFallback` means the supervisor produced an entry but the
agent's observed state didn't match any branch, so the agent used the safe
default. It *is* pre-staged, latency-hiding content — just not the
most-targeted one.

*Decision:* headline = `HitExact` only. `HitFallback`, `Miss`, and
`ExpiredMiss` are reported as separate categories alongside it.

*Rationale:* consistent with the strict stance in D1 — the headline measures
"the supervisor pre-staged the *right* thing," not merely "the supervisor
pre-staged *something*." Including `HitFallback` would inflate the number and
let a supervisor that always misses the branch but writes a generic fallback
score high, masking prediction-quality problems. The latency-hidden value of
`HitFallback` is not lost — it's still captured in the separate category and
in the prefetch-side `consumed` metrics; it just doesn't enter the headline
viability number (~70% bar).

## Resolved schema (decisions baked in)

```python
@dataclass
class BlackboardBranch:
    user_state: str
    action: str
    instruction: str
    data_payload_keys: dict[str, str]   # dep_name → DataFetch.cache_key
    source: str                         # mcts | empirical | both | cached_playbook
    confidence: float
    must_say: list[str] = field(default_factory=list)
    avoid:    list[str] = field(default_factory=list)

@dataclass
class BlackboardFallback:
    action: str
    instruction: str
    source: str                         # tier_1 | built_in | tier_2  (cheapest available — D2)

@dataclass
class BlackboardEntry:
    session_id:            str
    predicted_turn_index:  int
    cohort:                str
    predicted_user_states: list[tuple[str, float]]
    branches:              list[BlackboardBranch]
    fallback:              BlackboardFallback        # always present (D2)
    supervisor_decided_at: datetime
    expires_at:            datetime                  # decided_at + min(60s, payload freshness) (D3)
    supervisor_tier:       str
    version:               int = 1
    schema_version:        int = 1
```

Lookup algorithm (strict — D1):

```
1. Live entries for (session_id, turn_index)? none → Miss
2. now() > entry.expires_at?                        → ExpiredMiss
3. observed_cohort != entry.cohort?                 → HitFallback
4. branch with user_state == observed_user_state?   → HitExact(branch)
5. otherwise                                         → HitFallback
```

Persistence (D4): in-memory source of truth + awaited SQLite mirror for both
`blackboard_entries` and `blackboard_lookups`.

## Next steps (all decisions closed)

1. Alembic migration for `blackboard_entries` + `blackboard_lookups`
   (autogenerate; do not drop `planner.db`).
2. Implement `BlackboardManager` — in-memory dict + awaited SQLite mirror.
3. Wire `blackboard.lookup(...)` into the planner's turn-start path behind a
   `blackboard_enabled` flag (baseline pipeline unchanged when off).
4. Wire `blackboard.write(...)` into the pondering scheduler's emit path
   using existing trajectory predictions (no new prediction work).
5. Add a CLI/notebook that joins `blackboard_lookups` + `blackboard_entries`
   + `turns` + `data_fetches` and prints the per-experiment hit-rate
   breakdown — this is the SLI report.

Then run Experiment #1 from the kickoff note: baseline supervisor run on
`car_insurance_renewal`, 20-turn sessions × 10. The resulting hit-rate
distribution becomes the v0 baseline.

## Review checklist

All five decisions are locked (D1 strict, D2 cheapest-source fallback, D3
hardcoded 60s TTL, D4 durable writes, D5 HitExact-only headline). Items still
worth a second opinion before/while implementing:

- [ ] **D1 strict matching** — is forcing a fallback on cohort drift too
      conservative? It trades latency-hiding for tone safety. Revisit in v1
      if SLI shows cohort drift is common and low-risk.
- [ ] **D3 hardcoded TTL** — comfortable shipping without per-SOP tuning and
      letting `ExpiredMiss` counts drive v1 tuning?
- [ ] Anything in the resolved schema that wouldn't survive contact with the
      production RPC transport later.
- [ ] Reconcile the kickoff note + this report against the mood-diversity
      findings (`2026-05-23-stable-vs-transition-...`): the point-mass
      "blocking dependency" is lifted at offsets +2/+3, so the top-K branch
      source the blackboard depends on now exists.
