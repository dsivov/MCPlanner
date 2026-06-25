---
date: 2026-06-03
title: Sync fallback + async supervisor — remove live MCTS from the critical path
status: design decision — locked direction; small implementation pending (~1 hour code + N=5 verification)
tags: [architecture, supervisor, pondering, router, tier-3, fallback, voice-agent, design-decision]
related: [pool-cache-N5-verification, pool-based-cache-architecture, supervisor-research-framing-and-confirmation-criteria, voice-agent-supervisor-kickoff]
---

# Sync fallback + async supervisor

*Locks the architectural direction for the supervisor thread. Triggered by*
*the latency analysis in `2026-06-03-pool-cache-N5-verification.md` which*
*showed live MCTS on the critical path is the architecture's last unfixed*
*latency sink (60% of per-turn cost). User-driven discussion on 2026-06-03*
*reached the decision: remove tier-3 (live MCTS) from the critical path*
*entirely; the fallback path becomes a fast LLM synthesis from the pool.*

## TL;DR

**Decision: live MCTS is removed from the critical path.** When a turn
arrives and the pool/router can't immediately produce an answer, the
supervisor falls back to a fast LLM synthesis call enriched with pool items
— *not* to live MCTS. MCTS continues to run in the background via
pondering, where it seeds the pool and accumulates precedents for the
empirical predictor.

This is the natural endpoint of the supervisor thread: every slow piece of
work now happens off the critical path. The user's perceived latency is
bounded by the fast-LLM cost of classify + rerank + synthesis + response_gen
≈ **3–5 s/turn** vs today's ~18 s/turn.

**Trade-off accepted:** on rare genuinely-branching turns where MCTS would
have picked a different action than baseline, we accept a slightly worse
choice in exchange for predictable real-time latency. The measurement that
validates the trade-off is success-rate parity at N=5.

## The two questions that landed the decision

After laying out the architectural pattern, two open questions decided the
shape:

1. **Do we want tier-3 (live MCTS on critical path) to still exist?**
   → **Remove it entirely.** Even reserved-for-explicit-quality cases add
   reasoning surface area and complicate the contract. If we want
   MCTS-quality, we get it through pondering's pre-computation, never on
   the critical path.

2. **Should the "fallback" supervisor also include a fast LLM
   'synthesise-from-pool' call beyond pool rerank?**
   → **Yes, include.** This is the bridge to milestone (B). The pool
   rerank picks 0-3 items; the synthesis step takes them + classified
   `(cohort, mood, state)` + recent history and produces an action choice
   *and* an instruction. The weak agent uses the instruction directly.

## What the architecture looks like now

### Critical path (target: ≤ 5 s)

```
   User message arrives
       │
       ▼
   Cohort/state/mood classify   (~1.5 s, 1 fast LLM call)
       │
       ▼
   Pool rerank                  (~1.5 s, 1 fast LLM call)
       │
       │  picks 0-3 items (data payloads + eventually pre-staged instructions)
       ▼
   Sync supervisor synthesis    (~1-2 s, 1 fast LLM call)
       │
       │  takes pool picks + classification → action + instruction
       │
       ▼
   Response generation          (~1-2 s, 1 fast LLM call — uses instruction)
       │
       ▼
   Agent reply
```

Total: 5-7 s per turn for novel turns. Tier-1 (cached_playbook) skips
synthesis entirely and lands in 1-2 s. **Live MCTS does not appear here.**

### Async lane (between turns)

```
   Turn N completes
       │
       ▼
   Pondering: K MCTS searches for top-K predicted next states
       │
       │  Each search → rollouts → data prefetches → pool items
       │  Each search → final action choice → precedent_trace insert
       │
       ▼
   Pool grows with diverse data items + (eventually) instructions
   Precedent_traces accumulate → empirical predictor + router both improve
```

Pondering's *consume* path (cache lookup at next turn) becomes irrelevant —
we never need MCTS on the critical path. What pondering keeps doing is:

1. **Filling the pool.** Each pondering's MCTS rollouts trigger data
   prefetches keyed by the predicted future actions. Pool items show up by
   the time the actual next turn arrives.
2. **Feeding empirical priors.** Each pondering's chosen action is
   persisted to `precedent_traces`, which the router and empirical
   predictor read from for confidence scoring next turn.

Pondering is no longer "background MCTS cache for the next decision." It's
"background pool curator." Cleaner role.

## What's already built vs what needs to change

| Layer | State |
|---|---|
| Cohort/state/mood classify | ✓ existing |
| Pool rerank | ✓ existing (verified N=5 at 96% hit rate) |
| Data prefetch from rollouts | ✓ existing |
| Pondering (MCTS in background) | ✓ existing (consume path will atrophy unused, fine) |
| Pool items as response_gen context | ✓ existing |
| Multi-tier router | ✓ existing but escalates to tier-3 on uncertainty |
| **Tier-3 disabled when explicitly configured** | **needs config flag + router branch** |
| Synthesis step distinct from response_gen | partially — pool rerank already does this; could be merged with response_gen or kept distinct |

The minimum-viable build is **one config flag + one router branch.** The
"sync supervisor synthesis" step the architecture calls for is already
implemented as pool rerank → response_gen. Today response_gen takes pool
items as `prefetched_context` and synthesizes a reply that incorporates
them. That's the synthesis step.

## The implementation in concrete code terms

```python
# schemas.py MCTSConfig
tier3_enabled: bool = True   # legacy default; set False to remove live MCTS
                             # from critical path (sync-fallback architecture).
```

```python
# router.py — when picking a tier
def pick_tier(...):
    if matches_tier1_conditions: return "cached_playbook"
    if matches_tier2_conditions: return "baseline"
    # was: return "mcts" (tier-3)
    # now:
    if cfg.tier3_enabled:
        return "mcts"
    return "baseline"   # never escalate to live MCTS — synthesize from pool instead
```

That's the whole change. Pondering continues to run in background regardless
of the flag (it doesn't fire on the critical path either way).

## The measurement that validates the trade-off

Pre-commit thresholds (before running N=5 with the new architecture):

| Criterion | Threshold | Why |
|---|---|---|
| **Mean per-turn agent_ms < 6 s** | hard | If we removed tier-3 and didn't get the latency, something else is dominating; investigate. |
| **Success rate within 5 pp of pool baseline (80%)** | hard | If we drop below 75%, MCTS was doing real action-selection work on novel turns and we should reserve tier-3 for high-stakes cases. |
| **Pool effective hit rate ≥ 90%** | soft | Pool should still be useful since pondering still fills it. |
| **Live data-fetch fallback rate ≤ 10%** | soft | If high, the new sync-synthesis is choosing actions that weren't predicted; minor. |

If the hard criteria both pass, the architecture is locked in for the
production design. If success rate drops >10 pp, we revert and reserve tier-3
for an explicit "high-stakes" caller. If latency stays high, something else
needs investigation before adopting.

## What this resolves from earlier thread state

- **The pool-cache N=5 verification's latency FAIL** (p95 2441 ms vs 800 ms
  target) becomes moot. The right reframe is: the rerank's 1.5-2.5 s cost
  was being measured against a 16 s baseline that's about to disappear. In
  the new architecture, total critical path is 5-7 s, and rerank is just
  one of four fast LLM calls — its latency budget is no longer
  threshold-bound.
- **The kickoff note's "queue model for pondering"** becomes lower-priority.
  Pondering's job under this design is "fill the pool," not "be ready with
  next-decision-cache before the user finishes typing." 1-turn lookahead is
  enough for the pool-fill purpose; multi-turn queue is a refinement, not a
  load-bearing piece.
- **Milestone (B) instruction prefetch** has a much cleaner shape now.
  Pre-staged instructions go in the pool alongside data items. The rerank
  step naturally picks one when it matches; the synthesis step uses it
  directly when picked. Same mechanism, new payload kind.

## Why this is the right architectural endpoint

The supervisor research thread set out to apply AsyncMLD to dialogue
planning: fast actor on the critical path, slow planner running async,
shared blackboard. **Today's tier-3 is the only place where the slow
planner still sits on the critical path.** Removing it completes the
pattern.

Equivalent in plain language: *the weak agent never waits for the
supervisor to think. It uses whatever the supervisor has already prepared
in the pool; if nothing's there, the agent falls back to its own fast
synthesis.* The supervisor catches up async. Over many turns the pool
thickens and fallbacks become rare.

This is the architecture the production note has been pointing to since
2026-05-23. The current discussion just made the last decision explicit.

## Concrete next steps

1. **Build** — Add `tier3_enabled` flag to `MCTSConfig`; add router branch
   that respects it. Maybe 30 minutes.
2. **N=5 verification** — Same harness, same SOP, `--tier3-disabled` flag.
   ~5 min wall-clock.
3. **Analysis** — Verdict against the four criteria above. Write
   `2026-06-DD-no-tier3-N5.md`.
4. **Cross-SOP** — Repeat on credit_card and medical if the
   car_insurance numbers pass.
5. **Update framing note** — Once verified, mark the supervisor thread's
   architectural direction as locked.
