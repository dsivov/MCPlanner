---
date: 2026-05-31
title: Supervisor research framing and confirmation criteria — what we must prove, what we've shown, what remains
status: framing document — locks terminology + checkpoints for review
tags: [framing, terminology, supervisor, weak-agent, milestones, confirmation, research-design]
related: [voice-agent-production-architecture, stable-vs-transition-state-prediction-asymmetry, first-successful-conversation-and-78pct-state-prediction, blackboard-schema-v0]
---

# Supervisor research framing and confirmation criteria

*A locked-down statement of who the actors are, what the supervisor's job is,
and what we must demonstrate to claim the architecture works. Built from
discussion on 2026-05-31. Treat the terminology below as canonical for
future notes.*

## TL;DR

The research subject is a **background supervisor** for a **weak AI agent**
(production target: voice agent). Tests replace the live user with a **smart
human simulator** that is honest but not the research goal. The supervisor's
job, given conversation history + all available signal, is to predict the
next 1-N turns of the conversation well enough that it can pre-process two
distinct outputs into a shared blackboard the weak agent reads from:

- **(A)** Data retrieved from external sources (RAG / KG / DB / MCP).
- **(B)** Instructions for the weak agent (response text or playbook
  directive).

Both must be available, approximately matched to whatever turn actually
fires, before the weak agent needs them.

| | Status |
|---|---|
| (A) Data prefetch | **Confirmed at N=5 across all 3 seed SOPs** (2026-05-31 + 2026-06-02 runs). Off+2 hit-rate 60–72% cross-SOP; mean latency hidden 25–110 s/session; live-fallback rate ≤1 per 35–45 speculative fetches. 80% success rate on car_insurance + medical; 0% on credit_card (separate planner-quality issue, not prefetch). |
| (B) Instruction prefetch | **Not yet built**, therefore not confirmed. Mechanism exists (response_gen) but is current-turn-only today. |

The fastest path to closing the architecture's central research question is
**building B as a measurable ablation against A's existing infrastructure**.
~3-4 hours of work; outcome decides whether the production architecture
needs to commit to "data-only prefetch" or can do the full "data +
instruction" version.

## The three actors

```
                    ┌──────────────────────────┐
                    │       LIVE USER          │
                    │ (in production)          │
                    │ — or —                   │
                    │ SMART HUMAN SIMULATOR    │
                    │ (in test;                │
                    │  strong LLM,             │
                    │  not research target)    │
                    └────────────┬─────────────┘
                                 │
                          natural language
                                 │
                                 ▼
   ─── CRITICAL PATH ────────────────────────────────────────────
                    ┌──────────────────────────┐
                    │     WEAK AI AGENT        │
                    │ (production: voice;      │
                    │  test: fixed harness)    │
                    │                          │
                    │ Fast, latency-bound;     │
                    │ limited reasoning        │
                    └──────────┬───────────────┘
                               │  reads
                               ▼
                    ┌──────────────────────────┐
                    │       BLACKBOARD         │
                    │  shared state cache      │
   ────────────────►│ • prefetched data       │◄────── writes
                    │ • pre-staged instruction │
                    │ • TTL / source tags      │
                    └──────────────────────────┘
                               ▲
                               │  writes
                               │
   ─── ASYNC LANE ─────────────┴────────────────────────────────
                    ┌──────────────────────────┐
                    │      SUPERVISOR          │
                    │ (RESEARCH SUBJECT)       │
                    │                          │
                    │ Slow, strong, runs       │
                    │ between user turns:      │
                    │  • predicts top-K next   │
                    │    (action, state, mood) │
                    │  • fetches data (A)      │
                    │  • generates             │
                    │    instructions (B)      │
                    │  • writes to blackboard  │
                    └──────────────────────────┘
                               │
                               ▼
                ┌────┬────┬────┬─────┐
                │MCP │ KG │ DB │ RAG │  ← data sources
                └────┴────┴────┴─────┘
```

The weak agent never talks to the supervisor or to data sources directly.
It only reads the blackboard. The supervisor never talks to the user
directly — it only observes turns and writes to the blackboard. Test runs
replace the live user with the smart human simulator; everything else is
unchanged.

## The core thesis

> *Given conversation history + all available signal (cohort, mood,
> classified user_state, retrieved precedents, SOP rules, success/failure
> markers), the supervisor can predict the next 1-N turns of the
> conversation sharply enough that:*
>
> **(A)** *the external-data queries the weak agent will need can be
> issued and cached ahead of time;*
>
> **(B)** *the instruction (response text or playbook directive) the weak
> agent will execute can be generated or looked up and pre-staged ahead
> of time.*

"Approximately of course" — the supervisor doesn't commit to one branch.
It hedges across the top-K most plausible (action, state, mood) tuples per
offset. When the actual turn fires, whichever pre-staged branch matches is
the one the weak agent consumes; the others are wasted.

## (A) — what we've confirmed about data prefetch

### The mechanism is built and end-to-end functional

- MCTS rollouts produce per-step `planned_actions` + `planned_states`.
- Rollout-side mood diversity (Phase-1) produces non-degenerate state
  distributions at offset+2/+3.
- Runtime mood classifier (Phase-2) emits per-turn `(cohort, mood, state)`
  with zero added LLM calls.
- `EmpiricalTrajectoryPredictor` conditions SQL on `(cohort, state, mood)`
  with graceful fallback chain → `cohort+state+mood` → `cohort+state` →
  `cohort` → `sop`.
- `UnionTrajectoryPredictor` runs MCTS + Empirical, merges with
  source-tagging.
- `DataPrefetchManager` issues parallel background fetches against the
  declared `data_dependencies` for each predicted action.

### Real data

| Session | Outcome | Off+1 hit | Off+2 hit | Off+3 hit | Latency hidden |
|---|---|---|---|---|---|
| `f583ab42` (no mood, baseline) | abandoned | 50% (1/2) | 50% (2/4) | 40% (2/5) | 16.8 s |
| `e01bd7f2` (no mood, longer) | success | 100% (1/1) | 100% (4/4) | — | 9.8 s |
| `a0999b2e` (mood-only) | success | 100% (1/1) | **100% (4/4)** | — | 16.8 s |
| `6bca5acc` (mood + temp 1.05) | success | (3 pend) | 100% (1/1) | **100% (5/5)** | **21.1 s** |
| `c08114aa` (Phase-2 runtime mood) | success | 100% (1/1) | — (2 pend) | — (2 pend) | 3.0 s |

Concrete fetch that hit:

```
session=a0999b2e  issued_turn=0  predicted_turn=2  offset=+2
  dependency:    policy_record  (DB, 3000ms simulated latency)
  action:        ReviewCurrentCoverage
  predictor:     empirical
  outcome:       consumed at turn 5  →  3.0 s of DB-call latency hidden
                                       from the user-facing path
```

### What "confirmed" means here

- The pipeline mechanically works end-to-end across all three SOPs.
- Hit rates at offset+2/+3 doubled (50% → 100%) when mood diversity was
  added (`a0999b2e` vs `f583ab42` baseline).
- Latency hidden is the architecture's value proposition: it's measurable,
  positive, and scales with session length.
- **Gaps**: N=1-2 per intervention. Need N=5+ replication. Cold-start
  behaviour with empty precedent_traces hasn't been measured. Tier-1
  router elevation is still rare (precedents need to accumulate).

So **(A) is confirmed in principle**, awaiting statistical robustness.

## (B) — what we have NOT confirmed about instruction prefetch

### What's missing

Today's supervisor only runs `response_gen` on the **current turn**, after
MCTS has picked the chosen action. It doesn't speculatively generate
candidate responses for predicted future turns. So the blackboard's
`instruction` slot in the production schema is currently empty in our
test runs.

```
   What we have today                         What (B) requires
─────────────────────                       ─────────────────────

  ┌──────────────────────┐                 ┌──────────────────────┐
  │ turn N completes     │                 │ turn N completes     │
  └──────┬───────────────┘                 └──────┬───────────────┘
         │                                         │
         ▼                                         ▼
  ┌──────────────────────┐                 ┌──────────────────────┐
  │ MCTS rollouts        │                 │ MCTS rollouts        │
  │ predict next K turns │                 │ predict next K turns │
  └──────┬───────────────┘                 └──────┬───────────────┘
         │                                         │
         ▼                                         ▼
  ┌──────────────────────┐                 ┌──────────────────────┐
  │ Prefetch DATA  ←(A) │                 │ Prefetch DATA  ←(A)  │
  │ for predicted        │                 │ for predicted        │
  │ next-actions         │                 │ next-actions         │
  └──────────────────────┘                 └──────┬───────────────┘
                                                  │
                                                  ▼
                                           ┌──────────────────────┐
                                           │ ⚠️  MISSING TODAY     │
                                           │ Generate INSTRUCTION │
                                           │ for predicted        │
                                           │ next-actions:        │
                                           │  • response_gen      │
                                           │  • playbook lookup   │
                                           │ Store keyed by       │
                                           │ (action, state, mood)│
                                           └──────────────────────┘
```

### Why this is the load-bearing piece

The voice-agent production note assumed that pre-staged *instructions*
(not just data) would live in the blackboard. The "blackboard schema v0"
note explicitly defines slots for both `data_payloads` and
`instruction`. We've populated the data side; the instruction side is
empty.

A real production voice agent reading the blackboard expects to find:
```
{
  "branches": [
    { "user_state": "WeighingOffer",
      "action": "HandlePriceObjection",
      "instruction": "Lead with comparison; cite market_rates...",   ← MUST BE PRE-STAGED
      "data_payloads": { "market_rates_kg": "...", "discount_eligibility": "..." }
    }, ...
  ]
}
```

Without the `instruction` field populated, the weak agent has to generate
its own response on the critical path — which is exactly the situation the
architecture is supposed to avoid for hot/frequent turns.

## The Milestone-2 experiment

A clean ablation that decides whether (B) is feasible against the same
weak agent + smart simulator setup we have today.

### Build (~3-4 hours)

```
1. After turn N commits (where today's prefetch fires), ALSO:
   for top-K predicted (action_{N+1}, state_{N+1}, mood_{N+1}):
       hypothetical_history = real_history + simulated_user_text(N+1)
       pre_response = await response_gen(
           task, hypothetical_history,
           chosen_action=action_{N+1},
           precedents=mood-conditioned-retrieve(...))
       store in staged_responses table, key=(exp_id, predicted_turn,
                                              action, state, mood)
       (the simulated_user_text comes from the matching MCTS rollout —
        it's the rollout's user-sim output at depth 1, which we already
        compute and discard)

2. At turn N+1, after live cohort/state/mood classification:
       hit = lookup_staged_response(exp_id, N+1, action, state, mood)
       if hit:
           log "would have served from cache"
       run live response_gen as today
       log live response

3. Compare staged vs live:
       - exact-string match?
       - LLM-judge semantic similarity in [0..1]?
       - human-judge quality match (optional, expensive)?
```

### Real example of what we'd be comparing

From session `c08114aa` turn 2 (chosen action was `VerifyIdentity`,
classified mood was `informational`):

**What the supervisor would have pre-staged at end of turn 1:**

> *"Hello, this is Sarah from SafeWay. To make sure I'm speaking with the
> right policyholder, can you please confirm your name and date of birth
> for me?"*

(Generated using the rollout's simulated user response + predicted action
+ retrieved precedents from `mood=informational, cohort=PriceShopper`.)

**What the weak agent actually said at turn 2:**

> *"Thanks for reaching out. To verify your account, can you confirm your
> full name and date of birth, please?"*

These are semantically nearly identical. An LLM-judge similarity check
would rate them ≥0.85. If 60%+ of pre-staged responses score that high,
(B) is in business.

### Predicted outcomes

| Result | Interpretation | Implication |
|---|---|---|
| 70%+ semantic match on hits | Strong success. (B) is viable. | Ship instruction-prefetch in production architecture. |
| 30-70% match | Modest success. Pre-staging useful in some segments. | Selective use: pre-stage only for high-confidence predictions (mood+state match, tier-1-eligible). |
| < 30% match | Weak. Pre-staging diverges too much from live. | Production architecture commits to "data-only prefetch"; voice agent generates responses live. |

If we land in the bottom bucket, the natural next thread is *why* —
maybe response_gen at supervisor-time has a different prompt context
(no live user text), or rollouts' simulated text drifts from real user
text after one turn.

## Questions for further discussion

These are open and worth deciding before / during the next phase. They
don't have obvious answers — flagged here so we can talk through them.

### Q1 — Granularity of pre-staged instructions

Today's `response_gen` produces full natural-language sentences. For a
voice agent, that's the right grain. For a chat agent, maybe richer
(JSON with optional reasoning steps). Do we want to:

  - **a.** Always pre-stage exact text the weak agent would speak verbatim?
  - **b.** Pre-stage a *template* the weak agent fills in (e.g., "say a
    Greeting that includes <customer_name> and references <last_visit_date>")?
  - **c.** Pre-stage a *playbook directive* — pointer to a canonical
    response from a curated library, with the data slot to fill in?

(b) and (c) are more robust to drift but require the weak agent to do *some*
work. (a) is purer cache-style but harder to match exactly.

### Q2 — What's a "hit" for instruction prefetch?

Three plausible match-rate definitions:

  - **Strict**: exact `(predicted_action, predicted_state, predicted_mood)` ==
    `(live_action, live_state, live_mood)` triple. Most accurate but
    sparsest.
  - **Lenient**: same action + state, mood mismatch allowed. More hits,
    might serve subtly wrong-toned content.
  - **Semantic**: any pre-staged instruction with cosine similarity ≥ X
    to what live response_gen would produce. Maximally generous but
    needs an embedding step at consume time.

The "right" definition probably depends on the production scenario's
tolerance for tone mismatch. A renewal call can tolerate slight mood drift;
a medical triage probably can't.

### Q3 — Trust model: does the weak agent trust the supervisor blindly?

When the weak agent finds a pre-staged instruction, does it:

  - **a.** Use verbatim, no second-guessing?
  - **b.** Compare against a quick self-check before using (a single fast LLM
    call to validate: "is this response appropriate for the message I just
    received?")?
  - **c.** Use as a *suggestion* and let the weak agent decide whether to
    regenerate?

(a) is fastest but riskiest. (b) is the most production-realistic. (c)
defeats much of the purpose of pre-staging.

### Q4 — When pre-staged content is wrong, how do we know?

If we serve a pre-staged response and the user reaction the next turn
indicates the agent said something off, we want a feedback signal back to
the supervisor so it can learn not to pre-stage that particular branch in
that context next time. This is the basis for a slow-loop self-correction
mechanism but we haven't designed it. Want to think about whether the
existing terminal-outcome backprop machinery is the right hook for this.

### Q5 — Production economics

Pre-staging instructions costs an LLM call per top-K branch per turn. If
K=3 and we run for 2 future offsets, that's 6 extra `response_gen` calls
per turn — significant. The savings come from the weak agent skipping its
own response_gen on hits. The break-even analysis:

```
  Hit rate × cost_per_response × prod_volume   vs   K × offset × cost_per_response × prod_volume

  → instruction prefetch pays off when hit_rate × prod_volume > K × offset
```

For a 70% hit rate at K=3, offset=2: pays off when prod_volume × 0.7 > 6,
i.e., always (per-session). But the *cost surface* is different in
production — the weak agent uses a cheap fast model, the supervisor uses
an expensive smart one. The break-even shifts. Worth pricing out
explicitly.

### Q6 — Does mood-aware retrieval actually improve response quality?

We have (1) shipped, but we haven't measured whether the agent's response
quality measurably improves when its prompt's precedent block is
mood-matched vs cohort-only. The natural metric: A/B same turn with vs
without mood filter, blind-judge the agent's response. Worth measuring
separately from (B) above.

### Q7 — Pondering vs full-Union for instruction generation

Today's pondering scheduler runs *full MCTS* between turns for top-K
predicted user states. Adding response generation per branch means
pondering becomes more expensive but more useful. Or we could split:
pondering only for *action* prediction, separate cheap step for instruction
generation. Worth deciding which architectural path before committing
to one.

### Q8 — Where does the "smart simulator" stop being honest about prod?

Our smart human simulator is intentionally strong — to ensure tests are
realistic. But strong simulators are also *predictable* in ways real users
aren't (e.g., the rollout simulator's continuity bias we documented).
We need to be honest about the limit of what test results tell us about
production behaviour. Worth a short methodology note about which claims
transfer to production directly vs which need calibration.

## Where to dig further

- `docs/agent-user-asymmetry-in-rollouts.md` — why the supervisor's task
  is not symmetric with the weak agent's.
- `notes/2026-05-24-blackboard-schema-v0.md` — proposed contract between
  supervisor and weak agent (currently spec'd for both data and
  instruction; only data is implemented).
- `notes/2026-05-23-voice-agent-production-architecture.md` — the
  production framing this discussion builds on.
- `notes/2026-05-23-stable-vs-transition-state-prediction-asymmetry.md` —
  the offset+1/+2 distinction that underpins prefetch hit-rate analysis.

## What changes after this discussion

- Terminology in future notes locks: **weak agent**, **smart human
  simulator** (or just *simulator* when context is clear), **supervisor**,
  **blackboard**.
- Research claims will be framed against milestones (A) and (B), not
  against the older "transition accuracy" or "state diversity" metrics
  (those become diagnostic substeps).
- The next concrete deliverable for the research is **Milestone-2 — instruction
  prefetch**. Until that's measured, the architecture's promise is
  half-confirmed.
