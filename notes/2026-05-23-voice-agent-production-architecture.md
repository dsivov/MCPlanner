# Voice Agent + Background Supervisor Architecture

Discussion note mapping the POC research onto the production voice-agent product.
Not an action plan; the goal is to surface the load-bearing decisions and where the
research already de-risks them.

---

## Executive summary (½ page)

**The product problem.** The voice agent is fast but weak. To handle real
scenarios it needs both a smarter planner and the ability to query MCP / KG /
DB / RAG sources. Both add latency the voice budget can't absorb (planner: 1–25 s;
data lookups: 1–7 s; voice's per-turn budget: a few hundred ms).

**The pattern.** Run everything slow in the background. A supervising planner
decides, data sources are queried, and a per-scenario instruction is pre-staged
in a shared store *before* the user finishes their next turn. The voice agent
reads the store and uses what's there. This is published architecture —
**Asynchronous Multi-Level Decomposition (AsyncMLD)**, [arXiv:2312.13925](https://arxiv.org/html/2312.13925v1),
originally for robotics — adapted from a robot-controller setting to a voice +
LLM setting.

**Why it works for voice latency.** The voice agent stays on its real-time
budget because it never waits on the planner or on data. Slow work happens
during the natural gap of the user speaking or thinking. When the planner's
prediction is correct, the response feels instant. When wrong, the agent falls
back to a safe default — no worse than today.

**The economics.** A multi-tier router decides per turn whether the heavy
planner even needs to run. After enough sessions, most turns are routed to a
cached lookup (free, instant). Some go through a single fast LLM call. Only
novel / high-uncertainty turns invoke the full search. The cost story depends
on this distribution being heavily skewed toward the cheap tiers, which
simulation supports.

**What's de-risked.** Async actor/planner decoupling is a published pattern
(AsyncMLD). Every building block — multi-turn action prediction, branch
hedging across plausible user states, prefetching declared data dependencies,
learning empirical priors from past sessions — is implemented and validated in
the POC.

**The remaining bet.** Will the planner's hit rate stay high enough on real
calls to justify the engineering investment? Measurable from day one by
instrumenting how often a turn finds a useful pre-staged instruction.
Sustained **≥ 70 % hit rate** makes the architecture economically viable. The
POC has the measurement tooling.

**Production scope.** *Reuse* from the POC: planner internals, router,
trajectory predictors, prefetch queue, instrumentation. *Build new*: the
shared-store schema/API between voice agent and supervisor, and real fetcher
implementations for MCP/KG/DB/RAG (POC uses mock fetchers with realistic
latencies). The novel work is in the blackboard contents and supervisor stack;
the architecture itself is borrowed and battle-tested.

---

## Architecture at a glance

```
                              ┌──────────┐
                              │   USER   │
                              └────┬─────┘
                    speech ──────► │ ◄────── voice reply
                                   ▼
═══════ CRITICAL PATH (≤500 ms) ═══════════════════════════════════════════════
                              ┌──────────┐        lookup       ┌──────────────────┐
                              │  Voice   │ ◄────── (~ms) ───── │   Shared State   │
                              │  Agent   │                     │   "blackboard"   │
                              └──────────┘                     │                  │
═══════════════════════════════════════════════════════════════│ · cohort         │═══
                              user_state                       │ · top-K branches │
                              hint per turn                    │ · instruction    │
                                  │                            │ · data payloads  │
                                  ▼                            │ · fallback       │
                              ┌──────────┐                     │ · TTL            │
                              │  Router  │ ── tier-1 cached ──►│                  │
                              │  (gate)  │ ── tier-2 baseline ►│                  │
                              └────┬─────┘ ── tier-3 MCTS ────►└──────────▲───────┘
                                   │                                      │ writes
                                   ▼                                      │
                            ┌──────────────┐                              │
                            │  Trajectory  │ — predicted action + state — │
                            │   Predictor  │   for offsets +1, +2, +3     │
                            │ MCTS+Empir.  │   (state-aware Union)        │
                            └──────┬───────┘                              │
                                   │ plan items                           │
                                   ▼                                      │
                            ┌──────────────┐                              │
                            │   Prefetch   │ — parallel fetches ──────────┤ payloads
                            │   Manager    │                              │
                            └──────┬───────┘                              │
                                   │                                      │
                ┌──────────┬───────┼───────┬──────────┐                   │
                ▼          ▼       ▼       ▼          ▼                   │
              ┌─────┐  ┌─────┐ ┌─────┐ ┌─────┐    ┌─────┐                 │
              │ MCP │  │ KG  │ │ DB  │ │ RAG │    │ ... │                 │
              └──┬──┘  └──┬──┘ └──┬──┘ └──┬──┘    └──┬──┘                 │
                 └────────┴───────┴───────┴──────────┴─────────────────────┘
                          payloads (each 1–7 s; all in parallel)
═══════ ASYNC LANE (between user turns) ═══════════════════════════════════════
```

**How to read it.**

- **Top half (critical path)** is the only thing constrained by voice latency.
  Voice Agent talks to the user and reads the Shared State. That's it.
- **Bottom half (async lane)** is everything that's slow: Router → Predictor →
  Prefetch → external data sources. All of this runs *between* user turns,
  during the natural gap of the user speaking or thinking. It can take seconds
  without the user feeling it.
- **The Shared State is the only seam** between the two halves. The voice agent
  never directly invokes the planner or a data source — it just reads whatever
  the supervisor has already written.
- **The Router controls cost.** Most turns should hit tier-1 (cached lookup,
  ~ms, free). Tier-2 (single LLM call) handles moderate-uncertainty turns.
  Tier-3 (full MCTS, seconds, costly) only runs on novel turns. The
  distribution shifts toward tier-1 as the empirical priors accumulate from
  past calls.
- **Top-K branches** in the blackboard are the hedge against wrong predictions:
  the supervisor pre-stages instructions for the *most likely* user_state and
  for *one or two alternates*. Whichever branch the user actually triggers,
  the voice agent has the right instruction ready.

---

## Supervisor as a queue, not a deadline

A natural — but slightly wrong — way to read the diagram is *"the supervisor
has a deadline of one user turn to produce a pre-staged instruction."* That
framing breaks down once you accept that the supervisor is a **continuous
queue processor**, not a per-turn task runner.

**What's actually happening.**

- The supervisor maintains a queue of predicted upcoming turns: `[N+1, N+2, N+3, …]`.
- It works them in priority order, writing each result to the blackboard as
  soon as it completes. There is no hard "must finish by turn X" deadline.
- For any turn N+K, the supervisor has had however much time elapsed since the
  prediction was first queued. Deeper into the future = more elapsed time
  available, but also higher prediction error (the user may have diverged from
  the predicted branch by then).

**The only metric that matters is hit rate at lookup time.** The voice agent
reads the blackboard at moment T. Whatever's there is what it uses. The
question is "what fraction of lookups find something useful?" — not "did the
supervisor finish in time?"

**What constrains useful lookahead, then?** Not the clock — *compounding
prediction error*. The supervisor *could* spend unbounded CPU on predicting
turn N+10, but by then the user will likely have left the predicted branch.
Practical lookahead is 2–3 turns. Beyond that, work is wasted regardless of
how much time was available.

**One place a real deadline still exists:** the very first turn of a session.
At session start the queue is empty — the supervisor has only the time it
takes the user to compose their first message to populate any pre-staged
content. Cold-start latency hides here. After turn 1, the queue runs
continuously and the deadline framing stops applying.

**Implications for tier-3 (MCTS) cost.** A 25-second tier-3 search isn't "too
slow" in absolute terms. It's only too slow *if it's the only thing the
supervisor is doing and the result is needed before it finishes*. With a queue
of 2–3 predicted turns in flight, tier-3 can run in the background for 25 s
and still complete well before any of those turns arrive. The cost cap is the
*user-facing hit rate*, not the wall-clock per task.

**Why this matters for product design.**

- You don't size the supervisor by "what fits in one turn." You size it by
  "what work is worth doing across the rolling 2–3 turn horizon," and let the
  queue absorb variability in per-task duration.
- A pathological 60-second tier-3 search isn't automatically a failure. It's a
  failure only if its result wasn't ready by the lookup at turn N+1, *and* no
  cheaper tier had produced an answer in the meantime. The router handles the
  latter; the queue handles the former.
- When predictions are wrong (user diverges), the queue's invested work is
  wasted but **no user-facing latency was incurred** — the voice agent just
  reads whatever's there now. Waste matters for cost, not latency.

---

## Related work — AsyncMLD

The pattern described here ("fast actor on the critical path, slow planner
running asynchronously, both reading/writing a shared state") is exactly the
**Asynchronous Multi-Level Decomposition (AsyncMLD)** framework from
[arXiv:2312.13925](https://arxiv.org/html/2312.13925v1). That paper formalises
the actor/planner decoupling in robotics: a fast low-level controller acts in
real time while a slow high-level planner refines the long-horizon plan in
parallel, with synchronisation through a shared blackboard. Map onto our
setting:

| AsyncMLD term         | Voice-agent production analogue            |
|-----------------------|--------------------------------------------|
| Low-level controller  | Voice-LLM (real-time, weak)                |
| High-level planner    | MCTS supervisor (slow, strong)             |
| Shared blackboard     | Pre-staged-instruction store               |
| Replanning trigger    | New user_state classified per turn         |
| Plan refinement cycle | Pondering MCTS between turns               |

The novel bits in our research aren't in the async decoupling itself — that's
solved by AsyncMLD — but in the **content** of the shared state: state-aware
Union-predicted trajectories with empirical+MCTS confidence tagging, per-branch
hedged prefetch with declared data dependencies, and a multi-tier router that
gates how much planning work runs per turn. AsyncMLD doesn't prescribe the
planner internals; it prescribes the protocol.

So in a production conversation, the framing is:

> "We're implementing an AsyncMLD-style decoupling. The voice agent is the
> low-level controller; our supervisor is the high-level planner; the
> pre-staged-instruction store is the shared blackboard. The research
> contribution beyond AsyncMLD is what goes *in* the blackboard and how it
> gets there — multi-turn state-aware trajectory predictions with per-branch
> data prefetch, gated by a learned router."

That framing has two practical benefits:
1. **Risk reduction.** AsyncMLD is published and battle-tested in robotics;
   the async pattern itself isn't speculative.
2. **Clear scope of the bet.** The novel work is the blackboard contents and
   the supervisor's internals, not the architecture. If production wants to
   start with a thin slice and grow, AsyncMLD scaffolding can be built first
   with stub planners, then the supervisor can be incrementally enriched.

## Production scenario (as described)

- Real product is a **voice agent** in pre-configured scenarios (insurance,
  scheduling, account servicing, etc.). Each scenario has a well-defined SOP.
- The voice-LLM is **weak** — optimized for streaming latency, not reasoning. It
  cannot do non-trivial planning, nor wait on multi-second data lookups inline.
- Two missing capabilities the voice agent needs from elsewhere:
  1. **Decision-making supervision** — what to say next, when to escalate,
     which branch of the SOP we're on.
  2. **Knowledge retrieval** — MCP servers, KG queries, DB lookups, RAG over
     internal docs. Latencies typically 1–7 s; some external APIs > 6 s.
- Hard constraint: **voice is real-time**. TTFB budget per agent turn is ~200–500 ms.

## Proposed pattern

- **Supervisor runs out-of-band** (between user turns, in parallel with voice
  output streaming). It predicts the next 1–N turns ahead.
- For each predicted (action, user_state) tuple, the supervisor pre-fetches the
  data dependencies declared by the SOP and pre-composes the instruction the
  voice agent should follow if that branch is hit.
- A **shared state store** holds the supervisor's output keyed by predicted
  (cohort, user_state, action). The voice agent reads from it per turn.
- On each turn, the voice agent looks up the current cohort + classified
  user_state in shared state. If a useful pre-staged instruction is present, it
  uses it. Otherwise it falls back to its own (weaker) decision and the
  supervisor catches up out-of-band.

## How the POC research maps onto each piece

| Production piece | POC component(s) | Notes |
|---|---|---|
| Out-of-band supervisor | **MCTS planner + Pondering scheduler** | The pondering scheduler is already the background-MCTS pattern. Today it pre-computes the *next* turn's decision; production needs it to maintain a rolling queue of N predicted turns and process them continuously. |
| Multi-turn prediction | **planned_actions + planned_states arrays per rollout**, **state-aware Union predictor** | We already publish the joint (action, user_state) trajectory per rollout. Per-offset modal state + action is what the shared state should be keyed on. |
| Branch hedging | **Pondering top-K next user_states** | Pondering already fires K hypothesis MCTS searches per turn. K controls how many branches the shared state covers. |
| Data prefetch | **DataPrefetchManager + TrajectoryPredictor + plan items** | Direct port: replace MockDataFetcher with MCP/KG/DB/RAG implementations. Per-action `data_dependencies` declaration is already there. |
| Avoiding cold-start cost | **Multi-tier router (cached_playbook / baseline / mcts)** | This is the *load-bearing component for economics*. Steady-state production should hit tier_1 (free, instant) on the majority of turns; MCTS is the rare exception. See §"Why the router matters most" below. |
| Empirical fast path | **EmpiricalTrajectoryPredictor** | After enough sessions, empirical lookup over `precedent_traces` is ~ms-latency and answers "what action follows X in cohort C" without an LLM call. This is the *production* predictor; MCTS becomes the cold-start fallback. |
| Instruction-level cache | **PrefetchPlanItem.predictor_source + predicted_user_state + DataFetch row** | Each pre-staged instruction can be tagged with its source + state. The voice agent can compare its observed state against the tag and decide whether to trust the pre-staged instruction. |
| Continuous learning | **Slow-loop lift mining + Alembic schema for traces** | Over time, hot cohorts accumulate enough precedents to flip from MCTS → empirical → cached_playbook routes. The router does this automatically based on entropy. |

The POC already implements every block above. Production work would be:
swapping the mock fetchers for real ones, building the shared-state contract
between supervisor and voice agent, and tuning the router thresholds for the
voice latency budget.

## Why the router matters most for voice production

Per-turn MCTS in the POC takes 5–25 s and costs $0.02–0.05 in tokens. The
*latency* concern is handled by the queue model — tier-3 work doesn't block
voice. But the *cost* concern is real: doing tier-3 on every turn of a voice
call adds up to dollars per call. The product is only viable if:

- The vast majority of turns hit **tier_1 (cached_playbook)** — sub-ms lookup,
  free. Production target should be 70–80%+ tier_1 hit rate at steady state.
- A meaningful slice hits **tier_2 (baseline)** — single LLM call, ~1 s, low cost.
- **Tier_3 (MCTS)** is reserved for novel / high-entropy turns where the cost
  is justified.

The pondering scheduler runs in the background queue, so a 25-s tier-3 search
is *latency-fine* as long as the queue is sized for a 2–3 turn rolling
horizon. What the router controls is *how often that expensive search runs at
all* — not "in time" but "at all."

## What the shared state should contain

Per (session_id, predicted_turn_index), the supervisor should publish:

```
{
  "cohort":                "PriceShopper",
  "predicted_user_states": [{state: "WeighingOffer", prob: 0.7}, {state: "Skeptical", prob: 0.3}],
  "branches": [
    {
      "user_state":    "WeighingOffer",
      "action":        "HandlePriceObjection",
      "instruction":   "Lead with comparison: cite market_rates. Apply 18% discount stack
                        (multi-policy+telematics) before quoting the new number.",
      "data_payloads": {
        "market_rates_kg":      "MARKET: peer_avg=$1580/yr; Geico $1510, Progressive $1620, StateFarm $1490; rank 2nd",
        "discount_eligibility": "DISCOUNTS: multi-policy=8%, telematics=12%, ..."
      },
      "source":        "both",   // mcts + empirical agreed → highest confidence
      "ttl_s":         60
    },
    { /* second branch */ }
  ],
  "fallback": {
    "instruction": "Acknowledge price concern, then ask what specifically feels high...",
    "source": "tier_2"   // safety net if no branch matches the actual user_state
  },
  "supervisor_decided_at": "2026-05-23T14:32:11Z",
  "expires_at":            "2026-05-23T14:33:11Z"
}
```

Key properties:
- **TTL per entry** so stale predictions don't poison later turns.
- **Multiple branches** matching the top-K predicted states (covers the "I don't
  know which way the user will go" case — exactly what pondering+Union solves).
- **Fallback instruction** the voice agent uses if the actual classified state
  doesn't match any branch — graceful degradation rather than hard miss.
- **Source tag** lets the voice agent (or analytics) discount low-confidence
  predictions and gate behavior accordingly.

## The hit-rate question

The whole architecture stands or falls on one number: **what fraction of turns
the voice agent finds a useful pre-staged instruction**.

The POC instrumentation already measures this per source (mcts / empirical /
both / live) on the prefetch side via `DataFetch.consumed` vs `wasted`. The
pondering scheduler measures the analogous "MCTS reuse" hit rate. Production
should publish both as the primary SLI.

Reasonable target shape at steady state:
- **Tier 1 (cached_playbook) hit rate:** 70%+ of turns. Empirical priors over
  precedent_traces are confident enough — no supervisor work needed.
- **Tier 1 + Tier 2 combined:** 90%+. Most remaining turns can be handled with
  a single fast LLM call.
- **Tier 3 (MCTS):** under 10% of turns. Always runs in the background queue;
  never on the voice agent's critical path.

If those numbers don't hold, the architecture's latency story breaks and the
voice agent ends up doing more on-the-spot work than intended. The POC sweep
is exactly the right place to measure this baseline before production
commitment.

## Failure modes worth designing for

1. **Wrong prediction.** Supervisor predicted state=WeighingOffer, user
   actually said something matching state=Skeptical (which was branch #2). The
   voice agent picks branch #2 from shared state — costs nothing, no LLM call.
   This is the *normal* case the multi-branch shared state design handles.

2. **All predictions wrong.** Neither predicted state matches what the user
   said. Voice agent falls back to its built-in (weaker) policy or the
   pre-composed `fallback` instruction. The supervisor logs this as a miss and
   updates empirical priors for next time. **Critical:** this must not block;
   the voice agent has its own latency budget to stay within.

3. **Supervisor still running when user speaks.** The voice agent doesn't wait
   for anything — it reads the blackboard and uses whatever's there (possibly
   nothing). The supervisor's in-flight work for turn N+1 isn't discarded; it
   continues and lands in the blackboard for turn N+2 (still useful, just for
   a turn the user hasn't reached yet). No emergency, just a one-turn slip in
   the queue's effective lookahead.

4. **Stale shared state.** User paused 90 s mid-call. Predictions made before
   the pause shouldn't drive behavior after the pause; cohort might have
   shifted (e.g., became hesitant). TTL-based expiry handles this. After
   expiry, voice agent falls back; supervisor re-pondered on next signal.

5. **Cold cohort.** Brand-new SOP or rarely-seen cohort. No empirical priors
   → tier_3 MCTS runs every turn. Solutions: (a) seed empirical from
   simulator-generated precedents (we already have this; that's exactly what
   the POC sweep produces); (b) gate the SOP behind a "warm-up" period before
   exposing to live traffic.

6. **Voice agent disagreeing with supervisor.** Voice agent has its own
   reasoning, might pick differently than the supervisor's pre-staged branch.
   Log the divergence: it's training data for both the supervisor (was its
   prediction off?) and for diagnosing voice-agent miscalibration.

7. **Cost runaway.** If tier_3 fires too often (cold start, drift, or
   adversarial users), MCTS cost dominates. Hard cap on MCTS iterations per
   session, budget caps per minute, fall back to tier_2 if cap breached.

8. **Multi-turn drift.** Pre-staged instruction for turn N+2 was based on a
   prediction made at turn N. By turn N+1 we learn the prediction was wrong,
   but the N+2 staging is still based on the old branch. Either re-ponder
   eagerly when prediction errors are detected, or accept the staleness as
   bounded since real conversations rarely span > 3 predicted turns of
   coherent state.

## What this research does *not* answer

- **The voice agent's own runtime contract.** ASR/TTS latency, barge-in,
  streaming interruption, prosody — all out of scope. Our supervisor produces
  text instructions; how the voice agent renders them is its problem.
- **Real-world cohort + state classification accuracy.** The POC uses
  simulator-generated precedents; production needs real call data to validate
  that the cohort/state vocabulary maps cleanly to live calls.
- **Multi-tenant isolation.** Different scenarios may need different SOPs,
  different routers, different prefetch budgets. The POC is single-tenant.
- **Voice-specific reliability tradeoffs.** A wrong instruction in chat is
  recoverable in the next message; a wrong instruction in voice may end the
  call. The supervisor needs a confidence floor below which it returns the
  fallback rather than a low-confidence branch.

## Open research questions specific to this scenario

These would inform production design but are *research* questions, not
engineering ones:

1. **How many turns ahead is useful to predict?** POC sweeps with depth = 2–4.
   In voice, turn cadence is faster (5–15 s per turn vs minutes in chat) and
   user state shifts faster. Empirically, the marginal value of predicting
   turn N+3 may be near-zero. Worth measuring on the sweep we just ran.

2. **What's the right K for between-turn pondering?** Each unit of K is one
   parallel MCTS run between turns. K=1 covers the most likely branch; K=3
   covers ~80–90% of branches in practice; K=5+ is mostly wasted. POC has
   `pondering_k` configurable for ablation.

3. **State-aware Union vs empirical-only at scale.** Once you have ≥1k
   sessions per cohort, does MCTS still add anything over empirical? The POC
   measures `hit_rate_by_source`; running it on real call data would answer
   directly. Suspicion: empirical dominates once you have data, MCTS is purely
   cold-start.

4. **How does the SOP itself need to change for voice?** Voice users
   interrupt, mumble, change topics mid-sentence. The SOP graph may need
   tighter recovery edges (e.g., every action edge → AskClarification). Worth
   checking whether the SOP graph mined from text-chat sessions transfers to
   voice or if the topology shifts.

5. **What's the right unit of supervision?** Per-action (our default), per
   "strategy" (a group of semantically related actions), or per "turn-script"
   (multiple sentences of pre-composed text)? Trade-off: bigger units =
   cheaper but less responsive to mid-utterance corrections.

## TL;DR for the production conversation

Everything the proposed architecture asks for is already implemented in the
POC and validated in simulation. The two things production must commit to
before shipping are:

1. **Build the shared-state contract.** Define schema, TTL policy, write +
   read API. This is the only piece that crosses the voice-agent / supervisor
   boundary — get it right and the rest is internal.

2. **Measure hit rate from day one.** The architecture's value proposition
   *is* the hit rate. Without instrumentation showing it stays above ~70%, you
   don't know if the supervisor is paying for itself.

Everything else — router, predictors, prefetch budget, empirical learning —
the POC has working implementations and an experimental harness to tune them.
