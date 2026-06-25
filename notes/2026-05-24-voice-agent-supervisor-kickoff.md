---
date: 2026-05-24
title: Voice-agent supervisor queue — kickoff and open questions
status: kickoff — research plan, no code yet
tags: [voice-agent, supervisor, asyncmld, blackboard, queue, hit-rate, production, prefetch]
related: [voice-agent-production-architecture, stable-vs-transition-state-prediction-asymmetry, speculative-data-prefetch-pipeline, first-successful-conversation-and-78pct-state-prediction]
---

# Voice-agent supervisor queue — kickoff and open questions

*Marks the transition from the 2026-05-23 production-architecture proposal*
*(`2026-05-23-voice-agent-production-architecture.md`) to active research on*
*the supervisor-queue thread. This note frames the direction, lists the*
*load-bearing open questions, names the deliverables, and orders the*
*experiments. No code yet — this is the research plan.*

> **Canonical framing (supersedes parts of this note):** see
> `2026-05-31-supervisor-research-framing-and-confirmation-criteria.md`. That
> note locks terminology (**weak agent**, **smart human simulator**,
> **supervisor**, **blackboard**) and splits the research into milestones
> **(A) data prefetch** — confirmed at N=1-2, needs N=5+ replication — and
> **(B) instruction prefetch** — not yet built, identified as the load-bearing
> missing piece. The "voice agent" language used in this kickoff note maps
> 1:1 to "weak agent" in the canonical framing; the deliverables below remain
> valid as supervisor-side infrastructure but the load-bearing research
> question shifted from "build the blackboard contract" to "populate the
> instruction slot the schema already defines." See also
> `2026-05-31-query-aware-data-prefetch-Q6b.md` for the next refinement of
> milestone (A).

## TL;DR

The next research thread is the **background supervisor queue** that lets a
weak real-time voice agent benefit from a strong slow planner without paying
its latency. The thread's value proposition reduces to a single number: the
**blackboard hit rate** — the fraction of voice turns at which the agent
finds a useful pre-staged instruction (and/or pre-fetched data) waiting for
it.

The architecture itself is borrowed (AsyncMLD) and the POC has working
implementations of every internal piece. What's new for this thread is
committing to the *specific contract* between voice agent and supervisor,
instrumenting the SLI, and proving the hit rate stays above ~70% across
realistic call shapes.

**Deliverables, in priority order:**
1. Blackboard schema and read/write protocol — the only seam, get it right early.
2. Hit-rate SLI: per-turn, per-source, persisted, queryable.
3. Queue model: prioritization, backpressure, cancel-on-divergence.
4. Cold-start mitigation: simulator-seeded priors, warm-up, fallback policy.

**Blocking dependency — now largely lifted (updated 2026-05-29).** Hedging
across plausible user states requires non-degenerate rollout distributions.
At kickoff the rollout user-sim emitted point-mass predictions (null result
in `2026-05-23-stable-vs-transition-state-prediction-asymmetry.md`). The
mood-diversity work shipped since then produces non-degenerate distributions
at offsets +2/+3 — prefetch hit-rate doubled at off+2 (50%→100%). Offset+1
stays point-mass but that's resolved as not-a-bug (depth-1 state is
action-determined, and off+1 doesn't benefit from hedging anyway). So the
supervisor's top-K branches now have a real source of diversity at the
offsets where prefetch value lives. See the reconciliation section below.

## The direction in one paragraph

In production, a voice agent runs in pre-configured scenarios (insurance,
scheduling, account servicing). It is optimized for streaming latency, not
for reasoning. The hard constraint is voice real-time: TTFB budget per agent
turn is a few hundred milliseconds. The voice agent cannot wait inline on
multi-second planning or data lookups. So we move all of the slow work — the
strong planner, the external data fetches — off the critical path. A
supervisor runs continuously in the background, predicts the next 1–3 turns,
pre-stages an instruction (and pre-fetched data payloads) for each predicted
turn, and writes them to a shared blackboard. When the voice agent reaches
turn N+K, it looks up the blackboard, uses whatever is there, and falls back
to a built-in policy if nothing useful was pre-staged. The whole bet is that
the predictions hold up often enough.

This is the **predict → queue → consume-at-future-step** loop. Everything
else in this thread is in service of that loop.

## What the POC already supports

| Piece | Where | Maturity for this thread |
|---|---|---|
| Background planner | Pondering scheduler | Today pre-computes the *next* turn. Needs a rolling 2–3-turn queue. |
| Multi-turn action prediction | `planned_actions` per rollout | Populated end-to-end. Consumed by prefetch. |
| Joint (action, state) prediction | `planned_states` per rollout | Populated. Modal collapse is the bottleneck — see blocking dep. |
| Branch hedging | Pondering top-K user_states | `pondering_k` configurable. Needs non-degenerate rollouts to do real work. |
| Data prefetch | `DataPrefetchManager` + `TrajectoryPredictor` | Per-action `data_dependencies` declared in SOP. Mock fetchers cover rag/kg/db/api/mcp. |
| Cost gate | Multi-tier router | tier-1 cached / tier-2 baseline / tier-3 MCTS. The economics lever. |
| Empirical fast path | `EmpiricalTrajectoryPredictor` | ms-latency SQL over `precedent_traces`. Steady-state predictor; MCTS becomes cold-start fallback. |
| Audit trail | `data_fetches` table (consumed/wasted/source) | SLI computation is one SQL away once we define it. |

So the heavy lifting on the *internal* mechanics is done. What's missing is
the *external* contract (the blackboard) and the SLI machinery that proves
the architecture is actually doing what it claims to do.

## What's genuinely new work for this thread

Ordered from most upstream (everything else depends on it) to least.

### 1. The blackboard contract

The single seam between voice agent and supervisor. A strawman JSON shape
appears in the production note's "What the shared state should contain"
section. To turn it into a contract we need to settle:

- **Schema** with field types and required-vs-optional designations.
- **TTL semantics**: per-entry expiry; behavior when the voice agent reads
  an expired entry (silently drop vs. surface as miss).
- **Branch ordering**: when multiple plausible states are pre-staged, what
  does the voice agent use to pick one? Modal first then fallback? Or does
  the agent classify state itself and select the matching branch?
- **Fallback handling**: what does the agent do when zero branches match
  observed state? Use the entry's `fallback` instruction, or skip the
  blackboard entirely and use built-in policy?
- **Write protocol**: append-only? Last-writer-wins? Versioned so a late
  pondering result can supersede an earlier one without race?
- **Read API**: blocking lookup with timeout? Subscription? Hint-on-miss
  so the agent can prompt the user briefly while a fast fallback fires?

Open question: should the blackboard be a database row, an in-memory
keyed cache, or a message bus topic? Each has different failure modes and
different deployment stories. For the POC, an in-process dict is enough.
For production, the answer affects the SLA.

### 2. The hit-rate SLI

The architecture's value proposition reduces to one number. Define it
precisely so we can measure it:

- **Hit**: at voice turn N, the blackboard had an entry whose
  `(cohort, predicted_user_state)` matches the classified
  `(cohort, actual_user_state)`, *and* the entry has not expired, *and* the
  voice agent actually used it.
- **Miss**: any other case (no entry, expired entry, wrong-branch entry,
  agent ignored entry).
- **Per-source attribution**: which planner tier produced the entry —
  cached_playbook / baseline / mcts / pondering — and which predictor signal
  drove it (empirical / mcts / both).

Measure at consumption time, not at write time. Persist per-turn. Roll up
to session, to SOP, to cohort. The production note's 70% threshold is a
reasonable hypothesis; this thread *measures* it.

This is also the SLI the production team can later expose externally — it's
the single number that tells them whether the architecture is paying for
itself.

### 3. Queue model

The supervisor is a continuous queue processor, not a per-turn task runner.
The production note states this but never specifies it. Open decisions:

- **Priority ordering**. Nearest predicted turn first? Lowest-confidence
  first (largest information gain)? A bandit that learns the right
  trade-off?
- **Backpressure**. Tier-3 takes 5–25 s; the queue can fall behind on a
  pathologically deep search. Hard caps per turn, per session, per minute?
- **Cancel-on-divergence**. When the user takes a branch we didn't predict,
  the in-flight work for the discarded branches is moot. Cancel immediately
  to free budget, or let it land and GC later?
- **Idle behavior**. When nothing is pending, what does the supervisor do?
  Speculatively go deeper? Refine prior work? Sit idle to save cost?

### 4. Cold-start mitigation

Without precedents, the router degrades to "always tier-3" and the economics
collapse. The first successful session note showed 95% tier-3 rate even
after two successful sessions
(`2026-05-23-first-successful-conversation-and-78pct-state-prediction.md`).
Mitigations:

- **Simulator-seeded priors**: pre-populate `precedent_traces` with
  autopilot-generated trajectories before exposing the SOP to live traffic.
- **Warm-up period**: gate the SOP behind a feature flag until it has
  accumulated N sessions of real data.
- **Stall instructions**: when the blackboard is empty, the voice agent
  emits a brief filler ("let me pull that up for you") to buy the supervisor
  a one-shot cold path.

These are interventions, not research. The research question is which
combination produces the smoothest tier-1 ramp.

### 5. Voice-specific SOP topology

The SOP graphs we have were designed for text chat. Voice users interrupt,
mumble, change topics mid-sentence. The SOP graph may need:

- Tighter recovery edges (every action → `AskClarification` as an escape).
- Time-aware action filtering (`RequestRenewal` becomes force-allowed after
  N turns of probing — directly addresses the loop pathology observed in
  session `62463cd02b4f`).
- Barge-in handling: an explicit `Interrupted` state with its own recovery
  paths.

Open question: do these topology changes need to be hand-authored per SOP,
or can they be mined from real call data once available?

## The blocking dependency — user-sim diversity (RECONCILED 2026-05-29)

At kickoff this was framed as the hard blocker: the thread couldn't validate
the *hedge* part of its design until rollouts produced non-degenerate state
distributions, and the rollout user-sim emitted point-mass predictions
(top-3 transition accuracy 0%, identical to top-1, because 8 parallel
rollouts at a turn all emitted the same state).

**That blocker is now largely lifted.** The mood-diversity follow-ups in
`2026-05-23-stable-vs-transition-state-prediction-asymmetry.md` (dated
2026-05-28 and 2026-05-29) changed the picture:

- Per-cohort mood sampling shipped (71 mood entries across 3 SOPs); one mood
  per rollout, frozen for its duration; parallel rollouts get different
  moods.
- **State-distribution width nearly doubled at offsets +2/+3** (1.59 → 3.00
  distinct states on average).
- **Prefetch hit-rate doubled at offset+2 (50% → 100%)**; mood+temp went to
  5/5 wins at offset+3 with 21.1 s latency hidden.
- **Offset+1 stays point-mass — resolved as not-a-bug.** Depth-1 user state
  is action-determined (after `AskLifeChanges` the user reports a change →
  `Interested` regardless of mood; the state vocabulary is too coarse to
  capture mood-level variation in the immediate response). And offset+1
  doesn't benefit from hedging anyway — the agent has one turn of processing
  time, so a wrong off+1 prediction has almost no recovery room.

**What this means for the supervisor thread:**

- The top-K blackboard branches the design depends on *now have a real
  source of diversity* — at offsets +2/+3, which is exactly where prefetch
  value lives (more user-think-time to overlap, deeper hedging possible).
- The "0% transition accuracy" framing was measuring offset+1, the one
  offset where mood can't and needn't help. The production-relevant metric
  is **prefetch hit-rate by offset**, and that improved.
- Pondering top-K at the blackboard layer (separate parallel MCTS runs per
  hypothesised user_state at the *same* offset) is orthogonal to within-MCTS
  mood diversity and is unaffected either way.

**Revised sequencing.** Deliverables 1–4 remain independent of prediction
quality — do them first, measure baseline hit rate. But Experiment #3 below
("user-sim diversity fix shipped") is no longer a future gate: the fix has
landed and produced N=1 wins. What's left there is *confirming* the hit-rate
improvement at N=5, not discovering whether diversity is achievable at all.

## Research questions, in priority order

1. **What hit rate does the supervisor actually achieve at steady state?** —
   on a realistic-shaped SOP, with realistic-latency fetchers, instrumented
   end to end. The production-note 70% target is a hypothesis; this is the
   measurement.
2. **How fast does precedent accumulation drive tier-1 hit rate up?** —
   curve of `% turns at tier-1` vs `cumulative sessions on this SOP`. The
   shape of this curve is the cost story.
3. **Does the mood-diversity prefetch win (off+2 hit-rate 50%→100%, N=1)
   hold at N=5, and does it carry into blackboard hit rate?** — the fix has
   landed; this confirms the improvement isn't sample noise and that the
   per-offset gain translates into the supervisor's headline HitExact rate.
   (Revised from the original "how much recovers with the fix" — that
   question is answered; this is the confirmation.)
4. **What's the optimal prediction depth for voice?** — chat-cadence
   conversations tolerate depth = 3–4. Voice turns happen in 5–15 s; the
   user diverges faster. Hypothesis: prediction depth ≥ 3 is mostly wasted
   in voice; marginal value drops fast after offset+1.
5. **What's the optimal K for between-turn pondering?** — K = 1 covers the
   most likely branch; K = 3 covers ~80–90% in practice; K ≥ 5 is mostly
   wasted. Per-SOP because branching density varies.
6. **At what queue depth does the supervisor saturate?** — characterizes
   the cost ceiling under realistic call loads. Informs production budget
   caps on tier-3 invocations.

## Evaluation methodology

We do not have voice-agent access for this thread. All measurement reuses
the existing **autopilot harness**: the user side is the same LLM-driven
customer simulator that powers chat-mode rollouts, and the "voice agent"
side is the text-mode planner + responder reading from the blackboard
rather than invoking MCTS directly on the critical path.

This is a fair test of what the supervisor *contributes*, because the
quantities we care about — prediction quality, hit rate, latency-hidden,
tier distribution, transition-vs-stable accuracy split — are independent of
voice rendering. The supervisor's job is to produce a pre-staged
instruction; whether that instruction is later TTS'd or shown as text does
not affect whether it was the right instruction at the right turn.

**What this methodology can measure faithfully:**
- Blackboard hit rate per turn, per source (cached_playbook / baseline /
  mcts / pondering; empirical / mcts / both).
- Latency-hidden total per session — sum of prefetched payload latencies
  that would have blocked in a no-supervisor baseline.
- Tier distribution as precedents accumulate, including the tier-1 ramp
  curve.
- Transition vs stable accuracy split — the structural pattern that
  determines whether top-K hedging pays off.

**What this methodology cannot measure:**
- Real prosody, barge-in, or interruption-recovery behavior.
- Real ASR latency and its interaction with the supervisor's wall-clock
  budget.
- Real-user shape: mumbles, restarts, mid-utterance topic switches. The
  customer simulator is profile-conditioned and continuity-biased — the
  same asymmetry called out in `docs/agent-user-asymmetry-in-rollouts.md`.
  This is the same evaluation bias every prior note in this project has
  carried, so results compare cleanly across notes; just not against live
  voice traffic.

**Implication for external claims.** When this thread's results are handed
to a production team, the caveat is: *these numbers are
simulator-validated; we expect the qualitative shape to transfer to voice,
but the precise rates need to be re-measured against real call data.* That
re-measurement is a production-engineering step, not a research one, and is
out of scope for this thread.

## Sequence of experiments

Each is one autopilot-run-and-measure on top of the deliverables landing in
order. None require new ML; they're sweeps over the pipeline-as-built (plus
the user-sim fix).

| # | What | Depends on | What we learn |
|---|---|---|---|
| 1 | Baseline supervisor run on `car_insurance_renewal`, 20-turn sessions × 10, current config | Deliverables 1+2 | Cold-start hit-rate distribution; SLI shape |
| 2 | Same SOP after 50 cumulative sessions (simulator-seeded warm-up) | Deliverable 4 | Tier-1 ramp curve |
| 3 | Confirm mood-diversity prefetch win at N=5 (fix already shipped) | Mood diversity landed (done) | Whether off+2/+3 hit-rate gain is robust + carries into blackboard HitExact rate |
| 4 | Sweep `pondering_k ∈ {1, 2, 3, 5}` × `prediction_depth ∈ {1, 2, 3, 4}` | All of 1–3 | Optimal hedge breadth × depth per SOP |
| 5 | Same shape on a second SOP (`virtual_medical_assistant`) | All of 1–4 | Does the hit-rate curve transfer, or is it SOP-specific? |

## What this thread does *not* try to settle

Scope discipline so the thread doesn't sprawl.

- **The voice agent's own runtime contract.** ASR/TTS, barge-in mechanics,
  prosody, streaming interruption. The supervisor produces text; how the
  voice agent renders it is a different problem.
- **Multi-tenant deployment.** The POC is single-tenant. Multi-tenant
  routing, per-tenant SOPs, per-tenant priors are a separate concern.
- **Real fetcher implementations.** Mocks are sufficient for measuring hit
  rate. Real MCP / KG / DB / RAG implementations are engineering, not
  research.
- **Voice-specific safety tradeoffs.** A wrong instruction in chat is
  recoverable; a wrong one in voice may end the call. Confidence floors and
  fail-safe policies are downstream of this thread.
- **Reward shaping for voice.** Today's reward composition is text-chat
  oriented (success markers, rationality). Voice may need to weight
  conversational latency or interruption handling more explicitly. Flagged
  for later.

## Related notes

- `2026-05-23-voice-agent-production-architecture.md` — the proposal this
  thread executes against. Read first; this note assumes its framing.
- `2026-05-23-stable-vs-transition-state-prediction-asymmetry.md` — the
  null result that defines the blocking dependency.
- `2026-05-23-speculative-data-prefetch-pipeline.md` — the prefetch
  subsystem this thread relies on for data-payload pre-staging.
- `2026-05-23-first-successful-conversation-and-78pct-state-prediction.md` —
  the only end-to-end success trajectory observed; informs realistic
  expectations for hit rate and tier-1 ramp.
- `docs/how-prefetch-reads-mcts-rollouts.md` — how the prefetch pipeline
  consumes MCTS by-products. Same mechanism the blackboard writer will use.
- `docs/agent-user-asymmetry-in-rollouts.md` — why the user-sim diversity
  problem is structural, not just a tuning issue.

## What "done" looks like for this thread

The thread is done when:

1. The blackboard schema is a frozen contract with at least one (text-based)
   consumer in the POC reading from it.
2. The hit-rate SLI is queryable per session and per SOP, with per-source
   attribution.
3. There is a measured hit-rate curve over cumulative sessions on at least
   one SOP, with a verdict on whether the 70% threshold is reachable in
   this configuration.
4. The user-sim diversity fix has been attempted and its impact on
   transition-hit-rate has been measured.

After that, the thread either greenlights production engineering or
documents the gap that blocks it. Either outcome is a clear research result.
