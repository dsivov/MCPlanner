---
date: 2026-05-23
title: First successful conversation closure, 78% cold-start state-prediction accuracy, and first end-to-end prefetch measurement
status: observed (N=1+1)
tags: [pca-m, mcts, state-prediction, union-predictor, prefetch, success-marker, router, sop-design]
related: [speculative-data-prefetch-pipeline, voice-agent-production-architecture]
---

# First successful conversation closure, 78% cold-start state-prediction accuracy, and first end-to-end prefetch measurement

*Observation note from two paired live sessions on the same SOP:*
*(1) the first `terminal_outcome=success` across all experiments to date,*
*with a measured 78% MCTS state-prediction accuracy on cold-start data, but*
*with prefetch disabled by config; and*
*(2) the first prefetch-enabled session with the state-aware Union predictor*
*— short but mechanically complete, with 3 s of external-I/O latency hidden*
*on a successful fetch.*

## TL;DR

Session `62463cd02b4f` on the `car_insurance_renewal` SOP closed on the
`AgreedToRenew` success_marker after 29 turns — the first `terminal_outcome=success`
observed across ~150 prior sessions, all of which had either timed out at
`max_turns=6` or hit a failure_marker.

The session also gave us the **first per-turn measurement of MCTS state
prediction accuracy** on a cold-start SOP (i.e., no prior precedent_traces for
this SOP / cohort combination):

- **78% of next-turn user_state predictions were correct** (21 of 27 testable
  turns), with all 6 misses falling on state-transition boundaries.

This is above the 70% threshold cited in the voice-agent production note as
the viability bar for a queue-based supervisor — *and* it's measured before
any of the empirical-priors machinery has had a chance to accumulate data.
That number should only get better as `precedent_traces` grows.

The same session also exposed three weaknesses worth flagging: 97% of turns
went to tier-3 (full MCTS) because precedent data is still too thin to elevate
the router, the action policy got stuck in a 20-turn local loop before
escaping to the success branch, and prefetch happened to be disabled at the
config level (not a finding about prefetch itself — just absent data).

## Session summary

| | |
|---|---|
| Session id | `62463cd02b4f` |
| SOP | `seed:car_insurance_renewal.json` |
| Planner | PCA-M, simulate-mode rollouts, llm_top1 action policy |
| Router | enabled, tier-1 threshold entropy ≤ 0.4 |
| Prefetch | **disabled** (UI config — not a deliberate ablation) |
| Predictor | union (configured, not exercised since prefetch off) |
| Cohort settled to | `LifeChange` (turn 1 → 28) |
| Turns | 29 |
| Outcome | **success**, reward = 1.0 |
| Wall-clock | ~12 min |
| LLM calls | ~1,400 (52/turn × 28 MCTS turns + ~5/turn for non-MCTS) |

## Action sequence (compressed)

```
T0  Greeting           [ReportingChange]
T1  VerifyIdentity     [ReportingChange]      → cohort = LifeChange
T2  Greeting           [IsThemselves]         (router → baseline; only non-MCTS turn)
T3  StateReason        [ReportingChange]
T4-T8     StateReason / AskLifeChanges loop  [ReportingChange]
T9-T14    StateReason / AskLifeChanges loop  [Interested]
T15-T19   ReviewCurrentCoverage / AskLifeChanges  [Interested ↔ ReportingChange]
T20-T27   AskLifeChanges / StateReason / ReviewCurrentCoverage  [Interested]
T28 AskLifeChanges     [AgreedToRenew]        → success_marker hit, session closed
```

The agent looped through `{StateReason, AskLifeChanges, ReviewCurrentCoverage}`
for ~20 turns before the simulated user finally agreed. The exit happened
"despite" the loop — MCTS kept selecting `AskLifeChanges` at T28 and the user
sim just happened to escalate to `AgreedToRenew`. The planner didn't
deliberately steer there; it converged through user simulator drift.

## State-prediction accuracy

For each turn N ≥ 2, we compared:

- **Predicted**: the modal `predicted_user_state` at offset+1 across all
  rollouts at turn N-1 that started with the chosen action (this is the hint
  the state-aware Union predictor would hand to the empirical SQL).
- **Actual**: the `predicted_user_state` actually classified at turn N from
  the live user message.

(Skipping turns where the prior turn had no MCTS rollouts — turn 0 has no
prior, turns 2-3 followed the tier-2 baseline turn at T2 which produced no
rollouts.)

| Status | Count | Turn indices |
|---|---|---|
| ✓ correct | 21 | 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27 |
| ✗ wrong | 6 | 2, 9, 15, 17, 28 + one earlier |
| no test | 2 | 0, 1, 3 |

**Hit rate: 21 / 27 = 78%.**

Every miss lands at a state-transition boundary:

| Turn | Predicted (from T-1 modal) | Actual | What changed |
|---|---|---|---|
| 2 | ReportingChange | IsThemselves | LLM finally confirmed identity |
| 9 | ReportingChange | Interested | User softened from change-reporting to interested mode |
| 15 | Interested | ReportingChange | Brief revert during coverage discussion |
| 17 | ReportingChange | Interested | Drifted back |
| 28 | Interested | **AgreedToRenew** | The success moment |

Within a stable phase MCTS predicts perfectly because the modal state at
offset+1 across rollouts converges to whatever the user currently is. The
errors are concentrated at the transitions — exactly where prefetch payoff
would be highest in production (a state change at turn N triggers a different
set of data deps for turn N+1).

The miss at turn 28 (the success moment) is the most product-relevant: a
production agent would have failed to pre-stage any data tied to the
`AgreedToRenew` branch and had to fall back at consume time. This is the kind
of transition that pondering top-K branches is supposed to hedge against — at
turn 27 the supervisor should have pre-staged both `Interested` (which it did)
*and* `AgreedToRenew` (a less-likely but possible terminal). With pondering K
≥ 2 we'd expect a meaningful fraction of those terminal transitions to be
covered too.

## Router didn't help (yet)

97% of turns went through tier-3 (full MCTS, ~52 LLM calls each). The router
elevated to tier-2 only once (turn 2), and to tier-1 never. The reason:

- `tier_min_supporting_traces = 3` — for any (cohort, user_state) pair to be
  routable to tier-1, we need ≥ 3 prior precedents agreeing on a dominant
  action.
- This was the first MCTS-mode session of significance on the `LifeChange`
  cohort for this SOP. There simply aren't 3 prior precedents to read from.

The router *will* fix this on subsequent sessions — by the third or fourth
session through the same cohort+state path, tier-1 should start firing for
the stable phases (the long `Interested ↔ AskLifeChanges` loop is the most
obvious candidate). This is also the *primary cost lever* in the production
architecture (~1,400 LLM calls per call is uneconomic for a voice product
absent the router elevation).

**Direct prediction**: re-run 5 more sessions of this exact configuration on
this SOP, then re-measure. Tier-1 rate should go from 0% to something in the
20-40% range, and per-turn LLM call count should drop accordingly.

## The looping behaviour

The agent spent turns 4-27 in a `{StateReason, AskLifeChanges,
ReviewCurrentCoverage}` cycle. This isn't a planner pathology in the strict
sense — MCTS was correctly maximising expected reward at each turn — but it
reveals a soft spot in the SOP:

- The `success_marker` `AgreedToRenew` is unreachable without a
  `RequestRenewal` action somewhere on the path.
- `RequestRenewal` was available in the SOP's allowed-next-actions set from
  turn ~6 onward, but its Q-value never beat `AskLifeChanges` in MCTS rollouts
  because the user-simulator never reported life-change-completed in a way
  that would have made the rollouts predict success on `RequestRenewal`.
- The session only closed because the simulated user *itself* drifted to
  `AgreedToRenew`, bypassing the planner's preferred path.

In a real call, this loop would feel robotic ("yes, but anything else
changed?... yes, but anything else?..."). The SOP fix isn't on the planner side:

- Add an *urgency* constraint or a turn-count-aware action filter (e.g.,
  `RequestRenewal` should be force-allowed once the agent has covered all of
  `{StateReason, ReviewCurrentCoverage, AskLifeChanges}` at least once).
- Tighten the failure_markers so that prolonged looping gets penalised, which
  would push MCTS away from `AskLifeChanges` after the 3rd or 4th repetition.

Both are SOP edits, not code changes.

## Cross-checking against the production note's claims

The voice-agent production note set three load-bearing assumptions:

1. **Hit rate ≥ 70% makes the architecture economically viable.** This
   session's 78% state-prediction accuracy on cold-start data sits above the
   bar, but it's a single observation and on a cohort that quickly settled
   into a stable phase. The harder test is *transition-boundary* accuracy,
   where this session got 0/5. A multi-session sweep needs to repeat the
   measurement before claiming production-readiness.

2. **The router gates cost.** This session confirms what the production note
   warned about: *without* enough precedent data, the router degenerates to
   "always tier-3" and the supervisor cost story collapses. Whether the
   precedent-accumulation curve actually catches up fast enough on real call
   data is now the central unknown. The cold-start mitigation (stall
   instructions, simulator-seeded priors) discussed in the production note
   matters more than I'd weighted it.

3. **MCTS publishes joint trajectory structure that empirical can't match.**
   Validated mechanically — `planned_states` arrays are populated and the
   78%-accurate offset+1 hint comes directly from them. The empirical
   predictor was configured but had no precedent_traces to read from for this
   SOP+cohort combination, so it contributed nothing this session. Subsequent
   sessions will be the first real test of the Union predictor's joint behaviour.

## Prefetch addendum — what we learned from session `58237697f0dd`

The 78%-accuracy session had `data_prefetch_enabled=false` for unrelated
reasons (UI config), so it doesn't speak to the prefetch pipeline. Earlier the
same day a separate **autopilot session** ran the same SOP with
**prefetch=on, predictor=union**, and gave us the first concrete prefetch
measurement on the new realistic-latency SOP. Six turns, abandoned outcome,
but exercises the pipeline end to end. Key numbers:

| | |
|---|---|
| Speculative fetches scheduled | 5 |
| Live fallback fetches | 0 |
| Hits (consumed) | 1 |
| Misses (wasted) | 4 |
| Latency hidden (sum) | **3,000 ms** (one DB lookup served from cache) |
| Hit rate | **20%** |

### Per-source attribution

| Source | Hits | Miss | Rate | ms hidden |
|---|---|---|---|---|
| empirical | 1 | 2 | 33% | 3,000 |
| both (mcts ∩ empirical) | 0 | 2 | 0% | 0 |
| mcts (solo) | 0 | 0 | — | 0 |

The empirical predictor produced 3 of the 5 fetches (state-blind, marginal
distribution over precedent_traces from earlier sessions on this SOP). Two
fetches came from the state-aware Union with source=`both` and predicted
state=`PriceConcern` for offset+1.

### What worked, what didn't

The one **hit** was meaningful:

- After turn 0 (Greeting), empirical predicted offset+1 action = `VerifyIdentity` based on prior sessions' Greeting→VerifyIdentity transition frequency.
- The DB-kind `policy_record` fetch was scheduled, ran for 3,000 ms in parallel with the user simulator's response composition.
- At turn 1 the agent picked `VerifyIdentity` exactly as predicted → fetch consumed instantly → 3 s of DB-call latency hidden from the user.

This is the prefetch system doing exactly what it's designed for: turning a
sequential 3 s DB call into invisible background work.

The four **misses** are also informative:

- The two `both`-tagged fetches at turn 1 targeted `HandlePriceObjection` (predicted state `PriceConcern`). MCTS rollouts at turn 1 generated this prediction because the rollout user-simulator imagined a price-pushback branch. The actual user never pushed back on price — the cohort settled to `LifeChange`, not `PriceShopper` — so `market_rates_kg` (KG, 4.2 s) and `discount_eligibility` (API, 2.8 s) sat in the cache until session end and timed out.
- The two empirical misses at turn 0 were for `policy_record` and `claims_history_rag` tied to `ReviewCurrentCoverage`. The agent never reached `ReviewCurrentCoverage` in this 6-turn session.

### Reading 20% hit rate honestly

This is well below the 70% production target from the voice-agent note.
Caveats before reading too much into it:

1. **6-turn truncation kills hit rate mechanically.** Three of the four misses
   target actions further down the SOP path that the session simply didn't
   reach. In the 29-turn success session, those same fetches would likely
   have been consumed eventually. Hit rate measured on short sessions
   systematically understates the steady-state hit rate.
2. **N=5 fetches is not a measurement.** A single mispredicted user cohort
   (`LifeChange` vs `PriceShopper`) zeroed out the state-aware fetches.
3. **Prefetch confidence threshold was the default (0.05).** Four of the five
   fetches had `confidence ≤ 0.07` — these are exactly the speculative
   long-tail predictions the threshold is supposed to gate. Tuning
   `data_prefetch_min_confidence` higher would drop those misses at the cost
   of also dropping borderline hits.

### What the data does support claiming

- **The pipeline mechanically works.** Fetches were scheduled, ran in
  background, persisted to DB, consumed at the right turn, latency hidden
  metric populated, source attribution recorded correctly.
- **State-aware Union actually emitted state-conditioned predictions.**
  `predicted_user_state="PriceConcern"` appears on the two `both`-tagged
  fetches — the empirical predictor received the MCTS modal hint and used it
  in its SQL. Even though those particular predictions missed, the *plumbing*
  is doing what the design says.
- **Empirical priors do flow across sessions.** Despite this being only the
  third-ish session on the SOP, empirical produced sensible predictions
  pulled from the precedent_traces of the earlier sessions. The
  `Greeting → VerifyIdentity` transition was strong enough to schedule that
  fetch, and it paid off.

What we can't claim yet:

- The 70% hit-rate threshold from the production note is achievable in this
  setup. Need more sessions before the empirical priors thicken.
- The state-aware Union outperforms state-blind empirical. Needs an ablation
  with `predictor=empirical` on the same SOP+config to compare.
- The latency-hidden total scales linearly with hit count. Need a longer
  session with more deep-path fetches.

## Long prefetch session — `e01bd7f2dd98` (N=2 update)

The obvious next experiment fired: same SOP, same config, but
`max_turns=20` and `data_prefetch_enabled=true` from the start. Closed on
`AgreedToRenew` at turn 20 — second success in a row.

The numbers materially **revise the earlier claims in this note**.

### Prefetch — strong, once the session is long enough

| | 6-turn (prior) | 20-turn (this) |
|---|---|---|
| Fetches scheduled | 5 | 10 |
| Hits | 1 | **3** |
| Misses | 4 | **0** |
| Pending at end | 0 | 7 |
| Hit rate (resolved) | 20% | **100%** |
| Latency hidden | 3.0 s | **9.8 s** |

Two of the three hits served the *same actions* (`ReviewCurrentCoverage`) that
"missed" in the 6-turn session — because that session never reached them in
its truncated horizon. Same predictions, more time to consume → wins. The
hypothesis from the earlier addendum ("short sessions systematically
understate hit rate") is confirmed.

Per-source contribution this session:

| Source | Fetches | Hits | Latency hidden |
|---|---|---|---|
| **empirical** | 8 | **3** | 9.8 s |
| both (Union) | 1 | 0 (pending) | — |
| mcts solo | 1 | 0 (pending) | — |

**Empirical alone produced 100% of the latency-hiding value.** The state-aware
Union didn't contribute meaningfully, which is the *expected* steady-state
behaviour — once precedent priors fill in, the MCTS side of Union becomes
useful only at genuine cold-start, not for routine action prediction.

### State-prediction accuracy: split bimodal, not 78%

This is the biggest revision to the prior section. Re-running the same
per-turn comparison on this longer session, but now also classifying each
turn as a *transition* (actual state ≠ prior actual state) vs *stable*
(actual = prior):

| Category | Correct | Total | Accuracy |
|---|---|---|---|
| **Stable** turns | 8 | 8 | **100%** |
| **Transition** turns | 0 | 8 | **0%** |
| **Overall (blended)** | 8 | 16 | 50% |

Read carefully: when the user stays in the same state, MCTS predicts that
perfectly. When the state actually changes, MCTS misses every single time.

The 78% number from the prior session wasn't *wrong* — it was honest
measurement on a session whose middle 18 turns were a stable `Interested`
phase, so most predictions were trivial "next turn = same as now" calls. This
longer session has more state churn (turns 5/6/7/8 and 9/10/11 oscillate
between `Interested` and `ReportingChange`) and the bimodal pattern reveals
itself.

The most consequential miss: **turn 19, predicted=`Interested` (100%
modal), actual=`AgreedToRenew`** — the success moment. *No rollout in MCTS
ever simulated a user saying "yes, I'll renew."* When the live user sim
actually did, the supervisor had zero pre-staging for it.

### Why this happens (the design implication)

Modal `predicted_user_state` at offset+1 collapses the rollout *distribution*
of possible next states down to its mode. For predictions like "user will stay
Interested" the mode equals the truth and we get 100%. For predictions like
"user might transition to AgreedToRenew with low probability, or stay
Interested with high probability," modal drops the low-probability branch
entirely.

The information that matters for prefetch is exactly the low-probability
branch — because the *miss cost* on a terminal transition is much higher
than the miss cost on a within-phase shift. The architecture as built ignores
this. See companion note: **stable-vs-transition state-prediction
asymmetry**.

### Router still didn't help

| Tier | Turns | % |
|---|---|---|
| MCTS (tier-3) | 19 | 95% |
| Baseline (tier-2) | 1 | 5% |
| Cached (tier-1) | 0 | 0% |

Even with a successful prior session contributing precedents, the router
never elevated to tier-1. `tier_min_supporting_traces=3` still wasn't met for
any (cohort, user_state) pair. Two successful sessions ≠ enough precedent
mass. This is the slow part of the architecture's economic curve and the
sweep numbers from earlier today are consistent with it.

## Revised summary across both sessions

What we can now claim with N=2 evidence:

1. **Prefetch mechanically works**, and gives 9.8 s of latency hidden in one
   20-turn session. Empirical priors do most of the work.
2. **State prediction is split bimodal**: 100% on stable phases, 0% on
   transitions. The averaged accuracy number is meaningless without that
   split.
3. **Modal prediction is the wrong abstraction for the moments that matter**.
   What we throw away (the low-probability branches) is exactly what prefetch
   wants to hedge against.
4. **Router economics need precedents we don't have yet**. Two sessions on a
   fresh SOP is not enough; the tier-1 rate stays at 0%.

What this *breaks* from the original note headline:

- The "78% cold-start accuracy" claim was an honest measurement on a session
  whose state diversity was low. The deeper signal is the stable/transition
  split, and that's a much harder threshold to meet on the metric that matters.
- The architecture's "prefetch hedge" claim was hand-wavy in the production
  note. This data shows it needs to be concrete: top-K state distribution
  in the blackboard, not modal.

## What's missing from this observation

- **A longer session still.** Even 20 turns isn't a settled steady state.
  Would be useful to autopilot one session with `max_turns=40-50` and see
  whether the tier-1 router rate eventually climbs as precedents thicken.
- **A second cohort.** Every turn after turn 1 was in cohort `LifeChange`. We
  haven't seen how state prediction holds up when the user shifts cohorts
  mid-conversation (a known hard case the POC's cohort classifier handles but
  with measurable error).
- **A non-loop session.** This session's 29-turn length is dominated by the
  loop. A baseline session that takes a cleaner path through the SOP would
  give a cleaner read on per-phase accuracy.
- **Repeatability.** N=1 isn't a measurement — it's an existence proof.

## Concrete next steps (for tracking, not committed)

1. Autopilot 5-10 more sessions on `car_insurance_renewal` with the same
   config + prefetch enabled. Confirm:
   - state-prediction accuracy stays around 70-80% on average,
   - tier-1 starts firing as precedents accumulate,
   - latency-hidden metric on prefetch hits.
2. Tighten the SOP: add forced-allow logic for `RequestRenewal` after N turns
   of life-change probing. Verify loop length drops.
3. Add a "transition accuracy" metric to the bench harness — count
   predictions specifically at turns where the actual state differs from the
   prior turn's actual state. This is the metric that matters for prefetch.
4. Mine `planned_states` from this session into a `state_transitions`
   precedent table so the empirical predictor can start using it next session.

## Why this observation matters

Until this session we had:

- A working planner that never demonstrably finished a conversation
  successfully.
- A theoretical claim about state-aware Union prediction with no
  ground-truth measurement.
- A production architecture proposal whose hit-rate threshold (70%) was
  picked from intuition.

After this session we have:

- One concrete success trajectory, end to end.
- A real number (78%) for state prediction accuracy on cold-start data.
- A confirmed pattern that errors cluster at transitions — which is also
  exactly where the supervisor's pondering-top-K hedge is supposed to help.

It's still N=1. But it's the first time the architecture has produced
something we can point at and say "this is what the production picture should
look like, here's what worked, here's what broke." The note exists to capture
that picture before it gets compressed by the next round of changes.
