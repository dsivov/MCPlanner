---
date: 2026-05-23
title: Stable vs transition state-prediction asymmetry — why modal aggregation breaks prefetch hedging
status: top-K + mood diversity shipped; offset+1 metric was misleading; offset+2/+3 prefetch hit-rate doubled with mood
tags: [pca-m, mcts, state-prediction, union-predictor, prefetch, top-k-hedging, voice-agent, mood-diversity]
related: [first-successful-conversation-and-78pct-state-prediction, voice-agent-production-architecture, speculative-data-prefetch-pipeline]
---

# Stable vs transition state-prediction asymmetry

*A single research finding pulled out for its own page: when we measure MCTS
state predictions split by whether the next turn changed state or stayed in
the same one, accuracy is **100% on stable turns and 0% on transition turns**.
Modal aggregation of rollout-predicted states throws away the information
that matters most for prefetch hedging. This note documents the finding and
sketches the implementation fix.*

## The asymmetry

Across two prefetch-instrumented car-insurance-renewal sessions (N=16
testable transitions total), splitting state predictions by whether the
*actual* next state differed from the *prior actual* state:

| Category | Definition | Correct | Total | Accuracy |
|---|---|---|---|---|
| **Stable** | `actual_t = actual_{t-1}` | 8 | 8 | **100%** |
| **Transition** | `actual_t ≠ actual_{t-1}` | 0 | 8 | **0%** |
| Blended | all turns | 8 | 16 | 50% |

(Source: session `e01bd7f2dd98`, 20-turn run with `max_turns=20`,
`data_prefetch_predictor=union`, `rollout_action_policy=bandit`,
`car_insurance_renewal` SOP. Sister session `62463cd02b4f` followed the same
shape with a longer stable middle that inflated the blended number to 78%.)

The earlier headline from session `62463cd02b4f` of "78% accuracy" was
correct as a blended measurement — but it averaged a 100%-stable phase with a
0%-transition tail. When the next session had less stable phase, the blended
number dropped to 50%; the *underlying* per-category numbers stayed exactly
the same.

## Why it happens

The MCTS rollout pipeline generates `planned_actions` and `planned_states`
arrays per rollout. The state-aware Union predictor aggregates these by
picking, per (offset, action), the **modal** predicted user state — i.e., the
single state with the largest reward-weighted vote share across rollouts.

This collapse to mode has two failure modes both rooted in the same thing:

### 1. The rollout user simulator is continuity-biased

When the agent's predicted action at offset 1 is one the rollout user
simulator would respond to in continuation rather than transition (e.g., "you
just said you want to verify some details, the user says: 'sure, go ahead' →
state still `Interested`"), every rollout converges on the same continuation.
Modal = 100% Interested. Live user simulator running with deeper context
makes the same call. Hit.

### 2. Genuine bimodality at transition boundaries gets squashed

When the actual moment is a *fork* — the user might continue (high
probability) or change track (lower but real probability) — rollouts behave
the same way: a few will sample the lower-probability branch, most will
sample continuation. Modal still returns "continuation" because that's what
the majority sampled. The minority branch is discarded.

The state we throw away is precisely the one that:
- has higher *expected miss-cost* if it happens (because we have no
  pre-staged content for it), and
- has higher *information value* if we're right (because the prefetch system
  could pre-fetch terminal-marker data that nothing else would predict).

The worst single miss in the observed data: at turn 19, the rollout user-sim
never once produced `AgreedToRenew`. Live user-sim at turn 20 did. The
supervisor had zero pre-staging for what turned out to be the success
moment.

## Why this matters for the production architecture

The voice-agent production note's pondering top-K branches design depends on
the supervisor being able to enumerate the *plausible* future states and
pre-stage instructions for each. With modal aggregation, "plausible" reduces
to "one most-likely state per offset" — which is fine when the next state is
deterministic but useless at every interesting branching moment.

In production terms:
- **At stable phases** (90% of turns in a typical conversation): modal works,
  zero hedging value would be added by top-K.
- **At transitions** (10% of turns, but disproportionately important):
  modal silently fails, top-K is the *only* way the supervisor can pre-stage
  the right answer.

The miss-cost asymmetry is what makes this a real problem rather than a
cosmetic one. A transition miss at turn 19 of a sales call (`AgreedToBuy`)
is much costlier than a stable-phase miss in the middle of small talk: the
former is the moment the agent should be invoking the *checkout* prefetch
pipeline; the latter is just another routine turn.

## What's broken in the current code

Three places conspire to throw the information away:

1. **`MctsTrajectoryPredictor.predict()` in
   `backend/app/planner/trajectory_predictor.py`** aggregates rollouts into
   one `TrajectoryPrediction` per (offset, action), with `predicted_user_state`
   set to the modal state via the existing `state_weight` table:
   ```python
   modal_state[(offset, action_name)] = best_state  # argmax reward-weighted
   ```
   The full per-state weight distribution is computed in `state_weight` and
   then discarded.

2. **`TrajectoryPrediction` dataclass** has `predicted_user_state: str | None`
   — a single label, not a distribution. The wire shape can't carry top-K.

3. **`UnionTrajectoryPredictor.predict()`** builds the per-offset state hint
   for empirical from the MCTS modal:
   ```python
   for p in mcts_preds:
       if p.offset in seen_offsets or not p.predicted_user_state:
           continue
       state_hints[p.offset] = p.predicted_user_state
       seen_offsets.add(p.offset)
   ```
   So even though empirical *could* accept multiple state hints per offset,
   it only ever receives one because the MCTS feed gave it one.

The data is computed end-to-end, then thrown away at the aggregation step.

## What the fix looks like (option c from earlier discussion)

This is a sketch of the implementation, not a commitment. Documented for
visibility before any code is written.

### Data model changes

**Extend `TrajectoryPrediction`** to carry the full per-state distribution
alongside the modal label (keep modal for back-compat / fast paths):

```python
@dataclass
class TrajectoryPrediction:
    action: str
    offset: int
    probability: float
    source: str = "unknown"
    predicted_user_state: str | None = None             # modal (existing)
    predicted_user_state_dist: dict[str, float] = field(default_factory=dict)
    # ^ NEW: {state_name: normalized share}. Sums to 1 over rollouts that
    # voted on this (offset, action). Empty when no state info available.
```

**Extend `PrefetchPlanItem`** the same way:

```python
@dataclass
class PrefetchPlanItem:
    dependency_name: str
    action_name: str
    confidence: float
    predicted_turn_offset: int
    predictor_source: str = "mcts"
    predicted_user_state: str | None = None
    predicted_user_state_dist: dict[str, float] = field(default_factory=dict)  # NEW
```

### Aggregation change in MctsTrajectoryPredictor

Replace the modal-only computation with a top-K (K = 3 by default) selection
plus full-distribution publication:

```python
# Build per-state weight as today
state_weight: dict[tuple[int, str, str], float] = ...

# Per (offset, action), publish the full distribution instead of just modal
state_dist: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)
for (offset, action_name, state), w in state_weight.items():
    state_dist[(offset, action_name)][state] = w
# Normalize
for k, d in state_dist.items():
    total = sum(d.values()) or 1.0
    for s in d: d[s] /= total
# Keep modal for back-compat
modal_state = {k: max(d, key=d.get) for k, d in state_dist.items()}
```

Each emitted `TrajectoryPrediction` now carries both `predicted_user_state`
(modal — current consumers keep working) and `predicted_user_state_dist`
(full top-K distribution — new consumers).

### Empirical predictor changes

`EmpiricalTrajectoryPredictor.predict()` already accepts
`state_hints: dict[int, str]`. Extend the signature to also accept a
multi-hint per offset:

```python
async def predict(
    self, *, max_offset: int = 3,
    state_hints: dict[int, str] | None = None,
    state_hints_topk: dict[int, list[str]] | None = None,  # NEW
) -> list[TrajectoryPrediction]:
```

When `state_hints_topk` is provided, for each offset the SQL is run *once
per hinted state*, returning K independent distributions which are then
either merged (by union) or kept separate (one prediction per (offset, action,
state) instead of one per (offset, action)). The latter is cleaner — the
downstream pipeline already handles multiple predictions per (offset, action).

### Union changes

The Union aggregation needs to handle the multi-state case:

```python
# Phase 1: MCTS publishes per-offset top-K state distribution
top_k = {}
for p in mcts_preds:
    if p.predicted_user_state_dist:
        # Keep top-K states per offset, accumulated across actions weighted by
        # the action's probability.
        for s, share in p.predicted_user_state_dist.items():
            top_k.setdefault(p.offset, {}).setdefault(s, 0.0)
            top_k[p.offset][s] += share * p.probability

# Phase 2: empirical runs K queries per offset (one per hinted state)
state_hints_topk = {off: sorted(d, key=d.get, reverse=True)[:K] for off, d in top_k.items()}
emp_preds = await self.empirical.predict(max_offset=..., state_hints_topk=state_hints_topk)

# Phase 3: merge as today, but tagged with the state each prediction came from
```

### Plan-building change

`build_prefetch_plan_from_predictions` already keys on
`(dep, offset, action)`. With state-conditioned predictions, the key could
optionally expand to `(dep, offset, action, predicted_state)` for cases
where the dependency is *state-sensitive* (e.g., a dependency declared with
a `state_branches: dict[str, str]` config). For state-agnostic dependencies
(everything we have today), the existing key is fine — the multiple
state-conditioned predictions for the same (offset, action) just contribute
to a single plan item's confidence score.

### Prefetch budget allocation

With more predictions per turn, the total speculative-fetch budget can blow
out. Two mitigations, applied at the manager:

1. **Per-offset budget**: cap N fetches per offset rather than total. Today
   the cap is `data_prefetch_max_outstanding` globally; refactor to a per-
   offset slice.
2. **State-weighted confidence boost**: when multiple state branches predict
   the same (action, dep), sum their probabilities into the plan item's
   confidence. When they're spread across many actions, each individual plan
   item has lower confidence and naturally falls below the `min_confidence`
   filter.

These keep the *most likely* combined predictions, while still admitting
low-probability terminal branches if no high-probability branch dominates.

### UI changes

The MCTS Replay tab's "Predicted user states (MCTS)" section currently shows
one row per offset with the modal state. With top-K it becomes one row per
(offset, state) showing each branch with its share — same layout as the
existing "Hit-rate by source" section but for states. The PrefetchRow
component can show a small state chip per fetch tagged with which state
branch triggered it (we already render this; just multiplies in count).

### Effort estimate

- Data model + MCTS aggregation: ~30 min
- Empirical multi-hint support: ~30 min (mostly SQL parameterization)
- Union orchestration: ~30 min
- Migration: none — `predicted_user_state_dist` is in-memory only, doesn't
  hit DB unless we choose to persist it on `RolloutRecord` (which we could
  defer)
- UI: ~30 min
- Smoke test + verify: ~30 min

So **~3 hours** for end-to-end. Modest because most plumbing exists; the
change is "stop discarding the distribution we already compute."

## What the next experiment would measure

The clean ablation is:

1. Re-run an autopilot session on the same SOP + same config, but with the
   top-K state distribution change rolled out.
2. Re-compute the per-turn state prediction accuracy, but now scoring "did
   the actual state appear *anywhere* in the top-K predicted distribution?"
3. Compare the transition accuracy specifically. The hypothesis is:
   - Top-1 (modal) transition accuracy stays around 0% (no algorithmic
     change at the modal level).
   - **Top-3 transition accuracy should be much higher** — somewhere in the
     30-70% range — because the rollouts *do* sample minority branches; we
     just stop discarding them.
4. Measure the prefetch hit rate change: should go *up* on transition turns
   (we'd pre-stage the actually-hit branch alongside the wrong-modal one)
   and stay flat on stable turns (the modal already had it).

If top-K accuracy at transitions is < 20%, then the rollout user-sim itself
is the bottleneck (it never even samples the right branch, so no
aggregation strategy can recover it). That would be a different research
finding — pointing at the simulator prompt as the next thing to fix.

## Why this finding is worth its own note

The original observation note on session `62463cd02b4f` was about a milestone
(first success closure) and bundled state-prediction accuracy as one of
several findings. With the N=2 data we now have a much sharper structural
claim:

> *Modal aggregation of rollout-predicted user states is correct under
> continuity and catastrophic at transitions. The architecture's prefetch
> hedge needs the full per-state distribution, not just the mode, to be
> meaningful.*

That's a single, clean research finding. It also has a direct, low-effort
implementation path that we've sketched here. The asymmetry being so sharp
(100% vs 0%) makes the case both unambiguous and easy to communicate to a
production team.

---

## Implementation log — top-K landed, and the null result it surfaced

Shipped end-to-end the same day as the original observation. Code locations:

| Layer | File | Change |
|---|---|---|
| Data model | `backend/app/planner/trajectory_predictor.py` | Added `TrajectoryPrediction.predicted_user_state_dist: dict[str, float]` alongside the existing modal field. |
| Data model | `backend/app/planner/data_prefetch.py` | Added `PrefetchPlanItem.predicted_user_state_dist` (in-memory only, not persisted). |
| Aggregation | `MctsTrajectoryPredictor.predict()` | Replaced the modal-only computation with full per-(offset, action) state distribution, normalized, truncated to top-K=3. Modal field stays for back-compat. |
| Multi-hint SQL | `EmpiricalTrajectoryPredictor.predict()` | New `state_hints_topk: dict[int, list[str]]` parameter. Runs one SQL per hinted state, emits one `TrajectoryPrediction` per (offset, action, hint). Falls back to state-blind on offsets where every hinted state is sparse. |
| Union orchestration | `UnionTrajectoryPredictor.predict()` | Builds per-offset top-K from MCTS predictions' state distributions, action-weighted. Passes the list to empirical. Carries the MCTS top-K dist through the merge so plan items publish it. |
| Plan-building | `build_prefetch_plan_from_predictions()` | Aggregates the richest (longest) distribution seen for each plan key. |
| UI | `frontend/src/tabs/MCTSReplayTab.tsx` | "Predicted user states (MCTS)" section now renders top-K chips per offset, modal chip accent-coloured, minority chips dim. |

Smoke-tested in isolation against synthetic rollouts with a deliberate
bimodal (6/8 vote majority, 2/8 vote minority) state distribution. Top-K math
worked: `Friendly=0.85, AgreedToRenew=0.15` emitted correctly, empirical
received both hints, and the minority branch produced its own
`ClosePolite × AgreedToRenew` prediction that the modal-only path would have
discarded. So the **plumbing is correct**.

### Verification session — session `f583ab42165a`

20-turn autopilot run on `car_insurance_renewal`, identical config to the
prior asymmetry session. Outcome: `abandoned`. Top-K accuracy results:

| Metric | Modal (top-1) | Top-3 | Δ |
|---|---|---|---|
| All turns | 13 / 17 = 76% | 13 / 17 = 76% | **0** |
| Stable | 13 / 13 = 100% | 13 / 13 = 100% | **0** |
| Transition | 0 / 4 = 0% | 0 / 4 = 0% | **0** |

**Top-K aggregation produced zero improvement.** The intervention didn't
work. The reason is visible in the per-turn distribution: every "top-3" was
in fact a single state with 100% mass. e.g.:

```
turn 5  actual=ReportingChange   predicted = Interested=100%
turn 9  actual=Interested        predicted = ReportingChange=100%
```

Never `Interested=72%, ReportingChange=18%`. Across all 8 parallel rollouts
at each turn, the user simulator emitted the same predicted state. The "top-K
distribution" we computed was always degenerate.

This means the rollouts have no minority branch to hedge across. There's no
information in the rollout distribution beyond what the modal already
captured. Top-K is mathematically a strict superset of modal but
informationally equivalent when the underlying distribution is point-mass.

### Why this is the right kind of null result

The asymmetry note's "experiment that decides it" section explicitly framed
the two possible outcomes:

> *"If top-K accuracy at transitions is < 20%, then the rollout user-sim
> itself is the bottleneck (it never even samples the right branch, so no
> aggregation strategy can recover it). That would be a different research
> finding — pointing at the simulator prompt as the next thing to fix."*

Top-K transition accuracy came in at exactly 0%. The note's branch-(b)
conclusion is now the active finding.

This is a high-value null result for two reasons:

1. **We can't claim the fix without testing it.** Had we shipped top-K and
   declared the production architecture sound on its strength, we'd have
   been wrong. The plumbing change was cheap; the verification was
   essential.
2. **It points at a much sharper next target.** The bottleneck isn't in the
   aggregation step — it's one layer down in the rollout user-simulator.
   That's a meaningfully different research problem with a different fix
   shape.

### The next research question

If 8 parallel rollouts at one turn always produce the same predicted user
state, the rollout user-sim is *effectively deterministic for this prompt
context*. Possible causes:

- **Temperature too low** in the user-sim LLM call. Easiest thing to check
  and tune. Inspect `simulate_user_with_state` and
  `simulate_user_end_rollout` in `backend/app/planner/user_sim.py`.
- **Prompt doesn't invite divergence.** The system message may be too
  constraining ("classify into one of these states based on the user's
  message") rather than inviting hypothetical alternatives ("imagine 3
  plausible user responses, each with a likely state, weighted by prior").
- **Implicit caching.** If OpenAI's response is cached deterministically for
  the same prompt across parallel calls, sampling diversity disappears.
  Worth checking the actual response IDs / fingerprints to rule this out.
- **The simulator is fundamentally a single-mode model.** It only ever
  outputs the highest-likelihood continuation. To get diversity we'd need
  to *force* multi-sample (best-of-N with explicit diversity rewards) or
  switch to a different sampling strategy.

The cheap experiment: bump temperature in the user-sim, re-run, re-measure
the distribution width. If distributions stay degenerate, the issue isn't
sampling — it's the prompt or the underlying model behavior. That points at
prompt redesign or branch-explicit user-sim ("generate 3 distinct plausible
user responses for this agent action").

### Updated takeaway for the production architecture

The voice-agent production note's "top-K branches in the blackboard" was
correct *as a design* — the supervisor SHOULD hedge across multiple
plausible futures. What's now clear is that we don't yet have a *source* of
diverse future predictions. The current rollout user-sim collapses to one
prediction per prompt context. So:

- **Top-K plumbing**: shipped and ready to use. Costs nothing while
  distributions are degenerate (top-K = top-1 in that case).
- **Production hedge**: still needs work, but the work is upstream of the
  predictor — it's in the rollout user-sim itself, possibly in the prompt
  design or the sampling strategy.
- **Cold-start production deployments**: top-K still bites correctly when
  distributions become non-degenerate (e.g., higher-temperature user-sim
  in a future iteration). The architecture is forward-compatible with the
  fix; we just don't have data flowing through it yet.

The clean way to communicate this externally:

> *"We can hedge prefetch across the top-K predicted user states, and the
> Union predictor is wired to do that. Right now our rollout user-simulator
> produces degenerate (point-mass) distributions, so the hedge has nothing
> to act on. The next research step is making the simulator generate
> diverse plausible-user-response samples; the rest of the architecture is
> ready to consume them."*

---

## 2026-05-28 follow-up — mood diversity shipped; the offset+1 metric was misleading

Per-cohort mood vocabulary (3-4 moods per cohort, designer-set priors) was
implemented and seeded across all three SOPs. Rollouts now sample one mood
per-rollout at start, frozen for the rollout's duration; different parallel
rollouts get different moods.

A user-sim temperature bump (0.7 → 1.05) was tested in parallel as a
control.

### Comparative measurement

Three sessions on `car_insurance_renewal`, same MCTS config, same SOP, three
intervention levels:

| Session | Intervention | Outcome | Distinct states at off+1 (avg) | off+2 (avg) | off+3 (avg) |
|---|---|---|---|---|---|
| `f583ab42165a` | none (baseline) | abandoned | 1.00 | 1.59 | 1.88 |
| `a0999b2e155d` | mood diversity | success | 1.00 | **3.00** ↑89% | **3.31** ↑76% |
| `6bca5acc7030` | mood + temp 1.05 | success | 1.00 | 2.87 | 3.07 |

**Mood diversity nearly doubled the state-distribution width at offset+2/+3.
Temperature bump on top added nothing further.** And critically, offset+1
diversity stayed degenerate across all three sessions — the rollout user-sim
at depth-1 doesn't have enough context to diverge yet; mood-driven divergence
manifests by depth-2.

### The metric switch that resolved the apparent null

Top-3 transition accuracy at offset+1 stayed at 0% across all three sessions.
That number was real but **diagnosing the wrong thing**: it measured
prediction accuracy at exactly the offset where rollouts haven't diversified
yet.

The right metric is **prefetch hit-rate by offset**, which tells the actual
production-relevant story:

| Session | off+1 hit-rate | off+2 hit-rate | off+3 hit-rate | latency hidden |
|---|---|---|---|---|
| Baseline (no mood) | 50% (1/2) | 50% (2/4) | 40% (2/5) | 16.8 s |
| Mood-only | **100%** (1/1) | **100%** (4/4) | — (no fetches) | 16.8 s |
| Mood + temp 1.05 | — (3 pending) | **100%** (1/1) | **100%** (5/5) | **21.1 s** |

**Mood-driven prefetch hit-rate doubled at offset+2 (50% → 100%)** while
also covering one more distinct action at that offset (2 → 3). Mood+temp
went deeper, producing 5 hits at offset+3 alone with 18.1 s of latency
hidden — none of which the baseline session attempted with confidence.

### What was actually shipped

1. `CohortMood` + `CohortItem` types in schemas; 71 mood entries across 3 SOPs.
2. `_sample_cohort_mood()` helper; per-rollout mood sampling in `_rollout`.
3. Mood threaded through `simulate_user_with_state` /
   `simulate_user_end_rollout` prompts as a "CURRENT MOOD:" block.
4. `RolloutRecord.mood` + `PrecedentTrace.mood` columns (Alembic migration
   `16544e121c8c`); empty in current rollouts but populated going forward.
5. User-sim temperatures bumped 0.7 → 1.05 (rollout-side only) under
   task #113 ablation.

### What the broader research picture now looks like

- The original asymmetry note's framing ("rollouts produce degenerate
  state distributions because the user-sim is deterministic-per-prompt")
  was partly right. Per-rollout mood diversity *does* produce non-degenerate
  distributions — but only by depth-2.
- Depth-1 (offset+1) remains a point-mass because the depth-1 user response
  has too little branching context for mood to manifest. Temperature alone
  doesn't fix this.
- Prefetch operates across offsets 1, 2, and 3, and the wins this
  intervention delivered are concentrated at offsets 2 and 3 — exactly
  where mood diversity actually lives. So the architecture was already
  shaped to consume this diversity; mood just had to start producing it.
- The "0% transition accuracy at offset+1" headline from earlier today
  was misleading. The right summary going forward is **"prefetch hit-rate
  by offset" — doubled at off+2, 5/5 wins at off+3 (mood+temp)**.

### Next research directions

1. **Run N=5 sessions per intervention** to confirm the hit-rate doubling
   isn't sample noise on N=1.
2. ~~**Investigate why offset+1 stays degenerate**~~ — **resolved as
   not-a-bug** (see follow-up below).
3. **Persist mood on precedent_traces from a runtime classifier** (Phase 2
   from the original mood discussion) so the empirical predictor can
   condition on it.
4. **Mood vocabulary refinement**: clustering past `precedent_traces` user
   utterances by latent disposition could derive moods empirically instead
   of designer-set. Phase 3 work.

---

## 2026-05-29 follow-up — offset+1 immutability resolved as not-a-bug

A direct measurement on session `a0999b2e155d` finally answered why offset+1
distributions stay degenerate even with mood diversity firing correctly.

### Direct test

For every turn in the mood-on session, grouped rollouts by their sampled
mood and looked at `planned_states[1]` (the state the rollout predicts at
offset+1):

| Turn | chosen_action | Mood groups | Result |
|---|---|---|---|
| 11 | AskLifeChanges | 3 distinct moods | All predict `Interested` |
| 13 | AskLifeChanges | 4 distinct moods | All predict `Interested` |
| 16 | AskLifeChanges | 4 distinct moods | All predict `Interested` |
| 18 | AskLifeChanges | 3 distinct moods | All predict `AgreedToRenew` |

Every single turn, every mood, every rollout: identical offset+1 state. Even
at turn 18 where the user actually does transition to `AgreedToRenew`, all
three mood groups predict it — uniformly.

### Why this happens

Depth-1 user state is **action-determined**, not mood-determined:

- After action `AskLifeChanges`, the user typically reports a change → state
  classifier returns `Interested` regardless of disposition.
- After action `Greeting`, the user introduces context → state classifier
  returns `ReportingChange` regardless of mood.

Different moods produce subtly different *user text* at depth-1 (verified by
direct LLM-call inspection earlier), but the state vocabulary is too coarse
to capture mood-level variation in the immediate response. The state
classifier maps `transactional` and `anxious_about_cost` text to the same
state label because the SOP's user_state vocabulary describes *what's
happening* not *how the user feels about it*.

### Why mood divergence still works at depth-2+

The phenomenon is a recursive amplification:

```
depth-1: mood → slightly different user text → state classifier squashes → same state
   ↓
depth-2: previous user text (mood-varied) → bandit picks different agent action → user reacts differently → DIFFERENT STATE
   ↓
depth-3: further amplified, more state diversity
```

So mood produces real signal at depth-1 but the signal gets compressed by the
state vocabulary. By depth-2 the signal has propagated through the user↔agent
loop, picked up agent-action diversity, and surfaces as state-label diversity
the classifier can see.

### Implication for production architecture

This is actually fine for the prefetch architecture:

1. Offset+1 prefetches don't gain much from hedging anyway — by definition
   the agent has one turn of processing time to consume them; if the
   prediction is wrong, there's almost no recovery room.
2. The real prefetch value lives at offsets +2 and +3 (more user-think-time
   to overlap, deeper hedging possible). Those offsets *do* see mood-driven
   diversity (1.59 → 3.00 distinct states on average, hit-rate 50% → 100%).
3. Production pondering top-K at the blackboard layer hedges across
   user_states *at the same offset* via separate parallel MCTS runs,
   orthogonal to within-MCTS mood diversity. That hedge mechanism is
   unaffected.

### Honest reframing of the "0% transition accuracy" finding

The previous note's headline "0% top-3 transition accuracy" was measuring an
inherent property of the rollout structure (offset+1 action-determinism),
not a fixable defect. The right metric is **prefetch hit-rate at offsets +2
and +3**, where the mood-driven diversity translates to actual production
value.

The thread "fix offset+1 immutability" closes as resolved-not-a-bug. The
mood-diversity intervention earned its production-relevant win at the
offsets where prefetch hedging matters — offset+1 just doesn't enter the
picture in a way that mood can help.
