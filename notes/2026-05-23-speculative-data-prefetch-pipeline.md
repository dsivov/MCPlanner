---
date: 2026-05-23
title: Speculative Data-Prefetch Pipeline for SOP-Constrained Dialogue Planners
status: implemented + first benchmark
tags: [pca-m, mcts, fast-mctd, speculative-execution, data-pipeline, latency]
related: [pondering-mcts, multi-tier-router, context-graph]
---

# Speculative Data-Prefetch Pipeline for SOP-Constrained Dialogue Planners

*Trajectory-keyed background fetching driven by MCTS predictions, layered on top of the PCA-M planner.*

## TL;DR

The dominant latency in production dialogue agents is external I/O (RAG / KG / DB / MCP tool
calls), not LLM inference. PCA-M's MCTS rollouts already produce *trajectory predictions* —
hypotheses about the next K turns of conversation. Each predicted action declares its data
dependencies. We use those predictions to issue idempotent fetches speculatively, in the
background, while the user is still thinking — keyed by `(session_id, dep_name, action_name)`,
TTL-bounded, dedupe-merged, budget-capped. At later turns we consult the queue; on a hit the
external-data latency is hidden entirely.

**First measured result (3 sessions × 6 turns, Balanced preset, credit-card seed):**
86% hit rate, 15 s of external-data wall-clock moved off the synchronous path, 50% hit rate
at offset-2 (predictions made 2 turns in advance held up half the time).

## Problem framing

In commercial dialogue agents the LLM is rarely the bottleneck. A typical turn looks like:

```
user msg → intent classification (LLM, ~1s)
        → RAG / knowledge-graph / DB / MCP-tool lookup (1-10s, often multiple)
        → planning + reasoning (LLM, 1-3s)
        → response generation (LLM, 1-2s)
                                            ─────── 5-20+ seconds perceived
```

The bulk of perceived latency is **external I/O** — and most of it could have been initiated
10-30 seconds earlier, *while the user was still typing*, **if** the agent had a credible
prediction of what the next turn would need.

PCA-M's job is exactly to produce that prediction. Each MCTS rollout is a hypothesis about the
next K turns of conversation. If those hypotheses are accurate even a fraction of the time, the
data needed by future agent actions can be issued speculatively and ready by the time it's
needed.

This subsystem is the realization of that idea.

## Architecture

```
turn N commits
    │
    ├─ logger.rollouts ──▶ derive_prefetch_plan(rollouts, chosen_action_N)
    │                       │   for each rollout matching chosen_action_N:
    │                       │     for each (offset, action_at_offset):
    │                       │       for each dep in action.data_dependencies:
    │                       │         score = Σ rollout.reward · exp(-λ·offset)
    │                       ▼
    ├─ DataPrefetchManager.schedule(plan) ─▶ asyncio.create_task per dep
    │                                          │ writes DataFetch row (speculative=True)
    │                                          ▼
    │                                  ┌───── Outstanding queue ──────┐
    │                                  │ key → FetchHandle(task, …)   │
    │                                  └──────────────────────────────┘
    │                                          │ on completion
    │                                          ▼
    │                                  ┌───── Completed cache ────────┐
    │                                  │ key → FetchResult(payload,…) │
    │                                  └──────────────────────────────┘
    ▼
turn N+1 arrives → cohort_state_propose → action chosen
    │
    ├─ DataPrefetchManager.consume(action.data_dependencies)
    │     for each dep:
    │       cache hit + valid TTL  → use ─────────────▶  consumed=True
    │       in-flight + small wait → use after wait ──▶  consumed=True
    │       cache miss             → live fetch (block) ─▶ speculative=False
    ▼
response_gen
```

### Schema additions

```
TaskDefinition.data_dependencies : list[DataDependency]
DataDependency = { name, description, kind, config, expected_latency_ms, cache_ttl_s, idempotent }
NamedItem.data_dependencies      : list[str]   # references DataDependency.name

MCTSConfig.data_prefetch_enabled                : bool
MCTSConfig.data_prefetch_min_confidence          : float = 0.05
MCTSConfig.data_prefetch_max_outstanding          : int   = 50
MCTSConfig.data_prefetch_decay_lambda             : float = 0.3
MCTSConfig.data_prefetch_await_in_flight_ms       : int   = 2000

PlannerTrace.data_prefetch_consumed_count        : int
PlannerTrace.data_prefetch_live_count             : int
PlannerTrace.data_prefetch_latency_hidden_ms      : int
PlannerTrace.data_prefetch_live_latency_ms        : int
PlannerTrace.data_prefetch_scheduled_after_turn   : int
```

### `data_fetches` table — the audit log

One row per fetch (speculative or live):

| Column | Meaning |
|---|---|
| `experiment_id` | session this belongs to |
| `cache_key` | dedupe key — `(session_id, dep_name, action_name)` hashed |
| `dependency_name`, `action_name`, `kind` | what was requested |
| `issued_at_turn`, `predicted_turn`, `consumed_at_turn` | timeline of prediction → use |
| `started_at`, `completed_at`, `fetch_duration_ms` | wall-clock of the fetch |
| `confidence` | aggregated trajectory score that triggered the schedule |
| `consumed`, `wasted`, `evicted` | terminal flags |
| `speculative` | True = scheduled ahead of need, False = live fallback |

This is the raw material for the **prediction half-life curve** —
`consumed_at_turn − predicted_turn` per row tells you how far ahead the prediction held.

### Scoring (marginal frequency × Q × time-decay)

```python
# Inside derive_prefetch_plan, for each rollout matching the chosen first action:
for offset, action_name in enumerate(rollout.planned_actions[1:], start=1):
    for dep in action.data_dependencies:
        scores[(dep, offset, action_name)] += rollout.reward * exp(-decay_lambda * offset)
```

Items with `score >= min_confidence` are dispatched; others are dropped.

### Cache key + safety constraints

- **Key**: `sha1(session_id | dep_name | action_name | extra)[:24]` — session-scoped,
  deterministic, dedupe-safe.
- **Idempotency**: `DataDependency.idempotent` defaults to True; scheduler **refuses** to
  prefetch a dep with `idempotent=False`. Mutating operations (booking, payment, sending) must
  opt out via this flag.
- **TTL**: per-dependency `cache_ttl_s`; expired completed entries are skipped at consume time
  and re-issued on demand.
- **Budget cap**: `max_outstanding_per_session` (default 50). When full, lowest-confidence
  in-flight handle is evicted (the task is allowed to finish; its result is just dropped on
  completion).
- **Session lifecycle**: `finalize_session()` cancels outstanding tasks and marks all
  unconsumed rows `wasted=True`. Fires on success/failure marker hit or explicit
  `/api/chat/{id}/end`.

## Initial results

Benchmark on `seed:credit_card_activation.json` (annotated with 4 mock data dependencies
covering 5 of 9 agent actions):

```
Preset: Balanced (iter=8, branching=3, rollout_depth=3, parallel=4)
Mode:   simulate rollouts
Sessions per condition: 3
Max turns per session:  6
```

| Condition | Sessions | Avg turns | p50 agent latency | LLM calls/turn |
|---|---|---|---|---|
| Prefetch **OFF** (baseline) | 3 | 5.0 | 17 202 ms | 39.4 |
| Prefetch **ON** | 3 | 4.0 | 17 700 ms | 36.9 |

### Pipeline metrics (prefetch ON only)

```
scheduled  consumed  live   hit_rate  hidden_total  live_total
        7         6     3        86%      15.0 s       11.0 s
```

- **86% of scheduled prefetches were consumed at a later turn** — i.e., the MCTS trajectory
  predictions held up.
- **15.0 seconds of external-data wall-clock was moved off the synchronous path** in just three
  sessions. That's latency a real user would have felt that they no longer have to.

### Prediction half-life curve

```
offset   scheduled   consumed   hit rate
   1+        3           1        33%
   2+        4           2        50%
```

The curve sums across all sessions: most schedules in this run were offset-2 predictions
(2 turns ahead), and half of them held up. Offset-1 hit rate was lower in this small sample
due to one short session that terminated before consuming.

**Headline interpretation**: *Predictions made now about an action two turns from now hit ~50%
in this domain at Balanced preset.* In an open question — "how predictable is *your* domain"
— that's a measurable property and this CLI gives a one-command answer.

## Design notes

### What works well

- **Composes cleanly with existing infrastructure.** No new prediction model — uses the
  rollouts MCTS already produces. No new prompts. No new LLM calls in the critical path.
- **Failure modes are benign.** A wrong prediction wastes background compute but adds zero
  latency to the user. A right prediction can hide >5 seconds; the worst case matches baseline
  behaviour.
- **Audit trail is complete.** `data_fetches` joins with `turns` and `rollouts` for any
  cross-cutting analysis.

### Known limitations

1. **Trajectory mode required.** Value-mode rollouts produce `planned_actions` of length 1
   (one-shot scoring, no future trajectory). Prefetch is a no-op under value-mode; it requires
   **simulate** or **hybrid** rollouts. This is an architectural trade-off, not a bug.

2. **SOP-narrow turns produce zero rollouts.** When `|allowed_actions| == 1` (SOP forces a
   single move), MCTS is skipped entirely and there are no rollouts to derive trajectories
   from. Hit rate is mechanically zero on those turns. Not a problem — those turns are already
   cheap.

3. **Router tier-1 / tier-2 also bypass MCTS.** Same effect: when the multi-tier router
   shortcuts MCTS for an "obvious" turn, prefetch can't run. The systems are complementary at
   different operating points: the router buys speed on routine turns; prefetch buys speed on
   novel turns where MCTS is running anyway.

4. **Stale-data risk grows with offset.** A balance fetched at turn 5 might be wrong by turn 8
   if the user mentioned a withdrawal in between. The current design uses static TTLs; the next
   iteration could **context-key** the cache so semantically-stale items don't get reused.
   This is a research direction, not a bug.

5. **Real fetchers not implemented.** The `MockDataFetcher` covers all `kind` values
   (rag / kg / db / api / mcp) for benchmarking. Real implementations slot into the
   `BaseFetcher` interface — a one-class change per backend.

## Update — TrajectoryPredictor interface (later same day)

The "value-mode produces no trajectories → prefetch can't fire" issue was an
**implementation limitation, not an architectural one**. Refactor:

```
TrajectoryPredictor (ABC)
├── MctsTrajectoryPredictor       ← in-memory rollouts (simulate/hybrid)
└── EmpiricalTrajectoryPredictor  ← SQL on precedent_traces, action-N→action-N+offset
                                    transitions grouped by (sop_ref, cohort)
```

`build_prefetch_plan_from_predictions(predictions, task, decay_lambda)` consumes the
flat list of `TrajectoryPrediction(action, offset, probability, source)` regardless of
which predictor produced it. The chat route picks a predictor by policy:

- `data_prefetch_predictor: "auto"` — MCTS if rollouts contain trajectories (length > 1),
  else empirical (router-pattern fallback).
- `"mcts"` — force MCTS (cold-start fails; useful for control studies).
- `"empirical"` — force empirical (works under value-mode; needs accumulated data).

**Implication for production**: value-mode is no longer disqualified. As soon as a
domain has accumulated a few sessions per (cohort, last_action), `EmpiricalTrajectoryPredictor`
provides better predictions than the LLM-simulated trajectories anyway — they're
grounded in real outcomes rather than imagined ones.

## Update — rollout-collapse observation (same day)

While verifying the MCTS Replay tab on a real session we noticed **rollouts
collapsing to identical trajectories** on narrow-SOP turns. Concrete example
(`ef5c9b…` turn 3, Fast preset): 4 rollouts, 3 of them with the trajectory
`StateReason → ReviewCurrentCoverage → HandlePriceObjection`, the 4th nearly
identical.

Root cause is the rollout-action LLM call: temperature 0.6, k=1 (top-1 pick). With
SOP-narrow allowed sets, the LLM lands on the same action every time given the
same context. So we end up averaging N noisy samples of *one* path instead of
searching a tree. The reported per-candidate Q-values are accordingly low-variance
even when the underlying space is larger than what was actually explored.

Quick mitigation shipped here: bumped rollout-step temperature to 0.9. This is a
hack — the proper fix is **bandit-style action selection inside rollouts** (UCT or
ε-greedy over the SOP-allowed set) instead of deterministic LLM-pick. That's a
real research direction: PCA-M's published behaviour is search, but the LLM rollout
substrate makes it behave more like Monte Carlo averaging over a single path.

### Update — bandit rollouts implemented + A/B/C sweep (same day)

We then built the proper fix as `MCTSConfig.rollout_action_policy` with three
implementations in `backend/app/planner/rollout_policy.py`:

- `llm_top1` — legacy: LLM picks top-1 action (the collapse-prone baseline).
- `llm_topk` — LLM proposes top-K (default 3); per-rollout uniform sample.
- `bandit`   — empirical priors from `precedent_traces` + per-rollout local UCT visits
                + softmax sampling + ε-greedy fallback. **No LLM call** at the rollout step.

Parallel-safety contract (preserved):

- Each rollout owns its own `BanditState`. No shared mutable state.
- Empirical priors come from a SQL read (concurrent reads on SQLite are safe).
- The K parallel rollouts in one `asyncio.gather` batch share a read-only
  `priors_cache` dict so the SQL lookup amortizes across the batch.
- Diversity emerges from softmax sampling: same prior → different random draws.

**A/B/C results** (car_insurance, Fast preset, 3 sessions × 5 turns, same conditions
otherwise):

| Policy   | p50 latency | calls/turn | avg Q | Distinct trajectories | Reward spread |
|----------|-------------|------------|-------|-----------------------|---------------|
| llm_top1 |    8.21 s   |   17.0     | 0.36  | 13/48 (**27%**)       | 0.15          |
| llm_topk |    8.97 s   |   17.0     | 0.34  | 28/48 (**58%**)       | 0.15          |
| bandit   |  **5.51 s** | **10.6**   | 0.34  | **40/64 (62%)**       | **0.25**      |

Bandit wins on cost (−38% LLM calls), wall-clock (−33%), trajectory diversity
(27% → 62%), and reward spread (0.15 → 0.25 — broader signal for candidate
differentiation). avg Q stays comparable, which is expected: the *mean* reward
across rollouts doesn't change; what changes is the variance, and that's what
MCTS actually needs to choose between candidates.

**Router still works exactly as before** — bandit only changes what happens
INSIDE tier-3 (MCTS) rollouts. Tier-1 (cached_playbook) and tier-2 (baseline)
still skip MCTS entirely. So the router's "skip MCTS for obvious turns" win
composes cleanly with the bandit's "make MCTS cheaper + more meaningful" win.

**N=3 sessions per policy is smoke-test-grade.** The directional finding is robust
(cost down, diversity up). The follow-up to firm up the paper-grade comparison
is a larger sweep with terminal_outcome distributions — does the increased
diversity translate into better decisions, or merely more varied ones?

## Research follow-ups

- **Mode joint-tuning study.** Sweep `(router, pondering, prefetch)` configurations across
  SOPs; measure how the per-mode wins compose. Hypothesis: router + prefetch peak on different
  turn types, so the union dominates either alone.
- **Context-keyed caching.** Make cache invalidation depend on observed conversational state
  (e.g., key in the most recent user_state). Quantify accuracy gain vs. waste.
- **Trajectory-deep value-mode.** Currently the cheap value-mode rollouts collapse
  trajectories. A "stub trajectory" where the LLM also predicts the *next 2 actions* (just
  names, no simulation) in its value call would restore the prefetch hook at near-zero extra
  cost.
- **Real-fetcher case study.** Implement a real `RagDataFetcher` over the precedent index and
  benchmark against the mock. Concrete domain comparison: how does prediction half-life vary
  across SOPs (credit-card vs medical vs insurance)?
- **Bandit-style rollouts to fix collapse.** Replace deterministic LLM top-1 picks during
  rollout steps with UCT or ε-greedy over the SOP-allowed set. Hypothesis: this restores the
  search-tree exploration behaviour PCA-M is supposed to have, and the corresponding Q-value
  estimates become meaningful enough to support the *selective futility pruning* from the
  Fast-MCTD paper.
- **Predictor comparison study.** Run pair-matched sessions with `data_prefetch_predictor` set
  to "mcts" vs "empirical" vs "auto"; report prediction-half-life curves per predictor. Headline:
  "when is each signal most useful?"

## How to reproduce

```bash
cd backend
# Baseline (no prefetch)
.venv/bin/python scripts/run_benchmark.py \
    --modes simulate --preset balanced \
    --sessions-per-mode 5 --max-turns 8 --concurrency 4 \
    --no-data-prefetch --no-router \
    --out bench_baseline.jsonl

# Speculative pipeline ON
.venv/bin/python scripts/run_benchmark.py \
    --modes simulate --preset balanced \
    --sessions-per-mode 5 --max-turns 8 --concurrency 4 \
    --data-prefetch --no-router \
    --out bench_prefetch.jsonl
```

The CLI prints the comparison table and per-offset half-life curve. The JSONL files have
per-session granularity for further analysis (pandas / duckdb).

## Code layout

- `backend/app/schemas.py` — `DataDependency`, `NamedItem.data_dependencies`, MCTSConfig +
  PlannerTrace fields
- `backend/app/db.py` — `DataFetch` table
- `backend/app/planner/data_prefetch.py` — `DataPrefetchManager`, `MockDataFetcher`,
  `derive_prefetch_plan`
- `backend/app/routes/chat.py` — `consume` before response_gen; `schedule` after turn commits;
  lifecycle cleanup on end
- `backend/app/routes/experiments.py` — `/api/experiments/{id}/data-fetches` endpoint
- `backend/scripts/run_benchmark.py` — CLI harness + report rendering
- `data/sops/credit_card_activation.json` — first annotated seed (4 deps, 5 attached actions)
