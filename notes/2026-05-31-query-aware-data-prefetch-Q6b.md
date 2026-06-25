---
date: 2026-05-31
title: Query-aware data prefetch (Q6b) — moving from action-keyed to question-keyed retrieval
status: design proposal — biggest open architectural lever for closing milestone (A) for production
tags: [data-prefetch, rag, query-prediction, supervisor, weak-agent, blackboard, milestone-A]
related: [supervisor-research-framing-and-confirmation-criteria, speculative-data-prefetch-pipeline, voice-agent-production-architecture, stable-vs-transition-state-prediction-asymmetry]
---

# Query-aware data prefetch (Q6b)

*The single biggest architectural lever left for closing milestone (A) at
production grade: stop fetching generic action-keyed records and start
fetching the specific answers to specific questions the supervisor predicts
the user is about to ask. The rollout's currently-discarded simulated user
utterance becomes the load-bearing signal.*

## TL;DR

Today the supervisor's data-prefetch system is **action-keyed**: when MCTS
predicts action `PitchActivation` at offset+1, it fetches the dependencies
that action declared in the SOP (`customer_account`, `tailored_offer`).
Those returns are *generic* — the same canned record regardless of what
the user is about to ask.

In production, real RAG / KG queries are **question-keyed**: the user says
"Is windshield damage covered if I park on the street?" and the agent needs
a RAG hit specific to that question. Mood and cohort condition over many
possible questions; they're too coarse to drive the actual retrieval.

The architectural extension this note proposes:

1. **Capture and propagate the rollout's simulated user utterance** —
   we already generate these in `simulate_user_with_state` and currently
   discard them. They become the predicted *question*.
2. **Add query templates to `DataDependency`** so the supervisor can form
   parameterized queries from rollout-derived signal.
3. **Replace `MockDataFetcher` for `kind=rag` deps** with a real fetcher
   that takes a query string and returns embedded-search results.
4. **Cache results by `(dep, action, query_hash)`** so repeated/near-
   duplicate predicted questions reuse work.

This converts data-prefetch from "warm up generic records" to **"answer
the question before it's asked"** — the production-realistic claim
needed to confirm milestone (A) at scale.

## The distinction this note rests on

Mood-aware *precedent* retrieval (Q6a, already shipped) sharpens **response
style** — what tone the agent uses. It does not sharpen **data retrieval**
— what *facts* the agent has on hand to respond with.

| Retrieval target | Today's conditioning | Sufficient? |
|---|---|---|
| Precedent style block | `(cohort, mood, embedding_sim)` | Adequate for tone |
| Generic record (`customer_account`, `policy_record`) | `action_name` only | Adequate, the record itself is the same per session |
| **RAG/KG answer to a specific user question** | `action_name` only | **Inadequate** — same action covers many user questions |

The last row is the gap. It's the entire reason real production voice
agents need a query-aware data plane.

## Concrete real-data example

From the autopilot session `a0999b2e155d`, the rollout user-simulator
generated these utterances at depth-2/3 across mood-diverse rollouts:

| Mood | Predicted user utterance (from rollout) | What an action-keyed fetch would do | What a query-aware fetch should do |
|---|---|---|---|
| `transactional` | *"I'm ready to finalize my policy renewal. Please let me know the next steps to complete the process."* | Return canned `policy_disclosure_cache` | Return canned `policy_disclosure_cache` (no difference; question is procedural) |
| `anxious_about_cost` | *"I'm really anxious about the potential increase in my premium due to my new vehicle. Can you provide some reassurance?"* | Return canned `tailored_offer` (about a different topic) | RAG query: *"premium impact vehicle change reassurance"* → docs on new-vehicle rate calculation, telematics discount, similar-customer outcomes |
| `informational` | *"Can you confirm if my premium will remain stable through the renewal?"* | Same canned `tailored_offer` | RAG query: *"premium stability renewal"* → policy renewal stability rules + recent rate-change history |
| `stressed_general` | *"I'm still feeling overwhelmed, but I want to make sure I get the best deal possible. I hope the telematics program helps reduce my premium."* | Same canned `tailored_offer` | RAG query: *"telematics discount eligibility quick setup"* → fast-track enrollment doc |

Same action (`PitchDiscount`) in all four cases. Same generic
`tailored_offer` returned by the existing mock. **The four user questions
are entirely different.** A query-aware system would pre-fetch four
different RAG results in parallel, each keyed to one of the four
mood-branched predicted questions.

When the actual turn fires and the live user matches one of the predicted
moods/questions, the right pre-fetched answer is ready. When they don't
match, fall back to live fetch (current behaviour).

## The architectural extension — diagrams

### Today's pipeline (action-keyed)

```
   turn N completes
        ↓
   MCTS rollouts predict planned_actions = [..., "PitchDiscount", ...]
        ↓
   for each predicted action at offset+K:
     look up SOP.agent_actions[action].data_dependencies
        ↓                                              ┌─ user_text generated
   for each (dep_name):                                │  by simulate_user_with_state
     cache_key = hash(session, dep_name, action)      │  is computed and DISCARDED.
     issue fetch via _FETCHERS[dep.kind]              │
        ↓                                              │  ❌ wasted signal
   MockDataFetcher.fetch(dep, session_id, action_name)
        ↓
   sleep(dep.expected_latency_ms)
   return dep.config.text  ← canned, action-keyed
        ↓
   write DataFetch row, cache by (dep_name, action_name)
```

### Proposed pipeline (query-aware)

```
   turn N completes
        ↓
   MCTS rollouts predict trajectory:
     planned_actions      = [..., "PitchDiscount", ...]
     planned_states       = [..., "PriceConcern", ...]
     planned_user_texts   = [..., "I'm really anxious about the premium...", ...]
                              ↑ NEW — was discarded; now captured
        ↓
   for each predicted (action, state, mood, user_text) at offset+K:
     look up SOP.agent_actions[action].data_dependencies
        ↓
   for each (dep_name):
     if dep.query_template is set:
         query = render_query_template(
             dep.query_template,
             user_text=user_text,
             cohort=cohort,
             mood=mood,
             state=state,
             action=action,
         )
         cache_key = hash(session, dep_name, action, query)
     else:
         query = None
         cache_key = hash(session, dep_name, action)   ← old behaviour fallback
        ↓
   issue fetch via _FETCHERS[dep.kind]
   RagFetcher.fetch(dep, session_id, action_name, query=query)
        ↓
   if query: run semantic search over corpus, return top-K docs
   else:     return dep.config.text (legacy)
        ↓
   write DataFetch row, cache by (dep_name, action_name, query_hash)
```

### What the blackboard entry looks like after this change

```jsonc
{
  "session_id": "...",
  "predicted_turn": 5,
  "branches": [
    {
      "user_state": "PriceConcern",
      "mood":       "anxious_about_cost",
      "action":     "PitchDiscount",
      "instruction": "...",                 // milestone (B) — separate
      "data_payloads": {
        // NEW: keyed by (dep, query_hash), not just dep
        "rag:premium_impact_vehicle_change_reassurance": {
          "docs": ["...new-vehicle rate calc doc snippet...",
                   "...telematics-discount-eligibility...",
                   "...similar-customer-outcome stats..."],
          "query":        "premium impact vehicle change reassurance",
          "source":       "tailored_offer",
          "ttl_remaining_s": 235
        }
      }
    },
    { /* branch for mood=informational, different RAG query, different docs */ }
  ]
}
```

The voice agent reads the branch that matches the live (cohort, mood,
state), and gets back **the docs that actually answer the question that
was asked**, not a generic blob.

## Schema changes

### `DataDependency` — add `query_template`

```python
class DataDependency(BaseModel):
    name: str
    description: str = ""
    kind: Literal["mock", "rag", "kg", "db", "api", "mcp"] = "mock"
    config: dict = Field(default_factory=dict)
    expected_latency_ms: int = 1000
    cache_ttl_s: int = 300
    idempotent: bool = True
    # NEW: parameterized query template. When present, the supervisor renders this
    # with rollout-derived signal (predicted user_text, mood, cohort, action) and
    # passes the rendered string to the fetcher. When absent, falls back to the
    # legacy action-keyed canned fetch.
    #
    # Supported placeholders:
    #   {user_text}   — the rollout's predicted user utterance at the offset turn
    #   {cohort}      — runtime-classified cohort
    #   {mood}        — runtime-classified mood
    #   {state}       — predicted user_state at the offset turn
    #   {action}      — predicted agent action at the offset turn
    query_template: str | None = None
```

### Real example for `tailored_offer`

```jsonc
{
  "name": "tailored_offer",
  "kind": "rag",
  "expected_latency_ms": 4200,
  "cache_ttl_s": 300,
  "idempotent": true,
  "query_template": "Customer in cohort {cohort} with mood {mood} just said: {user_text}. What products, discounts, or program benefits would address their concern?",
  "config": { "rag_index": "products_and_offers_v3", "top_k": 5 }
}
```

At rollout-prediction time, this template is rendered four ways (one per
mood-diverse rollout sample), producing four distinct queries. Four
parallel RAG fetches; whichever predicted (cohort, mood) actually matches
at the next turn gets a cache hit.

### Rollout records — capture `planned_user_texts`

The rollout user-sim already generates `user_text` at each depth; today
that text only goes into the temporary in-memory history during the
rollout and is discarded at rollout end.

```python
class RolloutOutcome:
    planned: list[str]                # actions (already)
    planned_states: list[str]         # user states (already)
    planned_user_texts: list[str]     # NEW — aligned with planned_states
                                      # planned_user_texts[i] = simulated user
                                      # reply at depth i
```

DB column on `rollouts` for analysis:
```python
class RolloutRecord(Base):
    ...
    planned_user_texts = Column(JSON, nullable=False, default=list)
```

Alembic migration straightforward.

### Predictor — emit `predicted_user_text` per (offset, action)

```python
@dataclass
class TrajectoryPrediction:
    action: str
    offset: int
    probability: float
    predicted_user_state: str | None = None
    predicted_user_state_dist: dict[str, float] = field(default_factory=dict)
    # NEW: the user text the rollout generated at this (offset, action). Used as
    # the seed query for query-aware data prefetch.
    predicted_user_text: str | None = None
```

Aggregation strategy: when multiple rollouts converge on the same
(offset, action), keep the highest-reward rollout's text as the
representative. When they diverge, top-K state distribution already
captures the branching; texts can be unioned and the fetcher can run a
query per text.

### `PrefetchPlanItem` — carry the rendered query

```python
@dataclass
class PrefetchPlanItem:
    dependency_name: str
    action_name: str
    confidence: float
    predicted_turn_offset: int
    predictor_source: str
    predicted_user_state: str | None
    # NEW
    rendered_query: str | None = None     # populated when dep has query_template
```

### `DataFetch` row — log the query for analysis

```python
class DataFetch(Base):
    ...
    query_text = Column(Text, nullable=True)    # NEW
    query_hash = Column(String, nullable=True, index=True)  # NEW
```

So we can post-hoc compare: "at turn N+1, the LIVE user said X; the
supervisor had pre-fetched RAG results for predicted query Y; how similar
are X and Y?"

## Implementation order

| Step | Effort | Unlocks |
|---|---|---|
| 1. Capture `planned_user_texts` in rollouts; persist on `RolloutRecord` (with Alembic) | ~30 min | Diagnostic + future query plumbing |
| 2. Add `predicted_user_text` to `TrajectoryPrediction`; have `MctsTrajectoryPredictor` populate it | ~30 min | Predictor emits texts |
| 3. Add `query_template` to `DataDependency`; render at plan-build time | ~30 min | Plan items carry rendered queries |
| 4. Add `query` parameter to `BaseFetcher.fetch`; thread through `DataPrefetchManager` | ~20 min | Plumbing complete |
| 5. Replace `MockDataFetcher` (for `kind=rag`) with `RagFetcher` (embed → search over a small fixture corpus) | ~1-2 h | First real query-aware fetch |
| 6. Add `query_text`, `query_hash` columns on `DataFetch` (Alembic) | ~15 min | Analysis substrate |
| 7. Seed `query_template` on 1-2 deps in `car_insurance_renewal.json` for verification | ~15 min | Real test setup |
| 8. Autopilot + analyze: did query-aware prefetches hit? How similar was predicted vs live user text? | ~1 h | The measurement |

Total ~3-5 hours for first end-to-end. Then iterate based on the
predicted-vs-live similarity numbers.

## The measurement that closes Q6b

For each speculative query-aware fetch:

1. Record the **predicted_query** (rendered from rollout's user_text).
2. When the actual matching turn fires, classify the live user_text into
   the same (cohort, mood, state) bucket.
3. Compute a similarity metric between predicted and live user_text:
   - **Strict**: BLEU or exact phrase overlap (sparse)
   - **Embedding**: cosine similarity of OpenAI embeddings (continuous)
   - **LLM-judge**: "would these two queries return the same documents from
     the same RAG corpus?" yes/no (binary, expensive)
4. Compute **doc-overlap** between predicted-query RAG results and
   live-query RAG results (the real production-relevant metric — if both
   queries return ≥50% of the same docs, the cached answer is useful).

| Metric | Threshold for "Q6b confirmed" |
|---|---|
| Similarity (embedding cosine) avg ≥ 0.7 | Architecture has signal |
| Doc-overlap ≥ 50% on hits | Architecture has production-grade value |
| Live-fetch fallback rate ≤ 30% | Cache pays off more often than it doesn't |

Concrete: if 60%+ of cached query results overlap ≥50% with what live
RAG would have returned, we can claim **"query-aware data prefetch is
production-viable"** — milestone (A) closed for real-world deployments.

## Where this leaves mood

Mood is **still useful** but it's *one parameter in the query template*,
not the whole signal. The biggest single input is `{user_text}` from the
rollout. Mood, cohort, state shape *how* the question is formed and
*which* documents to bias toward — but the *what is being asked* lives
in the user_text.

So mood-driven rollout diversity (Phase 1) is even more valuable under
query-aware prefetch than under action-keyed prefetch: different moods →
different simulated user_texts → different rendered queries → different
parallel RAG fetches → wider hedge across the K most plausible
questions. The architecture composes nicely.

## Questions for further discussion

### Q-A — Real corpus or mocked?

The verification experiment can run on:

  - **(a)** A small hand-curated fixture corpus (a few dozen policy docs
    for car_insurance). Cheap to set up, faithful enough to demonstrate
    the mechanism.
  - **(b)** A real production-style RAG corpus (10k+ docs). Honest test
    of retrieval quality but adds setup complexity.
  - **(c)** Use a public corpus (e.g., a sampled IRS form set, public
    insurance regulator docs). Realistic but not aligned with the SOP.

(a) is the minimal viable test. (b) is the strongest claim. Worth deciding
which before building.

### Q-B — Query template language

Three plausible mechanisms:

  - **(a)** Python `str.format()` with named placeholders (`{user_text}`,
    `{mood}`). Simple, brittle.
  - **(b)** A small templating layer (e.g., Jinja2) for conditionals
    and loops over predicted-state distribution. Flexible.
  - **(c)** Have the supervisor's LLM *generate* the query at plan-build
    time, given all the signals. Most flexible, most expensive (1 extra
    LLM call per dep per offset).

(a) is good enough for v1. (c) is the production sweet spot eventually.

### Q-C — Fallback when prediction has no user_text

For deps where the predicted user_text is unavailable (cold-start
rollouts, value-mode without simulated user) — do we:

  - **(a)** Skip query-aware fetch, fall back to today's action-keyed?
  - **(b)** Render the template with empty `{user_text}` and let the
    fetcher handle it?
  - **(c)** Use the cohort/mood/action description as a synthesis seed
    ("a {mood} {cohort} customer in state {state}")?

(a) is the safe default. (c) might give partial useful results.

### Q-D — Cache key — query hash or query embedding?

Two near-identical predicted queries (e.g., "is windshield covered" vs
"does the policy cover windshield damage") will produce the same RAG
results. Do we:

  - **(a)** Hash the literal query string — different strings = different
    cache entries, may duplicate work.
  - **(b)** Hash the query *embedding* (rounded), so near-duplicate
    queries share a cache slot.

(b) is smarter but adds an embed call per cache key. For low traffic (a)
is fine; for high traffic (b) saves work.

### Q-E — Query-aware fetch with no rollout simulation (value mode)?

In `rollout_mode=value`, the rollouts don't generate user text — they just
produce a single value-score per rollout. So we have no predicted
user_text. Do we:

  - **(a)** Disable query-aware prefetch under value mode.
  - **(b)** Generate the user_text via a separate cheap LLM call from the
    classified (cohort, mood, predicted_action) tuple.
  - **(c)** Use a static mood-and-cohort-conditioned query without user_text.

Worth deciding what value-mode users get.

### Q-F — Predicted vs live query similarity — what threshold matters?

I proposed 70% embedding similarity / 50% doc overlap as the bar. These
are picked from intuition, not measurement. After we run the first
experiment we'll have a concrete distribution and can decide what
threshold is meaningful. Question: should we *commit* to a threshold
ahead of time, or measure first then debate?

### Q-G — Compose with milestone (B)?

If milestone (B) — instruction prefetch — is built later, the supervisor
will be pre-generating *responses* that reference the pre-fetched data.
That implies a dependency: instructions need the data they cite to be in
the same blackboard entry. Either:

  - **(a)** Generate (B) AFTER (A) succeeds for the same predicted turn,
    referencing the pre-fetched data in the response_gen prompt.
  - **(b)** Generate (B) in parallel with (A), accept that some pre-staged
    instructions may reference data that wasn't successfully fetched.

(a) is cleaner causally, slower wall-clock. (b) is sloppier but faster.

### Q-H — Effect on cost economics

Query-aware prefetch means K *parallel* RAG queries per future turn
(one per mood-diverse rollout sample, or one per top-K state branch).
A real RAG backend may cost $0.001-0.01 per query at scale; pondering
fires this on every turn, so production costs could be 6-30× higher than
today's mock setup. Voice production may need a cost cap on number of
RAG queries per session — same shape as the existing prefetch budget cap.

## Where this leaves the research

Saving this note locks the architectural direction. The minimum viable
implementation (steps 1-8 above, ~3-5 h) closes Q6b for the test setup.
A production-grade version would add real RAG corpus integration, query
embedding-cache, cost caps, and the connection to milestone (B).

When Q6b is confirmed, milestone (A) ("data prefetch") moves from
"action-keyed, generic records" to **"question-keyed, specific answers"**
— which is the production claim that matters.
