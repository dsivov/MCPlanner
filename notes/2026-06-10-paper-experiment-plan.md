# Experiment plan — completing the paper as a real publication

Goal: take the current preprint (N=5, simulator-only, internal baselines, journey-framing)
to a credible workshop/conference paper. Experiments below are grouped by necessity.

## Tier 1 — Necessary for credibility (must-have)

### E1. Statistical power
- Re-run every headline condition at **N ≥ 20 per SOP** (currently 5).
- Report **Wilson 95% CIs** on success/hit rates, **bootstrap 95% CIs** on latency p50/p95/p99.
- Replaces every "N=5" claim. Cost: autopilot time + API spend, no new code.

### E2. Prior-art baselines for the pool claim (kills the strawman objection)
The 11%→96% number compares two of our own variants. Implement + measure four caches over
the same supervisor prediction stream:
- **B0 No-cache** — live fetch on demand (measures latency the pool hides).
- **B1 Naive per-turn RAG** — top-K cosine over a session vector store, no prefetch, no
  supervisor curation (the obvious baseline; isolates the *speculative-population* value).
- **B2 Exact-key prefetch** — current "11%" variant (kept for the prefetch-vs-key ablation).
- **B3 Pool (ours)**.
Reframes Contribution 1 from "ours vs our old thing" to "ours vs the standard approach."

### E3. A real quality metric (beyond binary session success)
Define and measure per turn:
- **Data-grounding**: on data-dependent user questions, did the reply correctly use the
  available prefetched data? (LLM-judge rubric + the SOP's ground-truth data.)
- **Task efficiency**: turns-to-terminal, count of off-topic turns.
- Redefine **effective hit rate** as "picked item was relevant AND used in the reply,"
  not the current weak "≥1 item picked."

## Tier 2 — Controlled ablations (convert the journey into science)

### E4. Rerank-design ablation grid (replaces Iterations 1/2/3 narrative)
One table, one run: **LLM rerank | cosine-only | cosine+dedup | cosine+dedup+floor**.
Columns: mean/p95 latency, quality (E3), LLM-pick overlap. Presents the design choice as a
controlled comparison instead of a chronological story.

### E5. Speculative-context prompt ablation (the honest-prompts "principle" is currently N=1)
Prompt grid across **both SOPs, N ≥ 15**: **curated | neutral | speculative | speculative+examples**.
Show the effect is consistent (ideally monotonic). This is what turns one anecdote into a
defensible principle. Add a second task/architecture if time permits (transfer evidence).

### E6. Dedup-necessity ablation
cosine-only vs cosine+dedup — shows dedup recovers the LLM's unique contribution (the ~13%
of picks the LLM made differently). Supports the "the LLM's value was dedup, not relevance" claim.

## Tier 3 — External-validity upgrades (the differentiators)

### E7. Real-human evaluation via the avatar harness (NOW FEASIBLE — we built it)
Modest study: **10-20 human testers**, each runs 2-3 SOP scenarios on the live GPT-Realtime
avatar, **A/B with vs without supervisor prefetch**. Measure perceived latency, task success,
and whether data injection helped. Single biggest credibility jump; tooling exists.

### E8. MCTS pondering under realistic inter-turn pauses
Re-run with injected **2-5 s pauses** (real speech timing). Measure pondering completion rate
+ contribution. Either promotes pondering to a contribution or scopes it out with data
(currently 98% cancelled at zero-pause).

## Tier 4 — Breadth & systems rigor

### E9. Third+ SOP
The credit_card action-proposer issue is now FIXED (SOP-graph state-prereq gating). Re-include
credit_card + ideally a 4th domain to show the architecture isn't insurance-specific.

### E10. Cost / token analysis
LLM calls/turn, tokens, $/session: pool+cosine vs LLM-rerank vs naive RAG. Systems papers
need this; we can compute it from logged llm_calls.

## Suggested order
E1 (re-run at N) + E4/E5/E6 (ablations, mostly re-presentation of re-run data) →
E2 (baselines, new code) + E3 (quality metric, new judge) → E9 (breadth) + E10 (cost) →
E7 (human study, the differentiator) + E8 (pondering).

## What this does to the paper
- E2+E3 fix the two reviewer-fatal weaknesses (strawman baseline, weak metric).
- E4+E5+E6 let us delete the iteration narrative and present clean ablations.
- E7 converts "simulator-only" from a fatal limitation into a validated result.
- E1 makes every number defensible.
