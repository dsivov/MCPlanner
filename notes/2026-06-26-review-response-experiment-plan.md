# Experiment plan: response to the Codex review (2026-06-26)

The review (7/10) accepts the architecture and the candour about limits, but is explicit that
the evidence is **pilot-grade**: small N, simulator users, selected cells, internal baselines,
and a headline hit-rate stated on two denominators. This plan turns each scientific ask into a
concrete experiment. Code/artifact fixes from the review are already done (malformed JSONL,
structured failure logging, repro clarification, claim-tightening in paper Rev. 2, an analysis
bundle, and a unit-test suite); what remains below needs **runs** (OpenAI API budget / time)
or the **avatar** (humans), so it is staged for explicit go-ahead.

Priority order is by scientific leverage per the review.

---

## E-R1 — Common-metric baseline table  (HIGHEST leverage; the review's #1 ask)
**Question.** How much of the benefit is the *pool* vs. *any* retrieval layer?
**Design.** Four arms on one common metric, same SOPs, same simulator seeds:
1. no-cache (live fetch every data-dependent turn),
2. exact-key speculative cache,
3. naive per-turn session RAG (no prefetch),
4. pool prefetch (ours).
**Metric (single, shared).** *Fraction of data-dependent turns where the agent received a
relevant-and-used item before response generation.* (Requires the grounding labels of E-R3.)
**Report.** Per arm: that fraction, mean/p95 critical-path data latency, and task success — with
CIs (E-R2). This replaces the 11%/96% two-denominator framing with one apples-to-apples table.
**Cost.** 4 arms x 2-3 SOPs x N(>=20) — the biggest single API spend here. ~the dominant line item.
**Depends on.** E-R3 (the common metric); E-R2 (N + CIs).

## E-R2 — Statistical power: N>=20 + intervals
**Question.** Are the headline rates/latencies stable, or N=5 noise?
**Design.** Re-run every locked condition at **N>=20 per SOP per arm**. Wilson intervals on
rates (success, hit, grounding); bootstrap intervals on latency percentiles (p95/p99).
**Report.** Every headline number gets a CI; drop any cell that can't clear a pre-registered
inclusion rule (see E-R4). Replaces "5 of 8 successful runs" selection.
**Cost.** Multiplies whatever arms are run by ~4x vs current N=5.

## E-R3 — Per-turn response-grounding metric  (unlocks E-R1)
**Question.** Not "was an item returned" but "was the *right* item used, correctly?"
**Design.** For each data-dependent turn, a blind LLM judge (or rubric) labels a 4-level chain:
relevant item *available* -> *selected* by rerank -> *used* in the reply -> *used correctly*.
Define **effective hit rate = relevant-and-used / data-dependent turns** (replaces "returned").
**Report.** Redefine the pool result on this metric; use it as the shared metric for E-R1.
**Cost.** Judge calls per turn (cheap model) over the logged transcripts — modest, mostly offline
re-scoring of existing + new runs.
**Note.** Also covers the review's "Is cosine+dedup enough?" by scoring LLM-rerank vs cosine vs
cosine+dedup on grounding, not just latency.

## E-R4 — Pre-registered run protocol (fixes the selection caveat)
**Not an experiment — a procedure** applied to all of the above. Before running: fix N, seeds,
and an inclusion rule (e.g. "a run counts if the simulator reaches a terminal marker within K
turns; incomplete runs are reported as a separate failure column, never silently dropped").
Main results table includes *all* attempted runs; exploratory/filtered/locked are clearly
separated. The new structured failure logs (retrieval_prefetch.py) feed a failure column.

## E-R5 — Speculative-prompt framing grid (generalisation)
**Question.** Does the speculative-vs-curated framing effect generalise?
**Design.** 2x2+ grid {curated, neutral, speculative, speculative+examples} x {>=2 SOPs},
scored on grounding (E-R3) and closing-turn over-use specifically.
**Cost.** Moderate; reuses the simulator.

## E-R6 — Pondering under realistic timing
**Question.** Does background MCTS pondering help when it actually has time to finish?
**Design.** Inject 2-5 s inter-turn pauses (vs the current zero-pause autopilot that cancels
~98% of pondering). Report completion rate, hit-rate contribution, token cost, marginal benefit
over the empirical predictor.
**Cost.** Low-moderate; mostly a config change + re-run. Settles whether pondering earns its
place at all (current data can't speak to it).

## E-R7 — Human A/B via the avatar harness  (largest credibility upgrade; needs humans)
**Question.** Does prefetch help *real* voice interaction (ASR error, barge-in, hesitation)?
**Design.** A/B with vs. without prefetch through the GPT-Realtime avatar; humans complete a
fixed SOP task. Primary outcome: perceived latency / task completion; secondary: grounding.
**Cost.** Recruiting + sessions; not an API-only run. Stage last, but it is the one study that
converts "simulator-only" from a stated limitation into a validated voice-agent claim.

---

## Reproducibility deliverables (mostly done / cheap)
- [x] `scripts/regen_paper_numbers.py` — regenerates pool/regret/Table-1 numbers from committed
  `bench_*.jsonl` + `table1_sessions.jsonl`; lists what needs the (un-shipped) trace DB.
- [ ] **Sanitized trace bundle**: ship per-turn pool picks, fetch rows, rerank latencies (incl.
  the per-turn values p95 needs), terminal outcomes, exact configs — so the rerank-latency and
  ablation tables also regenerate from committed data, not API replay. This is the missing piece
  that would make every paper number machine-auditable.

## Suggested sequencing
1. E-R3 + E-R4 (define the metric + protocol) — cheap, unblocks everything.
2. E-R1 + E-R2 (the common-metric baseline table at N>=20+CIs) — the headline upgrade.
3. E-R5, E-R6 — generalisation + pondering, in parallel as budget allows.
4. E-R7 — human avatar study, once the above hold up.

## Rough cost note
The dominant spend is E-R1 x E-R2 (4 arms x >=2 SOPs x >=20 runs x ~8 turns x ~2 LLM calls/turn,
plus judge calls). Worth a budget estimate before launch — flag for decision.
