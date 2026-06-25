# Research notes

Dated research notes for the PCA Planner POC. Each file is one experiment,
design proposal, or measured finding. Filenames are `YYYY-MM-DD-slug.md`.
Each note has a frontmatter block with `date`, `title`, `status`, `tags`,
and `related` — used for quick scanning.

## Index

| Date | Title | Status |
|---|---|---|
| 2026-05-23 | [Speculative Data-Prefetch Pipeline for SOP-Constrained Dialogue Planners](2026-05-23-speculative-data-prefetch-pipeline.md) | implemented + first benchmark |
| 2026-05-23 | [Voice Agent + Background Supervisor Architecture](2026-05-23-voice-agent-production-architecture.md) | proposal |
| 2026-05-23 | [First successful conversation closure, 78% state-prediction accuracy, and first end-to-end prefetch measurement](2026-05-23-first-successful-conversation-and-78pct-state-prediction.md) | observed (N=2) |
| 2026-05-23 | [Stable vs transition state-prediction asymmetry — why modal aggregation breaks prefetch hedging](2026-05-23-stable-vs-transition-state-prediction-asymmetry.md) | top-K + mood diversity shipped; off+1 metric misleading; off+2/+3 prefetch hit-rate doubled with mood |
| 2026-05-24 | [Voice-agent supervisor queue — kickoff and open questions](2026-05-24-voice-agent-supervisor-kickoff.md) | kickoff — research plan, no code yet |
| 2026-05-24 | [Blackboard schema v0 — contract between supervisor and voice agent](2026-05-24-blackboard-schema-v0.md) | design proposed v0 — pending review, no code yet |
| 2026-05-28 | [Blackboard schema v0 — design review](2026-05-28-blackboard-schema-design-review.md) | design review — all 5 decisions locked, ready for implementation |
| 2026-05-31 | [Supervisor research framing and confirmation criteria](2026-05-31-supervisor-research-framing-and-confirmation-criteria.md) | framing — locks terminology; defines milestones (A) data prefetch confirmed, (B) instruction prefetch not yet built |
| 2026-05-31 | [Query-aware data prefetch (Q6b)](2026-05-31-query-aware-data-prefetch-Q6b.md) | design proposal — moves data prefetch from action-keyed to question-keyed; rollout's user_text becomes load-bearing signal |
| 2026-05-31 | [Data prefetch N=5 replication — milestone (A) closed for production candidate](2026-05-31-data-prefetch-N5-replication.md) | measured (N=5) — 2 of 3 criteria pass; off+2 hit-rate regressed 100%→60% as expected; mean 25.2s latency hidden per session |
| 2026-06-02 | [Data prefetch cross-SOP N=5 — milestone (A) closed across all three seed SOPs](2026-06-02-data-prefetch-cross-SOP-N5.md) | measured (N=5 × 3 SOPs) — milestone (A) closed; off+2 hit-rate 60–72% cross-SOP; credit_card has unrelated 0% success rate worth flagging |
| 2026-06-02 | [Q6b verification — N=5 results](2026-06-02-q6b-N5-verification-results.md) | measured (N=5) — plumbing PASS; committed cosine ≥ 0.70 + doc-overlap ≥ 0.50 thresholds FAIL (observed 0.547 mean cosine, 0.157 mean Jaccard); 60% of predicted queries hit ≥1 shared doc with live; Greeting-loop fix delivered 5/5 success closure |
| 2026-06-03 | [Pool-based cache architecture](2026-06-03-pool-based-cache-architecture.md) | design proposal — replaces key-lookup with supervisor-curated pool; reframes Q6b verification; 60% "≥1 shared doc" becomes new baseline hit rate; pending build |
| 2026-06-03 | [Pool-based cache N=5 verification](2026-06-03-pool-cache-N5-verification.md) | measured (N=5) — architecture PASS (96% effective hit rate, +84pp vs key-lookup); latency FAIL (p95 2441ms vs 800ms target — fixable upstream); pool utilisation 44% mean |
| 2026-06-03 | [Sync fallback + async supervisor — remove live MCTS from critical path](2026-06-03-sync-fallback-async-supervisor-decision.md) | design decision — direction locked; small build pending (add tier3_enabled flag, route fallback through pool synthesis instead of live MCTS) |
| 2026-06-04 | [Comprehensive session summary — voice-agent supervisor architecture, blackboard schema, pool cache, sync-fallback decision, G fix, and cross-SOP verification](2026-06-04-session-comprehensive-summary.md) | summary — captures all work, decisions, builds, measurements, open issues, and plans across the session |
