# MCPlanner — Speculative Context for SOP-Constrained Voice Agents

> **Teaching a slow brain to help a fast mouth.**
> A research POC for low-latency, SOP-constrained voice dialogue: a fast on-path agent
> answers the user in real time while a slower off-path **supervisor** speculatively
> predicts where the conversation is going and pre-stages the data and text the agent
> will need — into a shared, session-scoped **pool**.

This repo accompanies the paper *"Speculative Context: A Pool-Based Cache for
SOP-Constrained Voice Agents"* (`paper/`) and the research write-up
*"Supervising the Fast Mouth"* (`blog/`).

- **📖 Research story (blog):** https://dsivov.github.io/MCPlanner/
- **📄 Paper (PDF):** https://dsivov.github.io/MCPlanner/paper/paper.pdf

*(published via GitHub Pages from the `docs/` folder on `main`)*

---

## The problem

A voice agent has one job that fights itself: **be fast** (sub-second, or the dialogue
feels broken) and **be right** (follow the Standard Operating Procedure, fetch the
customer's real data, don't hallucinate policy). The data lookups and careful planning
that make it *right* are exactly what make it *slow*.

## The idea: two systems, one pool

Instead of one model trying to do both, split the work across the user's natural speaking
and thinking pauses:

```
                 CRITICAL PATH (latency-bounded)
  LIVE USER ──speech──▶ WEAK VOICE AGENT ──reads──▶  ┌──────────────┐
                        (fast model)                  │  BLACKBOARD  │  session pool:
                                                       │   (POOL)     │  data + pre-staged
                 ASYNC LANE (off-path, speculative)    └──────────────┘  text · TTL · tags
  SUPERVISOR ──predicts next turns · prefetches · writes──────▲
  (slow/strong)  ──queries──▶  MCP · KG · DB · RAG ───────────┘
```

- The **weak agent** never blocks on a lookup — it reads whatever is already in the pool.
- The **supervisor** runs during the user's speech-time, predicts the next few turns over
  the SOP graph, and fills the pool ahead of need.
- The pool is **misprediction-tolerant**: a wrong guess just goes unused (TTL evicts it);
  a right guess turns a multi-second lookup into a pool hit.

## What we found

The research story (and the reason the design looks the way it does) is a sequence of
results, documented in the blog and paper:

- **The pool works.** Speculative prefetch turns blocking external-data lookups into pool
  hits across three SOPs (car-insurance renewal, credit-card activation, medical booking).
- **A null result that mattered:** a confidence-gated rerank (v2) didn't help — the cosine
  distribution didn't separate the way we assumed.
- **Removing the LLM from the rerank** (v3: cosine + dedup, a thresholded MMR) recovered
  the latency without the quality regression a naive threshold caused.
- **MCTS does *not* predict retrieval.** A decisive ablation: for predicting the agent's
  next action (which data to prefetch), a **cheap empirical predictor** — a SQL count over
  precedent traces — hits **88% recall@3** at ~0 tokens, versus **33%** for MCTS
  (~25.6K tokens) and **38%** for a naive LLM. MCTS reasons about a hypothetical future;
  retrieval needs to predict the agent's *own* learned policy, which counting memorizes.
- **The one place a generative model earns its cost** is the free-text RAG query slot
  (`{user_text}`): a single small-model call to predict the user's next utterance adds
  **+11–14pp** doc-overlap on fine-grained corpora at ~85× lower cost than MCTS.

**Net design:** empirical predictor for actions + structured params; a cheap single LLM
call only for the generative query slot; MCTS retired from the retrieval path (kept only
for cold-start planning). ~99% token reduction on the prediction path.

Background LLM work (pondering / prefetch) is bounded and preemptible via a PASTE-style
scheduler: speculative calls run on a budget and yield to the live turn.

---

## Repository layout

```
backend/    FastAPI + SQLite (aiosqlite) + OpenAI SDK — planner, supervisor, pool, scheduler
frontend/   Vite + React + Tailwind — config / chat / avatar tabs, SOP graph, live blackboard
Avatar/     Node service — GPT-Realtime (WebRTC) + TalkingHead 3D avatar, steered by the supervisor
data/        Seed SOPs (data/sops/) and RAG corpora (data/rag_corpus/); planner.db is generated locally
paper/       arXiv paper (paper.tex / paper.pdf) + figures
blog/        "Supervising the Fast Mouth" — the research story (standalone HTML)
docs/        Design notes on MCTS / rollouts / prefetch
notes/       Dated lab notebook: experiments, decisions, N=5 verifications
```

`bench_*.jsonl` at the root are benchmark outputs referenced by the dated notes.

---

## Setup

Requires **Python 3.10+** and **Node 18+**, and an OpenAI API key.

### 1. Keys

```bash
cp .env.example .env                  # repo-root key (OPENAI_API_KEY)
cp backend/.env.example backend/.env  # backend models + DB url
cp Avatar/.env.example Avatar/.env    # realtime + avatar-manager models
# then edit each .env and paste your sk-... key
```

### 2. Backend (planner + supervisor API)

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Schema changes go through Alembic (`alembic revision --autogenerate` → `alembic upgrade head`) —
never drop `data/planner.db`, it holds accumulated research traces.

### 3. Frontend (research UI)

```bash
cd frontend
npm install
npm run dev   # http://127.0.0.1:5173
```

Tabs: **Configuration** (build an SOP by chat), **Chat** (test the planner with a human or
simulated user, watch state prediction / action Q-values / pool hits per turn), and
**Avatar** (live GPT-Realtime + 3D avatar, with the SOP, MCTS actions, and blackboard
contents shown live).

### 4. Avatar service (optional, for the live demo)

```bash
cd Avatar
npm install
npm start     # serves the realtime + avatar manager (see Avatar/INSTALL.md)
```

All API keys stay server-side; the browser only ever receives a short-lived ephemeral
Realtime token.

---

## Reproducing the numbers

**On the database:** `data/planner.db` is **not shipped** — a fresh clone starts with an
empty local DB that fills as you run sessions (it accumulates research traces over time, which
is why it is gitignored, not because it is required to read the repo). The committed artifacts
that back the paper are the `bench_*.jsonl` session outputs and `paper/table1_sessions.jsonl`
(true JSONL, one session object per line — the export behind Table 1). Full benchmark
*reproduction* additionally requires OpenAI API calls and an accumulated trace DB, so the
committed JSONL artifacts are the auditable record; live re-runs will differ in the stochastic
cells (`N=5`).

The dated files in `notes/` are the lab notebook — each major claim links to a
verification run (`notes/2026-06-03-pool-cache-N5-verification.md`,
`notes/2026-06-02-data-prefetch-cross-SOP-N5.md`, etc.) and the corresponding `bench_*.jsonl`.

> The fine/large RAG corpora under `data/rag_corpus/` are **constructed stress-tests**
> (realistic situation-variants we authored to probe corpus granularity), not production
> data — labeled as such in the paper.

## Author

Dima (Dan) Sivov.
