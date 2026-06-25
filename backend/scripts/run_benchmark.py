#!/usr/bin/env python3
"""Batch benchmarking harness for PCA-Planner experiments.

Runs N auto-mode sessions per (sop_ref, MCTSConfig) combination, with autopilot
on, until each session terminates (success/failure marker hit, max turns, or
explicit cap). Aggregates per-mode statistics and emits a comparison report:

  - avg LLM calls / turn          (cost)
  - avg agent latency             (perceived latency for a real user)
  - terminal outcome distribution (quality: success/failure/abandoned/open)
  - avg rationality per turn      (quality: planner's own scoring)
  - cohort hit rate from pondering (efficiency)
  - cache miss latency vs hit latency

Run with:
    cd backend
    .venv/bin/python scripts/run_benchmark.py --help

Example — sweep three rollout modes on the credit-card seed, 8 sessions each:
    .venv/bin/python scripts/run_benchmark.py \\
        --sop seed:credit_card_activation.json \\
        --modes simulate,value,hybrid \\
        --sessions-per-mode 8 \\
        --max-turns 12

The script talks to a running backend over HTTP, so start the server first:
    .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations
import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import httpx


# ---------------------------------------------------------------------------
# Defaults

DEFAULT_BASE = "http://127.0.0.1:8000"
DEFAULT_SOP = "seed:credit_card_activation.json"
DEFAULT_SESSIONS = 6
DEFAULT_MAX_TURNS = 12

PRESETS = {
    "fast":     {"iterations": 4,  "branching": 2, "rollout_depth": 2, "parallel_rollouts": 4},
    "balanced": {"iterations": 8,  "branching": 3, "rollout_depth": 3, "parallel_rollouts": 4},
    "thorough": {"iterations": 16, "branching": 4, "rollout_depth": 4, "parallel_rollouts": 8},
}


# ---------------------------------------------------------------------------
# Data model

@dataclass
class SessionResult:
    session_id: str
    turns: int
    terminal_outcome: str | None
    total_agent_ms: int = 0
    total_user_sim_ms: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    pondering_hits: int = 0
    pondering_eligible: int = 0
    per_turn_latency_ms: list[int] = field(default_factory=list)
    per_turn_calls: list[int] = field(default_factory=list)
    rationality_means: list[float] = field(default_factory=list)
    tier_counts: dict[str, int] = field(default_factory=lambda: {"cached_playbook": 0, "baseline": 0, "mcts": 0})
    # Data-prefetch aggregates
    prefetch_consumed: int = 0
    prefetch_live: int = 0
    prefetch_latency_hidden_ms: int = 0
    prefetch_live_latency_ms: int = 0
    prefetch_scheduled: int = 0
    # Per-offset hit / miss counters: offset → (consumed_at_that_offset, scheduled_at_that_offset)
    prefetch_by_offset: dict[int, dict[str, int]] = field(default_factory=dict)
    error: str | None = None


@dataclass
class ModeReport:
    label: str
    config: dict
    sessions: list[SessionResult] = field(default_factory=list)

    def aggregate(self) -> dict:
        succ = [s for s in self.sessions if s.terminal_outcome == "success"]
        fail = [s for s in self.sessions if s.terminal_outcome == "failure"]
        aband = [s for s in self.sessions if s.terminal_outcome == "abandoned"]
        open_ = [s for s in self.sessions if s.terminal_outcome is None]
        all_turns_lat = [t for s in self.sessions for t in s.per_turn_latency_ms]
        all_turns_calls = [c for s in self.sessions for c in s.per_turn_calls]
        all_rationality = [r for s in self.sessions for r in s.rationality_means]
        total_eligible = sum(s.pondering_eligible for s in self.sessions)
        total_hits = sum(s.pondering_hits for s in self.sessions)
        tier_totals = {"cached_playbook": 0, "baseline": 0, "mcts": 0}
        for s in self.sessions:
            for k, v in s.tier_counts.items():
                tier_totals[k] = tier_totals.get(k, 0) + v
        total_tier_turns = sum(tier_totals.values()) or 1
        n = len(self.sessions)
        return {
            "n_sessions": n,
            "success": len(succ),
            "failure": len(fail),
            "abandoned": len(aband),
            "open": len(open_),
            "success_rate": (len(succ) / n) if n else 0.0,
            "avg_turns": (statistics.mean([s.turns for s in self.sessions]) if self.sessions else 0),
            "avg_agent_ms_per_turn": (statistics.mean(all_turns_lat) if all_turns_lat else 0),
            "p50_agent_ms_per_turn": (statistics.median(all_turns_lat) if all_turns_lat else 0),
            "avg_llm_calls_per_turn": (statistics.mean(all_turns_calls) if all_turns_calls else 0),
            "avg_rationality_per_turn": (statistics.mean(all_rationality) if all_rationality else 0),
            "pondering_hit_rate": (total_hits / total_eligible) if total_eligible else 0.0,
            "tier_cached_pct": round(100 * tier_totals["cached_playbook"] / total_tier_turns, 1),
            "tier_baseline_pct": round(100 * tier_totals["baseline"] / total_tier_turns, 1),
            "tier_mcts_pct": round(100 * tier_totals["mcts"] / total_tier_turns, 1),
            "total_tokens_in": sum(s.total_tokens_in for s in self.sessions),
            "total_tokens_out": sum(s.total_tokens_out for s in self.sessions),
            # Data prefetch
            "prefetch_consumed": sum(s.prefetch_consumed for s in self.sessions),
            "prefetch_live": sum(s.prefetch_live for s in self.sessions),
            "prefetch_scheduled": sum(s.prefetch_scheduled for s in self.sessions),
            "prefetch_latency_hidden_ms_total": sum(s.prefetch_latency_hidden_ms for s in self.sessions),
            "prefetch_live_latency_ms_total": sum(s.prefetch_live_latency_ms for s in self.sessions),
            "prefetch_by_offset": self._aggregate_offset_curve(),
        }

    def _aggregate_offset_curve(self) -> dict[int, dict[str, int]]:
        agg: dict[int, dict[str, int]] = {}
        for s in self.sessions:
            for offset, slot in s.prefetch_by_offset.items():
                cur = agg.setdefault(offset, {"scheduled": 0, "consumed": 0})
                cur["scheduled"] += slot["scheduled"]
                cur["consumed"] += slot["consumed"]
        return agg


# ---------------------------------------------------------------------------
# Driver

async def fetch_calls_for_turn(client: httpx.AsyncClient, base: str, exp_id: str, turn_id: str) -> int:
    """Count llm_calls rows attributed to this turn. We use the /api/experiments endpoint to
    get the turn list with their ids, then ask for llm-calls and filter."""
    # Not strictly needed — we can also rely on trace.tokens_in/out. Compute from llm-calls
    # so the cost number includes auto_user + embed + everything attributable to the turn.
    r = await client.get(f"{base}/api/experiments/{exp_id}/llm-calls", params={"limit": 2000})
    if r.status_code != 200:
        return 0
    rows = r.json()
    return sum(1 for row in rows if row.get("turn_id") == turn_id)


async def run_one_session(
    client: httpx.AsyncClient, base: str, *, sop_ref: str, cfg: dict, max_turns: int,
    user_think_time_s: float = 0.0,
) -> SessionResult:
    # Start
    body = {
        "sop_id": sop_ref,
        "planner_mode": "mcts",
        "chat_mode": "auto",
        "mcts": cfg,
    }
    r = await client.post(f"{base}/api/chat/start", json=body)
    r.raise_for_status()
    exp_id = r.json()["session_id"]
    result = SessionResult(session_id=exp_id, turns=0, terminal_outcome=None)

    try:
        for i in range(max_turns):
            # Simulate a human user's natural pause before typing — gives background
            # pondering / prefetch a window to complete. i==0 has no prior agent reply
            # to ponder against, so we only delay on subsequent turns.
            if i > 0 and user_think_time_s > 0:
                await asyncio.sleep(user_think_time_s)
            tr = await client.post(f"{base}/api/chat/{exp_id}/turn", json={}, timeout=300.0)
            if tr.status_code >= 400:
                msg = tr.text
                if "already ended" in msg.lower():
                    break
                result.error = f"{tr.status_code} {msg[:200]}"
                break
            data = tr.json()
            tt = data.get("trace") or {}
            result.turns += 1
            agent_ms = int(tt.get("agent_duration_ms") or 0)
            sim_ms = int(tt.get("user_sim_ms") or 0)
            result.total_agent_ms += agent_ms
            result.total_user_sim_ms += sim_ms
            result.per_turn_latency_ms.append(agent_ms)
            result.total_tokens_in += int(tt.get("tokens_in") or 0)
            result.total_tokens_out += int(tt.get("tokens_out") or 0)
            if tt.get("pondering_hit_state") is not None or tt.get("from_pondering"):
                result.pondering_hits += int(bool(tt.get("from_pondering")))
            if i > 0:  # turn 0 has no precomputed cache
                result.pondering_eligible += 1
            # rationality: avg over MCTS candidates' q_value if any, otherwise 0
            cands = tt.get("candidates") or []
            if cands:
                qs = [c.get("q_value", 0.0) for c in cands]
                result.rationality_means.append(sum(qs) / len(qs))
            tier = tt.get("tier_used")
            if tier in result.tier_counts:
                result.tier_counts[tier] += 1
            # Data-prefetch per-turn aggregates
            result.prefetch_consumed += int(tt.get("data_prefetch_consumed_count") or 0)
            result.prefetch_live += int(tt.get("data_prefetch_live_count") or 0)
            result.prefetch_latency_hidden_ms += int(tt.get("data_prefetch_latency_hidden_ms") or 0)
            result.prefetch_live_latency_ms += int(tt.get("data_prefetch_live_latency_ms") or 0)
            result.prefetch_scheduled += int(tt.get("data_prefetch_scheduled_after_turn") or 0)

            # Check session terminal via /api/chat/{id}
            ss = await client.get(f"{base}/api/chat/{exp_id}")
            if ss.status_code == 200:
                sj = ss.json()
                if sj.get("terminal_outcome"):
                    result.terminal_outcome = sj["terminal_outcome"]
                    break
    finally:
        # If still open after max_turns, mark abandoned for clean accounting
        if result.terminal_outcome is None and result.turns >= max_turns:
            try:
                er = await client.post(f"{base}/api/chat/{exp_id}/end")
                if er.status_code == 200:
                    result.terminal_outcome = er.json().get("outcome") or "abandoned"
            except Exception:
                pass

    # Compute per-offset prefetch hit rate from data_fetches (the half-life curve)
    try:
        df = await client.get(f"{base}/api/experiments/{exp_id}/data-fetches")
        if df.status_code == 200:
            rows = df.json()
            for row in rows:
                if not row.get("speculative"):
                    continue
                issued = row.get("issued_at_turn")
                consumed_at = row.get("consumed_at_turn")
                if issued is None:
                    continue
                # Offset = how many turns AHEAD we predicted. Use predicted_turn - issued_at_turn.
                pred = row.get("predicted_turn")
                offset = (pred - issued) if (pred is not None and issued is not None) else None
                if offset is None or offset < 1:
                    continue
                slot = result.prefetch_by_offset.setdefault(offset, {"scheduled": 0, "consumed": 0})
                slot["scheduled"] += 1
                if consumed_at is not None:
                    slot["consumed"] += 1
    except Exception:
        pass

    # Compute per-turn LLM calls count from llm_calls (best signal of actual cost)
    try:
        # API caps limit at 2000; pull pages until we run out.
        per_turn: dict[str, int] = defaultdict(int)
        offset = 0
        page_size = 1000
        while True:
            r = await client.get(
                f"{base}/api/experiments/{exp_id}/llm-calls",
                params={"limit": page_size, "offset": offset},
            )
            if r.status_code != 200:
                print(f"   llm-calls fetch returned {r.status_code}: {r.text[:120]}", flush=True)
                break
            page = r.json()
            for row in page:
                tid = row.get("turn_id") or "_no_turn"
                per_turn[tid] += 1
            if len(page) < page_size:
                break
            offset += page_size
        result.per_turn_calls = [c for tid, c in per_turn.items() if tid != "_no_turn"]
    except Exception as e:
        print(f"   llm-calls fetch error: {type(e).__name__}: {e}", flush=True)

    return result


async def run_benchmark(
    base: str,
    sop_ref: str,
    modes: list[str],
    sessions_per_mode: int,
    base_cfg: dict,
    max_turns: int,
    concurrency: int = 8,
    user_think_time_s: float = 0.0,
) -> dict[str, ModeReport]:
    """Dispatch the full (modes × sessions) pool in parallel, bounded by `concurrency`.

    All sessions across all modes share a single global semaphore so the OpenAI rate
    limit is respected regardless of how many modes we sweep.
    """
    reports: dict[str, ModeReport] = {label: ModeReport(label=label, config={**base_cfg, "rollout_mode": label}) for label in modes}
    sem = asyncio.Semaphore(concurrency)
    done_counter = {label: 0 for label in modes}
    total = sessions_per_mode * len(modes)
    print(
        f"Dispatching {total} sessions across {len(modes)} modes with concurrency={concurrency}",
        flush=True,
    )

    async with httpx.AsyncClient(timeout=300.0) as client:
        h = await client.get(f"{base}/api/health")
        h.raise_for_status()

        async def one(mode: str, i: int):
            async with sem:
                cfg = reports[mode].config
                t0 = time.perf_counter()
                res = await run_one_session(client, base, sop_ref=sop_ref, cfg=cfg, max_turns=max_turns, user_think_time_s=user_think_time_s)
                dt = time.perf_counter() - t0
                done_counter[mode] += 1
                done_total = sum(done_counter.values())
                print(
                    f"  [{done_total}/{total}] {mode} ({done_counter[mode]}/{sessions_per_mode}): "
                    f"turns={res.turns} outcome={res.terminal_outcome} "
                    f"agent_ms_total={res.total_agent_ms} wall={dt:.1f}s"
                    + (f"  ERROR: {res.error}" if res.error else ""),
                    flush=True,
                )
                return mode, res

        # Build the full pool, shuffle interleaving so progress is visible across modes
        tasks: list[asyncio.Task] = []
        for i in range(sessions_per_mode):
            for mode in modes:
                tasks.append(asyncio.create_task(one(mode, i)))
        results = await asyncio.gather(*tasks)

        for mode, res in results:
            reports[mode].sessions.append(res)

    return reports


def format_report(reports: dict[str, ModeReport]) -> str:
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("BENCHMARK SUMMARY")
    lines.append("=" * 100)
    header = (
        f"{'mode':<10} {'n':>3} {'succ':>4} {'fail':>4} {'succ%':>6} "
        f"{'avg_turns':>9} {'p50_lat':>8} {'avg_calls':>9} {'avg_Q':>6} {'pond_hit':>8} "
        f"{'t1%':>5} {'t2%':>5} {'t3%':>5}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for label, rep in reports.items():
        a = rep.aggregate()
        lines.append(
            f"{label:<10} {a['n_sessions']:>3} {a['success']:>4} {a['failure']:>4} "
            f"{a['success_rate']*100:>5.0f}% {a['avg_turns']:>9.1f} "
            f"{a['p50_agent_ms_per_turn']:>6.0f}ms {a['avg_llm_calls_per_turn']:>8.1f} "
            f"{a['avg_rationality_per_turn']:>6.2f} {a['pondering_hit_rate']*100:>7.0f}% "
            f"{a['tier_cached_pct']:>4.0f}% {a['tier_baseline_pct']:>4.0f}% {a['tier_mcts_pct']:>4.0f}%"
        )
    lines.append("\nLegend: t1=cached_playbook, t2=baseline, t3=mcts (only counted when router is enabled)")

    # Data-prefetch detail
    any_prefetch = any(rep.aggregate()["prefetch_scheduled"] > 0 for rep in reports.values())
    if any_prefetch:
        lines.append("")
        lines.append("=" * 100)
        lines.append("DATA PREFETCH (speculative external-data pipeline)")
        lines.append("=" * 100)
        head = (
            f"{'mode':<10} {'scheduled':>10} {'consumed':>10} {'live':>6} "
            f"{'hit_rate':>10} {'hidden_s':>10} {'live_s':>9}"
        )
        lines.append(head)
        lines.append("-" * len(head))
        for label, rep in reports.items():
            a = rep.aggregate()
            sched = a["prefetch_scheduled"]
            cons = a["prefetch_consumed"]
            live = a["prefetch_live"]
            rate = (cons / sched) if sched else 0.0
            lines.append(
                f"{label:<10} {sched:>10} {cons:>10} {live:>6} "
                f"{rate*100:>8.0f}% {a['prefetch_latency_hidden_ms_total']/1000:>9.1f}s "
                f"{a['prefetch_live_latency_ms_total']/1000:>8.1f}s"
            )

        # Prediction-half-life curve: per-offset hit rate, aggregated across modes
        lines.append("")
        lines.append("Prediction half-life (per turn-offset, across all modes):")
        merged: dict[int, dict[str, int]] = {}
        for rep in reports.values():
            for offset, slot in rep._aggregate_offset_curve().items():
                cur = merged.setdefault(offset, {"scheduled": 0, "consumed": 0})
                cur["scheduled"] += slot["scheduled"]
                cur["consumed"] += slot["consumed"]
        lines.append(f"  {'offset':>6} {'sched':>8} {'cons':>8} {'rate':>8}")
        for offset in sorted(merged):
            s = merged[offset]
            r = (s['consumed'] / s['scheduled']) if s['scheduled'] else 0.0
            lines.append(f"  {offset:>6}+ {s['scheduled']:>8} {s['consumed']:>8} {r*100:>6.0f}%")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Batch benchmark for PCA-Planner rollout modes.")
    p.add_argument("--base", default=DEFAULT_BASE, help="Backend URL")
    p.add_argument("--sop", default=DEFAULT_SOP, help="sop_ref (e.g. 'seed:credit_card_activation.json' or a saved sop_id)")
    p.add_argument("--modes", default="simulate,value,hybrid",
                   help="Comma-separated rollout modes to compare (simulate/value/hybrid)")
    p.add_argument("--preset", default="fast", choices=list(PRESETS.keys()),
                   help="MCTS preset (fast/balanced/thorough)")
    p.add_argument("--sessions-per-mode", type=int, default=DEFAULT_SESSIONS)
    p.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    p.add_argument("--concurrency", type=int, default=8,
                   help="Total concurrent sessions across ALL modes (default 8). "
                        "Bound by OpenAI rate limits; lower to 2-4 if you hit 429s.")
    p.add_argument("--pondering", action="store_true", help="Enable pondering")
    p.add_argument("--no-pondering", dest="pondering", action="store_false")
    p.set_defaults(pondering=False)
    p.add_argument("--precedents", action="store_true", help="Enable precedent injection")
    p.set_defaults(precedents=True)
    p.add_argument("--router", action="store_true", help="Enable the multi-tier router")
    p.add_argument("--no-router", dest="router", action="store_false")
    p.set_defaults(router=True)
    p.add_argument("--granularity", choices=["action", "strategy"], default="action",
                   help="Plan over individual actions or Strategy groups (requires SOP.strategies populated)")
    p.add_argument("--rollout-policy", choices=["llm_top1", "llm_topk", "bandit"], default="llm_top1",
                   help="Per-step rollout action policy. 'bandit' uses empirical priors + per-rollout local visits (parallel-safe).")
    p.add_argument("--data-prefetch", action="store_true",
                   help="Enable speculative data prefetch (requires SOP.data_dependencies populated)")
    p.add_argument("--no-data-prefetch", dest="data_prefetch", action="store_false")
    p.set_defaults(data_prefetch=False)
    p.add_argument("--predictor", choices=["auto", "mcts", "empirical", "union"], default="auto",
                   help="Trajectory predictor driving prefetch scheduling. 'union' runs MCTS + empirical and merges.")
    p.add_argument("--user-think-time-s", type=float, default=0.0,
                   help="Inject artificial think-time (seconds) BEFORE each non-initial turn's POST. "
                        "Simulates a real voice user's natural pause and gives background pondering/prefetch "
                        "a window to complete. Default 0 (legacy autopilot behaviour).")
    p.add_argument("--pondering-wait-ms", type=int, default=1500,
                   help="How long consume() blocks waiting for in-flight pondering before falling back to "
                        "live MCTS. Default 1500ms (legacy). For pondering-realistic tests bump to 10000-15000.")
    p.add_argument("--tier3-disabled", action="store_true",
                   help="Sync-fallback architecture: router never escalates to live MCTS on the critical "
                        "path. Sparse/high-entropy turns fall back to tier-2 (baseline LLM) with pool synthesis. "
                        "Pondering still runs in background to fill the pool.")
    p.set_defaults(tier3_disabled=False)
    p.add_argument("--out", default=None, help="Optional JSONL path to dump full per-session results")
    args = p.parse_args(argv)

    preset = PRESETS[args.preset]
    base_cfg = {
        **preset,
        "c_uct": 1.4,
        "nee_threshold": 0.15,
        "nee_min_visits": 2,
        "top_k_precedents": 3,
        "use_precedents_expand": args.precedents,
        "use_precedents_score": False,
        "use_precedents_response": args.precedents,
        "pondering_enabled": args.pondering,
        "pondering_k": 2,
        "router_enabled": args.router,
        "tier_entropy_max_t1": 0.4,
        "tier_entropy_max_t2": 1.2,
        "tier_min_supporting_traces": 3,
        "planning_granularity": args.granularity,
        "rollout_action_policy": args.rollout_policy,
        "data_prefetch_enabled": args.data_prefetch,
        "data_prefetch_min_confidence": 0.05,
        "data_prefetch_max_outstanding": 50,
        "data_prefetch_decay_lambda": 0.3,
        "data_prefetch_await_in_flight_ms": 2000,
        "data_prefetch_predictor": args.predictor,
        "pondering_await_in_flight_ms": args.pondering_wait_ms,
        "tier3_enabled": not args.tier3_disabled,
    }
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    print(f"Running benchmark", flush=True)
    print(f"  base:               {args.base}", flush=True)
    print(f"  sop:                {args.sop}", flush=True)
    print(f"  preset:             {args.preset} {preset}", flush=True)
    print(f"  modes:              {modes}", flush=True)
    print(f"  sessions per mode:  {args.sessions_per_mode}", flush=True)
    print(f"  max turns:          {args.max_turns}", flush=True)
    print(f"  pondering enabled:  {args.pondering}", flush=True)
    print(f"  precedents enabled: {args.precedents}", flush=True)
    print(f"  router enabled:     {args.router}", flush=True)
    print(f"  granularity:        {args.granularity}", flush=True)
    print(f"  rollout policy:     {args.rollout_policy}", flush=True)
    print(f"  data prefetch:      {args.data_prefetch}", flush=True)
    print(f"  predictor:          {args.predictor}", flush=True)
    print(f"  user_think_time_s:  {args.user_think_time_s}", flush=True)
    print(f"  tier3 enabled:      {not args.tier3_disabled}", flush=True)

    reports = asyncio.run(run_benchmark(
        args.base, args.sop, modes, args.sessions_per_mode, base_cfg, args.max_turns,
        concurrency=args.concurrency, user_think_time_s=args.user_think_time_s,
    ))

    print(format_report(reports))

    if args.out:
        with open(args.out, "w") as f:
            for label, rep in reports.items():
                for s in rep.sessions:
                    f.write(json.dumps({
                        "mode": label,
                        "config": rep.config,
                        "session_id": s.session_id,
                        "turns": s.turns,
                        "terminal_outcome": s.terminal_outcome,
                        "total_agent_ms": s.total_agent_ms,
                        "total_user_sim_ms": s.total_user_sim_ms,
                        "total_tokens_in": s.total_tokens_in,
                        "total_tokens_out": s.total_tokens_out,
                        "pondering_hits": s.pondering_hits,
                        "pondering_eligible": s.pondering_eligible,
                        "per_turn_latency_ms": s.per_turn_latency_ms,
                        "per_turn_calls": s.per_turn_calls,
                        "rationality_means": s.rationality_means,
                        "tier_counts": s.tier_counts,
                        "prefetch_scheduled": s.prefetch_scheduled,
                        "prefetch_consumed": s.prefetch_consumed,
                        "prefetch_live": s.prefetch_live,
                        "prefetch_latency_hidden_ms": s.prefetch_latency_hidden_ms,
                        "prefetch_live_latency_ms": s.prefetch_live_latency_ms,
                        "prefetch_by_offset": s.prefetch_by_offset,
                        "error": s.error,
                    }) + "\n")
        print(f"\nWrote per-session JSONL to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
