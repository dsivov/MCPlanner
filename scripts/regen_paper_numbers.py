#!/usr/bin/env python3
"""Regenerate the paper's headline numbers directly from committed artifacts.

Run from the repo root:  python3 scripts/regen_paper_numbers.py

This makes every reconstructable paper number machine-auditable from the committed
`bench_*.jsonl` session outputs and `paper/table1_sessions.jsonl`, per the review's
reproducibility request. Each line prints the regenerated value next to the value claimed
in the paper, and flags any number that is NOT reconstructable from committed artifacts
(those require the un-shipped trace DB and are listed explicitly as gaps).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# The cross-SOP / pool-composition / regret runs (car + credit-card + medical).
CROSS_SOP = [
    "bench_no_tier3_g_n5.jsonl",
    "bench_no_tier3_g_credit_card.jsonl",
    "bench_no_tier3_g_medical.jsonl",
]


def load(name: str) -> list[dict]:
    p = ROOT / name
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def check(label: str, got, claimed) -> None:
    print(f"  {label:42s} regenerated={got!s:>10}   paper={claimed}")


def main() -> int:
    print("== Pool composition + realized regret (cross-SOP: car+credit-card+medical) ==")
    miss = miss_ms = hidden = cons = sched = sess = 0
    for f in CROSS_SOP:
        for r in load(f):
            sess += 1
            miss += r.get("prefetch_live", 0) or 0
            miss_ms += r.get("prefetch_live_latency_ms", 0) or 0
            hidden += r.get("prefetch_latency_hidden_ms", 0) or 0
            cons += r.get("prefetch_consumed", 0) or 0
            sched += r.get("prefetch_scheduled", 0) or 0
    total_fetches = sched + miss  # empirical/pondering scheduled + blocking misses
    check("sessions", sess, "15 (3 SOPs x 5)")
    check("blocking misses (regret events)", miss, "9")
    check("realized latency regret (s)", round(miss_ms / 1000, 1), "34.7")
    check("latency hidden by prefetch (s)", round(hidden / 1000, 1), "738.1")
    check("regret fraction (%)", round(100 * miss_ms / (miss_ms + hidden), 1), "4.5")
    check("captured (%)", round(100 * hidden / (miss_ms + hidden), 1), "95.5")
    check("mean latency / miss (ms)", round(miss_ms / miss), "3858")
    check("mean latency / hit (ms)", round(hidden / cons), "3865")

    print("\n== Cross-SOP Table 1 (from paper/table1_sessions.jsonl) ==")
    rows = [json.loads(l) for l in (ROOT / "paper/table1_sessions.jsonl").read_text().splitlines() if l.strip()]
    by_sop: dict[str, list[dict]] = {}
    for r in rows:
        key = "car" if "Insurance" in r["sop_name"] else ("medical" if "Medical" in r["sop_name"] else r["sop_name"])
        by_sop.setdefault(key, []).append(r)
    for sop, rs in by_sop.items():
        succ = sum(1 for r in rs if r["terminal_outcome"] == "success")
        mean_rerank = sum(r["mean_rerank_ms"] for r in rs) / len(rs)
        print(f"  {sop:10s} success={succ}/{len(rs)}  mean rerank (mean of per-session means)={mean_rerank:.0f} ms")
    agg = sum((rs for rs in by_sop.values()), [])
    print(f"  aggregate mean rerank = {sum(r['mean_rerank_ms'] for r in agg)/len(agg):.0f} ms   (paper: 299 ms mean)")

    print("\n== NOT reconstructable from committed artifacts (need the un-shipped trace DB) ==")
    for gap in [
        "rerank p95/p99 (table1 stores per-session MEAN rerank only, not per-turn values)",
        "the matched 2167->940 ms LLM-vs-cosine rerank comparison (per-turn rerank traces)",
        "predictor recall@3 ablations (88/38/33) and corpus-granularity table (per-turn judge data)",
        "cold-start convergence curve (regenerate via backend/scripts/eval_predictor_adaptation.py + DB)",
    ]:
        print(f"  - {gap}")
    print("\nThese gaps are the auditability limit the review flagged; the analysis bundle covers"
          "\nthe pool/regret/Table-1 numbers, the rest need a shipped trace bundle (future work).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
