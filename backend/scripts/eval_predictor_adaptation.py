"""Cold-start convergence for the online-adaptive empirical predictor (held-out, temporal).

Method. Order all one-step transitions (turn N action -> turn N+1 action) chronologically.
Hold out the most recent TEST_FRAC as a fixed test set. Then sweep a training cutoff over the
remaining history: at each cutoff the predictor may only count precedents created strictly
before it (the new `created_before` as-of filter), so this is a genuine train-on-past /
test-on-future evaluation -- no instance leaks from test into the counts. We report
instance-weighted recall@3 (state-blind: cohort+action conditioning) on the held-out test set
as the amount of training history grows, for baseline vs decay+shrink vs explore.

This answers: (a) does decay+shrink converge to the same place as baseline (safety), and
(b) does it help when history is thin (the cold-start regime that would justify going further
toward a full bandit)?
"""
from __future__ import annotations
import asyncio
from collections import defaultdict

from sqlalchemy import text
from app.db import SessionLocal
from app.planner.trajectory_predictor import EmpiricalTrajectoryPredictor

VARIANTS = {
    "baseline":     dict(recency_half_life_days=0.0, shrinkage_kappa=0.0, explore=False),
    "decay+shrink": dict(recency_half_life_days=30.0, shrinkage_kappa=2.0, explore=False),
    "explore":      dict(recency_half_life_days=30.0, shrinkage_kappa=2.0, explore=True),
}
TEST_FRAC = 0.20
CUTOFF_FRACS = [0.10, 0.20, 0.35, 0.50, 0.65, 0.80]  # of the training portion


async def transitions(db):
    rows = (await db.execute(text("""
        SELECT p.sop_ref, p.cohort, p.action AS cur, np.action AS nxt, p.created_at AS ts
        FROM precedent_traces p
        JOIN turns t  ON t.id = p.turn_id
        JOIN turns nt ON nt.experiment_id = t.experiment_id AND nt.turn_index = t.turn_index + 1
        JOIN precedent_traces np ON np.turn_id = nt.id
        WHERE p.action <> '' AND np.action <> ''
        ORDER BY p.created_at
    """))).all()
    return [(r[0], r[1] or "", r[2], r[3], r[4]) for r in rows]


async def recall_on_test(db, test, cutoff_ts, kw, trials):
    hit = 0; tot = 0
    # cache top-3 per (context) since many test instances share a context
    cache: dict[tuple, set] = {}
    for sop, cohort, cur, nxt, _ in test:
        key = (sop, cohort, cur)
        if key not in cache:
            score = defaultdict(float)
            for _ in range(trials):
                p = EmpiricalTrajectoryPredictor(
                    db=db, sop_ref=sop, cohort=cohort, chosen_action=cur,
                    min_supporting=1, created_before=cutoff_ts, **kw)
                for x in await p.predict(max_offset=1):
                    if x.offset == 1:
                        score[x.action] += x.probability
            cache[key] = {a for a, _ in sorted(score.items(), key=lambda kv: -kv[1])[:3]}
        tot += 1
        if nxt in cache[key]:
            hit += 1
    return hit / tot if tot else float("nan")


async def main():
    async with SessionLocal() as db:
        tr = await transitions(db)
        n = len(tr); split = int(n * (1 - TEST_FRAC))
        train, test = tr[:split], tr[split:]
        print(f"transitions={n}  train={len(train)}  test(held-out, latest {int(TEST_FRAC*100)}%)={len(test)}")
        print(f"test spans contexts={len({(s,c,a) for s,c,a,_,_ in test})}\n")
        cutoffs = [(f, train[int(len(train)*f)-1][4]) for f in CUTOFF_FRACS]
        hdr = "train cutoff".ljust(16) + "".join(n_.ljust(15) for n_ in VARIANTS)
        print(hdr); print("-" * len(hdr))
        for f, ts in cutoffs:
            line = f"{int(f*100)}% ({str(ts)[5:10]})".ljust(16)
            for name, kw in VARIANTS.items():
                r = await recall_on_test(db, test, ts, kw, trials=15 if kw["explore"] else 1)
                line += f"{r*100:5.1f}%".ljust(15)
            print(line)
        print("\nheld-out recall@3 on the fixed latest-20% test set, vs amount of training history. "
              "explore = mean of 15 draws.")
asyncio.run(main())
