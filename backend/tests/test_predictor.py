"""Tests for the empirical trajectory predictor: shrinkage/explore math (pure) and the
SQL fallback order + decay/shrinkage on an in-memory database (integration)."""
import math
import random
from datetime import datetime, timedelta

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.db import Base, Experiment, TurnRecord, PrecedentTrace
from app.planner.trajectory_predictor import EmpiricalTrajectoryPredictor


def make_predictor(**kw):
    # db is unused by the pure _finalize_probs path.
    return EmpiricalTrajectoryPredictor(db=None, sop_ref="s", cohort="c", chosen_action="a", **kw)


# ---------------- pure: shrinkage blend math ----------------

def test_shrinkage_kappa_zero_is_plain_normalization():
    p = make_predictor(shrinkage_kappa=0.0)
    dist = [("X", 3, 3.0), ("Y", 1, 1.0)]
    out = dict(p._finalize_probs(dist, prior={}))
    assert abs(out["X"] - 0.75) < 1e-9 and abs(out["Y"] - 0.25) < 1e-9


def test_shrinkage_blends_toward_prior_for_sparse_cell():
    p = make_predictor(shrinkage_kappa=2.0)
    dist = [("X", 1, 1.0)]                 # W=1, very sparse
    prior = {"X": 0.5, "Z": 0.5}          # prior also likes an unseen action Z
    out = dict(p._finalize_probs(dist, prior))
    # P(X) = (1 + 2*0.5)/(1+2) = 2/3 ; P(Z) = (0 + 2*0.5)/3 = 1/3
    assert abs(out["X"] - 2 / 3) < 1e-9
    assert abs(out["Z"] - 1 / 3) < 1e-9
    assert abs(sum(out.values()) - 1.0) < 1e-9


def test_shrinkage_negligible_for_rich_cell():
    p = make_predictor(shrinkage_kappa=2.0)
    dist = [("X", 100, 100.0), ("Y", 100, 100.0)]  # W=200 >> kappa
    prior = {"Z": 1.0}
    out = dict(p._finalize_probs(dist, prior))
    assert abs(out["X"] - 0.5) < 0.02 and abs(out["Y"] - 0.5) < 0.02


def test_explore_is_a_valid_distribution_and_seed_deterministic():
    dist = [("X", 5, 5.0), ("Y", 2, 2.0), ("Z", 1, 1.0)]
    p = make_predictor(shrinkage_kappa=1.0, explore=True)
    random.seed(42); a = dict(p._finalize_probs(dist, {"X": 0.6, "Y": 0.4}))
    random.seed(42); b = dict(p._finalize_probs(dist, {"X": 0.6, "Y": 0.4}))
    assert a == b                                  # same seed -> same draw
    assert abs(sum(a.values()) - 1.0) < 1e-9       # proper distribution
    assert all(v >= 0 for v in a.values())


# ---------------- integration: SQL fallback + decay on in-memory DB ----------------

@pytest.fixture
async def seeded_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    @event.listens_for(engine.sync_engine, "connect")
    def _reg_exp(dbapi, _rec):  # the predictor's decay term needs exp() in SQLite
        dbapi.create_function("exp", 1, math.exp, deterministic=True)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    # One SOP "s". Sessions where action A is followed by B (mostly) or C (rarely),
    # all in cohort "c". Plus one session in a different cohort "other".
    async with Session() as db:
        async def session(exp_id, cohort, seq, days_ago=0):
            db.add(Experiment(id=exp_id, sop_ref="s", sop_snapshot={},
                              planner_mode="mcts", chat_mode="auto"))
            ts = datetime.utcnow() - timedelta(days=days_ago)
            for i, act in enumerate(seq):
                tid = f"{exp_id}-t{i}"
                db.add(TurnRecord(id=tid, experiment_id=exp_id, turn_index=i))
                db.add(PrecedentTrace(turn_id=tid, experiment_id=exp_id, sop_ref="s",
                                      cohort=cohort, action=act, created_at=ts))
        for k in range(4):
            await session(f"e{k}", "c", ["A", "B", "D"])     # A->B at offset 1
        await session("e4", "c", ["A", "C", "D"])            # one A->C
        await session("eo", "other", ["A", "Z"])             # different cohort
        await db.commit()
        yield Session
    await engine.dispose()


async def test_predicts_modal_next_action(seeded_session):
    async with seeded_session() as db:
        p = EmpiricalTrajectoryPredictor(db=db, sop_ref="s", cohort="c",
                                         chosen_action="A", min_supporting=2,
                                         shrinkage_kappa=0.0, recency_half_life_days=0.0)
        preds = [x for x in await p.predict(max_offset=1) if x.offset == 1]
        top = max(preds, key=lambda x: x.probability)
        assert top.action == "B"   # 4x B vs 1x C in cohort c


async def test_cohort_fallback_to_sop_when_unknown_cohort(seeded_session):
    async with seeded_session() as db:
        # cohort "ghost" has no rows -> must fall back to SOP-level and still predict.
        p = EmpiricalTrajectoryPredictor(db=db, sop_ref="s", cohort="ghost",
                                         chosen_action="A", min_supporting=2,
                                         shrinkage_kappa=0.0, recency_half_life_days=0.0)
        preds = [x for x in await p.predict(max_offset=1) if x.offset == 1]
        actions = {x.action for x in preds}
        assert "B" in actions   # recovered via SOP-level fallback, not empty


async def test_decay_runs_and_ranks(seeded_session):
    async with seeded_session() as db:
        # half_life on exercises the exp() UDF path; should still rank B on top.
        p = EmpiricalTrajectoryPredictor(db=db, sop_ref="s", cohort="c",
                                         chosen_action="A", min_supporting=2,
                                         shrinkage_kappa=2.0, recency_half_life_days=30.0)
        preds = [x for x in await p.predict(max_offset=1) if x.offset == 1]
        assert preds and max(preds, key=lambda x: x.probability).action == "B"
