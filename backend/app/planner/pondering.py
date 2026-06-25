"""Pondering MCTS — speculative search executed between user turns.

After a turn finishes we don't know what the user will say, but the SOP constrains
the next user_state to a small vocabulary. We pre-compute MCTS results for the
top-K most likely next states. When the real turn arrives, if the classified state
matches a hypothesis, we reuse the cached MCTS decision and skip the live rollouts.

The K next-states are picked by EMPIRICAL TRANSITION FREQUENCY from accumulated
precedent_traces (the context graph): for the current (cohort, last_action) tuple,
look at immediate_state distribution in past traces. Cold start falls back to
uniform over the SOP's user_states vocabulary.
"""

from __future__ import annotations
import asyncio
import time
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..schemas import TaskDefinition, MCTSConfig, PlannerTrace, RetrievedPrecedent
from ..db import PrecedentTrace, PonderingRun, SessionLocal
from ..logger import ExperimentLogger
from .mcts import run_mcts
from .data_prefetch import derive_prefetch_plan, manager as data_prefetch_manager


# ---------- Likely next-state prediction ----------

async def predict_likely_next_states(
    db: AsyncSession,
    *,
    sop_ref: str,
    cohort: str,
    last_action: str,
    k: int,
    fallback_vocab: list[str],
) -> list[tuple[str, float]]:
    """Returns [(state_name, prior_prob), ...] sorted descending by prior_prob, length <= k.

    Empirical: counts immediate_state values in precedent_traces matching (sop_ref, cohort,
    action=last_action) where immediate_state is not null. Probability = count / total.
    Cold start: uniform over fallback_vocab (which should be the SOP's user_states names).
    """
    if k <= 0:
        return []
    q = (
        select(PrecedentTrace.immediate_state, func.count(PrecedentTrace.id))
        .where(
            PrecedentTrace.sop_ref == sop_ref,
            PrecedentTrace.cohort == cohort,
            PrecedentTrace.action == last_action,
            PrecedentTrace.immediate_state.is_not(None),
        )
        .group_by(PrecedentTrace.immediate_state)
        .order_by(func.count(PrecedentTrace.id).desc())
        .limit(k)
    )
    rows = (await db.execute(q)).all()
    total = sum(c for _, c in rows)
    if rows and total > 0:
        return [(s, round(c / total, 4)) for s, c in rows]

    # Cold start: SOP vocabulary, uniform
    if not fallback_vocab:
        return []
    p = round(1.0 / min(k, len(fallback_vocab)), 4)
    return [(s, p) for s in fallback_vocab[:k]]


# ---------- Pondering scheduler ----------

@dataclass
class PonderingEntry:
    """In-memory handle to a running pondering task."""
    run_id: str
    predicted_cohort: str
    predicted_state: str
    task: asyncio.Task
    started_at: datetime
    rank: int


class PonderingScheduler:
    """Per-process scheduler keyed by experiment_id -> list of active ponderings.

    Methods are designed to be called from the chat-route handler:
      schedule_after_turn(...)   — fire K background tasks, return immediately
      consume(...)               — at next turn, look up the cache by (cohort, state)
      cancel_all(experiment_id)  — on session end / restart
    """

    def __init__(self) -> None:
        self._by_exp: dict[str, list[PonderingEntry]] = defaultdict(list)
        self._lock = asyncio.Lock()

    def active_for(self, experiment_id: str) -> list[PonderingEntry]:
        return list(self._by_exp.get(experiment_id, []))

    async def cancel_all(self, experiment_id: str) -> None:
        # Cancel AND await the tasks: a bare .cancel() returns before the task unwinds,
        # so a pondering run mid-DB-write can leave its SQLite write transaction open
        # until it actually finishes cancelling. Awaiting (with return_exceptions=True to
        # swallow CancelledError) guarantees each task's `async with SessionLocal()` exits
        # and releases the lock before the next session's chat-start runs. Fixes the
        # back-to-back-session 500 race (task #135).
        async with self._lock:
            entries = self._by_exp.pop(experiment_id, [])
        tasks = []
        for e in entries:
            if not e.task.done():
                e.task.cancel()
                tasks.append(e.task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def schedule_after_turn(
        self,
        *,
        experiment_id: str,
        after_turn_index: int,
        task_def: TaskDefinition,
        sop_ref: str,
        history: list[dict[str, str]],
        state_log: list[str],
        cohort: str,
        last_action: str,
        mcts_cfg: MCTSConfig,
        precedents: list[RetrievedPrecedent],
    ) -> None:
        """Run inside an asyncio session (fire-and-forget). Picks K most-likely next states
        and spawns one pondering task per state. Each task owns its own DB session."""
        if not mcts_cfg.pondering_enabled or mcts_cfg.pondering_k <= 0:
            return
        async with SessionLocal() as db:
            fallback_vocab = [s.name for s in task_def.user_states]
            predicted = await predict_likely_next_states(
                db,
                sop_ref=sop_ref,
                cohort=cohort,
                last_action=last_action,
                k=mcts_cfg.pondering_k,
                fallback_vocab=fallback_vocab,
            )
            if not predicted:
                return

        # Cancel any leftover ponderings for this experiment (a session shouldn't
        # have two ponder waves outstanding at once).
        await self.cancel_all(experiment_id)

        for rank, (state, prob) in enumerate(predicted):
            task = asyncio.create_task(self._run_one(
                experiment_id=experiment_id,
                after_turn_index=after_turn_index,
                task_def=task_def,
                sop_ref=sop_ref,
                history=history,
                state_log=state_log,
                cohort=cohort,
                predicted_state=state,
                rank=rank,
                prior_prob=prob,
                mcts_cfg=mcts_cfg,
                precedents=precedents,
            ))
            entry = PonderingEntry(
                run_id="",  # filled by the task once it inserts the DB row
                predicted_cohort=cohort,
                predicted_state=state,
                task=task,
                started_at=datetime.utcnow(),
                rank=rank,
            )
            async with self._lock:
                self._by_exp[experiment_id].append(entry)

    async def _run_one(
        self,
        *,
        experiment_id: str,
        after_turn_index: int,
        task_def: TaskDefinition,
        sop_ref: str,
        history: list[dict[str, str]],
        state_log: list[str],
        cohort: str,
        predicted_state: str,
        rank: int,
        prior_prob: float,
        mcts_cfg: MCTSConfig,
        precedents: list[RetrievedPrecedent],
    ) -> None:
        """One MCTS pondering, on its own DB session, results persisted to PonderingRun."""
        # Mark this whole task tree speculative so its rollout LLM calls run on slack only
        # (PASTE-style budget + critical-path preemption; see llm/scheduler.py).
        from ..llm.scheduler import speculative_mode
        speculative_mode.set(True)
        async with SessionLocal() as db:
            run = PonderingRun(
                experiment_id=experiment_id,
                after_turn_index=after_turn_index,
                predicted_cohort=cohort,
                predicted_state=predicted_state,
                rank=rank,
                prior_prob=prior_prob,
                started_at=datetime.utcnow(),
            )
            db.add(run)
            await db.flush()
            run_id = run.id
            await db.commit()

            # Build a hypothetical state_log: add the predicted state as if it had been observed.
            hypothetical_state_log = list(state_log) + [predicted_state]
            trace = PlannerTrace(mode="mcts")  # type: ignore[arg-type]
            logger = ExperimentLogger(experiment_id=experiment_id)
            t0 = time.perf_counter()
            cancelled = False
            try:
                chosen, _cohort_out, _state_out, _mood_out, _state_rat = await run_mcts(
                    task_def,
                    history,
                    hypothetical_state_log,
                    mcts_cfg,
                    trace,
                    precedents=precedents,
                    db=db,
                    sop_ref=sop_ref,
                    cohort_for_bandit=cohort,
                    logger=logger,
                )
                duration_ms = int((time.perf_counter() - t0) * 1000)
                # G fix (2026-06-03): mine pondering's rollouts for prefetch plan and schedule.
                # Without this, pondering's MCTS predictions never feed the pool — they're only
                # persisted to PonderingRun for analysis. With it, the architecture's removal of
                # tier-3 doesn't starve the data-prefetch pipeline (see N=5 result showing 59%
                # live-fallback rate before this fix).
                rollouts_snapshot = list(logger.rollouts)
                # Persist the result + LLM calls for analysis. Note these calls are tagged
                # with experiment_id but NOT with a real turn_id — they're pondering work.
                await logger.flush(db, turn_id=None)
                run = await db.get(PonderingRun, run_id)
                if run is None:
                    return
                run.finished_at = datetime.utcnow()
                run.duration_ms = duration_ms
                run.result_json = {
                    "chosen_action": chosen,
                    "candidates": [c.model_dump() for c in trace.candidates],
                    "mcts_iterations": trace.mcts_iterations,
                    "rollouts": trace.rollouts,
                }
                run.llm_calls_count = len(logger.calls)
                run.tokens_in = trace.tokens_in
                run.tokens_out = trace.tokens_out
                await db.commit()
                # Schedule prefetches from this pondering's rollouts. Tagged "pondering" so
                # the data_fetches table can attribute pool growth to pondering vs main turn.
                try:
                    plan = derive_prefetch_plan(
                        rollouts_snapshot,
                        task=task_def,
                        chosen_action_now=chosen,
                        decay_lambda=mcts_cfg.data_prefetch_decay_lambda,
                    )
                    for item in plan:
                        item.predictor_source = "pondering"
                    if plan and mcts_cfg.data_prefetch_enabled:
                        data_prefetch_manager.max_outstanding = mcts_cfg.data_prefetch_max_outstanding
                        await data_prefetch_manager.schedule(
                            experiment_id=experiment_id,
                            sop_ref=sop_ref,
                            task=task_def,
                            plan=plan,
                            current_turn_index=after_turn_index,
                            min_confidence=mcts_cfg.data_prefetch_min_confidence,
                        )
                except Exception:
                    # Pool population is best-effort — don't crash pondering if scheduling fails.
                    pass
            except asyncio.CancelledError:
                cancelled = True
                run = await db.get(PonderingRun, run_id)
                if run is not None:
                    run.cancelled = True
                    run.finished_at = datetime.utcnow()
                    run.duration_ms = int((time.perf_counter() - t0) * 1000)
                    await db.commit()
                raise
            except Exception as e:
                # Mark cancelled-ish for analysis purposes; surface the error in logs.
                run = await db.get(PonderingRun, run_id)
                if run is not None:
                    run.cancelled = True
                    run.finished_at = datetime.utcnow()
                    run.duration_ms = int((time.perf_counter() - t0) * 1000)
                    run.result_json = {"error": f"{type(e).__name__}: {e}"}
                    await db.commit()

    async def consume(
        self,
        db: AsyncSession,
        *,
        experiment_id: str,
        after_turn_index: int,
        cohort: str,
        state: str,
        consuming_turn_id: str,
        wait_in_flight_ms: int = 1500,
    ) -> Optional[dict]:
        """Look up the cache. If a finished PonderingRun matches (cohort, state), return its
        result. If a matching run is still in-flight, await briefly. Mark consumed in DB.

        Returns the result_json (dict) on hit, or None on miss.
        """
        # 1) Check in-memory tasks for a matching in-flight ponder.
        entries = list(self._by_exp.get(experiment_id, []))
        for e in entries:
            if e.predicted_cohort == cohort and e.predicted_state == state and not e.task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(e.task), timeout=wait_in_flight_ms / 1000.0)
                except asyncio.TimeoutError:
                    # Don't cancel — let it finish for future hit, just don't consume now.
                    pass
                except asyncio.CancelledError:
                    pass

        # 2) Query DB for a matching finished, non-consumed run.
        q = (
            select(PonderingRun)
            .where(
                PonderingRun.experiment_id == experiment_id,
                PonderingRun.after_turn_index == after_turn_index,
                PonderingRun.predicted_cohort == cohort,
                PonderingRun.predicted_state == state,
                PonderingRun.consumed.is_(False),
                PonderingRun.cancelled.is_(False),
                PonderingRun.finished_at.is_not(None),
            )
            .order_by(PonderingRun.rank.asc())
            .limit(1)
        )
        run = (await db.execute(q)).scalar_one_or_none()
        if run is None or not run.result_json:
            return None
        run.consumed = True
        run.consumed_turn_id = consuming_turn_id
        return run.result_json


# Process-wide singleton (FastAPI is a single process).
scheduler = PonderingScheduler()
