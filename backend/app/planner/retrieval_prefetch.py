"""Background retrieval-prefetch scheduling — empirical action prediction plus an optional
cheap-LLM generative {user_text} slot, replacing MCTS for the retrieval path.

Findings (June 2026 ablation): for predicting the next agent action (which data source),
the empirical predictor dominates MCTS and a naive LLM (88% vs 33% vs 38% recall@3) at zero
token cost. For the free-text RAG query, the structured template slots suffice on coarse
corpora; on fine-grained corpora a single small-model call to predict the user's next
utterance recovers a +14pp doc-overlap gain — matching MCTS rollouts at ~85x lower cost.
MCTS is dropped from the retrieval path; this module is its replacement.

Runs as a fire-and-forget background task: the prefetch targets FUTURE turns, so a ~1s
scheduling delay is harmless, and the work is marked speculative so it is bounded and
preempted by the live turn (PASTE-style scheduler, llm/scheduler.py).
"""
from __future__ import annotations
import asyncio
import logging
from typing import Optional

from ..schemas import TaskDefinition, MCTSConfig
from ..db import SessionLocal
from .trajectory_predictor import (
    EmpiricalTrajectoryPredictor, build_prefetch_plan_from_predictions,
)
from .next_utterance import predict_next_utterance
from .data_prefetch import manager as data_prefetch_manager

_log = logging.getLogger(__name__)


async def schedule_retrieval_prefetch_bg(
    *,
    experiment_id: str,
    sop_ref: str,
    task: TaskDefinition,
    history: list[dict],
    cohort: str,
    state: str,
    mood: str,
    chosen_action: str,
    current_turn_index: int,
    mcts_cfg: MCTSConfig,
    min_supporting: int = 2,
) -> None:
    """Empirical-predict the next-turn data deps, optionally fill the generative RAG query
    slot via a cheap LLM call, build the plan, and schedule. Best-effort, off-path."""
    from ..llm.scheduler import speculative_mode
    from ..logger import ExperimentLogger
    speculative_mode.set(True)
    logger = ExperimentLogger(experiment_id=experiment_id)
    try:
        async with SessionLocal() as db:
            predictor = EmpiricalTrajectoryPredictor(
                db=db, sop_ref=sop_ref, cohort=cohort or "", chosen_action=chosen_action,
                min_supporting=min_supporting, mood=mood or None,
                recency_half_life_days=mcts_cfg.predictor_recency_half_life_days,
                shrinkage_kappa=mcts_cfg.predictor_shrinkage_kappa,
                explore=mcts_cfg.predictor_explore,
                explore_alpha0=mcts_cfg.predictor_explore_alpha0,
            )
            predictions = await predictor.predict(
                max_offset=mcts_cfg.rollout_depth or 3,
                state_hints_topk={1: [state]} if state else None,
            )
        if not predictions:
            return

        # Cheap-LLM {user_text}: only when enabled AND a predicted action actually needs a
        # query-template dependency (otherwise the LLM call would be wasted).
        override: Optional[str] = None
        if getattr(mcts_cfg, "use_llm_query_predictor", False):
            tmpl_deps = {d.name for d in task.data_dependencies if getattr(d, "query_template", None)}
            if tmpl_deps:
                # A query-template dep may be keyed to an action predicted at ANY offset
                # (e.g. claims_history_rag → ReviewCurrentCoverage, often 2-3 turns out),
                # so check the whole predicted horizon, not just offset 1.
                pred_actions = {p.action for p in predictions}
                needs = any(
                    set(a.data_dependencies or []) & tmpl_deps
                    for a in task.agent_actions if a.name in pred_actions
                )
                if needs:
                    # Seed {user_text} with the user's predicted NEXT utterance (offset-1
                    # modal action gives the context); the pool tolerates the imprecision
                    # for slightly-further-out RAG needs.
                    top = max(
                        (p for p in predictions if p.offset == 1),
                        key=lambda p: p.probability, default=None,
                    )
                    # The call is speculative (slack-gated). Bound the wait: if no slack
                    # appears before the deadline (e.g. zero-pause stress conditions), fall
                    # back to the structured stub rather than blocking the prefetch. Real
                    # calls have a 2-5s user pause, comfortably within this budget.
                    try:
                        override = await asyncio.wait_for(
                            predict_next_utterance(
                                task, history,
                                predicted_action=(top.action if top else chosen_action),
                                cohort=cohort, state=state, mood=mood, logger=logger,
                            ),
                            timeout=getattr(mcts_cfg, "data_prefetch_await_in_flight_ms", 2000) / 1000.0 + 1.5,
                        )
                    except asyncio.TimeoutError:
                        override = None  # no slack — use the stub

        plan = build_prefetch_plan_from_predictions(
            predictions, task=task, decay_lambda=mcts_cfg.data_prefetch_decay_lambda,
            cohort=cohort or "", mood=mood or "", user_text_override=override,
        )
        if plan:
            data_prefetch_manager.max_outstanding = mcts_cfg.data_prefetch_max_outstanding
            await data_prefetch_manager.schedule(
                experiment_id=experiment_id, sop_ref=sop_ref, task=task, plan=plan,
                current_turn_index=current_turn_index,
                min_confidence=mcts_cfg.data_prefetch_min_confidence,
            )
        # Persist the next_utterance LLM call for token accounting (background work, no turn).
        try:
            async with SessionLocal() as db:
                await logger.flush(db, turn_id=None)
        except Exception as e:
            _log.warning(
                "retrieval_prefetch logger.flush failed: experiment_id=%s turn=%s source=empirical err=%s: %s",
                experiment_id, current_turn_index, type(e).__name__, e,
            )
    except Exception as e:
        # Prefetch is best-effort and must never break the live session, but a silent failure
        # here can quietly bias hit-rate/latency measurements (a dropped predictor or fetch is
        # invisible). Log a structured event so benchmark summaries can count these.
        _log.warning(
            "retrieval_prefetch failed: experiment_id=%s turn=%s cohort=%s state=%s action=%s "
            "source=empirical err=%s: %s",
            experiment_id, current_turn_index, cohort, state, chosen_action, type(e).__name__, e,
        )
