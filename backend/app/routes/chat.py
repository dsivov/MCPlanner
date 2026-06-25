from __future__ import annotations
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from ..db import get_session, SOPRecord, Experiment, TurnRecord, RouterDecision, PoolPick
from ..schemas import (
    TaskDefinition,
    ChatStartRequest,
    ChatStartResponse,
    TurnRequest,
    TurnResponse,
    PlannerTrace,
    MCTSConfig,
    RetrievedPrecedent,
)
from ..config import settings
from ..logger import ExperimentLogger
from ..planner.baseline import select_action_baseline
from ..planner.mcts import run_mcts
from ..planner.responder import generate_response
from ..planner.user_sim import simulate_user_utterance
from ..planner.precedents import (
    embed_text, situation_text_from_history, write_precedent, write_retrieval_log,
    retrieve_precedents, fill_previous_immediate_outcome, finalize_experiment,
)
from ..planner.pondering import scheduler as pondering_scheduler
from ..planner.router import route_turn
from ..planner.data_prefetch import manager as data_prefetch_manager
from ..planner.trajectory_predictor import (
    MctsTrajectoryPredictor, EmpiricalTrajectoryPredictor, UnionTrajectoryPredictor,
    build_prefetch_plan_from_predictions,
)
import asyncio

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _models_snapshot() -> dict[str, str]:
    return {
        "selection": settings.MODEL_SELECTION,
        "rollout": settings.MODEL_ROLLOUT,
        "user_sim": settings.MODEL_USER_SIM,
        "state": settings.MODEL_STATE,
        "builder": settings.MODEL_BUILDER,
        "embedding": "text-embedding-3-small",
    }


async def _load_experiment(exp_id: str, db: AsyncSession) -> tuple[Experiment, TaskDefinition]:
    exp = await db.get(Experiment, exp_id)
    if not exp:
        raise HTTPException(404, "experiment not found")
    return exp, TaskDefinition.model_validate(exp.sop_snapshot)


@router.post("/start", response_model=ChatStartResponse)
async def start_chat(req: ChatStartRequest, db: AsyncSession = Depends(get_session)) -> ChatStartResponse:
    if req.sop_id.startswith("seed:"):
        p: Path = settings.DATA_DIR / "sops" / req.sop_id[len("seed:"):]
        if not p.exists():
            raise HTTPException(404, "seed not found")
        task = TaskDefinition.model_validate(json.loads(p.read_text()))
    else:
        rec = await db.get(SOPRecord, req.sop_id)
        if not rec:
            raise HTTPException(404, "sop not found")
        task = TaskDefinition.model_validate(rec.payload)

    exp = Experiment(
        sop_ref=req.sop_id,
        sop_name=task.name,
        sop_snapshot=task.model_dump(),
        planner_mode=req.planner_mode,
        chat_mode=req.chat_mode,
        mcts_config=req.mcts.model_dump(),
        models=_models_snapshot(),
        history=[],
        notes="",
    )
    db.add(exp)
    await db.commit()
    await db.refresh(exp)
    return ChatStartResponse(session_id=exp.id, sop=task)


@router.post("/{session_id}/turn", response_model=TurnResponse)
async def take_turn(
    session_id: str, req: TurnRequest, db: AsyncSession = Depends(get_session)
) -> TurnResponse:
    exp, task = await _load_experiment(session_id, db)
    if exp.terminal_outcome is not None:
        raise HTTPException(400, f"session already ended ({exp.terminal_outcome})")

    planner_mode = exp.planner_mode
    chat_mode = exp.chat_mode
    mcts_cfg = MCTSConfig.model_validate(exp.mcts_config or {})

    history = list(exp.history or [])
    state_log = await _state_log(db, exp.id)

    logger = ExperimentLogger(experiment_id=exp.id)

    # 1) Produce the user message — TIMED SEPARATELY from agent work.
    user_sim_t0 = time.perf_counter()
    if chat_mode == "auto":
        user_text, _ = await simulate_user_utterance(task, history, logger=logger, call_site="auto_user")
    else:
        user_text = (req.user_message or "").strip()
        if not user_text:
            raise HTTPException(400, "user_message required in human mode")
    user_sim_ms = int((time.perf_counter() - user_sim_t0) * 1000) if chat_mode == "auto" else 0
    history.append({"role": "user", "content": user_text})

    # Agent timer begins HERE — duration_ms reflects only what a real human user would wait
    # for (embed → planner → response_gen). Excludes the auto-user simulator's runtime.
    started_at = datetime.utcnow()
    t0 = time.perf_counter()

    # 2) Embed the current situation + retrieve top-K precedents
    #    (cohort isn't known yet; retrieval is SOP-filtered. The planner can re-rank by
    #    cohort downstream since each precedent carries its own cohort label.)
    retrieval_t0 = time.perf_counter()
    situation_text = situation_text_from_history(
        history, cohort="", user_state="",  # pre-classification
        last_n=4,
    )
    try:
        situation_embedding = await embed_text(situation_text, logger=logger)
    except Exception:
        situation_embedding = b""  # tolerate embedding failures; retrieval falls back
    # Pull the most-recent classified mood from this session (if any) to use as a
    # retrieval hint. Mood is medium-moving — using prior turn's mood as the proxy is
    # a worse approximation than knowing this turn's mood, but it's strictly better than
    # mood-blind retrieval, and the retrieve function falls back when matches are too sparse.
    prior_mood = await db.scalar(
        select(TurnRecord.mood)
        .where(TurnRecord.experiment_id == exp.id)
        .where(TurnRecord.mood != "")
        .order_by(TurnRecord.turn_index.desc())
        .limit(1)
    )
    precedents: list[RetrievedPrecedent] = []
    if situation_embedding:
        precedents = await retrieve_precedents(
            db,
            sop_ref=exp.sop_ref,
            cohort="",  # unknown — accept any cohort; results carry their own cohort label
            query_text=situation_text,
            query_embedding=situation_embedding,
            k=mcts_cfg.top_k_precedents,
            mood_hint=(prior_mood or None),
        )
    retrieval_ms = int((time.perf_counter() - retrieval_t0) * 1000)

    # 3) Predict cohort + user state + select action
    #
    # If pondering is enabled and we have at least one prior turn, we attempt cache consume.
    # The cache key is (experiment_id, after_turn_index, cohort, state). We need cohort+state
    # FIRST (cheap call), then either use cached MCTS result or run live.
    trace = PlannerTrace(mode=planner_mode)  # type: ignore[arg-type]
    trace.precedents = precedents

    current_turn_index = await db.scalar(
        select(func.count()).select_from(TurnRecord).where(TurnRecord.experiment_id == exp.id)
    ) or 0

    router_decision = None
    mood: str = ""    # Phase-2 classified user mood; populated by MCTS-path classifier, "" otherwise.
    if planner_mode == "mcts":
        # Classify cohort + state + propose candidates ONCE. This single call is reused by:
        #   (a) the pondering-cache lookup,
        #   (b) the multi-tier router, and
        #   (c) the fallback live MCTS (via pre_classified=).
        from ..planner.mcts import _cohort_state_propose, _cohort_state_propose_strategy
        from ..planner.sop_graph import SOPGraph
        sg = SOPGraph(task)
        visited = sg.visited_from_history(history, state_log)
        if mcts_cfg.planning_granularity == "strategy":
            allowed = [s.name for s in sg.allowed_strategies(visited)]
            cohort, state, state_rat, cands, _res = await _cohort_state_propose_strategy(
                task, history, allowed, k=mcts_cfg.branching,
                precedents=precedents, use_precedents=mcts_cfg.use_precedents_expand,
                logger=logger,
            )
        else:
            allowed = sg.allowed_actions(visited)
            cohort, state, mood, state_rat, cands, _res = await _cohort_state_propose(
                task, history, allowed, k=mcts_cfg.branching,
                precedents=precedents, use_precedents=mcts_cfg.use_precedents_expand,
                logger=logger,
            )
        trace.tokens_in += _res.tokens_in
        trace.tokens_out += _res.tokens_out
        chosen: Optional[str] = None

        # (a) Pondering cache: did we pre-compute MCTS for this (cohort, state)?
        if mcts_cfg.pondering_enabled and current_turn_index > 0:
            cached = await pondering_scheduler.consume(
                db,
                experiment_id=exp.id,
                after_turn_index=current_turn_index - 1,
                cohort=cohort,
                state=state,
                consuming_turn_id="",
                wait_in_flight_ms=mcts_cfg.pondering_await_in_flight_ms,
            )
            if cached:
                chosen = cached.get("chosen_action") or (allowed[0] if allowed else "")
                cand_list = cached.get("candidates") or []
                from ..schemas import CandidateAction
                trace.candidates = [CandidateAction(**c) for c in cand_list]
                trace.chosen_action = chosen
                trace.mcts_iterations = int(cached.get("mcts_iterations") or 0)
                trace.rollouts = int(cached.get("rollouts") or 0)
                trace.from_pondering = True
                trace.pondering_hit_state = state

        # (b) Router: if we didn't hit the pondering cache, consult precedent-derived tiers.
        if chosen is None:
            router_decision = await route_turn(
                db, sop_ref=exp.sop_ref, cohort=cohort, state=state,
                allowed_actions=allowed, cfg=mcts_cfg,
            )
            trace.tier_used = router_decision.tier  # type: ignore[assignment]
            trace.tier_entropy = router_decision.stats.entropy
            trace.tier_supporting_traces = router_decision.stats.n_supporting
            trace.tier_dominant_action = router_decision.stats.dominant_action
            trace.tier_dominant_agreement = router_decision.stats.dominant_agreement
            trace.tier_rationale = router_decision.rationale

            from ..schemas import CandidateAction
            if router_decision.tier == "cached_playbook" and router_decision.stats.dominant_action:
                dom = router_decision.stats.dominant_action
                # In strategy mode the dominant name is a strategy (or action — precedents
                # store the concrete instantiated action). Make sure the executed `chosen`
                # is a concrete action that's currently SOP-allowed.
                action_allowed_now = sg.allowed_actions(visited)
                if dom in action_allowed_now:
                    chosen = dom
                elif mcts_cfg.planning_granularity == "strategy" and dom in [s.name for s in sg.allowed_strategies(visited)]:
                    chosen = sg.instantiate_strategy(dom, visited)
                    trace.chosen_strategy = dom
                else:
                    chosen = (action_allowed_now[0] if action_allowed_now else "")
                trace.chosen_action = chosen
                trace.candidates = [CandidateAction(
                    action=dom, q_value=router_decision.stats.dominant_agreement, visits=router_decision.stats.n_supporting,
                    rationale=f"cached playbook: dominant historical ({router_decision.stats.dominant_agreement*100:.0f}% of {router_decision.stats.n_supporting} traces)",
                )]
            elif router_decision.tier == "baseline":
                # Use the top candidate from the cohort_state_propose call we already paid for.
                top = cands[0] if cands else (allowed[0] if allowed else "", "fallback")
                top_name = top[0] if isinstance(top, tuple) else top
                if mcts_cfg.planning_granularity == "strategy":
                    chosen = sg.instantiate_strategy(top_name, visited)
                    trace.chosen_strategy = top_name
                else:
                    chosen = top_name
                trace.chosen_action = chosen
                trace.candidates = [CandidateAction(
                    action=a, q_value=(1.0 if a == top_name else 0.0), visits=0,
                    rationale=(r if a == top_name else ""),
                ) for a, r in cands]
                trace.mode = "baseline"  # honest reporting — we didn't actually run MCTS
            # else router_decision.tier == "mcts" — fall through to (c)

        # (c) Full MCTS if we didn't short-circuit.
        if chosen is None:
            chosen, cohort, state, _mcts_mood, state_rat = await run_mcts(
                task, history, state_log, mcts_cfg, trace,
                precedents=precedents,
                pre_classified=(cohort, state, state_rat, cands),
                db=db,
                sop_ref=exp.sop_ref,
                cohort_for_bandit=cohort or "",
                logger=logger,
            )
            # run_mcts uses pre_classified so the mood we captured upstream is the
            # canonical one; _mcts_mood is "" when pre_classified is supplied.
    else:
        chosen, cohort, state, state_rat = await select_action_baseline(
            task, history, state_log, trace,
            precedents=precedents,
            use_precedents_expand=mcts_cfg.use_precedents_expand,
            logger=logger,
        )
    trace.predicted_user_state = state
    trace.state_rationale = state_rat
    trace.cohort = cohort
    trace.mood = mood    # phase-2 classified mood, "" if no cohort moods or baseline path
    state_log.append(state)

    # 4) Backfill the PREVIOUS precedent's immediate_state now that we observe it.
    await fill_previous_immediate_outcome(
        db, experiment_id=exp.id, new_user_state=state, immediate_reward=0.5,
    )

    # 4b) Consume any speculatively-prefetched data for `chosen` before response_gen.
    prefetch_stats = {"consumed": 0, "live": 0, "latency_hidden_ms": 0, "live_latency_ms": 0}
    if mcts_cfg.data_prefetch_enabled and task.data_dependencies:
        _payloads, prefetch_stats = await data_prefetch_manager.consume(
            experiment_id=exp.id,
            sop_ref=exp.sop_ref,
            task=task,
            action_name=chosen,
            current_turn_index=current_turn_index,
            await_in_flight_ms=mcts_cfg.data_prefetch_await_in_flight_ms,
            live_fallback=True,
        )
        trace.data_prefetch_consumed_count = int(prefetch_stats["consumed"])
        trace.data_prefetch_live_count = int(prefetch_stats["live"])
        trace.data_prefetch_latency_hidden_ms = int(prefetch_stats["latency_hidden_ms"])
        trace.data_prefetch_live_latency_ms = int(prefetch_stats["live_latency_ms"])

    # 4c) Pool rerank — supervisor curates 0-3 items from the session's prefetch pool
    # for this turn's response_gen. Replaces the legacy key-lookup model documented in
    # notes/2026-06-03-pool-based-cache-architecture.md. Pool-rerank also runs when the
    # legacy consume() finds no exact match — they're complementary, not exclusive.
    pool_picks_payloads: list[str] = []
    pool_pick_ids: list[str] = []
    pool_rerank_ms = 0
    pool_size_at_pick = 0
    pool_rerank_rationale = ""
    if mcts_cfg.data_prefetch_enabled and task.data_dependencies:
        from ..planner.pool_rerank import rerank_pool_for_turn
        live_pool = data_prefetch_manager.get_pool(exp.id)
        pool_size_at_pick = len(live_pool)
        if live_pool:
            picks, pool_rerank_rationale, pool_rerank_ms, _ = await rerank_pool_for_turn(
                live_pool,
                live_user_message=user_text,
                classified_cohort=cohort or "",
                classified_mood=mood or "",
                classified_state=state or "",
                chosen_action=chosen,
                max_picks=3,
                logger=logger,
            )
            for p in picks:
                # Compose a short payload-with-tag for the response_gen prompt block.
                pool_picks_payloads.append(
                    f"[{p.dependency_name} — {p.source_action}] {p.payload_summary}"
                )
                if p.fetch_id:
                    pool_pick_ids.append(p.fetch_id)

    # 5) Generate response (precedents injected for style/must_say handling).
    # Milestone B (instruction prefetch): before paying for the live response_gen call,
    # check whether the supervisor pre-generated a response for this exact (action, state).
    # Exact-match POC: hit → use payload verbatim, skip response_gen entirely. Miss → fall
    # through to the live call as before. Tracked via trace.instruction_hit for analytics.
    instruction_item = data_prefetch_manager.lookup_instruction(
        exp.id, chosen_action=chosen, classified_state=state or "",
    ) if mcts_cfg.data_prefetch_enabled else None
    if instruction_item is not None and instruction_item.payload:
        resp_text = str(instruction_item.payload).strip()
        # Synthetic LLMResult-shaped stub so downstream tokens accounting stays consistent.
        from types import SimpleNamespace
        res_resp = SimpleNamespace(tokens_in=0, tokens_out=0)
        trace.precedents_used_response = False
        trace.instruction_hit = True
        trace.instruction_fetch_id = instruction_item.fetch_id
        trace.instruction_data_count = getattr(instruction_item, "instr_data_count", 0)
    else:
        resp_text, res_resp = await generate_response(
            task, history, chosen,
            precedents=precedents,
            use_precedents=mcts_cfg.use_precedents_response,
            prefetched_context=pool_picks_payloads or None,
            logger=logger,
        )
        trace.precedents_used_response = bool(mcts_cfg.use_precedents_response and precedents)
        trace.instruction_hit = False
    trace.tokens_in += res_resp.tokens_in
    trace.tokens_out += res_resp.tokens_out

    # 6) Persist turn + flush logger
    duration_ms = int((time.perf_counter() - t0) * 1000)
    ended_at = datetime.utcnow()
    trace.agent_duration_ms = duration_ms
    trace.user_sim_ms = user_sim_ms
    history.append({"role": "assistant", "content": resp_text, "action": chosen})

    turn_count = await db.scalar(
        select(func.count()).select_from(TurnRecord).where(TurnRecord.experiment_id == exp.id)
    )
    turn = TurnRecord(
        experiment_id=exp.id,
        turn_index=turn_count or 0,
        started_at=started_at,
        ended_at=ended_at,
        duration_ms=duration_ms,
        user_sim_ms=user_sim_ms,
        user_message=user_text,
        assistant_message=resp_text,
        chosen_action=chosen,
        predicted_user_state=state,
        state_rationale=state_rat,
        mood=mood,
        mode=planner_mode,
        tokens_in=trace.tokens_in,
        tokens_out=trace.tokens_out,
        mcts_iterations=trace.mcts_iterations,
        rollouts=trace.rollouts,
        trace=trace.model_dump(),
    )
    db.add(turn)
    await db.flush()  # to get turn.id

    # Snapshot rollouts BEFORE the logger.flush clears its buffer — the data-prefetch
    # scheduler below needs them to derive trajectory predictions.
    rollouts_snapshot = list(logger.rollouts)

    await logger.flush(db, turn_id=turn.id)

    # Pool-architecture v1: persist this turn's pool-rerank decision for analysis.
    # Skipped when prefetch is disabled OR the pool was empty at consume time (no LLM
    # call fired). pool_size_at_pick=0 acts as a sentinel for "rerank skipped".
    if pool_size_at_pick > 0:
        db.add(PoolPick(
            experiment_id=exp.id,
            turn_index=current_turn_index,
            picked_fetch_ids=pool_pick_ids,
            pool_size_at_pick=pool_size_at_pick,
            rationale=pool_rerank_rationale[:1000],
            pick_duration_ms=pool_rerank_ms,
        ))

    # 7) Emit the new precedent
    if situation_embedding:
        # Re-derive situation_text with the now-known cohort + state for embedding consistency on writes.
        situation_text_for_write = situation_text_from_history(history[:-1], cohort=cohort, user_state=state)
        await write_precedent(
            db,
            turn_id=turn.id,
            experiment_id=exp.id,
            sop_ref=exp.sop_ref,
            cohort=cohort or "unknown",
            mood=mood or None,
            situation_text=situation_text_for_write,
            embedding_bytes=situation_embedding,
            action=chosen,
            response_text=resp_text,
        )

    # 7b) Persist router decision (if the router was consulted this turn).
    if router_decision is not None:
        db.add(RouterDecision(
            turn_id=turn.id,
            experiment_id=exp.id,
            sop_ref=exp.sop_ref,
            cohort=cohort or "",
            state=state or "",
            tier_used=router_decision.tier,
            entropy=router_decision.stats.entropy,
            supporting_traces=router_decision.stats.n_supporting,
            dominant_action=router_decision.stats.dominant_action,
            dominant_agreement=router_decision.stats.dominant_agreement,
            action_distribution=router_decision.stats.distribution,
            rationale=router_decision.rationale,
        ))

    # 8) Log the retrieval (which precedents were used, by which injection points)
    await write_retrieval_log(
        db,
        turn_id=turn.id,
        experiment_id=exp.id,
        sop_ref=exp.sop_ref,
        cohort=cohort or "",
        query_text=situation_text,
        top_k_requested=mcts_cfg.top_k_precedents,
        results=precedents,
        used_expand=trace.precedents_used_expand,
        used_score=trace.precedents_used_score,
        used_response=trace.precedents_used_response,
        duration_ms=retrieval_ms,
    )

    exp.history = history
    flag_modified(exp, "history")

    # 9) Auto-finalize if predicted user state hits a marker.
    success_markers = set(task.conversation_profile.success_markers)
    failure_markers = set(task.conversation_profile.failure_markers)
    if state in success_markers:
        await finalize_experiment(db, experiment_id=exp.id, outcome="success", reward=1.0)
        await data_prefetch_manager.finalize_session(exp.id)
        await pondering_scheduler.cancel_all(exp.id)
    elif state in failure_markers:
        await finalize_experiment(db, experiment_id=exp.id, outcome="failure", reward=0.0)
        await data_prefetch_manager.finalize_session(exp.id)
        await pondering_scheduler.cancel_all(exp.id)

    await db.commit()

    # 10) Schedule pondering for the NEXT turn (fire-and-forget). The scheduler owns its
    # own DB sessions so the response can return without waiting.
    if planner_mode == "mcts" and mcts_cfg.pondering_enabled and exp.terminal_outcome is None:
        await pondering_scheduler.schedule_after_turn(
            experiment_id=exp.id,
            after_turn_index=current_turn_index,  # the index we just committed
            task_def=task,
            sop_ref=exp.sop_ref,
            history=list(history),
            state_log=list(state_log),
            cohort=cohort or "unknown",
            last_action=chosen,
            mcts_cfg=mcts_cfg,
            precedents=precedents,
        )

    # 11) Schedule speculative data prefetches based on MCTS rollouts' predicted trajectories.
    # Mirrors pondering but for external I/O — items go in a rolling per-session queue and
    # are consumed at later turns (possibly turns N+2, N+3 etc.) when the prediction holds.
    if (
        planner_mode == "mcts"
        and mcts_cfg.data_prefetch_enabled
        and task.data_dependencies
        and exp.terminal_outcome is None
        and chosen
    ):
        # Pick a trajectory predictor based on configured policy + what data is available.
        mode = getattr(mcts_cfg, "data_prefetch_predictor", "auto")
        has_deep_rollouts = any(len(getattr(r, "planned_actions", []) or []) > 1 for r in rollouts_snapshot)
        min_supp = max(2, mcts_cfg.tier_min_supporting_traces - 1)
        predictor = None
        # Phase-2: the empirical predictor takes the classified mood as a sharper conditioning
        # variable. Falls back through (cohort+state+mood → cohort+state → cohort → sop) on
        # sparse hits, so cold-start is safe (mood-tagged precedents are still rare).
        if mode == "union":
            predictor = UnionTrajectoryPredictor(
                mcts=MctsTrajectoryPredictor(
                    rollouts=rollouts_snapshot, chosen_action=chosen,
                    decay_lambda=mcts_cfg.data_prefetch_decay_lambda,
                ),
                empirical=EmpiricalTrajectoryPredictor(
                    db=db, sop_ref=exp.sop_ref, cohort=cohort or "", chosen_action=chosen,
                    min_supporting=min_supp, mood=mood or None,
                ),
            )
        elif mode == "mcts" or (mode == "auto" and has_deep_rollouts):
            predictor = MctsTrajectoryPredictor(
                rollouts=rollouts_snapshot, chosen_action=chosen,
                decay_lambda=mcts_cfg.data_prefetch_decay_lambda,
            )
        elif mode in ("empirical", "auto"):
            # Empirical retrieval path: run off the critical path via the background helper,
            # which adds the optional cheap-LLM {user_text} slot and replaces MCTS. (The
            # legacy union/mcts modes below stay inline for backward compatibility.)
            from ..planner.retrieval_prefetch import schedule_retrieval_prefetch_bg
            asyncio.create_task(schedule_retrieval_prefetch_bg(
                experiment_id=exp.id, sop_ref=exp.sop_ref, task=task,
                history=list(history), cohort=cohort or "", state=state or "",
                mood=mood or "", chosen_action=chosen, current_turn_index=current_turn_index,
                mcts_cfg=mcts_cfg, min_supporting=min_supp,
            ))
            predictor = None  # handled in background; skip the inline path below

        if predictor is not None:
            predictions = await predictor.predict(max_offset=mcts_cfg.rollout_depth or 3)
            # Q6b: pass classified cohort + mood so query_template can render with
            # production-grade context (not just rollout-derived). Both flow into the
            # {cohort} and {mood} placeholders that the seed SOPs use.
            plan = build_prefetch_plan_from_predictions(
                predictions, task=task, decay_lambda=mcts_cfg.data_prefetch_decay_lambda,
                cohort=cohort or "", mood=mood or "",
            )
            if plan:
                data_prefetch_manager.max_outstanding = mcts_cfg.data_prefetch_max_outstanding
                launched = await data_prefetch_manager.schedule(
                    experiment_id=exp.id,
                    sop_ref=exp.sop_ref,
                    task=task,
                    plan=plan,
                    current_turn_index=current_turn_index,
                    min_confidence=mcts_cfg.data_prefetch_min_confidence,
                )
                trace.data_prefetch_scheduled_after_turn = len(launched)

            # Milestone B (instruction prefetch): also pre-generate the agent's response
            # text for the top-1 predicted next-turn (action, state). On the next live
            # turn, if the classification matches, we skip the response_gen call and use
            # the pre-generated text verbatim. Empirical-path POC — exact match only.
            #
            # State seeding: the bare empirical predictor produces (action, offset) tuples
            # without conditioning on a specific user_state at offset 1, so we seed the
            # state independently using the same helper that pondering uses, then re-run
            # the empirical predictor with state_hints_topk to get the conditioned action.
            try:
                from ..planner.pondering import predict_likely_next_states
                next_states = await predict_likely_next_states(
                    db,
                    sop_ref=exp.sop_ref,
                    cohort=cohort or "unknown",
                    last_action=chosen,
                    k=1,
                    fallback_vocab=[s.name for s in task.user_states],
                )
                if next_states:
                    top_state, state_prob = next_states[0]
                    state_predictor = EmpiricalTrajectoryPredictor(
                        db=db, sop_ref=exp.sop_ref, cohort=cohort or "",
                        chosen_action=chosen, min_supporting=min_supp, mood=mood or None,
                    )
                    state_conditioned = await state_predictor.predict(
                        max_offset=1, state_hints_topk={1: [top_state]},
                    )
                    top_actions = sorted(
                        [p for p in state_conditioned if p.offset == 1 and p.action and p.predicted_user_state == top_state],
                        key=lambda p: -p.probability,
                    )
                    if top_actions:
                        top = top_actions[0]
                        asyncio.create_task(data_prefetch_manager.schedule_instruction_prefetch(
                            experiment_id=exp.id,
                            task=task,
                            history=list(history),
                            predicted_action=top.action,
                            predicted_state=top_state,
                            confidence=float(top.probability * state_prob),
                        ))
            except Exception:
                pass

    return TurnResponse(user_message=user_text, assistant_message=resp_text, trace=trace)


@router.post("/{session_id}/end")
async def end_session(session_id: str, db: AsyncSession = Depends(get_session)) -> dict:
    """Explicit user-driven end. Marks session abandoned unless already finalized.
    Triggers terminal back-propagation to all precedents in the session."""
    exp = await db.get(Experiment, session_id)
    if not exp:
        raise HTTPException(404, "session not found")
    if exp.terminal_outcome is not None:
        return {"ok": True, "outcome": exp.terminal_outcome, "already_ended": True}
    n = await finalize_experiment(db, experiment_id=exp.id, outcome="abandoned", reward=0.25)
    await db.commit()
    await pondering_scheduler.cancel_all(exp.id)
    await data_prefetch_manager.finalize_session(exp.id)
    return {"ok": True, "outcome": "abandoned", "precedents_updated": n}


async def _state_log(db: AsyncSession, exp_id: str) -> list[str]:
    res = await db.execute(
        select(TurnRecord.predicted_user_state)
        .where(TurnRecord.experiment_id == exp_id)
        .order_by(TurnRecord.turn_index)
    )
    return [r[0] for r in res.all() if r[0]]


@router.get("/{session_id}")
async def get_session_state(session_id: str, db: AsyncSession = Depends(get_session)) -> dict:
    exp, task = await _load_experiment(session_id, db)
    rows = (await db.execute(
        select(TurnRecord).where(TurnRecord.experiment_id == exp.id).order_by(TurnRecord.turn_index)
    )).scalars().all()
    return {
        "session_id": exp.id,
        "config": {
            "planner_mode": exp.planner_mode,
            "chat_mode": exp.chat_mode,
            "mcts": exp.mcts_config,
            "models": exp.models,
        },
        "notes": exp.notes,
        "terminal_outcome": exp.terminal_outcome,
        "terminal_reward": exp.terminal_reward,
        "ended_at": exp.ended_at.isoformat() if exp.ended_at else None,
        "history": exp.history,
        "traces": [r.trace for r in rows],
        "sop": task.model_dump(),
    }
