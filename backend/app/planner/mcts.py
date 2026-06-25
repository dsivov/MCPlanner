"""PCA-M: Monte Carlo Tree Search over agent-action sequences, constrained by the SOP."""

from __future__ import annotations
import asyncio
import math
import random
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from ..schemas import (
    TaskDefinition,
    MCTSConfig,
    PlannerTrace,
    CandidateAction,
    RetrievedPrecedent,
)
from ..config import settings
from ..logger import ExperimentLogger, CandidateEntry, RolloutEntry
from ..llm.client import chat_json, LLMResult
from ..llm.prompts import (
    MCTS_PROPOSE_SYSTEM, mcts_propose_user_prompt,
    COHORT_STATE_PROPOSE_SYSTEM, cohort_state_propose_user_prompt,
    COHORT_STATE_PROPOSE_STRATEGY_SYSTEM, cohort_state_propose_strategy_user_prompt,
    MCTS_ROLLOUT_STRATEGY_SYSTEM, mcts_rollout_strategy_user_prompt,
    VALUE_SCORE_SYSTEM, value_score_user_prompt,
    format_precedents_block,
)
from .sop_graph import SOPGraph
from .user_sim import simulate_user_with_state, simulate_user_end_rollout
from .reward import task_progress_bonus, combine_reward
from .state_predictor import predict_user_state


@dataclass
class Node:
    action: Optional[str]
    parent: Optional["Node"]
    visited: set[str]
    history: list[dict[str, str]]
    state_log: list[str]
    children: list["Node"] = field(default_factory=list)
    visits: int = 0
    in_flight: int = 0          # WU-PUCT: rollouts dispatched but not yet completed
    total_q: float = 0.0
    expanded: bool = False

    @property
    def effective_visits(self) -> int:
        """Visits + in-flight (Kim et al. WU-PUCT). Used in UCT denominators."""
        return self.visits + self.in_flight

    @property
    def mean_q(self) -> float:
        return self.total_q / self.visits if self.visits else 0.0

    def uct(self, c: float) -> float:
        """WU-PUCT-flavoured UCT: in-flight rollouts count as virtual visits so that K
        concurrent selectors don't pile onto the same leaf. Children with rollouts in
        flight get a stale-but-pessimistic mean_q (we treat in-flight as reward 0 for
        ranking purposes) and inflated visit counts that penalize re-selection."""
        ev = self.effective_visits
        if ev == 0:
            return float("inf")
        parent_visits = (self.parent.visits + self.parent.in_flight) if self.parent else 1
        # Treat in-flight as reward-0 virtual visits when computing the exploitation term:
        # mean over (visits + in_flight) with the in_flight contributions = 0.
        eff_mean = self.total_q / ev
        return eff_mean + c * math.sqrt(math.log(max(parent_visits, 1)) / ev)


async def _propose_strategies(
    task: TaskDefinition,
    history: list[dict[str, str]],
    predicted_user_state: str,
    allowed_strategy_names: list[str],
    k: int,
    *,
    logger: Optional[ExperimentLogger] = None,
    call_site: str = "mcts_rollout_strategy",
) -> tuple[list[tuple[str, str]], LLMResult]:
    """Strategy-level analogue of _propose_actions. Returns top-K (strategy_name, rationale)."""
    parsed, res = await chat_json(
        model=settings.MODEL_ROLLOUT,
        system=MCTS_ROLLOUT_STRATEGY_SYSTEM,
        user=mcts_rollout_strategy_user_prompt(task, history, predicted_user_state, allowed_strategy_names),
        temperature=0.4,
        max_tokens=300,
        logger=logger,
        call_site=call_site,
    )
    # The single-strategy variant returns {"strategy": ..., "rationale": ...} not a candidates list
    # — be tolerant of both shapes.
    out: list[tuple[str, str]] = []
    if "candidates" in parsed and isinstance(parsed["candidates"], list):
        seen: set[str] = set()
        for c in parsed["candidates"]:
            s = (c or {}).get("strategy") or (c or {}).get("action") or ""
            if s in allowed_strategy_names and s not in seen:
                out.append((s, (c or {}).get("rationale", "")))
                seen.add(s)
            if len(out) >= k:
                break
    else:
        s = (parsed.get("strategy") or "").strip()
        if s in allowed_strategy_names:
            out.append((s, parsed.get("rationale", "")))
    if not out and allowed_strategy_names:
        out = [(allowed_strategy_names[0], "fallback")]
    return out, res


async def _propose_actions(
    task: TaskDefinition,
    history: list[dict[str, str]],
    predicted_user_state: str,
    allowed: list[str],
    k: int,
    *,
    logger: Optional[ExperimentLogger] = None,
    call_site: str = "mcts_propose_root",
) -> tuple[list[tuple[str, str]], LLMResult]:
    # Root proposal stays moderate-temperature for stable candidate selection.
    # Rollout steps use a HIGHER temperature so parallel rollouts diverge across
    # the small SOP-allowed action set instead of all collapsing to the LLM's
    # top-1 pick — without this, rollouts produce ~identical trajectories on
    # narrow-SOP turns and MCTS averages 4 noisy samples of one path instead
    # of searching a tree. See research note 2026-05-23 §rollout-collapse.
    temperature = 0.6 if call_site == "mcts_propose_root" else 0.9
    parsed, res = await chat_json(
        model=settings.MODEL_SELECTION if call_site == "mcts_propose_root" else settings.MODEL_ROLLOUT,
        system=MCTS_PROPOSE_SYSTEM,
        user=mcts_propose_user_prompt(task, history, predicted_user_state, allowed, k),
        temperature=temperature,
        max_tokens=600,
        logger=logger,
        call_site=call_site,
    )
    cands = parsed.get("candidates", []) or []
    out: list[tuple[str, str]] = []
    seen = set()
    for c in cands:
        a = (c or {}).get("action", "")
        if a in allowed and a not in seen:
            out.append((a, (c or {}).get("rationale", "")))
            seen.add(a)
        if len(out) >= k:
            break
    if not out and allowed:
        # Greeting-loop fix (2026-06-02): when the LLM proposed only SOP-illegal actions
        # and we'd fall back to alphabetically-first allowed, prefer an UNVISITED action.
        # Otherwise we loop on no-prereq actions (Greeting) that are technically allowed
        # but already-done.
        visited = {h.get("action") for h in history if h.get("action")}
        unvisited = [a for a in allowed if a not in visited]
        out = [(unvisited[0] if unvisited else allowed[0], "fallback")]
    return out, res


@dataclass
class RolloutOutcome:
    reward: float
    planned: list[str]
    # Aligned with `planned`. planned_states[i] = user_state at the moment action planned[i]
    # was taken. planned_states[0] = the root (real, classified) user_state; subsequent
    # entries are simulated by the rollout's user-sim calls. Used by the state-aware
    # Union predictor to condition empirical lookups per offset.
    planned_states: list[str]
    # Aligned with planned[1:]. planned_user_texts[i] = simulated user reply AFTER action
    # planned[i+1] was taken. Index 0 of this list corresponds to the user-sim reply at
    # depth 1 of the rollout. Used by Q6b query-aware data prefetch as the seed text for
    # parameterized RAG/KG queries. Empty for value-mode rollouts (no simulation).
    planned_user_texts: list[str]
    final_state: str
    depth_completed: int
    hit_failure: bool
    hit_success: bool
    rationality: Optional[float]
    progress_bonus: float
    tokens_in: int
    tokens_out: int
    # Mood sampled per-rollout from the cohort's mood prior. Different parallel rollouts
    # get different moods. Persisted on RolloutRecord for analysis (per-mood transition
    # accuracy, per-mood prefetch hit rate, etc.).
    mood: str = ""


def _sample_cohort_mood(task: TaskDefinition, cohort_name: str) -> tuple[Optional[str], str]:
    """Sample one mood for this rollout from the named cohort's mood list, weighted by
    each mood's `prior`. Returns (name, description) or (None, "") when:
      - cohort_name is empty or doesn't match any cohort in the SOP
      - the matched cohort has no moods declared (older SOPs)
    Different `_rollout` invocations sample independently → diverse parallel rollouts.
    """
    if not cohort_name:
        return None, ""
    cohort_obj = next((c for c in task.cohorts if c.name == cohort_name), None)
    if cohort_obj is None or not cohort_obj.moods:
        return None, ""
    moods = cohort_obj.moods
    weights = [max(1e-6, m.prior) for m in moods]   # guard against all-zero priors
    chosen = random.choices(moods, weights=weights, k=1)[0]
    return chosen.name, chosen.description


async def _value_score(
    task: TaskDefinition,
    history: list[dict[str, str]],
    first_action: str,
    predicted_user_state: str,
    *,
    precedents_block: str = "",
    logger: Optional[ExperimentLogger] = None,
) -> tuple[float, float, str, LLMResult]:
    """Single LLM call returning (sequence_quality, success_probability, rationale).

    Replaces a depth-k step-by-step rollout with one value-model call. Used by
    rollout_mode="value" and the tail of "hybrid".
    """
    parsed, res = await chat_json(
        model=settings.MODEL_ROLLOUT,
        system=VALUE_SCORE_SYSTEM,
        user=value_score_user_prompt(
            task, history, first_action,
            planned_remaining_actions=[],  # planner may extend later
            predicted_user_state=predicted_user_state,
            precedents_block=precedents_block,
        ),
        temperature=0.3,
        max_tokens=200,
        logger=logger,
        call_site="mcts_value_rollout",
    )
    try:
        sq = float(parsed.get("sequence_quality", 0.0))
    except (TypeError, ValueError):
        sq = 0.0
    try:
        sp = float(parsed.get("success_probability", 0.0))
    except (TypeError, ValueError):
        sp = 0.0
    sq = max(0.0, min(1.0, sq))
    sp = max(0.0, min(1.0, sp))
    return sq, sp, (parsed.get("rationale") or ""), res


async def _rollout(
    task: TaskDefinition,
    g: SOPGraph,
    node: Node,
    depth: int,
    predicted_user_state: str,
    *,
    precedents_block: str = "",
    rollout_mode: str = "simulate",
    planning_granularity: str = "action",
    rollout_action_policy: str = "llm_top1",
    cfg: Optional[MCTSConfig] = None,
    db = None,
    sop_ref: str = "",
    cohort: str = "",
    priors_cache: Optional[dict] = None,
    logger: Optional[ExperimentLogger] = None,
) -> RolloutOutcome:
    """One rollout. Behaviour depends on rollout_mode:

      "simulate" — per-step rollout_action + user_sim_with_state for `depth` turns,
                   then user_sim_end_rollout for rationality. Most LLM calls (~2k+1).
      "value"    — single value-scoring LLM call that estimates Q directly.
      "hybrid"   — simulate the FIRST step (grounding), then value-score the rest.

    Short-circuits to reward 0 on failure-marker hits in simulate/hybrid modes.
    """
    tokens_in = 0
    tokens_out = 0
    history = list(node.history)
    visited = set(node.visited)
    state_log = list(node.state_log)
    # `planned` always tracks CONCRETE action names (not strategy names), so downstream
    # prompts that reference past actions see SOP-vocabulary names regardless of mode.
    # In strategy mode, the concrete action was already inserted into node.history with
    # an `action` field — pull from there. In action mode, node.action is the action.
    if node.action:
        if planning_granularity == "strategy" and node.history:
            last_msg = node.history[-1]
            concrete0 = (last_msg.get("action") or "") if last_msg.get("role") == "assistant" else ""
            planned: list[str] = [concrete0] if concrete0 else [node.action]
        else:
            planned = [node.action]
    else:
        planned = []
    last_state = predicted_user_state
    # planned_states[i] mirrors planned[i]: the user_state at the moment action planned[i]
    # was taken. planned_states[0] = the real root user state. Subsequent entries are filled
    # in by the rollout loop right before each action is appended.
    planned_states: list[str] = [predicted_user_state] if planned else []
    # Q6b: capture simulated user replies for query-aware data prefetch. Aligned to
    # planned[1:]; depth-1 reply goes at index 0. Empty for value-mode (no simulation).
    planned_user_texts: list[str] = []
    failure_markers = set(task.conversation_profile.failure_markers)
    success_markers = set(task.conversation_profile.success_markers)
    hit_failure = False
    hit_success = False

    last_step_rationality: Optional[float] = None
    depth_completed = 0
    # Bandit policy's per-rollout local state. Only used when rollout_action_policy=="bandit".
    from .rollout_policy import BanditState, bandit_step, llm_top1_step, llm_topk_step
    bandit_state = BanditState()
    last_action_taken_in_rollout: Optional[str] = node.action  # the root child's first action
    if planning_granularity == "strategy":
        # Strategy mode keeps LLM-only behaviour for now; bandit is action-mode only.
        rollout_action_policy = "llm_top1"

    # Sample a per-rollout mood from the cohort's mood prior. Each parallel rollout calls
    # this independently so they get diverse moods → diverse simulated user behaviour →
    # non-degenerate state distributions for top-K hedging. Empty when the cohort has no
    # moods declared (back-compat with older SOPs).
    mood_name, mood_description = _sample_cohort_mood(task, cohort)

    # ---- "value" mode: one LLM call, then exit ----
    if rollout_mode == "value":
        # In strategy mode, node.action is a strategy name; instantiate to a concrete action
        # so the value scorer sees vocabulary the SOP defines as agent_actions.
        if planning_granularity == "strategy":
            first_action = g.instantiate_strategy(node.action or "", node.visited)
        else:
            first_action = node.action or ""
        sq, sp, _rat, res_v = await _value_score(
            task, history, first_action, last_state,
            precedents_block=precedents_block, logger=logger,
        )
        tokens_in += res_v.tokens_in
        tokens_out += res_v.tokens_out
        # Reward composition: half-weight the LLM rationality, half-weight a "progress" proxy from
        # success_probability. Lands in roughly the same [0,0.75] band as simulate-mode rewards.
        reward = combine_reward(sq, sp)
        return RolloutOutcome(
            reward=reward,
            planned=planned,
            planned_states=planned_states,
            planned_user_texts=planned_user_texts,
            final_state=last_state,
            depth_completed=0,
            hit_failure=False,
            hit_success=False,
            rationality=sq,
            progress_bonus=sp,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            mood=mood_name or "",
        )

    # Hybrid: depth becomes "1 simulated step then value-score" — clamp depth to >=1.
    if rollout_mode == "hybrid":
        depth = 1

    for step in range(depth):
        if planning_granularity == "strategy":
            allowed_strats = g.allowed_strategies(visited)
            allowed_names = [s.name for s in allowed_strats]
            if not allowed_names:
                break
            cands, res = await _propose_strategies(
                task, history, last_state, allowed_names, k=1,
                logger=logger, call_site="mcts_rollout_strategy",
            )
            tokens_in += res.tokens_in
            tokens_out += res.tokens_out
            if not cands:
                break
            strategy_name, _ = cands[0]
            # Instantiate the strategy to a concrete action via SOP-allowed members.
            concrete = g.instantiate_strategy(strategy_name, visited)
            if not concrete:
                break
            action = concrete
            planned.append(strategy_name)   # track the search-tree path in strategies
            planned_states.append(last_state)
        else:
            allowed = g.allowed_actions(visited)
            if not allowed:
                break
            # Policy dispatch — see rollout_policy.py for parallel-safety notes.
            if rollout_action_policy == "bandit" and db is not None and cfg is not None:
                pr = await bandit_step(
                    task=task, db=db, sop_ref=sop_ref, cohort=cohort,
                    last_action_taken=last_action_taken_in_rollout,
                    allowed_actions=allowed, bandit_state=bandit_state,
                    cfg=cfg, priors_cache=priors_cache,
                )
            elif rollout_action_policy == "llm_topk" and cfg is not None:
                pr = await llm_topk_step(
                    task=task, history=history, last_user_state=last_state,
                    allowed_actions=allowed, k=cfg.rollout_action_topk, logger=logger,
                )
            else:
                pr = await llm_top1_step(
                    task=task, history=history, last_user_state=last_state,
                    allowed_actions=allowed, logger=logger,
                )
            if pr.llm_result is not None:
                tokens_in += pr.llm_result.tokens_in
                tokens_out += pr.llm_result.tokens_out
            if not pr.action:
                break
            action = pr.action
            last_action_taken_in_rollout = action
            planned.append(action)
            planned_states.append(last_state)
        visited.add(action)

        history = history + [{"role": "assistant", "content": f"<{action}>", "action": action}]

        is_last_step = (step == depth - 1)
        if is_last_step and rollout_mode == "simulate":
            # Merged user_sim + rationality at the end of the rollout: one call instead of two.
            user_text, state, rationality_score, res_us = await simulate_user_end_rollout(
                task, history, planned, logger=logger, precedents_block=precedents_block,
                mood_name=mood_name, mood_description=mood_description,
            )
            last_step_rationality = rationality_score
        else:
            user_text, state, res_us = await simulate_user_with_state(
                task, history, logger=logger,
                mood_name=mood_name, mood_description=mood_description,
            )
        tokens_in += res_us.tokens_in
        tokens_out += res_us.tokens_out
        history = history + [{"role": "user", "content": user_text}]
        # Q6b: capture for query-aware prefetch. Aligned with planned[1:] — depth-1's
        # user reply becomes planned_user_texts[0], depth-2's becomes [1], etc.
        planned_user_texts.append(user_text or "")
        depth_completed += 1
        if state:
            visited.add(state)
            state_log.append(state)
            last_state = state
            if state in failure_markers:
                hit_failure = True
                break
            if state in success_markers:
                hit_success = True
                break

    # Reward selection. Short-circuit when we already know the outcome.
    if hit_failure:
        reward = 0.0
        rationality_used: Optional[float] = None
        progress = 0.0
    elif hit_success:
        rationality_used = 1.0
        progress = 1.0
        reward = combine_reward(1.0, 1.0)
    elif rollout_mode == "hybrid":
        # After one simulated step we hand off to the value model for the tail.
        sq, sp, _rat, res_v = await _value_score(
            task, history,
            first_action=(planned[0] if planned else (node.action or "")),
            predicted_user_state=last_state,
            precedents_block=precedents_block, logger=logger,
        )
        tokens_in += res_v.tokens_in
        tokens_out += res_v.tokens_out
        rationality_used = sq
        progress = max(task_progress_bonus(task, state_log), sp)
        reward = combine_reward(sq, progress)
    else:
        rationality_used = last_step_rationality if last_step_rationality is not None else 0.0
        progress = task_progress_bonus(task, state_log)
        reward = combine_reward(rationality_used, progress)

    return RolloutOutcome(
        reward=reward,
        planned=planned,
        planned_states=planned_states,
        planned_user_texts=planned_user_texts,
        final_state=last_state,
        depth_completed=depth_completed,
        hit_failure=hit_failure,
        hit_success=hit_success,
        rationality=rationality_used,
        progress_bonus=progress,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        mood=mood_name or "",
    )


def _select(node: Node, c: float) -> Node:
    cur = node
    while cur.expanded and cur.children:
        cur = max(cur.children, key=lambda n: n.uct(c))
    return cur


def _backprop(node: Node, reward: float) -> None:
    cur: Optional[Node] = node
    while cur is not None:
        cur.visits += 1
        cur.total_q += reward
        cur = cur.parent


async def _cohort_state_propose_strategy(
    task: TaskDefinition,
    history: list[dict[str, str]],
    allowed_strategies: list[str],
    k: int,
    *,
    precedents: list[RetrievedPrecedent] | None = None,
    use_precedents: bool = False,
    logger: Optional[ExperimentLogger] = None,
) -> tuple[str, str, str, list[tuple[str, str]], LLMResult]:
    """Strategy-level merged cohort + state + propose. Returns (cohort, state, state_rationale, strategy_candidates, llm_result)."""
    block = format_precedents_block(precedents) if (use_precedents and precedents) else ""
    parsed, res = await chat_json(
        model=settings.MODEL_SELECTION,
        system=COHORT_STATE_PROPOSE_STRATEGY_SYSTEM,
        user=cohort_state_propose_strategy_user_prompt(task, history, allowed_strategies, k, precedents_block=block),
        temperature=0.5,
        max_tokens=800,
        logger=logger,
        call_site="cohort_state_propose_strategy",
    )
    cohort = (parsed.get("cohort") or "").strip() or "unknown"
    if task.cohorts:
        vocab = {c.name for c in task.cohorts}
        if cohort not in vocab:
            cohort = next(iter(vocab))
    state = (parsed.get("user_state") or "").strip()
    valid_states = {s.name for s in task.user_states}
    if state not in valid_states and valid_states:
        state = next(iter(valid_states))
    state_rat = parsed.get("state_rationale", "") or ""
    cands_raw = parsed.get("candidates", []) or []
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for c in cands_raw:
        name = (c or {}).get("strategy") or (c or {}).get("action") or ""
        if name in allowed_strategies and name not in seen:
            out.append((name, (c or {}).get("rationale", "")))
            seen.add(name)
        if len(out) >= k:
            break
    if not out and allowed_strategies:
        out = [(allowed_strategies[0], "fallback")]
    return cohort, state, state_rat, out, res


async def _cohort_state_propose(
    task: TaskDefinition,
    history: list[dict[str, str]],
    allowed: list[str],
    k: int,
    *,
    precedents: list[RetrievedPrecedent] | None = None,
    use_precedents: bool = False,
    logger: Optional[ExperimentLogger] = None,
) -> tuple[str, str, str, str, list[tuple[str, str]], LLMResult]:
    """Merged cohort + state + propose. Returns (cohort, state, mood, state_rationale, candidates, llm_result).

    `mood` is the classified user-disposition within the cohort. Empty string when the
    chosen cohort has no moods declared OR the LLM emitted a mood not in that cohort's
    vocabulary. Phase 2 of the disposition-diversity programme — see asymmetry note dated
    2026-05-23 for the rationale and 2026-05-28 follow-up for the offset+1 immutability
    finding that motivates the runtime-classifier addition.
    """
    block = format_precedents_block(precedents) if (use_precedents and precedents) else ""
    parsed, res = await chat_json(
        model=settings.MODEL_SELECTION,
        system=COHORT_STATE_PROPOSE_SYSTEM,
        user=cohort_state_propose_user_prompt(task, history, allowed, k, precedents_block=block),
        temperature=0.5,
        max_tokens=800,
        logger=logger,
        call_site="cohort_state_propose",
    )
    cohort = (parsed.get("cohort") or "").strip() or "unknown"
    if task.cohorts:
        vocab = {c.name for c in task.cohorts}
        if cohort not in vocab:
            cohort = next(iter(vocab))  # snap to vocabulary
    state = (parsed.get("user_state") or "").strip()
    valid_states = {s.name for s in task.user_states}
    if state not in valid_states and valid_states:
        state = next(iter(valid_states))
    # Mood validation: must belong to the chosen cohort's mood vocabulary. Falls back to
    # "" when the cohort has no moods or the LLM emitted an invalid label.
    mood = (parsed.get("mood") or "").strip()
    if mood:
        cohort_obj = next((c for c in task.cohorts if c.name == cohort), None)
        valid_moods = {m.name for m in (cohort_obj.moods if cohort_obj else [])}
        if mood not in valid_moods:
            mood = ""
    state_rat = parsed.get("state_rationale", "") or ""
    cands_raw = parsed.get("candidates", []) or []
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for c in cands_raw:
        a = (c or {}).get("action", "")
        if a in allowed and a not in seen:
            out.append((a, (c or {}).get("rationale", "")))
            seen.add(a)
        if len(out) >= k:
            break
    if not out and allowed:
        # Greeting-loop fix (2026-06-02): prefer UNVISITED action over alphabetically-
        # first. Without this, when LLM proposes only SOP-illegal actions the fallback
        # repeatedly picks no-prereq actions like Greeting that are already done.
        visited = {h.get("action") for h in history if h.get("action")}
        unvisited = [a for a in allowed if a not in visited]
        out = [(unvisited[0] if unvisited else allowed[0], "fallback")]
    return cohort, state, mood, state_rat, out, res


async def run_mcts(
    task: TaskDefinition,
    history: list[dict[str, str]],
    state_log: list[str],
    cfg: MCTSConfig,
    trace: PlannerTrace,
    *,
    precedents: list[RetrievedPrecedent] | None = None,
    pre_classified: Optional[tuple[str, str, str, list[tuple[str, str]]]] = None,
    db = None,
    sop_ref: str = "",
    cohort_for_bandit: str = "",
    logger: Optional[ExperimentLogger] = None,
) -> tuple[str, str, str, str]:
    """Returns (chosen_action, cohort, predicted_user_state, state_rationale).

    If `pre_classified=(cohort, state, state_rationale, candidates)` is provided, the
    internal cohort_state_propose call is skipped. The caller is responsible for having
    already updated trace.tokens_in/out for that earlier call.
    """
    precedents = precedents or []
    precedents_block_expand = format_precedents_block(precedents) if cfg.use_precedents_expand and precedents else ""
    precedents_block_score = format_precedents_block(precedents) if cfg.use_precedents_score and precedents else ""
    trace.precedents_used_expand = bool(precedents_block_expand)
    trace.precedents_used_score = bool(precedents_block_score)
    g = SOPGraph(task)
    root_visited = g.visited_from_history(history, state_log)
    root = Node(action=None, parent=None, visited=root_visited, history=list(history), state_log=list(state_log))
    root.expanded = True

    is_strategy_mode = (cfg.planning_granularity == "strategy")
    if is_strategy_mode:
        # In strategy mode, search nodes are strategies; we still validate via action SOP.
        allowed_action_set = set(g.allowed_actions(root.visited))
        allowed_strategy_objs = g.allowed_strategies(root.visited)
        allowed = [s.name for s in allowed_strategy_objs]
    else:
        allowed = g.allowed_actions(root.visited)
    candidate_rationales: dict[str, str] = {}

    # Short-circuit: when the SOP forces a single action/strategy, MCTS rollouts would all
    # evaluate the same path. Skip the rollouts entirely and run only a single state-predictor
    # call so the trace still carries the predicted user state for downstream analysis.
    if len(allowed) == 1:
        only_name = allowed[0]
        state, state_rat, _ = await predict_user_state(task, history, logger=logger)
        # In strategy mode, instantiate the lone strategy to its concrete action.
        only_action = g.instantiate_strategy(only_name, root.visited) if is_strategy_mode else only_name
        trace.chosen_action = only_action
        if is_strategy_mode:
            trace.chosen_strategy = only_name
        trace.candidates = [CandidateAction(
            action=only_name, q_value=0.0, visits=0,
            rationale="Only SOP-legal candidate at this turn; MCTS skipped.",
        )]
        trace.mcts_iterations = 0
        trace.rollouts = 0
        trace.mode = "mcts"
        trace.planning_granularity = cfg.planning_granularity
        if logger is not None:
            logger.record_candidate(CandidateEntry(
                action=only_name, q_value=0.0, visits=0,
                rationale="Only SOP-legal candidate at this turn; MCTS skipped.",
                was_chosen=True,
            ))
        return only_action, "", state, "", state_rat

    # Branching=1: skip the root proposal entirely. Each parallel rollout picks its own first
    # action via mcts_rollout_action, giving us implicit branching across iterations.
    # State prediction still needs to happen — do it in the first rollout's user_sim step
    # would be wrong (no agent action yet), so we use a one-shot state classifier here as a
    # minimum. Cheapest path: run state_and_propose with k=1, accept its candidate as a hint,
    # but treat all iterations as independent rollouts from a virtual "no-children" root.
    classified_mood: str = ""    # populated by action-mode classifier only; "" otherwise
    if cfg.branching <= 1:
        if pre_classified is not None:
            cohort, predicted_state, state_rat, cands = pre_classified
        elif is_strategy_mode:
            cohort, predicted_state, state_rat, cands, res = await _cohort_state_propose_strategy(
                task, history, allowed, k=1,
                precedents=precedents, use_precedents=cfg.use_precedents_expand,
                logger=logger,
            )
            trace.tokens_in += res.tokens_in
            trace.tokens_out += res.tokens_out
        else:
            cohort, predicted_state, classified_mood, state_rat, cands, res = await _cohort_state_propose(
                task, history, allowed, k=1,
                precedents=precedents, use_precedents=cfg.use_precedents_expand,
                logger=logger,
            )
            trace.tokens_in += res.tokens_in
            trace.tokens_out += res.tokens_out
        for a, r in cands:
            candidate_rationales[a] = r
    else:
        if pre_classified is not None:
            cohort, predicted_state, state_rat, cands = pre_classified
        elif is_strategy_mode:
            cohort, predicted_state, state_rat, cands, res = await _cohort_state_propose_strategy(
                task, history, allowed, k=cfg.branching,
                precedents=precedents, use_precedents=cfg.use_precedents_expand,
                logger=logger,
            )
            trace.tokens_in += res.tokens_in
            trace.tokens_out += res.tokens_out
        else:
            cohort, predicted_state, classified_mood, state_rat, cands, res = await _cohort_state_propose(
                task, history, allowed, k=cfg.branching,
                precedents=precedents, use_precedents=cfg.use_precedents_expand,
                logger=logger,
            )
            trace.tokens_in += res.tokens_in
            trace.tokens_out += res.tokens_out
        for name, rat in cands:
            # name == strategy name in strategy mode, action name in action mode
            if is_strategy_mode:
                concrete = g.instantiate_strategy(name, root.visited)
            else:
                concrete = name
            child_visited = set(root.visited)
            if concrete:
                child_visited.add(concrete)
            child_history = history + [{"role": "assistant", "content": f"<{concrete}>", "action": concrete}]
            child = Node(
                action=name,                 # SEARCH-TREE identifier (strategy or action name)
                parent=root,
                visited=child_visited,
                history=child_history,
                state_log=list(state_log),
            )
            root.children.append(child)
            candidate_rationales[name] = rat
    trace.cohort = cohort
    trace.planning_granularity = cfg.planning_granularity

    if cfg.branching > 1 and not root.children:
        fallback = next(iter(g.action_names), "")
        trace.chosen_action = fallback
        trace.candidates = []
        trace.mode = "mcts"
        return fallback, cohort, predicted_state, classified_mood, state_rat

    rollout_count = 0
    iters_done = 0
    parallel = max(1, getattr(cfg, "parallel_rollouts", 1) or 1)
    nee_threshold = float(getattr(cfg, "nee_threshold", 0.0) or 0.0)
    nee_min_visits = int(getattr(cfg, "nee_min_visits", 0) or 0)

    # Aggregate (q-sum, visits) per first-action when branching=1; rollouts pick their own first.
    b1_q: dict[str, float] = {}
    b1_v: dict[str, int] = {}

    nee_triggered = False
    remaining = cfg.iterations
    while remaining > 0:
        n = min(parallel, remaining)
        if cfg.branching <= 1:
            leaves: list[Node] = [root] * n
        else:
            # WU-PUCT: select K leaves under current stats + in_flight virtual losses,
            # then add in_flight=1 immediately so subsequent selectors see "this leaf is busy".
            leaves = []
            unvisited = [c for c in root.children if c.effective_visits == 0]
            for c in unvisited[:n]:
                leaves.append(c)
                c.in_flight += 1
            while len(leaves) < n:
                pick = _select(root, cfg.c_uct)
                leaves.append(pick)
                pick.in_flight += 1

        # Time the batch so we can record per-rollout duration approximately
        batch_t0 = time.perf_counter()
        batch_started_at = datetime.utcnow()

        # Per-iteration priors cache, shared across the K parallel rollouts in this batch.
        # Safe because it's READ-ONLY after population and rollouts only mutate their own
        # BanditState. SQLite concurrent reads are fine.
        priors_cache: dict = {}
        outcomes: list[RolloutOutcome] = await asyncio.gather(*[
            _rollout(
                task=task, g=g, node=leaf, depth=cfg.rollout_depth,
                predicted_user_state=predicted_state,
                precedents_block=precedents_block_score,
                rollout_mode=cfg.rollout_mode,
                planning_granularity=cfg.planning_granularity,
                rollout_action_policy=cfg.rollout_action_policy,
                cfg=cfg,
                db=db,
                sop_ref=sop_ref,
                cohort=cohort_for_bandit,
                priors_cache=priors_cache,
                logger=logger,
            )
            for leaf in leaves
        ])

        batch_duration_ms = int((time.perf_counter() - batch_t0) * 1000)

        for leaf, out in zip(leaves, outcomes):
            trace.tokens_in += out.tokens_in
            trace.tokens_out += out.tokens_out
            rollout_count += 1
            iters_done += 1
            if cfg.branching > 1:
                leaf.in_flight = max(0, leaf.in_flight - 1)
            if cfg.branching <= 1:
                first = out.planned[0] if out.planned else ""
                if first:
                    b1_q[first] = b1_q.get(first, 0.0) + out.reward
                    b1_v[first] = b1_v.get(first, 0) + 1
            else:
                _backprop(leaf, out.reward)

            if logger is not None:
                logger.record_rollout(RolloutEntry(
                    rollout_index=rollout_count - 1,
                    started_at=batch_started_at,
                    duration_ms=batch_duration_ms,  # batch wall-clock (rollouts ran concurrently)
                    first_action=(out.planned[0] if out.planned else (leaf.action or "")),
                    planned_actions=list(out.planned),
                    planned_states=list(out.planned_states),
                    planned_user_texts=list(out.planned_user_texts),
                    mood=out.mood or "",
                    final_state=out.final_state or "",
                    depth_completed=out.depth_completed,
                    hit_failure=out.hit_failure,
                    hit_success=out.hit_success,
                    rationality=out.rationality,
                    progress_bonus=out.progress_bonus,
                    reward=out.reward,
                    rollout_mode=cfg.rollout_mode,
                    rollout_action_policy=cfg.rollout_action_policy,
                ))
        remaining -= n

        # Strict Negative Early Exit: if every well-visited candidate is below threshold,
        # the rest of the iterations are wasted. Stop searching.
        if nee_threshold > 0 and remaining > 0:
            if cfg.branching > 1 and root.children:
                eligible = [c for c in root.children if c.visits >= nee_min_visits]
                if eligible and max(c.mean_q for c in eligible) < nee_threshold:
                    nee_triggered = True
                    remaining = 0
            elif cfg.branching <= 1 and b1_v:
                eligible_actions = [a for a, v in b1_v.items() if v >= nee_min_visits]
                if eligible_actions:
                    best_q = max(b1_q[a] / b1_v[a] for a in eligible_actions)
                    if best_q < nee_threshold:
                        nee_triggered = True
                        remaining = 0

    if cfg.branching <= 1:
        # Build pseudo-children from aggregated rollouts so the trace / candidate table look the same.
        actions_seen = sorted(b1_v.keys(), key=lambda a: (-b1_q[a] / max(b1_v[a], 1), -b1_v[a]))
        if not actions_seen:
            fallback = next(iter(g.action_names), "")
            trace.chosen_action = fallback
            trace.candidates = []
            trace.mode = "mcts"
            trace.mcts_iterations = iters_done
            trace.rollouts = rollout_count
            return fallback, cohort, predicted_state, classified_mood, state_rat
        best_name = actions_seen[0]
        # In strategy mode the names returned are strategies — instantiate to a concrete action.
        best_action = g.instantiate_strategy(best_name, root_visited) if is_strategy_mode else best_name
        trace.chosen_action = best_action
        if is_strategy_mode:
            trace.chosen_strategy = best_name
        trace.candidates = [
            CandidateAction(
                action=a,
                q_value=round(b1_q[a] / max(b1_v[a], 1), 4),
                visits=b1_v[a],
                rationale=candidate_rationales.get(a, ""),
            )
            for a in actions_seen
        ]
        if logger is not None:
            for a in actions_seen:
                logger.record_candidate(CandidateEntry(
                    action=a,
                    q_value=round(b1_q[a] / max(b1_v[a], 1), 4),
                    visits=b1_v[a],
                    rationale=candidate_rationales.get(a, ""),
                    was_chosen=(a == best_name),
                ))
        trace.mcts_iterations = iters_done
        trace.rollouts = rollout_count
        trace.mode = "mcts"
        trace.nee_triggered = nee_triggered
        return best_action, cohort, predicted_state, classified_mood, state_rat

    # Standard branching>=2 path
    best = max(root.children, key=lambda n: (n.mean_q, n.visits))
    best_name = best.action or ""
    # In strategy mode `best_name` is a strategy — instantiate it to a concrete action.
    best_action = g.instantiate_strategy(best_name, root_visited) if is_strategy_mode else best_name
    trace.chosen_action = best_action
    if is_strategy_mode:
        trace.chosen_strategy = best_name
    sorted_children = sorted(root.children, key=lambda n: (-n.mean_q, -n.visits))
    trace.candidates = [
        CandidateAction(
            action=c.action or "",
            q_value=round(c.mean_q, 4),
            visits=c.visits,
            rationale=candidate_rationales.get(c.action or "", ""),
        )
        for c in sorted_children
    ]
    trace.mcts_iterations = iters_done
    trace.rollouts = rollout_count
    trace.mode = "mcts"
    trace.nee_triggered = nee_triggered

    if logger is not None:
        for c in sorted_children:
            logger.record_candidate(CandidateEntry(
                action=c.action or "",
                q_value=round(c.mean_q, 4),
                visits=c.visits,
                rationale=candidate_rationales.get(c.action or "", ""),
                was_chosen=(c.action == best_name),
            ))

    return best_action, cohort, predicted_state, classified_mood, state_rat
