"""Rollout-step action selection policies.

A "policy" decides which action the agent takes at one step inside an MCTS rollout.
The default `llm_top1` policy collapses parallel rollouts on narrow-SOP turns (the
LLM picks the same top action given the same prompt). The `bandit` policy uses
empirical priors + per-rollout local visit counts + softmax sampling so parallel
rollouts diverge.

Parallel-safety contract:
  - Each rollout owns its OWN `BanditState` (local visits dict).
  - Empirical priors come from a SQL read (concurrent reads on SQLite are safe).
  - There is no shared mutable state between rollouts in a batch.
  - Diversity emerges from softmax sampling — each rollout samples independently
    from the same distribution, naturally landing on different actions.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ..schemas import MCTSConfig, TaskDefinition
from ..llm.client import LLMResult


@dataclass
class PolicyResult:
    """Decision + diagnostics from a single rollout-step policy call."""
    action: str
    rationale: str = ""
    llm_result: Optional[LLMResult] = None     # None when policy is "bandit" (no LLM call)


@dataclass
class BanditState:
    """Per-rollout local visit counts. Mutated only by the rollout that owns it.

    Lives for the duration of one rollout (depth k steps). Resets to empty for the
    next rollout — that's the whole point of independent rollouts.
    """
    visits: dict[str, int] = field(default_factory=dict)

    def total(self) -> int:
        return sum(self.visits.values())

    def increment(self, action: str) -> None:
        self.visits[action] = self.visits.get(action, 0) + 1


# ---------- LLM-top-1 (legacy) ----------


async def llm_top1_step(
    *,
    task: TaskDefinition,
    history: list[dict[str, str]],
    last_user_state: str,
    allowed_actions: list[str],
    logger,
) -> PolicyResult:
    """Current behavior — LLM picks top-1 action. Same code path as before for backwards-compat."""
    from .mcts import _propose_actions   # avoid circular import at module load
    cands, res = await _propose_actions(
        task, history, last_user_state, allowed_actions, k=1,
        logger=logger, call_site="mcts_rollout_action",
    )
    if not cands:
        return PolicyResult(action=allowed_actions[0] if allowed_actions else "", llm_result=res)
    action, rationale = cands[0]
    return PolicyResult(action=action, rationale=rationale, llm_result=res)


# ---------- LLM-top-K with random sample ----------


async def llm_topk_step(
    *,
    task: TaskDefinition,
    history: list[dict[str, str]],
    last_user_state: str,
    allowed_actions: list[str],
    k: int,
    logger,
) -> PolicyResult:
    """LLM proposes top-K candidates; we uniformly sample one. Diversity comes from
    the per-rollout random sample, not the LLM call. Each rollout makes its own LLM
    call (same parallel pattern as today) and samples independently from its top-K."""
    from .mcts import _propose_actions
    cands, res = await _propose_actions(
        task, history, last_user_state, allowed_actions,
        k=min(k, len(allowed_actions)),
        logger=logger, call_site="mcts_rollout_action",
    )
    if not cands:
        return PolicyResult(action=allowed_actions[0] if allowed_actions else "", llm_result=res)
    chosen = random.choice(cands)
    return PolicyResult(action=chosen[0], rationale=chosen[1], llm_result=res)


# ---------- Bandit (UCT + empirical priors + per-rollout local state) ----------


def _softmax_sample(items: list[tuple[str, float]], temperature: float) -> str:
    """Stable softmax over scores, then weighted random pick."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0][0]
    temperature = max(temperature, 1e-6)
    max_score = max(s for _, s in items)
    exps = [math.exp((s - max_score) / temperature) for _, s in items]
    total = sum(exps) or 1.0
    weights = [e / total for e in exps]
    return random.choices([a for a, _ in items], weights=weights)[0]


async def bandit_step(
    *,
    task: TaskDefinition,
    db: AsyncSession,
    sop_ref: str,
    cohort: str,
    last_action_taken: Optional[str],     # action the rollout took at the PREVIOUS step (None on first step)
    allowed_actions: list[str],
    bandit_state: BanditState,
    cfg: MCTSConfig,
    priors_cache: Optional[dict] = None,   # OPTIONAL caller-supplied {(cohort,last_action) -> {action: P}} for cheap repeat-lookups in one rollout
) -> PolicyResult:
    """No LLM call. Pick an action via UCT scored by empirical priors + local visits.

    Parallel-safe: bandit_state is per-rollout; SQL on precedent_traces is concurrent-read.
    """
    # ε-greedy: with small probability take a uniformly-random action — keeps exploration
    # alive on cold-start when the empirical prior is uninformative.
    if random.random() < cfg.rollout_bandit_epsilon:
        action = random.choice(allowed_actions) if allowed_actions else ""
        if action:
            bandit_state.increment(action)
        return PolicyResult(action=action, rationale="bandit:ε-greedy uniform")

    # Pull empirical prior P(next_action | cohort, last_action). Falls back to uniform
    # when there's no historical data for this (cohort, last_action).
    priors = await _get_priors(
        db=db, sop_ref=sop_ref, cohort=cohort, last_action=last_action_taken,
        allowed_actions=allowed_actions, cache=priors_cache,
    )

    total_visits = max(1, bandit_state.total())
    # UCT: prior + c * sqrt(log(total) / max(visits, 1))
    scored: list[tuple[str, float]] = []
    for a in allowed_actions:
        p = priors.get(a, 1.0 / max(1, len(allowed_actions)))
        v = bandit_state.visits.get(a, 0)
        exploration = cfg.rollout_bandit_c_uct * math.sqrt(math.log(total_visits + 1) / max(v, 1))
        scored.append((a, p + exploration))

    action = _softmax_sample(scored, temperature=cfg.rollout_bandit_softmax_temp)
    if action:
        bandit_state.increment(action)
    return PolicyResult(action=action, rationale="bandit:UCT(priors+local visits)")


async def _get_priors(
    *,
    db: AsyncSession,
    sop_ref: str,
    cohort: str,
    last_action: Optional[str],
    allowed_actions: list[str],
    cache: Optional[dict] = None,
) -> dict[str, float]:
    """Query empirical P(next_action | cohort, last_action) restricted to SOP-allowed.

    Returns a normalized probability dict. Uniform over allowed_actions when the query
    is empty (cold start).
    """
    key = (cohort or "", last_action or "")
    if cache is not None and key in cache:
        # cached entry already filtered by allowed_actions for THIS rollout step? No — we
        # filter on lookup. So caching is keyed by (cohort, last_action) only; we re-filter.
        raw_dist = cache[key]
    else:
        raw_dist = await _lookup_distribution(db, sop_ref=sop_ref, cohort=cohort, last_action=last_action)
        if cache is not None:
            cache[key] = raw_dist

    # Filter to currently-allowed actions; renormalize.
    filtered = {a: float(c) for a, c in raw_dist.items() if a in set(allowed_actions)}
    total = sum(filtered.values())
    if total <= 0:
        return {a: 1.0 / max(1, len(allowed_actions)) for a in allowed_actions}
    return {a: v / total for a, v in filtered.items()}


async def _lookup_distribution(
    db: AsyncSession,
    *,
    sop_ref: str,
    cohort: str,
    last_action: Optional[str],
) -> dict[str, int]:
    """Empirical distribution of the action taken at turn N+1, given that action at turn N
    was `last_action`, within sessions on this SOP and (optionally) this cohort."""
    from sqlalchemy import text as sql_text
    # When last_action is None (very first rollout step from the candidate's root), we
    # query at the SOP+cohort level for opening actions.
    if last_action is None:
        sql = sql_text("""
            SELECT p.action, COUNT(*) AS freq
            FROM precedent_traces p
            JOIN turns t ON t.id = p.turn_id
            WHERE p.sop_ref = :sop_ref
              AND t.turn_index = 0
              AND (:cohort = '' OR p.cohort = :cohort)
            GROUP BY p.action
        """)
        params = {"sop_ref": sop_ref, "cohort": cohort or ""}
    else:
        sql = sql_text("""
            SELECT next_p.action, COUNT(*) AS freq
            FROM precedent_traces p
            JOIN turns t                ON t.id = p.turn_id
            JOIN turns next_t           ON next_t.experiment_id = t.experiment_id
                                       AND next_t.turn_index = t.turn_index + 1
            JOIN precedent_traces next_p ON next_p.turn_id = next_t.id
            WHERE p.sop_ref = :sop_ref
              AND p.action  = :last_action
              AND (:cohort = '' OR p.cohort = :cohort)
            GROUP BY next_p.action
        """)
        params = {"sop_ref": sop_ref, "cohort": cohort or "", "last_action": last_action}
    res = await db.execute(sql, params)
    return {row[0]: int(row[1]) for row in res.all()}
