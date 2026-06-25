"""Multi-tier router — decides whether a turn needs full MCTS or can be served
by a cheaper path, based on empirical agreement in precedent_traces.

Three tiers:
  tier_1  cached_playbook  — historical evidence is overwhelming for one action.
                              Skip MCTS, take the dominant action directly.
  tier_2  baseline         — moderate evidence. Skip MCTS, use the LLM's one-shot pick
                              from cohort_state_propose's candidates.
  tier_3  mcts             — novel situation or high disagreement. Run live MCTS.

Decision metric is Shannon entropy of the action distribution observed at
(sop_ref, cohort, state). Lower entropy = more agreement = cheaper tier.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import PrecedentTrace
from ..schemas import MCTSConfig


@dataclass
class TierStats:
    """Empirical statistics consulted by the router. All fields are populated even
    when the chosen tier is mcts (the data is useful for analysis either way)."""
    n_supporting: int = 0
    entropy: float = 0.0
    dominant_action: Optional[str] = None
    dominant_agreement: float = 0.0   # fraction of n_supporting that picked dominant_action
    distribution: dict[str, int] = field(default_factory=dict)


@dataclass
class TierDecision:
    tier: str          # "cached_playbook" | "baseline" | "mcts"
    rationale: str
    stats: TierStats


def _shannon_entropy(counts: dict[str, int]) -> float:
    """Shannon entropy in bits."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


async def gather_tier_stats(
    db: AsyncSession,
    *,
    sop_ref: str,
    cohort: str,
    state: str,
    weight_by_outcome: bool = True,
) -> TierStats:
    """Build the action distribution observed at (sop_ref, cohort, state).

    With weight_by_outcome=True (default), successful precedents count 2x and failures count 0.
    Open / abandoned precedents count 1x. This keeps the router honest about *good* historical
    moves rather than just *common* ones — important for safety against early-session mistakes
    getting baked in.
    """
    if not cohort or not state:
        return TierStats()

    q = (
        select(PrecedentTrace.action, PrecedentTrace.terminal_outcome, func.count(PrecedentTrace.id))
        .where(
            PrecedentTrace.sop_ref == sop_ref,
            PrecedentTrace.cohort == cohort,
            PrecedentTrace.immediate_state == state,   # IMPORTANT: matches the OBSERVED state after the action
        )
        .group_by(PrecedentTrace.action, PrecedentTrace.terminal_outcome)
    )
    rows = (await db.execute(q)).all()

    weighted: dict[str, float] = {}
    raw: dict[str, int] = {}
    for action, outcome, n in rows:
        if not action:
            continue
        if weight_by_outcome:
            mult = 2.0 if outcome == "success" else (0.0 if outcome == "failure" else 1.0)
        else:
            mult = 1.0
        weighted[action] = weighted.get(action, 0.0) + float(n) * mult
        raw[action] = raw.get(action, 0) + int(n)

    n_supporting = sum(raw.values())
    if n_supporting == 0:
        return TierStats()

    # Convert weighted to integer-ish for entropy (we care about ratios).
    # Drop zero-weighted actions so they don't artificially raise entropy.
    nonzero = {a: max(0.0, v) for a, v in weighted.items() if v > 0}
    if not nonzero:
        # All supporting precedents were failures — no positive signal.
        return TierStats(n_supporting=n_supporting, entropy=0.0, distribution=raw)

    total_w = sum(nonzero.values())
    # Compute entropy + dominant
    h = 0.0
    for v in nonzero.values():
        p = v / total_w
        h -= p * math.log2(p)
    dominant_action, dominant_w = max(nonzero.items(), key=lambda kv: kv[1])
    agreement = dominant_w / total_w if total_w > 0 else 0.0

    return TierStats(
        n_supporting=n_supporting,
        entropy=round(h, 4),
        dominant_action=dominant_action,
        dominant_agreement=round(agreement, 4),
        distribution=raw,
    )


def decide_tier(
    stats: TierStats,
    cfg: MCTSConfig,
    *,
    allowed_actions: list[str],
) -> TierDecision:
    """Choose a tier from precomputed stats. The dominant action MUST be in the
    SOP-allowed set for tier_1 — if it isn't (SOP has progressed and this action is no
    longer legal), fall through to tier_2 or tier_3 based on remaining signal.
    """
    if stats.n_supporting < cfg.tier_min_supporting_traces:
        if not cfg.tier3_enabled:
            return TierDecision(
                tier="baseline",
                rationale=f"only {stats.n_supporting} supporting precedents "
                          f"(< min {cfg.tier_min_supporting_traces}) — tier3 disabled, "
                          f"falling back to baseline + pool synthesis",
                stats=stats,
            )
        return TierDecision(
            tier="mcts",
            rationale=f"only {stats.n_supporting} supporting precedents "
                      f"(< min {cfg.tier_min_supporting_traces}) — running live MCTS",
            stats=stats,
        )

    dom_ok = (
        stats.dominant_action is not None
        and stats.dominant_action in allowed_actions
    )

    if dom_ok and stats.entropy <= cfg.tier_entropy_max_t1:
        return TierDecision(
            tier="cached_playbook",
            rationale=(
                f"entropy {stats.entropy:.2f} ≤ {cfg.tier_entropy_max_t1} and "
                f"{stats.dominant_agreement*100:.0f}% of {stats.n_supporting} traces agree on "
                f"{stats.dominant_action} — using cached playbook"
            ),
            stats=stats,
        )

    if stats.entropy <= cfg.tier_entropy_max_t2:
        return TierDecision(
            tier="baseline",
            rationale=(
                f"entropy {stats.entropy:.2f} ≤ {cfg.tier_entropy_max_t2} "
                f"(but {'not legal' if not dom_ok else 'no clear winner'}) — using baseline"
            ),
            stats=stats,
        )

    if not cfg.tier3_enabled:
        return TierDecision(
            tier="baseline",
            rationale=f"entropy {stats.entropy:.2f} > {cfg.tier_entropy_max_t2} "
                      f"— tier3 disabled, falling back to baseline + pool synthesis",
            stats=stats,
        )

    return TierDecision(
        tier="mcts",
        rationale=f"entropy {stats.entropy:.2f} > {cfg.tier_entropy_max_t2} — falling back to MCTS",
        stats=stats,
    )


async def route_turn(
    db: AsyncSession,
    *,
    sop_ref: str,
    cohort: str,
    state: str,
    allowed_actions: list[str],
    cfg: MCTSConfig,
) -> TierDecision:
    """Top-level entry: build stats + decide. Convenience wrapper."""
    if not cfg.router_enabled:
        if not cfg.tier3_enabled:
            return TierDecision(
                tier="baseline",
                rationale="router disabled, tier3 disabled — baseline",
                stats=TierStats(),
            )
        return TierDecision(
            tier="mcts",
            rationale="router disabled",
            stats=TierStats(),
        )
    stats = await gather_tier_stats(db, sop_ref=sop_ref, cohort=cohort, state=state)
    return decide_tier(stats, cfg, allowed_actions=allowed_actions)
