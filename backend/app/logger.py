"""Per-experiment logging: records each LLM call, turn, and MCTS candidate.

Buffers writes in memory and flushes them when the turn / experiment commits.
This keeps each LLM call non-blocking on DB I/O during a turn.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from .db import LLMCallRecord, MCTSCandidateRecord, RolloutRecord


@dataclass
class LLMCallEntry:
    call_site: str
    model: str
    started_at: datetime
    duration_ms: int
    tokens_in: int
    tokens_out: int
    temperature: Optional[float]
    max_tokens: Optional[int]
    system_prompt: str
    user_prompt: str
    response_text: str
    response_json: Optional[dict[str, Any]]
    is_json_mode: bool
    ok: bool
    error: Optional[str] = None


@dataclass
class CandidateEntry:
    action: str
    q_value: float
    visits: int
    rationale: str = ""
    was_chosen: bool = False


@dataclass
class RolloutEntry:
    rollout_index: int
    started_at: datetime
    duration_ms: int
    first_action: str
    planned_actions: list[str]
    final_state: str
    depth_completed: int
    hit_failure: bool
    hit_success: bool
    rationality: Optional[float]
    progress_bonus: float
    reward: float
    rollout_mode: str = "simulate"
    rollout_action_policy: str = "llm_top1"
    # Aligned with planned_actions; planned_states[i] is the user state when action
    # planned_actions[i] was taken. planned_states[0] = the real root user state.
    planned_states: list[str] = field(default_factory=list)
    # Q6b: aligned with planned_actions[1:]. The user-sim reply text generated after
    # each agent action inside the rollout. Used as query seed for query-aware prefetch.
    # Empty in value mode (no simulation).
    planned_user_texts: list[str] = field(default_factory=list)
    # Mood sampled at the start of this rollout from the cohort's mood prior. Different
    # parallel rollouts get different moods. Empty when cohort has no moods declared.
    mood: str = ""


class ExperimentLogger:
    """A logger attached to one experiment + (optionally) the current turn.

    Usage:
        logger = ExperimentLogger(experiment_id=...)
        # LLM client appends entries via logger.record_llm_call(...)
        # at end of turn, call logger.bind_turn(turn_id) then logger.flush(db)
    """

    def __init__(self, experiment_id: Optional[str] = None) -> None:
        self.experiment_id = experiment_id
        self.current_turn_id: Optional[str] = None
        self.calls: list[LLMCallEntry] = []
        self.candidates: list[CandidateEntry] = []
        self.rollouts: list[RolloutEntry] = []

    # ---- recording (non-async, called from inside LLM client) ----

    def record_llm_call(self, entry: LLMCallEntry) -> None:
        self.calls.append(entry)

    def record_candidate(self, entry: CandidateEntry) -> None:
        self.candidates.append(entry)

    def record_rollout(self, entry: RolloutEntry) -> None:
        self.rollouts.append(entry)

    def reset_turn_buffer(self) -> None:
        self.calls.clear()
        self.candidates.clear()
        self.rollouts.clear()

    # ---- DB flush ----

    async def flush(self, db: AsyncSession, *, turn_id: Optional[str] = None) -> None:
        """Persist all buffered LLM calls + candidates. Caller must commit.
        Pass turn_id to attribute calls/candidates to that turn."""
        tid = turn_id or self.current_turn_id
        for c in self.calls:
            db.add(LLMCallRecord(
                experiment_id=self.experiment_id,
                turn_id=tid,
                call_site=c.call_site,
                model=c.model,
                started_at=c.started_at,
                duration_ms=c.duration_ms,
                tokens_in=c.tokens_in,
                tokens_out=c.tokens_out,
                temperature=c.temperature,
                max_tokens=c.max_tokens,
                system_prompt=c.system_prompt,
                user_prompt=c.user_prompt,
                response_text=c.response_text,
                response_json=c.response_json,
                is_json_mode=c.is_json_mode,
                ok=c.ok,
                error=c.error,
            ))
        for i, ce in enumerate(self.candidates):
            if tid is None:
                continue
            db.add(MCTSCandidateRecord(
                turn_id=tid,
                rank=i,
                action=ce.action,
                q_value=ce.q_value,
                visits=ce.visits,
                rationale=ce.rationale,
                was_chosen=ce.was_chosen,
            ))
        for re in self.rollouts:
            if tid is None:
                continue
            db.add(RolloutRecord(
                turn_id=tid,
                rollout_index=re.rollout_index,
                started_at=re.started_at,
                duration_ms=re.duration_ms,
                first_action=re.first_action,
                planned_actions=re.planned_actions,
                planned_states=re.planned_states,
                planned_user_texts=re.planned_user_texts,
                final_state=re.final_state,
                depth_completed=re.depth_completed,
                hit_failure=re.hit_failure,
                hit_success=re.hit_success,
                rationality=re.rationality,
                progress_bonus=re.progress_bonus,
                reward=re.reward,
                rollout_mode=re.rollout_mode,
                rollout_action_policy=re.rollout_action_policy,
                mood=re.mood,
            ))
        self.reset_turn_buffer()
