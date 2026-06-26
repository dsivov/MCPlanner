from __future__ import annotations
import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, DateTime, JSON, Integer, Float, Boolean, ForeignKey, Index, LargeBinary,
    event, text,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
import sqlite_vec
from .config import settings

EMBED_DIM = 1536  # text-embedding-3-small native dim

engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)


# Load the sqlite-vec extension on every new connection. With aiosqlite, the
# SQLAlchemy adapter wraps the real sqlite3.Connection a couple of layers deep.
# We probe a few well-known attribute paths and pick the first that exposes
# enable_load_extension. Falls back silently if the extension can't be loaded
# (retrieval downgrades to SQL ORDER BY ... LIMIT).
def _resolve_real_sqlite_conn(dbapi_connection):
    """Reach through the SQLAlchemy + aiosqlite wrappers to the underlying sqlite3.Connection.

    aiosqlite wraps a sqlite3 connection that lives in a dedicated worker thread; its
    own methods are coroutines, so we need the raw sqlite3.Connection where
    enable_load_extension is a sync method.
    """
    import sqlite3
    candidates = []
    paths = (
        lambda c: c,
        lambda c: getattr(c, "_connection", None),                                    # AsyncAdapt -> aiosqlite.Connection
        lambda c: getattr(getattr(c, "_connection", None), "_conn", None),            # aiosqlite -> sqlite3.Connection
        lambda c: getattr(getattr(c, "driver_connection", None), "_conn", None),
    )
    for path in paths:
        try:
            cand = path(dbapi_connection)
        except Exception:
            cand = None
        if isinstance(cand, sqlite3.Connection):
            return cand
        candidates.append(cand)
    return None


@event.listens_for(engine.sync_engine, "connect")
def _load_sqlite_vec(dbapi_connection, _connection_record) -> None:
    real = _resolve_real_sqlite_conn(dbapi_connection)
    if real is None:
        import logging
        logging.getLogger(__name__).warning(
            "sqlite-vec: could not resolve underlying sqlite3 connection; vec queries will fall back"
        )
        return
    try:
        real.enable_load_extension(True)
        sqlite_vec.load(real)
        real.enable_load_extension(False)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("sqlite-vec load failed: %s", e)
    # exp() is not a SQLite built-in; the empirical predictor uses it for recency-decayed
    # weighting (SUM(reward * exp(-age/half_life))). Register math.exp as a scalar UDF.
    try:
        import math
        real.create_function("exp", 1, math.exp, deterministic=True)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("exp() UDF registration failed: %s", e)


SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
Base = declarative_base()


def _uid() -> str:
    return uuid.uuid4().hex[:12]


# --- SOP catalog ---

class SOPRecord(Base):
    __tablename__ = "sops"
    id = Column(String, primary_key=True, default=_uid)
    name = Column(String, nullable=False, default="Untitled SOP")
    description = Column(Text, default="")
    payload = Column(JSON, nullable=False)  # full TaskDefinition JSON
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# --- Experiments (one per chat session) ---

class Experiment(Base):
    __tablename__ = "experiments"
    id = Column(String, primary_key=True, default=_uid)
    sop_ref = Column(String, nullable=False)            # "seed:<file>" or sop_id
    sop_name = Column(String, nullable=False, default="")
    sop_snapshot = Column(JSON, nullable=False)         # frozen TaskDefinition at start

    planner_mode = Column(String, nullable=False)       # "mcts" | "baseline"
    chat_mode = Column(String, nullable=False)          # "human" | "auto"
    mcts_config = Column(JSON, nullable=False, default=dict)
    models = Column(JSON, nullable=False, default=dict)  # snapshot of model env at start

    history = Column(JSON, nullable=False, default=list)  # [{role, content, action?}] for quick replay
    notes = Column(Text, default="")

    # Session finalization. Set automatically when a success/failure marker is hit, or
    # manually via /api/chat/{id}/end. Drives terminal-outcome back-propagation to precedents.
    terminal_outcome = Column(String, nullable=True, index=True)  # "success" | "failure" | "abandoned"
    terminal_reward = Column(Float, nullable=True)
    ended_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# --- Turns: one row per agent turn ---

class TurnRecord(Base):
    __tablename__ = "turns"
    id = Column(String, primary_key=True, default=_uid)
    experiment_id = Column(String, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True)
    turn_index = Column(Integer, nullable=False)

    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    # duration_ms = AGENT-ONLY work (embed → propose → MCTS → response_gen).
    # user_sim_ms = the auto-user simulator's wall-clock (auto chat-mode only; 0 in human mode).
    # Together they recover the total turn wall-clock when needed.
    duration_ms = Column(Integer, default=0)
    user_sim_ms = Column(Integer, default=0)

    user_message = Column(Text, default="")
    assistant_message = Column(Text, default="")
    chosen_action = Column(String, default="")
    predicted_user_state = Column(String, default="")
    state_rationale = Column(Text, default="")
    # Phase-2 classified user mood within the cohort. Empty when the cohort has no moods or
    # the path didn't run mood classification (baseline planner, older SOPs).
    mood = Column(String, default="", index=True)

    mode = Column(String, default="mcts")               # snapshot of planner_mode used this turn
    tokens_in = Column(Integer, default=0)
    tokens_out = Column(Integer, default=0)
    mcts_iterations = Column(Integer, default=0)
    rollouts = Column(Integer, default=0)

    trace = Column(JSON, nullable=False, default=dict)  # full PlannerTrace JSON (denormalized for convenience)


Index("ix_turns_exp_turn", TurnRecord.experiment_id, TurnRecord.turn_index)


# --- LLM Calls: one row per OpenAI call ---

class LLMCallRecord(Base):
    __tablename__ = "llm_calls"
    id = Column(String, primary_key=True, default=_uid)
    experiment_id = Column(String, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=True, index=True)
    turn_id = Column(String, ForeignKey("turns.id", ondelete="CASCADE"), nullable=True, index=True)

    # call_site values (controlled vocabulary):
    #   "sop_builder", "state_predictor", "baseline_select",
    #   "mcts_propose_root", "mcts_rollout_action", "user_sim", "rollout_state_classify",
    #   "rationality", "response_gen"
    call_site = Column(String, nullable=False, index=True)
    model = Column(String, nullable=False)
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    duration_ms = Column(Integer, default=0)
    tokens_in = Column(Integer, default=0)
    tokens_out = Column(Integer, default=0)
    temperature = Column(Float, nullable=True)
    max_tokens = Column(Integer, nullable=True)

    system_prompt = Column(Text, default="")
    user_prompt = Column(Text, default="")
    response_text = Column(Text, default="")
    response_json = Column(JSON, nullable=True)         # parsed dict if JSON mode
    is_json_mode = Column(Boolean, default=False)
    ok = Column(Boolean, default=True)
    error = Column(Text, nullable=True)


# --- Rollouts: one row per MCTS rollout ---

class RolloutRecord(Base):
    __tablename__ = "rollouts"
    id = Column(String, primary_key=True, default=_uid)
    turn_id = Column(String, ForeignKey("turns.id", ondelete="CASCADE"), nullable=False, index=True)
    rollout_index = Column(Integer, default=0)
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    duration_ms = Column(Integer, default=0)

    first_action = Column(String, default="")           # the action this rollout estimates
    planned_actions = Column(JSON, nullable=False, default=list)
    # Aligned with planned_actions: planned_states[i] is the user state at the moment
    # planned_actions[i] was taken. Index 0 = the real root user_state; later entries are
    # simulated by user-sim calls inside the rollout. Used by state-aware Union predictor.
    planned_states = Column(JSON, nullable=False, default=list)
    # Q6b: aligned with planned_actions[1:]. The simulated user-sim reply at each depth
    # of the rollout. Used as query seed for query-aware data prefetch. Empty for value
    # mode (no simulation).
    planned_user_texts = Column(JSON, nullable=False, default=list)
    final_state = Column(String, default="")
    depth_completed = Column(Integer, default=0)
    hit_failure = Column(Boolean, default=False)
    hit_success = Column(Boolean, default=False)

    rationality = Column(Float, nullable=True)          # from end-of-rollout, if computed
    progress_bonus = Column(Float, default=0.0)
    reward = Column(Float, default=0.0)
    # "simulate" | "value" | "hybrid" — which rollout strategy produced this row
    rollout_mode = Column(String, default="simulate", index=True)
    # "llm_top1" | "llm_topk" | "bandit" — which per-step action policy was used.
    # Stored on each rollout so ablation comparisons can group by policy.
    rollout_action_policy = Column(String, default="llm_top1", index=True)
    # Mood sampled at the start of this rollout from the cohort's mood prior. Different
    # parallel rollouts get different moods. Empty for older SOPs or rollouts without
    # cohort context. Indexed for per-mood analysis (transition accuracy, prefetch hit rate).
    mood = Column(String, default="", index=True)


# --- MCTS candidates: one row per candidate action per turn ---

class MCTSCandidateRecord(Base):
    __tablename__ = "mcts_candidates"
    id = Column(String, primary_key=True, default=_uid)
    turn_id = Column(String, ForeignKey("turns.id", ondelete="CASCADE"), nullable=False, index=True)
    rank = Column(Integer, default=0)                   # 0 = best by mean Q
    action = Column(String, nullable=False)
    q_value = Column(Float, default=0.0)
    visits = Column(Integer, default=0)
    rationale = Column(Text, default="")
    was_chosen = Column(Boolean, default=False)


# --- Precedent / Context Graph tables (Fast loop) ---

class PrecedentTrace(Base):
    """One emitted precedent per agent turn. The substrate of the context graph."""
    __tablename__ = "precedent_traces"
    id = Column(String, primary_key=True, default=_uid)
    turn_id = Column(String, ForeignKey("turns.id", ondelete="CASCADE"), nullable=False, index=True, unique=True)
    experiment_id = Column(String, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True)
    sop_ref = Column(String, nullable=False, index=True)
    cohort = Column(String, nullable=False, default="", index=True)
    # Mood classified or inferred for this turn. Phase 1 leaves this NULL because runtime
    # mood classification isn't implemented yet — the column exists so that Phase 2 can
    # populate it without a separate migration, and so that current schema reflects the
    # full intended data model.
    mood = Column(String, nullable=True, index=True)

    situation_text = Column(Text, nullable=False, default="")
    # Float32 little-endian packed embedding. 1536 dims × 4 bytes = 6144 bytes.
    situation_embedding = Column(LargeBinary, nullable=True)

    action = Column(String, nullable=False, default="", index=True)
    response_text = Column(Text, default="")

    # Immediate signal: the user_state predicted at the START of the NEXT turn (if any).
    # Populated when the next turn fires; null while this is the last turn of the session.
    immediate_state = Column(String, nullable=True, index=True)
    immediate_reward = Column(Float, default=0.0)

    # Terminal signal: filled by session finalization (back-prop). Null until then.
    terminal_outcome = Column(String, nullable=True, index=True)
    terminal_reward = Column(Float, nullable=True)
    turn_distance_to_terminal = Column(Integer, nullable=True)  # how far we were from the end

    created_at = Column(DateTime, default=datetime.utcnow, index=True)


Index("ix_prec_cohort_sop", PrecedentTrace.sop_ref, PrecedentTrace.cohort)
Index("ix_prec_action_outcome", PrecedentTrace.action, PrecedentTrace.terminal_outcome)


class PrecedentRetrieval(Base):
    """One row per turn recording WHICH precedents were retrieved and which injection
    points consumed them. Join with precedent_traces to study influence."""
    __tablename__ = "precedent_retrievals"
    id = Column(String, primary_key=True, default=_uid)
    turn_id = Column(String, ForeignKey("turns.id", ondelete="CASCADE"), nullable=False, index=True, unique=True)
    experiment_id = Column(String, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True)
    sop_ref = Column(String, nullable=False)
    cohort = Column(String, nullable=False, default="")

    query_situation_text = Column(Text, default="")
    top_k_requested = Column(Integer, default=0)

    # Ordered list of {trace_id, similarity, action, terminal_outcome}. Denormalized for fast UI.
    results = Column(JSON, nullable=False, default=list)
    used_expand = Column(Boolean, default=False)
    used_score = Column(Boolean, default=False)
    used_response = Column(Boolean, default=False)

    duration_ms = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class RouterDecision(Base):
    """One row per turn capturing which tier the multi-tier router chose and why.

    Persisted so we can compute hit-rate / cost-savings empirically across sessions.
    """
    __tablename__ = "router_decisions"
    id = Column(String, primary_key=True, default=_uid)
    turn_id = Column(String, ForeignKey("turns.id", ondelete="CASCADE"), nullable=False, index=True, unique=True)
    experiment_id = Column(String, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True)
    sop_ref = Column(String, nullable=False, index=True)
    cohort = Column(String, nullable=False, default="", index=True)
    state = Column(String, nullable=False, default="", index=True)

    tier_used = Column(String, nullable=False, index=True)   # cached_playbook | baseline | mcts
    entropy = Column(Float, nullable=True)                    # Shannon bits over action distribution
    supporting_traces = Column(Integer, default=0)
    dominant_action = Column(String, nullable=True)
    dominant_agreement = Column(Float, nullable=True)         # fraction in [0,1] supporting dominant_action
    action_distribution = Column(JSON, nullable=True)         # {action_name: count} for analysis
    rationale = Column(Text, default="")

    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class PonderingRun(Base):
    """One speculative MCTS run launched between user turns.

    Created when a turn finishes; consumed (or not) when the NEXT turn fires.
    Persisting every run — even unconsumed — lets us measure cache-hit rate per
    cohort, latency saved, wasted tokens, and counterfactuals ("the user actually
    went into state X but we'd predicted Y").
    """
    __tablename__ = "pondering_runs"
    id = Column(String, primary_key=True, default=_uid)
    experiment_id = Column(String, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True)
    after_turn_index = Column(Integer, nullable=False, index=True)

    predicted_cohort = Column(String, nullable=False, default="", index=True)
    predicted_state = Column(String, nullable=False, default="", index=True)
    rank = Column(Integer, default=0)            # 0 = most-likely, 1 = second, etc.
    prior_prob = Column(Float, default=0.0)       # empirical P(state | cohort, last_action)

    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    duration_ms = Column(Integer, default=0)
    cancelled = Column(Boolean, default=False)

    result_json = Column(JSON, nullable=True)     # {chosen_action, candidates, mcts_iterations, rollouts}
    llm_calls_count = Column(Integer, default=0)
    tokens_in = Column(Integer, default=0)
    tokens_out = Column(Integer, default=0)

    consumed = Column(Boolean, default=False, index=True)
    consumed_turn_id = Column(String, nullable=True)


Index("ix_ponder_cache_key", PonderingRun.experiment_id, PonderingRun.after_turn_index,
      PonderingRun.predicted_cohort, PonderingRun.predicted_state)


class DataFetch(Base):
    """Persisted audit of every speculative or live data fetch — the raw material for
    the prediction-half-life curve and latency-hidden metric."""
    __tablename__ = "data_fetches"
    id = Column(String, primary_key=True, default=_uid)
    experiment_id = Column(String, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True)

    cache_key = Column(String, nullable=False, index=True)
    dependency_name = Column(String, nullable=False, index=True)
    action_name = Column(String, nullable=False, default="")
    kind = Column(String, default="mock")              # mock / rag / kg / db / api / mcp

    issued_at_turn = Column(Integer, nullable=False)   # turn index AFTER which we scheduled
    predicted_turn = Column(Integer, nullable=True)    # offset turn we expected it to serve
    consumed_at_turn = Column(Integer, nullable=True)  # turn it was actually consumed, NULL if never

    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    fetch_duration_ms = Column(Integer, default=0)    # wall-clock of the fetch itself
    confidence = Column(Float, default=0.0)            # aggregated trajectory score that triggered the schedule
    payload_summary = Column(Text, default="")        # short human-readable summary; full payload not persisted

    # Outcome flags. At most one of {consumed, wasted, evicted} is set when the fetch is finalized.
    consumed = Column(Boolean, default=False, index=True)
    wasted = Column(Boolean, default=False, index=True)       # never consumed before TTL or session end
    evicted = Column(Boolean, default=False)                   # dropped from queue due to cap

    # Mode flags for analysis.
    speculative = Column(Boolean, default=True, index=True)    # False = live (blocking) fetch issued at consume time
    fetch_error = Column(Text, nullable=True)

    # Which trajectory predictor scheduled this fetch: "mcts" / "empirical" / "both" / "live".
    # "live" = fallback live-fetch issued at consume time (not predictor-driven).
    predictor_source = Column(String, default="mcts", index=True)
    # The predicted user_state that conditioned this prediction (MCTS rollout-derived).
    # NULL when the predictor didn't supply one (e.g., state-blind empirical or live fetch).
    predicted_user_state = Column(String, nullable=True)
    # Q6b: rendered RAG/KG query string + cache hash. When NULL, the fetch was action-keyed
    # (legacy behaviour). When set, it's the specific question the supervisor predicted the
    # user would ask, used both as the fetcher's input AND for predicted-vs-live comparison.
    query_text = Column(Text, nullable=True)
    query_hash = Column(String, nullable=True, index=True)


Index("ix_datafetch_exp_key", DataFetch.experiment_id, DataFetch.cache_key)


class PoolPick(Base):
    """Pool-architecture v1: per-turn rerank decisions on the supervisor's prefetch pool.

    One row per turn that had data_prefetch_enabled and a non-empty pool. Records what
    the rerank step picked, what was available, how long it took, and the rationale.
    The substrate for pool-utilisation analysis (see notes/2026-06-03).
    """
    __tablename__ = "pool_picks"
    id = Column(String, primary_key=True, default=_uid)
    experiment_id = Column(String, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True)
    turn_index = Column(Integer, nullable=False, index=True)
    picked_fetch_ids = Column(JSON, nullable=False, default=list)   # list of DataFetch.id refs
    pool_size_at_pick = Column(Integer, default=0)
    rationale = Column(Text, default="")
    pick_duration_ms = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class LearningRun(Base):
    """One slow-loop run. Persisted regardless of whether proposals were accepted."""
    __tablename__ = "learning_runs"
    id = Column(String, primary_key=True, default=_uid)
    sop_ref = Column(String, nullable=False, index=True)
    started_at = Column(DateTime, default=datetime.utcnow, index=True)
    duration_ms = Column(Integer, default=0)

    n_precedents = Column(Integer, default=0)
    n_sessions = Column(Integer, default=0)
    summary = Column(JSON, nullable=False, default=dict)  # per-cohort, per-action lift table
    proposals = Column(JSON, nullable=False, default=list)  # list of {action, cohort, must_say_add, must_not_say_add, citations}
    accepted_proposal_ids = Column(JSON, nullable=False, default=list)


def _alembic_upgrade_head() -> None:
    """Run `alembic upgrade head` against our DATABASE_URL.

    Fresh DB → creates all tables from the migration history.
    Existing DB → applies any pending deltas.
    Already-current DB → no-op.

    Runs in a thread because alembic uses sync APIs internally.
    """
    from pathlib import Path
    from alembic import command
    from alembic.config import Config as AlembicConfig

    backend_root = Path(__file__).resolve().parents[1]
    cfg_path = backend_root / "alembic.ini"
    if not cfg_path.exists():
        # Alembic not initialized — fall back to legacy bootstrap.
        return
    alembic_cfg = AlembicConfig(str(cfg_path))
    # env.py reads settings.DATABASE_URL itself.
    command.upgrade(alembic_cfg, "head")


async def init_db() -> None:
    # 1) Apply any pending Alembic migrations BEFORE touching the engine.
    #    Alembic is sync; run it in a worker thread so we don't block the loop.
    import asyncio as _asyncio
    try:
        await _asyncio.to_thread(_alembic_upgrade_head)
    except Exception as e:
        # If migrations fail, fall back to legacy create_all so the app still starts.
        # Surface a clear log message.
        import logging
        logging.getLogger(__name__).warning("alembic upgrade head failed: %s — falling back to create_all", e)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    # 2) Create vec0 virtual table (idempotent). Excluded from Alembic autogenerate
    #    because virtual tables don't survive autogenerate's diffing.
    async with engine.begin() as conn:
        await conn.execute(text(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_precedents USING vec0(
                trace_id TEXT PRIMARY KEY,
                sop_ref  TEXT PARTITION KEY,
                cohort   TEXT PARTITION KEY,
                embedding FLOAT[{EMBED_DIM}]
            )
            """
        ))


async def get_session() -> AsyncSession:
    async with SessionLocal() as s:
        yield s
