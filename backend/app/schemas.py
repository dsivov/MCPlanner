from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ---------- SOP / Task Definition (paper's schema) ----------

class UserProfile(BaseModel):
    """Who the user is — used by the user-simulator and surfaced to the planner."""
    name: Optional[str] = None
    description: str = ""
    demographics: dict[str, str] = Field(default_factory=dict)


class ConversationProfile(BaseModel):
    """The agent's role, goal, and success criteria."""
    agent_role: str = ""
    goal: str = ""
    success_markers: list[str] = Field(default_factory=list)
    failure_markers: list[str] = Field(default_factory=list)
    knowledge: str = ""


class NamedItem(BaseModel):
    name: str
    description: str = ""
    # Slow-loop learned content. Promoted phrases (`must_say`) are injected into
    # response generation; demoted phrases (`must_not_say`) are explicitly avoided.
    must_say: list[str] = Field(default_factory=list)
    must_not_say: list[str] = Field(default_factory=list)
    # Names of DataDependency entries this action requires at response_gen time.
    # The runtime speculatively prefetches these during pondering and rollout
    # trajectories so the data is already in hand when the action actually executes.
    data_dependencies: list[str] = Field(default_factory=list)


class CohortMood(BaseModel):
    """A named user-disposition sample for a cohort.

    A cohort describes WHO the user is (LoyalCustomer, PriceShopper, …). A mood
    describes HOW they're feeling within that identity at this moment. Moods are
    sampled per-rollout (different parallel rollouts get different moods) so that
    state predictions across the 8 parallel rollouts diverge meaningfully — the
    POC's standing fix for the rollout-determinism null result documented in
    notes/2026-05-23-stable-vs-transition-state-prediction-asymmetry.md.
    """
    name: str
    description: str = ""
    # Sampling weight within the cohort. Per-rollout sampling is proportional to
    # `prior` (re-normalized across the cohort's mood list). Designer-set initially;
    # in a later phase replaced by empirical priors mined from precedent_traces.
    prior: float = 1.0


class CohortItem(NamedItem):
    """A cohort. Same shape as NamedItem (name + description) plus an optional
    list of moods that the rollout user-simulator can sample from."""
    moods: list[CohortMood] = Field(default_factory=list)


class DataDependency(BaseModel):
    """An external data lookup that one or more agent_actions need at execution time.

    The runtime can speculatively pre-fetch idempotent dependencies during user
    think-time, keyed off MCTS trajectory predictions. Mutating operations (e.g. booking,
    payment, sending) MUST set idempotent=False; the prefetch system skips them.
    """
    name: str
    description: str = ""
    kind: Literal["mock", "rag", "kg", "db", "api", "mcp"] = "mock"
    config: dict = Field(default_factory=dict)   # fetcher-specific (endpoint, query template, …)
    expected_latency_ms: int = 1000
    cache_ttl_s: int = 300
    idempotent: bool = True                       # must be true to be eligible for prefetch
    # Q6b: parameterized query template. When present, the supervisor renders this string
    # with rollout-derived signal at plan-build time and passes the rendered string to
    # the fetcher as a real query (e.g., the input to embedding-based RAG search).
    # Supported placeholders (Python str.format):
    #   {user_text}  — rollout's predicted user utterance at the offset turn
    #   {cohort}     — runtime-classified cohort
    #   {mood}       — runtime-classified mood
    #   {state}      — predicted user_state at the offset turn
    #   {action}     — predicted agent action at the offset turn
    # When absent, falls back to today's action-keyed canned-fetch behaviour.
    query_template: Optional[str] = None


class Strategy(BaseModel):
    """A named group of related agent_actions used by hierarchical MCTS.

    Each strategy is a logical phase of dialogue (e.g. "Identify", "Persuade", "Close").
    The MCTS planner can be configured to plan over strategies instead of individual
    actions: search a coarser tree (fewer branches at each depth), then instantiate the
    chosen strategy to a concrete action via SOP filtering.

    member_actions is a list of agent_action.name values. Strategies don't need to be
    disjoint — an action can belong to multiple strategies if it makes sense.
    """
    name: str
    description: str = ""
    member_actions: list[str] = Field(default_factory=list)


# Edge directionality between SOP nodes (from the paper):
#   "forward"  : src must precede dst (src -> dst)
#   "backward" : dst must precede src (src <- dst)
#   "both"     : either order allowed (bidirectional)
#   "none"     : no relationship (default; omit edges of this type)
EdgeDir = Literal["forward", "backward", "both"]


class SOPEdge(BaseModel):
    src: str  # node name (agent_action or user_state)
    dst: str
    direction: EdgeDir = "forward"
    note: str = ""


class SOPGraphSchema(BaseModel):
    nodes: list[str] = Field(default_factory=list)  # union of action + state names
    edges: list[SOPEdge] = Field(default_factory=list)


class TaskDefinition(BaseModel):
    """The full Standard Operating Procedure definition for one task."""
    name: str = "Untitled SOP"
    description: str = ""
    user_profile: UserProfile = Field(default_factory=UserProfile)
    conversation_profile: ConversationProfile = Field(default_factory=ConversationProfile)
    agent_actions: list[NamedItem] = Field(default_factory=list)
    user_states: list[NamedItem] = Field(default_factory=list)
    # Cohort vocabulary for the precedent / context graph. Optional — if empty, the
    # per-turn classifier may emit a free-form cohort tag (less retrievable by exact match).
    cohorts: list[CohortItem] = Field(default_factory=list)
    # Strategy vocabulary for hierarchical MCTS. Optional; when empty and granularity=
    # "strategy", a one-strategy-per-action fallback is auto-derived at planning time.
    strategies: list[Strategy] = Field(default_factory=list)
    # External data lookups that agent_actions can declare via NamedItem.data_dependencies.
    data_dependencies: list[DataDependency] = Field(default_factory=list)
    sop: SOPGraphSchema = Field(default_factory=SOPGraphSchema)


# ---------- API: SOP build (Configuration tab) ----------

class BuildTurnRequest(BaseModel):
    history: list[dict[str, str]]  # [{"role": "user"|"assistant", "content": "..."}]
    current_sop: TaskDefinition


class BuildTurnResponse(BaseModel):
    assistant_message: str
    updated_sop: TaskDefinition
    is_complete: bool = False


class SOPSaveRequest(BaseModel):
    sop: TaskDefinition


class SOPMeta(BaseModel):
    id: str
    name: str
    description: str = ""
    updated_at: str


# ---------- API: Chat (Chat tab) ----------

PlannerMode = Literal["mcts", "baseline"]
ChatMode = Literal["human", "auto"]


class MCTSConfig(BaseModel):
    iterations: int = 8
    branching: int = 3          # d
    rollout_depth: int = 3      # k
    c_uct: float = 1.4
    parallel_rollouts: int = 4  # concurrent rollouts per UCT batch
    nee_threshold: float = 0.0
    nee_min_visits: int = 1
    # ---- Precedent / context-graph injection (Fast loop) ----
    top_k_precedents: int = 3
    use_precedents_expand: bool = True
    use_precedents_score: bool = False      # off by default — feedback collapse risk
    use_precedents_response: bool = True
    # ---- Pondering MCTS (speculative search between turns) ----
    pondering_enabled: bool = True
    pondering_k: int = 2                    # top-K most-likely next states to pre-compute
    # At consume time, how long the agent will block waiting for an in-flight pondering
    # task to finish before falling back to live MCTS. In voice production this is
    # essentially "max acceptable user-perceived MCTS latency" — the user's natural
    # pause covers the pondering work in parallel. In autopilot the user-sim returns
    # quickly so this knob has to be large enough to bridge the gap. Default 1500ms
    # matches legacy behaviour; production-realistic test runs use 10000-15000ms.
    pondering_await_in_flight_ms: int = 1500
    # ---- Rollout mode (Fast-MCTD: value-only rollouts) ----
    #   "simulate"  current behaviour: per-step rollout_action + user_sim_with_state + end-rationality
    #   "value"     a single value-scoring LLM call per rollout; collapses ~7 calls into 1
    #   "hybrid"    simulate first step (for grounding), value-score the rest
    rollout_mode: Literal["simulate", "value", "hybrid"] = "simulate"
    # ---- Rollout-step ACTION-SELECTION POLICY (independent of rollout_mode) ----
    # Affects what happens at each STEP inside a simulate/hybrid rollout. Default keeps
    # legacy behaviour so existing benchmarks are unchanged.
    #   "llm_top1"  LLM picks top-1 action; deterministic-ish → parallel rollouts often
    #               collapse to identical trajectories on narrow-SOP turns. Current default.
    #   "llm_topk"  LLM proposes top-K (cheap diversity); per-rollout uniform sample.
    #   "bandit"    UCT over SOP-allowed set with priors from EmpiricalTrajectoryPredictor +
    #               per-rollout local visits + softmax sampling. Removes the per-step LLM
    #               call entirely once enough precedents accumulate — saves ~half the rollout
    #               call budget AND restores tree-search diversity.
    rollout_action_policy: Literal["llm_top1", "llm_topk", "bandit"] = "llm_top1"
    rollout_action_topk: int = 3            # K for llm_topk
    rollout_bandit_c_uct: float = 1.4       # exploration constant
    rollout_bandit_softmax_temp: float = 0.7  # higher → more random, lower → more deterministic
    rollout_bandit_epsilon: float = 0.1     # ε-greedy uniform fallback to handle cold-start sparsity
    # ---- Planning granularity (Fast-MCTD: sparse / hierarchical planning) ----
    #   "action"   plan over individual agent_actions (current default)
    #   "strategy" plan over Strategy groups; each strategy expands to a concrete action
    #              via SOP filtering at execution time. Same call count per rollout, but
    #              each rollout step represents a coarser conversation move, so depth=k
    #              covers more "real" dialogue distance.
    planning_granularity: Literal["action", "strategy"] = "action"
    # ---- Speculative data prefetch ----
    # When enabled, after each turn the runtime walks the MCTS rollouts, scores each
    # (data_dependency, predicted_turn_offset) tuple by Σ rollout.reward · exp(-decay·offset),
    # and dispatches the highest-scoring prefetches into a per-session background queue.
    # At the next turn(s), the queue is consulted before falling back to a live fetch.
    data_prefetch_enabled: bool = False                 # opt-in; off by default
    data_prefetch_min_confidence: float = 0.05          # threshold under which we don't schedule
    data_prefetch_max_outstanding: int = 50             # per-session queue cap
    data_prefetch_decay_lambda: float = 0.3             # time-decay rate for trajectory offset
    data_prefetch_await_in_flight_ms: int = 2000        # at consume time, how long we wait for in-flight
    # Which trajectory predictor drives prefetch scheduling:
    #   "auto"      — MCTS rollouts if deep, else empirical-from-precedents
    #   "mcts"      — only the in-memory rollouts (current legacy behaviour; needs simulate/hybrid)
    #   "empirical" — only the empirical predictor (works with value-mode but needs accumulated data)
    #   "union"     — run both in parallel and merge: (action,offset) emitted by either contributes;
    #                 when both agree, probabilities combine via independent-sources union
    #                 1-(1-p_mcts)(1-p_emp). Source tagged "mcts" / "empirical" / "both" for analysis.
    data_prefetch_predictor: Literal["auto", "mcts", "empirical", "union"] = "auto"
    # Cheap LLM next-utterance predictor for the generative {user_text} slot of query-aware
    # (RAG/KG) prefetch. Empirical counting predicts the next ACTION + structured params;
    # it cannot produce the free-text query a RAG dep needs. A single small-model call fills
    # {user_text}, replacing MCTS rollouts as that slot's source at ~85x lower cost. Off by
    # default (on coarse corpora the structured stub suffices; the value scales with corpus
    # granularity — see retrieval ablation). Runs off the critical path, slack-gated.
    use_llm_query_predictor: bool = False
    # ---- Multi-tier router (Strategic-Supervisor-style fast paths) ----
    # When enabled, after classifying cohort+state the router checks precedent_traces for
    # historical agreement at this (cohort, state) and routes to one of three tiers:
    #   tier_1 ("cached_playbook"): trust the dominant historical action — skip MCTS entirely.
    #   tier_2 ("baseline"):        use cohort_state_propose's top candidate — skip MCTS but
    #                               trust the LLM's one-shot judgment.
    #   tier_3 ("mcts"):            run live MCTS as the safety net for novel/disagreed turns.
    router_enabled: bool = True
    tier_entropy_max_t1: float = 0.4        # Shannon bits; ≤ this → tier_1 eligible
    tier_entropy_max_t2: float = 1.2        # ≤ this (and not t1) → tier_2 eligible
    tier_min_supporting_traces: int = 3     # need this many past precedents to leave tier_3
    # Sync-fallback / async-supervisor architecture (2026-06-03 decision). When False, the
    # router NEVER elevates to tier_3 (live MCTS on critical path) — sparse-precedent or
    # high-entropy turns fall back to tier_2 (baseline LLM) instead, with pool rerank + pool
    # synthesis providing context. Pondering still runs in background to fill the pool.
    # Default True keeps legacy behaviour; set False for the supervisor-architecture endpoint.
    tier3_enabled: bool = True


class ChatStartRequest(BaseModel):
    sop_id: str
    planner_mode: PlannerMode = "mcts"
    chat_mode: ChatMode = "human"
    mcts: MCTSConfig = Field(default_factory=MCTSConfig)


class ChatStartResponse(BaseModel):
    session_id: str
    sop: TaskDefinition


class TurnRequest(BaseModel):
    user_message: Optional[str] = None  # ignored if chat_mode == "auto"


class CandidateAction(BaseModel):
    action: str
    q_value: float
    visits: int
    rationale: str = ""


class RetrievedPrecedent(BaseModel):
    id: str
    cohort: str
    action: str
    immediate_state: str = ""
    terminal_outcome: str | None = None
    immediate_reward: float = 0.0
    terminal_reward: float | None = None
    similarity: float = 0.0
    response_text: str = ""


class PlannerTrace(BaseModel):
    predicted_user_state: str = ""
    state_rationale: str = ""
    cohort: str = ""
    # Phase-2 runtime mood classification. Picked from the chosen cohort's mood vocabulary
    # by the cohort_state_propose classifier. Empty when the cohort has no moods declared
    # (older SOPs) or the path doesn't run mood classification (baseline planner).
    mood: str = ""
    chosen_action: str = ""
    candidates: list[CandidateAction] = Field(default_factory=list)
    mcts_iterations: int = 0
    rollouts: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    mode: PlannerMode = "mcts"
    nee_triggered: bool = False
    precedents: list[RetrievedPrecedent] = Field(default_factory=list)
    precedents_used_expand: bool = False
    precedents_used_score: bool = False
    precedents_used_response: bool = False
    # ---- Pondering telemetry ----
    from_pondering: bool = False              # MCTS result came from a precomputed pondering run
    pondering_hit_state: str | None = None    # which predicted state was reused
    # ---- Latency (split agent vs user simulator) ----
    agent_duration_ms: int = 0                # embed + planner + response_gen (what a real user would wait)
    user_sim_ms: int = 0                      # the auto-user simulator's time (0 in human mode)
    # ---- Router telemetry ----
    tier_used: Literal["cached_playbook", "baseline", "mcts"] | None = None
    tier_entropy: float | None = None         # Shannon bits of past action distribution at this (cohort, state)
    tier_supporting_traces: int = 0           # how many precedent rows the router consulted
    tier_dominant_action: str | None = None
    tier_dominant_agreement: float | None = None  # fraction supporting the dominant action
    tier_rationale: str | None = None
    # ---- Planning granularity (Option B) ----
    planning_granularity: Literal["action", "strategy"] = "action"
    chosen_strategy: str | None = None       # set when granularity="strategy"
    # ---- Data prefetch telemetry (per turn) ----
    data_prefetch_consumed_count: int = 0           # how many deps were served from the queue
    data_prefetch_live_count: int = 0               # how many fell back to a live fetch
    data_prefetch_latency_hidden_ms: int = 0        # sum of fetch_duration_ms over consumed items
    data_prefetch_live_latency_ms: int = 0          # sum of live-fetch wall-clock this turn
    data_prefetch_scheduled_after_turn: int = 0     # new items put in queue at end of this turn
    # ---- Instruction prefetch (milestone B) ----
    instruction_hit: bool = False                   # True if a pre-generated instruction was used verbatim (response_gen skipped)
    instruction_fetch_id: str | None = None         # fetch_id of the matched instruction item, for analytics
    instruction_data_count: int = 0                 # Fix A: how many pool data items were baked into the used instruction (data-on-hit utilisation)


class TurnResponse(BaseModel):
    user_message: str
    assistant_message: str
    trace: PlannerTrace
