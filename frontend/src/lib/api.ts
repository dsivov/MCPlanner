const BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000';

export type NamedItem = { name: string; description?: string };
export type EdgeDir = 'forward' | 'backward' | 'both';
export type SOPEdge = { src: string; dst: string; direction: EdgeDir; note?: string };

export type TaskDefinition = {
  name: string;
  description: string;
  user_profile: {
    name?: string | null;
    description: string;
    demographics: Record<string, string>;
  };
  conversation_profile: {
    agent_role: string;
    goal: string;
    success_markers: string[];
    failure_markers: string[];
    knowledge: string;
  };
  agent_actions: NamedItem[];
  user_states: NamedItem[];
  sop: { nodes: string[]; edges: SOPEdge[] };
};

export type PlannerMode = 'mcts' | 'baseline';
export type ChatMode = 'human' | 'auto';

export type MCTSConfig = {
  iterations: number;
  branching: number;
  rollout_depth: number;
  c_uct: number;
  parallel_rollouts: number;
  nee_threshold: number;
  nee_min_visits: number;
  top_k_precedents: number;
  use_precedents_expand: boolean;
  use_precedents_score: boolean;
  use_precedents_response: boolean;
  pondering_enabled: boolean;
  pondering_k: number;
  rollout_mode: 'simulate' | 'value' | 'hybrid';
  router_enabled: boolean;
  tier_entropy_max_t1: number;
  tier_entropy_max_t2: number;
  tier_min_supporting_traces: number;
  planning_granularity: 'action' | 'strategy';
  rollout_action_policy?: 'llm_top1' | 'llm_topk' | 'bandit';
  data_prefetch_enabled?: boolean;
  data_prefetch_predictor?: 'auto' | 'mcts' | 'empirical' | 'union';
};

export type PredictorSource = 'mcts' | 'empirical' | 'both' | 'live' | 'unknown';

export type DataFetchRow = {
  id: string;
  cache_key: string;
  dependency_name: string;
  action_name: string;
  kind: string;
  issued_at_turn: number;
  predicted_turn: number | null;
  consumed_at_turn: number | null;
  started_at: string | null;
  completed_at: string | null;
  fetch_duration_ms: number;
  confidence: number;
  consumed: boolean;
  wasted: boolean;
  speculative: boolean;
  fetch_error: string | null;
  predictor_source: PredictorSource;
  predicted_user_state: string | null;
};

export type CandidateAction = {
  action: string;
  q_value: number;
  visits: number;
  rationale: string;
};

export type RetrievedPrecedent = {
  id: string;
  cohort: string;
  action: string;
  immediate_state: string;
  terminal_outcome: string | null;
  immediate_reward: number;
  terminal_reward: number | null;
  similarity: number;
  response_text: string;
};

export type PlannerTrace = {
  predicted_user_state: string;
  state_rationale: string;
  cohort: string;
  mood: string;
  chosen_action: string;
  candidates: CandidateAction[];
  mcts_iterations: number;
  rollouts: number;
  tokens_in: number;
  tokens_out: number;
  mode: PlannerMode;
  nee_triggered: boolean;
  precedents: RetrievedPrecedent[];
  precedents_used_expand: boolean;
  precedents_used_score: boolean;
  precedents_used_response: boolean;
  from_pondering: boolean;
  pondering_hit_state: string | null;
  agent_duration_ms: number;
  user_sim_ms: number;
  tier_used: 'cached_playbook' | 'baseline' | 'mcts' | null;
  tier_entropy: number | null;
  tier_supporting_traces: number;
  tier_dominant_action: string | null;
  tier_dominant_agreement: number | null;
  tier_rationale: string | null;
  planning_granularity: 'action' | 'strategy';
  chosen_strategy: string | null;
};

export type TurnResponse = {
  user_message: string;
  assistant_message: string;
  trace: PlannerTrace;
};

export type AvatarDataContextItem = {
  dependency_name: string;
  source_action: string;
  summary: string;
};

export type AvatarPlan = {
  chosen_action: string;
  action_description: string;
  must_say: string[];
  must_not_say: string[];
  data_context: AvatarDataContextItem[];
  agent_role: string;
  goal: string;
  cohort: string;
  user_state: string;
  mood: string;
  state_rationale: string;
  pool_size: number;
  pool_rerank_ms: number;
  classify_ms: number;
  prefetch_consumed: number;
  prefetch_live: number;
  prefetch_latency_hidden_ms: number;
  next_prefetch_scheduled: number;
  turn_index: number;
  terminal_outcome: string | null;
  trace: PlannerTrace;
};

export type AvatarBlackboardItem = {
  kind: string;
  dependency_name: string;
  source_action: string;
  predicted_user_state: string | null;
  summary: string;
  confidence: number;
  predictor_source: string;
};

export type AvatarBlackboard = {
  session_id: string;
  pool_size: number;
  items: AvatarBlackboardItem[];
};

export type SOPMeta = { id: string; name: string; description: string; updated_at: string };

export type LiftRow = {
  cohort: string; action: string;
  n_total: number; n_success: number; n_failure: number; n_abandoned: number; n_open: number;
  success_rate: number; cohort_baseline: number; action_baseline: number;
  lift_vs_cohort: number; lift_vs_action: number;
};

export type LearningProposal = {
  id: string; cohort: string; action: string;
  must_say_add: string[]; must_not_say_add: string[];
  citations_success: string[]; citations_failure: string[];
  rationale: string;
};

export type MineResponse = {
  run_id: string; sop_ref: string;
  n_precedents: number; n_sessions: number;
  lift_table: LiftRow[]; proposals: LearningProposal[]; duration_ms: number;
};

export type GraphNode = { id: string; kind: 'cohort'|'action'|'outcome'; label: string; count: number };
export type GraphEdge = { src: string; dst: string; count: number; lift: number; success_rate: number };
export type GraphResponse = { sop_ref: string; n_precedents: number; nodes: GraphNode[]; edges: GraphEdge[] };

export type ScatterPoint = { trace_id: string; cohort: string; action: string; outcome: string; x: number; y: number };
export type ScatterResponse = { sop_ref: string; n_precedents: number; points: ScatterPoint[] };

export type ExperimentSummary = {
  id: string;
  sop_ref: string;
  sop_name: string;
  planner_mode: string;
  chat_mode: string;
  created_at: string;
  updated_at: string;
  notes: string;
  turn_count: number;
  tokens_in_total: number;
  tokens_out_total: number;
  llm_calls_total: number;
  duration_ms_total: number;
};

export type MCTSReplayCandidate = {
  rank: number;
  action: string;
  q_value: number;
  visits: number;
  rationale: string;
  was_chosen: boolean;
};

export type MCTSReplayRollout = {
  rollout_index: number;
  first_action: string;
  planned_actions: string[];
  planned_states: string[];        // aligned with planned_actions
  final_state: string;
  depth_completed: number;
  hit_failure: boolean;
  hit_success: boolean;
  rationality: number | null;
  progress_bonus: number;
  reward: number;
  rollout_mode: string;
  duration_ms: number;
};

export type MCTSReplay = {
  experiment_id: string;
  turn_id: string;
  turn_index: number;
  chosen_action: string;
  predicted_user_state: string;
  state_rationale: string;
  mode: string;
  config: Record<string, unknown>;
  trace: Record<string, unknown>;
  duration_ms: number;
  user_message: string;
  assistant_message: string;
  candidates: MCTSReplayCandidate[];
  rollouts: MCTSReplayRollout[];
};

export type TurnIndexEntry = {
  turn_index: number;
  chosen_action: string;
  mode: string;
  rollouts: number;
};

export type TraceDetail = {
  id: string; experiment_id: string; sop_ref: string; cohort: string; action: string;
  situation_text: string; response_text: string;
  immediate_state: string | null; terminal_outcome: string | null;
  immediate_reward: number; terminal_reward: number | null;
  created_at: string;
};

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch { /* noop */ }
    throw new Error(`${r.status} ${detail}`);
  }
  return r.json();
}

export const api = {
  health: () => req<{ ok: boolean; models: Record<string, string> }>('/api/health'),
  listSeeds: () => req<{ file: string; name: string; description: string }[]>('/api/sop/seeds'),
  getSeed: (file: string) => req<TaskDefinition>(`/api/sop/seeds/${file}`),
  listSops: () => req<SOPMeta[]>('/api/sop'),
  getSop: (id: string) => req<TaskDefinition>(`/api/sop/${id}`),
  saveSop: (sop: TaskDefinition) =>
    req<SOPMeta>('/api/sop', { method: 'POST', body: JSON.stringify({ sop }) }),
  updateSop: (id: string, sop: TaskDefinition) =>
    req<SOPMeta>(`/api/sop/${id}`, { method: 'PUT', body: JSON.stringify({ sop }) }),
  deleteSop: (id: string) => req<{ ok: boolean }>(`/api/sop/${id}`, { method: 'DELETE' }),
  buildTurn: (history: { role: string; content: string }[], current: TaskDefinition) =>
    req<{ assistant_message: string; updated_sop: TaskDefinition; is_complete: boolean }>(
      '/api/sop/build-turn',
      { method: 'POST', body: JSON.stringify({ history, current_sop: current }) },
    ),
  startChat: (sop_id: string, planner_mode: PlannerMode, chat_mode: ChatMode, mcts: MCTSConfig) =>
    req<{ session_id: string; sop: TaskDefinition }>('/api/chat/start', {
      method: 'POST',
      body: JSON.stringify({ sop_id, planner_mode, chat_mode, mcts }),
    }),
  turn: (session_id: string, user_message?: string, signal?: AbortSignal) =>
    req<TurnResponse>(`/api/chat/${session_id}/turn`, {
      method: 'POST',
      body: JSON.stringify({ user_message }),
      signal,
    }),
  endSession: (session_id: string) =>
    req<{ ok: boolean; outcome: string }>(`/api/chat/${session_id}/end`, { method: 'POST' }),

  // ---- Avatar (live voice testing) ----
  mintRealtimeSession: (session_id?: string) =>
    req<{ value: string; model: string; voice: string }>(`/api/avatar/realtime-session`, {
      method: 'POST',
      body: JSON.stringify({ session_id: session_id ?? null }),
    }),
  avatarPlanTurn: (session_id: string, user_message: string, avatar_prev_response?: string | null) =>
    req<AvatarPlan>(`/api/avatar/${session_id}/plan-turn`, {
      method: 'POST',
      body: JSON.stringify({ user_message, avatar_prev_response: avatar_prev_response ?? null }),
    }),
  avatarBlackboard: (session_id: string) =>
    req<AvatarBlackboard>(`/api/avatar/${session_id}/blackboard`),

  // ---- Context Graph + Learning ----
  graph: (sop_ref: string, cohort?: string) => {
    const u = new URL('/api/context-graph', BASE);
    u.searchParams.set('sop_ref', sop_ref);
    if (cohort) u.searchParams.set('cohort', cohort);
    return req<GraphResponse>(u.pathname + '?' + u.searchParams.toString());
  },
  scatter: (sop_ref: string) =>
    req<ScatterResponse>(`/api/context-graph/scatter?sop_ref=${encodeURIComponent(sop_ref)}`),
  traces: (params: { sop_ref: string; cohort?: string; action?: string; outcome?: string }) => {
    const u = new URLSearchParams();
    u.set('sop_ref', params.sop_ref);
    if (params.cohort) u.set('cohort', params.cohort);
    if (params.action) u.set('action', params.action);
    if (params.outcome) u.set('outcome', params.outcome);
    return req<TraceDetail[]>(`/api/context-graph/traces?${u.toString()}`);
  },
  mine: (sop_ref: string) =>
    req<MineResponse>(`/api/learn/mine?sop_ref=${encodeURIComponent(sop_ref)}`, { method: 'POST' }),
  applyProposals: (run_id: string, sop_id: string, proposal_ids: string[]) =>
    req<{ sop_id: string; accepted: string[]; actions_updated: number }>(
      `/api/learn/apply`,
      { method: 'POST', body: JSON.stringify({ run_id, sop_id, proposal_ids }) },
    ),
  saveAndApply: (run_id: string, source_sop_ref: string, proposal_ids: string[], new_name?: string) =>
    req<{ sop_id: string; sop_name: string; accepted: string[]; actions_updated: number }>(
      `/api/learn/save-and-apply`,
      { method: 'POST', body: JSON.stringify({ run_id, source_sop_ref, proposal_ids, new_name }) },
    ),

  // ---- Experiments ----
  listExperiments: (limit = 50) =>
    req<ExperimentSummary[]>(`/api/experiments?limit=${limit}`),

  // ---- MCTS Replay ----
  turnIndices: (exp_id: string) =>
    req<TurnIndexEntry[]>(`/api/experiments/${exp_id}/turn-indices`),
  mctsReplay: (exp_id: string, turn_index: number) =>
    req<MCTSReplay>(`/api/experiments/${exp_id}/mcts-replay/${turn_index}`),
  dataFetches: (exp_id: string) =>
    req<DataFetchRow[]>(`/api/experiments/${exp_id}/data-fetches`),
};

export function emptySop(): TaskDefinition {
  return {
    name: 'New SOP',
    description: '',
    user_profile: { name: '', description: '', demographics: {} },
    conversation_profile: {
      agent_role: '',
      goal: '',
      success_markers: [],
      failure_markers: [],
      knowledge: '',
    },
    agent_actions: [],
    user_states: [],
    sop: { nodes: [], edges: [] },
  };
}
