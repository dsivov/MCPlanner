import { useEffect, useMemo, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  GitBranchPlus, Play, Pause, SkipForward, RotateCcw, Gauge, CheckCircle2, XCircle, AlertTriangle, Loader2, FileText,
} from 'lucide-react';
import ReactFlow, {
  Background, BackgroundVariant, Controls, MarkerType, Handle, Position,
  type Node, type Edge, useNodesState, useEdgesState,
} from 'reactflow';
import 'reactflow/dist/style.css';
import dagre from '@dagrejs/dagre';
import {
  api,
  type ExperimentSummary,
  type MCTSReplay, type MCTSReplayRollout, type TurnIndexEntry,
  type DataFetchRow, type PredictorSource,
} from '../lib/api';

// ─────────────────────────────────────────────────────────────────────────────
// Layout: root → candidates (root children) → rollout trajectories
// We materialize each rollout as a chain of phantom nodes under its first-action
// candidate. Each chain is one rollout's planned_actions[1:] (positions beyond
// the candidate). Reward "orbs" animate from chain tips back up to the root.

const ROOT_W = 240, NODE_W = 200, ROLL_W = 150, NODE_H = 60;

type RootData = { kind: 'root'; cohort: string; state: string };
type CandData = {
  kind: 'cand';
  action: string;
  qLive: number;    // current animated Q
  visitsLive: number;
  qFinal: number;
  visitsFinal: number;
  isChosen: boolean;
  rationale: string;
  pulse: number;     // increments to retrigger animation
};
type RollData = {
  kind: 'roll';
  rolloutIndex: number;
  action: string;          // action at this depth in the rollout's plan
  depthCompleted: number;
  reward: number;
  hitSuccess: boolean;
  hitFailure: boolean;
  active: boolean;          // whether currently fading-in / pulsing
};

function RootNode({ data }: { data: RootData }) {
  return (
    <div
      className="relative rounded-xl border border-accent/60 bg-bg-elevated px-4 py-3 text-center shadow-glow"
      style={{ width: ROOT_W }}
    >
      <div className="text-[10px] uppercase tracking-wider text-accent">root</div>
      <div className="mt-1 text-[12px] text-fg font-medium">{data.cohort || 'cohort: —'}</div>
      <div className="text-[11px] text-fg-dim">user_state: {data.state || '—'}</div>
      <Handle type="source" position={Position.Bottom} style={{ opacity: 0, width: 1, height: 1 }} />
    </div>
  );
}

function CandNode({ data }: { data: CandData; selected: boolean }) {
  const fill = data.isChosen ? '#7c5cff' : 'rgba(124,92,255,0.45)';
  return (
    <motion.div
      className={`relative rounded-lg border ${data.isChosen ? 'border-accent shadow-glow' : 'border-bg-border'} bg-bg-panel px-3 py-2 cursor-pointer`}
      style={{ width: NODE_W, minHeight: NODE_H }}
      animate={data.pulse ? { boxShadow: [
        '0 0 0px rgba(124,92,255,0)',
        '0 0 22px rgba(124,92,255,0.85)',
        '0 0 0px rgba(124,92,255,0)',
      ] } : undefined}
      transition={{ duration: 0.7 }}
      key={`pulse-${data.pulse}`}
    >
      <Handle type="target" position={Position.Top} style={{ opacity: 0, width: 1, height: 1 }} />
      <Handle type="source" position={Position.Bottom} style={{ opacity: 0, width: 1, height: 1 }} />
      <div className="flex items-center justify-between">
        <span className="text-[12px] font-medium truncate text-fg" title={data.action}>{data.action}</span>
        {data.isChosen && <span className="chip chip-accent !py-0 ml-1">chosen</span>}
      </div>
      <div className="mt-1 flex items-center justify-between text-[10px] text-fg-dim font-mono">
        <span>Q <span className="text-fg">{data.qLive.toFixed(3)}</span></span>
        <span>v <span className="text-fg">{data.visitsLive}</span></span>
      </div>
      <div className="mt-1 h-1.5 w-full rounded-full bg-bg-elevated overflow-hidden">
        <motion.div
          animate={{ width: `${Math.min(100, Math.max(2, data.qLive * 100))}%` }}
          transition={{ duration: 0.4, ease: 'easeOut' }}
          style={{ background: fill, height: '100%' }}
        />
      </div>
    </motion.div>
  );
}

function RollNode({ data }: { data: RollData; selected: boolean }) {
  const outcomeColor = data.hitSuccess ? '#34d399' : data.hitFailure ? '#f87171' : '#9b9ba6';
  return (
    <motion.div
      initial={{ opacity: 0, y: -6 }}
      animate={{ opacity: data.active ? 1 : 0.55, y: 0 }}
      transition={{ duration: 0.2 }}
      className="relative rounded-md border border-bg-border bg-bg-elevated/70 px-2.5 py-1.5 text-[10.5px] cursor-pointer"
      style={{ width: ROLL_W }}
    >
      <Handle type="target" position={Position.Top} style={{ opacity: 0, width: 1, height: 1 }} />
      <Handle type="source" position={Position.Bottom} style={{ opacity: 0, width: 1, height: 1 }} />
      <div className="flex items-center justify-between text-fg">
        <span className="truncate">{data.action}</span>
        <span className="font-mono text-fg-dim">r{data.rolloutIndex}</span>
      </div>
      <div className="mt-0.5 flex items-center justify-between text-fg-dim">
        <span style={{ color: outcomeColor }}>r {data.reward.toFixed(2)}</span>
        <span>d {data.depthCompleted}</span>
      </div>
    </motion.div>
  );
}

const nodeTypes = { mctsRoot: RootNode, mctsCand: CandNode, mctsRoll: RollNode };

// ─────────────────────────────────────────────────────────────────────────────
// Tree building (positions + edges) via dagre

function buildGraph(
  replay: MCTSReplay,
  liveStats: Map<string, { q: number; visits: number; pulse: number }>,
  activeRolloutIndices: Set<number>,
) {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: 'TB', nodesep: 18, ranksep: 70, marginx: 24, marginy: 24 });

  const nodes: Node[] = [];
  const edges: Edge[] = [];

  // Root
  g.setNode('root', { width: ROOT_W, height: NODE_H });

  // Candidates (root children)
  for (const c of replay.candidates) {
    const id = `cand:${c.action}`;
    g.setNode(id, { width: NODE_W, height: NODE_H });
    g.setEdge('root', id);
  }

  // Rollout trajectories: planned_actions[1:] under their first-action candidate.
  // We chain phantom roll nodes per rollout. node id includes rollout_index + position.
  for (const r of replay.rollouts) {
    const candId = `cand:${r.first_action}`;
    const path = r.planned_actions.slice(1);
    if (path.length === 0) continue;
    let prevId = candId;
    for (let p = 0; p < path.length; p++) {
      const id = `roll:${r.rollout_index}:${p}`;
      g.setNode(id, { width: ROLL_W, height: NODE_H * 0.7 });
      g.setEdge(prevId, id);
      prevId = id;
    }
  }

  dagre.layout(g);

  // Root
  const rootNode = g.node('root');
  nodes.push({
    id: 'root',
    type: 'mctsRoot',
    position: { x: rootNode.x - ROOT_W / 2, y: rootNode.y - NODE_H / 2 },
    data: {
      kind: 'root',
      cohort: (replay.trace as { cohort?: string } | undefined)?.cohort || '',
      state: replay.predicted_user_state,
    } as RootData,
  });

  // Candidates
  for (const c of replay.candidates) {
    const id = `cand:${c.action}`;
    const n = g.node(id);
    const live = liveStats.get(c.action) || { q: 0, visits: 0, pulse: 0 };
    nodes.push({
      id,
      type: 'mctsCand',
      position: { x: n.x - NODE_W / 2, y: n.y - NODE_H / 2 },
      data: {
        kind: 'cand',
        action: c.action,
        qLive: live.q,
        visitsLive: live.visits,
        qFinal: c.q_value,
        visitsFinal: c.visits,
        isChosen: c.was_chosen,
        rationale: c.rationale,
        pulse: live.pulse,
      } as CandData,
    });
    edges.push({
      id: `e:root->${id}`,
      source: 'root',
      target: id,
      type: 'smoothstep',
      animated: c.was_chosen,
      style: { stroke: c.was_chosen ? '#7c5cff' : '#3a3a44', strokeWidth: c.was_chosen ? 1.8 : 1.2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: c.was_chosen ? '#7c5cff' : '#3a3a44' },
    });
  }

  // Rollout trajectories
  for (const r of replay.rollouts) {
    const path = r.planned_actions.slice(1);
    if (path.length === 0) continue;
    const candId = `cand:${r.first_action}`;
    const active = activeRolloutIndices.has(r.rollout_index);
    let prevId = candId;
    for (let p = 0; p < path.length; p++) {
      const id = `roll:${r.rollout_index}:${p}`;
      const n = g.node(id);
      const isTail = p === path.length - 1;
      nodes.push({
        id,
        type: 'mctsRoll',
        position: { x: n.x - ROLL_W / 2, y: n.y - (NODE_H * 0.7) / 2 },
        data: {
          kind: 'roll',
          rolloutIndex: r.rollout_index,
          action: path[p],
          depthCompleted: r.depth_completed,
          reward: r.reward,
          hitSuccess: !!r.hit_success && isTail,
          hitFailure: !!r.hit_failure && isTail,
          active,
        } as RollData,
      });
      edges.push({
        id: `e:${prevId}->${id}`,
        source: prevId,
        target: id,
        type: 'smoothstep',
        animated: active,
        style: {
          stroke: active ? '#7c5cff' : '#23232a',
          strokeWidth: active ? 1.4 : 1,
          strokeDasharray: active ? undefined : '4 4',
        },
      });
      prevId = id;
    }
  }

  return { nodes, edges };
}

// ─────────────────────────────────────────────────────────────────────────────
// Replay engine

type LiveStats = Map<string, { q: number; visits: number; pulse: number }>;

function initialLive(replay: MCTSReplay): LiveStats {
  const m: LiveStats = new Map();
  for (const c of replay.candidates) m.set(c.action, { q: 0, visits: 0, pulse: 0 });
  return m;
}

function advanceOne(live: LiveStats, rollout: MCTSReplayRollout): LiveStats {
  const next = new Map(live);
  const cur = next.get(rollout.first_action) || { q: 0, visits: 0, pulse: 0 };
  const visits = cur.visits + 1;
  const q = ((cur.q * cur.visits) + rollout.reward) / visits;
  next.set(rollout.first_action, { q, visits, pulse: cur.pulse + 1 });
  return next;
}

function finalLive(replay: MCTSReplay): LiveStats {
  const m: LiveStats = new Map();
  for (const c of replay.candidates) m.set(c.action, { q: c.q_value, visits: c.visits, pulse: 0 });
  return m;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main component

export default function MCTSReplayTab() {
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [expId, setExpId] = useState<string>('');
  const [turns, setTurns] = useState<TurnIndexEntry[]>([]);
  const [turnIndex, setTurnIndex] = useState<number | null>(null);

  const [replay, setReplay] = useState<MCTSReplay | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // All data fetches for this experiment — we filter by turn_index client-side.
  // Loaded once per experiment selection (the list is bounded by a session's lifespan).
  const [dataFetches, setDataFetches] = useState<DataFetchRow[]>([]);

  // Replay state
  const [liveStats, setLiveStats] = useState<LiveStats>(new Map());
  const [activeRollouts, setActiveRollouts] = useState<Set<number>>(new Set());
  const [step, setStep] = useState(0);     // number of rollouts replayed so far
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const playRef = useRef(false);
  useEffect(() => { playRef.current = playing; }, [playing]);

  // Inspector
  const [selected, setSelected] = useState<{ kind: 'cand'; action: string } | { kind: 'roll'; rolloutIndex: number } | null>(null);

  useEffect(() => {
    api.listExperiments(100).then(setExperiments).catch(() => {});
  }, []);
  useEffect(() => {
    if (!expId) { setTurns([]); setTurnIndex(null); setDataFetches([]); return; }
    api.turnIndices(expId).then((rs) => {
      const mctsTurns = rs.filter((t) => (t.rollouts ?? 0) > 0);
      setTurns(mctsTurns);
      setTurnIndex(mctsTurns.length ? mctsTurns[0].turn_index : null);
    }).catch((e) => setError(String(e)));
    api.dataFetches(expId).then(setDataFetches).catch(() => setDataFetches([]));
  }, [expId]);

  async function load(exp: string, ti: number) {
    setLoading(true); setError(null);
    try {
      const r = await api.mctsReplay(exp, ti);
      setReplay(r);
      setLiveStats(initialLive(r));
      setActiveRollouts(new Set());
      setStep(0);
      setPlaying(false);
      setSelected(null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally { setLoading(false); }
  }
  useEffect(() => { if (expId && turnIndex !== null) load(expId, turnIndex); }, [expId, turnIndex]);

  // Replay loop (kicked by setTimeout chain so changes to playing / speed are honoured)
  useEffect(() => {
    if (!playing || !replay) return;
    if (step >= replay.rollouts.length) { setPlaying(false); return; }
    const intervalMs = Math.max(150, 1100 / speed);
    const t = window.setTimeout(() => {
      if (!playRef.current) return;
      doStep();
    }, intervalMs);
    return () => window.clearTimeout(t);
  }, [playing, step, speed, replay]);

  function doStep() {
    if (!replay) return;
    if (step >= replay.rollouts.length) { setPlaying(false); return; }
    const r = replay.rollouts[step];
    // Mark the rollout as active for ~one beat
    setActiveRollouts((s) => new Set([...s, r.rollout_index]));
    setLiveStats((cur) => advanceOne(cur, r));
    setStep((s) => s + 1);
    // Fade trajectory after a moment
    window.setTimeout(() => setActiveRollouts((s) => {
      const n = new Set(s); n.delete(r.rollout_index); return n;
    }), Math.max(300, 900 / speed));
  }

  function reset() {
    if (!replay) return;
    setPlaying(false);
    setStep(0);
    setActiveRollouts(new Set());
    setLiveStats(initialLive(replay));
  }
  function jumpToEnd() {
    if (!replay) return;
    setPlaying(false);
    setStep(replay.rollouts.length);
    setActiveRollouts(new Set());
    setLiveStats(finalLive(replay));
  }

  // Build the visual graph
  const graph = useMemo(() => replay ? buildGraph(replay, liveStats, activeRollouts) : { nodes: [], edges: [] }, [replay, liveStats, activeRollouts]);
  const [nodes, setNodes, onNodesChange] = useNodesState(graph.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(graph.edges);
  useEffect(() => { setNodes(graph.nodes); setEdges(graph.edges); }, [graph, setNodes, setEdges]);

  const selectedCand = (selected && selected.kind === 'cand' && replay)
    ? replay.candidates.find((c) => c.action === selected.action) : null;
  const selectedRoll = (selected && selected.kind === 'roll' && replay)
    ? replay.rollouts.find((r) => r.rollout_index === selected.rolloutIndex) : null;

  // Prefetch attribution for the currently selected turn.
  // Issued = scheduled after this turn (i.e., based on this turn's chosen action).
  // Consumed = used at this turn (i.e., the prediction was correct N turns ago).
  const ti = replay?.turn_index ?? null;
  const issuedHere = useMemo(() => (
    ti === null ? [] : dataFetches.filter((f) => f.issued_at_turn === ti && f.speculative)
  ), [dataFetches, ti]);
  const consumedHere = useMemo(() => (
    ti === null ? [] : dataFetches.filter((f) => f.consumed_at_turn === ti)
  ), [dataFetches, ti]);
  const sourceCounts = useMemo(() => {
    const c: Record<PredictorSource, number> = { mcts: 0, empirical: 0, both: 0, live: 0, unknown: 0 };
    for (const f of issuedHere) c[f.predictor_source] = (c[f.predictor_source] ?? 0) + 1;
    return c;
  }, [issuedHere]);
  // Predicted user-state-by-offset, mined from MCTS rollouts of the current turn.
  // Per offset, return the TOP-K reward-weighted states (not just the mode) so the UI can
  // surface the minority branches that state-aware Union now hedges across in prefetch.
  // K=3 matches the backend's MctsTrajectoryPredictor.TOP_K_STATES.
  const TOP_K_STATES = 3;
  const stateByOffset = useMemo(() => {
    if (!replay) return [] as { offset: number; branches: { state: string; share: number }[] }[];
    const tallies: Map<number, Map<string, number>> = new Map();
    for (const r of replay.rollouts) {
      if (!r.planned_actions.length || r.planned_actions[0] !== replay.chosen_action) continue;
      const states = r.planned_states ?? [];
      const weight = Math.max(0.0001, r.reward);
      for (let offset = 1; offset < states.length; offset++) {
        const s = states[offset];
        if (!s) continue;
        if (!tallies.has(offset)) tallies.set(offset, new Map());
        const m = tallies.get(offset)!;
        m.set(s, (m.get(s) ?? 0) + weight);
      }
    }
    const out: { offset: number; branches: { state: string; share: number }[] }[] = [];
    for (const [offset, m] of [...tallies.entries()].sort((a, b) => a[0] - b[0])) {
      const total = [...m.values()].reduce((acc, v) => acc + v, 0) || 1;
      const branches = [...m.entries()]
        .map(([state, w]) => ({ state, share: w / total }))
        .sort((a, b) => b.share - a.share)
        .slice(0, TOP_K_STATES);
      out.push({ offset, branches });
    }
    return out;
  }, [replay]);
  const hitRateBySource = useMemo(() => {
    // Among fetches whose issued_at_turn ≤ ti and that have a non-null consumed/wasted/in-flight resolution,
    // group by predictor_source and compute consumed / (consumed + wasted). In-flight (still pending) excluded.
    const buckets: Record<PredictorSource, { hits: number; misses: number }> = {
      mcts: { hits: 0, misses: 0 }, empirical: { hits: 0, misses: 0 },
      both: { hits: 0, misses: 0 }, live: { hits: 0, misses: 0 }, unknown: { hits: 0, misses: 0 },
    };
    for (const f of dataFetches) {
      if (!f.speculative) continue;
      if (f.consumed) buckets[f.predictor_source].hits += 1;
      else if (f.wasted) buckets[f.predictor_source].misses += 1;
    }
    return buckets;
  }, [dataFetches]);

  return (
    <div className="h-full grid grid-cols-12 gap-4">
      {/* Left: canvas + controls */}
      <div className="col-span-8 flex flex-col gap-4 overflow-hidden">
        <div className="card px-3 py-2.5 flex items-center gap-2 flex-wrap">
          <GitBranchPlus size={14} className="text-accent" />
          <span className="text-sm font-medium">MCTS Replay</span>
          <select
            value={expId}
            onChange={(e) => setExpId(e.target.value)}
            className="input py-1.5 max-w-[260px]"
          >
            <option value="">Select experiment…</option>
            {experiments.map((e) => (
              <option key={e.id} value={e.id}>
                {e.sop_name} · {e.id.slice(0, 6)} · {e.planner_mode}
              </option>
            ))}
          </select>
          <select
            value={turnIndex ?? ''}
            onChange={(e) => setTurnIndex(e.target.value === '' ? null : Number(e.target.value))}
            className="input py-1.5 max-w-[260px]"
            disabled={!expId}
          >
            <option value="">Pick turn…</option>
            {turns.map((t) => (
              <option key={t.turn_index} value={t.turn_index}>
                turn {t.turn_index} · {t.chosen_action} ({t.rollouts} rollouts)
              </option>
            ))}
          </select>

          <div className="grow" />
          {replay && (
            <>
              <button className="btn btn-ghost text-xs" onClick={reset} title="Reset to step 0">
                <RotateCcw size={12} /> Reset
              </button>
              <button className="btn btn-ghost text-xs" onClick={doStep} disabled={!replay || step >= replay.rollouts.length}>
                <SkipForward size={12} /> Step
              </button>
              <button
                className={`btn text-xs ${playing ? 'btn-primary' : ''}`}
                onClick={() => setPlaying((p) => !p)}
                disabled={!replay || step >= replay.rollouts.length}
              >
                {playing ? <Pause size={12} /> : <Play size={12} />}
                {playing ? 'Pause' : 'Play'}
              </button>
              <button className="btn btn-ghost text-xs" onClick={jumpToEnd}>
                <Gauge size={12} /> Jump to end
              </button>
              <div className="flex items-center gap-1.5 text-[11px] text-fg-dim">
                <span>speed</span>
                <input
                  type="range" min={0.25} max={4} step={0.25} value={speed}
                  onChange={(e) => setSpeed(Number(e.target.value))}
                  className="accent-violet-500"
                />
                <span className="font-mono">{speed.toFixed(2)}×</span>
              </div>
            </>
          )}
        </div>

        <div className="card flex flex-col overflow-hidden flex-1">
          <div className="flex items-center justify-between px-4 py-2 border-b border-bg-border text-[11.5px]">
            {replay ? (
              <>
                <div className="flex items-center gap-2">
                  <span className="text-fg">turn {replay.turn_index} · {replay.chosen_action}</span>
                  <span className="text-fg-dim">|</span>
                  <span className="text-fg-dim">{replay.candidates.length} candidates · {replay.rollouts.length} rollouts</span>
                </div>
                <div className="text-fg-dim font-mono">step {step}/{replay.rollouts.length}</div>
              </>
            ) : <span className="text-fg-dim">No replay loaded</span>}
          </div>
          <div className="relative flex-1 gradient-panel">
            {loading && (
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <Loader2 size={20} className="animate-spin text-fg-dim" />
              </div>
            )}
            {error && (
              <div className="absolute top-2 left-2 right-2 text-[11px] text-accent-err flex items-start gap-1">
                <AlertTriangle size={12} className="mt-0.5 flex-none" /> {error}
              </div>
            )}
            {replay && (
              <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                nodeTypes={nodeTypes}
                fitView
                fitViewOptions={{ padding: 0.16 }}
                proOptions={{ hideAttribution: true }}
                minZoom={0.25}
                maxZoom={1.6}
                nodesDraggable={false}
                nodesConnectable={false}
                onNodeClick={(_, n) => {
                  const id = n.id;
                  if (id.startsWith('cand:')) setSelected({ kind: 'cand', action: id.slice('cand:'.length) });
                  else if (id.startsWith('roll:')) {
                    const r = Number(id.split(':')[1]);
                    setSelected({ kind: 'roll', rolloutIndex: r });
                  } else {
                    setSelected(null);
                  }
                }}
              >
                <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#1f1f26" />
                <Controls position="bottom-right" showInteractive={false} />
              </ReactFlow>
            )}
          </div>
        </div>
      </div>

      {/* Right: inspector */}
      <div className="col-span-4 card flex flex-col overflow-hidden">
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-bg-border">
          <div className="flex items-center gap-2 text-sm">
            <FileText size={14} className="text-accent-alt" />
            <span className="font-medium">Inspector</span>
          </div>
          {selected && <button className="btn btn-ghost text-xs" onClick={() => setSelected(null)}>Clear</button>}
        </div>
        <div className="flex-1 overflow-y-auto p-4 text-[12px] space-y-3">
          {!selected && !replay && <div className="text-fg-dim text-sm">Pick an experiment + turn to begin.</div>}
          {!selected && replay && (
            <div className="space-y-2">
              <div className="text-[10px] uppercase tracking-wider text-fg-dim">Turn</div>
              <KV k="user" v={replay.user_message} />
              <KV k="assistant" v={replay.assistant_message} />
              <div className="h-px bg-bg-border my-2" />
              <div className="text-[10px] uppercase tracking-wider text-fg-dim">Config</div>
              <KV k="mode" v={replay.mode} />
              <div className="grid grid-cols-2 gap-2">
                {Object.entries(replay.config).slice(0, 8).map(([k, v]) => (
                  <KV key={k} k={k} v={String(v)} />
                ))}
              </div>
              <AnimatePresence>
                {replay.candidates.map((c) => (
                  <motion.div key={c.action} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-[11px] text-fg-dim">
                    <span className="text-fg">{c.action}</span>
                    {c.was_chosen && <span className="ml-2 chip chip-accent !py-0">chosen</span>}
                    <div className="ml-1 text-[10.5px] mt-0.5">Q {c.q_value.toFixed(3)} · visits {c.visits}</div>
                    {c.rationale && <div className="ml-1 text-[10.5px] italic">{c.rationale}</div>}
                  </motion.div>
                ))}
              </AnimatePresence>
              {stateByOffset.length > 0 && (
                <>
                  <div className="h-px bg-bg-border my-2" />
                  <div className="text-[10px] uppercase tracking-wider text-fg-dim">
                    Predicted user states (MCTS, top-{TOP_K_STATES} per offset)
                  </div>
                  <div className="space-y-1 mt-1">
                    {stateByOffset.map((row) => (
                      <div key={row.offset} className="flex items-center gap-1.5 flex-wrap text-[11px]">
                        <span className="text-fg-dim font-mono w-7">+{row.offset}</span>
                        {row.branches.map((b, i) => (
                          <span
                            key={b.state}
                            className={`rounded border px-1.5 py-0.5 text-[10.5px] ${
                              i === 0
                                ? 'border-accent/50 bg-accent/10 text-accent'
                                : 'border-bg-border bg-bg-elevated/40 text-fg-muted'
                            }`}
                          >
                            {b.state}
                            <span className="ml-1 opacity-70">{Math.round(b.share * 100)}%</span>
                          </span>
                        ))}
                      </div>
                    ))}
                  </div>
                  <div className="mt-1 text-[10px] text-fg-dim italic">
                    Modal (accent) + minority branches. Union predictor now hedges prefetch across all top-K.
                  </div>
                </>
              )}
              {(issuedHere.length > 0 || consumedHere.length > 0) && (
                <>
                  <div className="h-px bg-bg-border my-2" />
                  <div className="text-[10px] uppercase tracking-wider text-fg-dim flex items-center justify-between">
                    <span>Speculative prefetch</span>
                    <span className="text-fg-muted normal-case">
                      issued {issuedHere.length} · consumed {consumedHere.length}
                    </span>
                  </div>
                  {issuedHere.length > 0 && (
                    <div className="flex items-center gap-1.5 mt-1 flex-wrap">
                      {(['both', 'mcts', 'empirical'] as const).map((src) =>
                        sourceCounts[src] > 0 ? (
                          <SourceBadge key={src} source={src} count={sourceCounts[src]} />
                        ) : null
                      )}
                    </div>
                  )}
                  {issuedHere.length > 0 && (
                    <div className="mt-2 space-y-1">
                      <div className="text-[10px] uppercase tracking-wider text-fg-dim">Issued after this turn</div>
                      {issuedHere.map((f) => (
                        <PrefetchRow key={f.id} f={f} />
                      ))}
                    </div>
                  )}
                  {consumedHere.length > 0 && (
                    <div className="mt-2 space-y-1">
                      <div className="text-[10px] uppercase tracking-wider text-fg-dim">Consumed here (issued earlier)</div>
                      {consumedHere.map((f) => (
                        <PrefetchRow key={f.id} f={f} showHit />
                      ))}
                    </div>
                  )}
                  <div className="mt-3">
                    <div className="text-[10px] uppercase tracking-wider text-fg-dim">Hit-rate by source (session-wide)</div>
                    <div className="grid grid-cols-3 gap-1 mt-1">
                      {(['both', 'mcts', 'empirical'] as const).map((src) => {
                        const b = hitRateBySource[src];
                        const total = b.hits + b.misses;
                        const rate = total > 0 ? Math.round((b.hits / total) * 100) : null;
                        return (
                          <div key={src} className="rounded border border-bg-border bg-bg-elevated/40 p-1.5">
                            <div className="text-[10px] text-fg-dim">{src}</div>
                            <div className="text-fg font-mono text-[11.5px]">
                              {rate === null ? '—' : `${rate}%`}
                              <span className="text-fg-muted ml-1 text-[10px]">({b.hits}/{total})</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </>
              )}
            </div>
          )}
          {selectedCand && (
            <div className="space-y-2">
              <div className="text-[10px] uppercase tracking-wider text-fg-dim">Candidate</div>
              <KV k="action" v={selectedCand.action} />
              <KV k="Q (final)" v={selectedCand.q_value.toFixed(4)} />
              <KV k="visits (final)" v={selectedCand.visits.toString()} />
              <KV k="was_chosen" v={selectedCand.was_chosen ? 'yes' : 'no'} />
              {selectedCand.rationale && (
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-fg-dim">Rationale</div>
                  <div className="text-fg">{selectedCand.rationale}</div>
                </div>
              )}
              <div className="h-px bg-bg-border my-2" />
              <div className="text-[10px] uppercase tracking-wider text-fg-dim">Source rollouts</div>
              {replay?.rollouts.filter((r) => r.first_action === selectedCand.action).map((r) => (
                <div key={r.rollout_index} className="text-[11px] border border-bg-border rounded p-2 bg-bg-elevated/40">
                  <div className="flex justify-between text-fg-muted">
                    <span>r{r.rollout_index} · {r.depth_completed}/{(r.planned_actions.length || 1) - 1}</span>
                    <span className={`font-mono ${r.hit_success ? 'text-accent-ok' : r.hit_failure ? 'text-accent-err' : ''}`}>
                      reward {r.reward.toFixed(3)}
                    </span>
                  </div>
                  <div className="text-fg-dim text-[10.5px] mt-0.5">
                    plan: <TrajectoryWithStates actions={r.planned_actions} states={r.planned_states} />
                  </div>
                </div>
              ))}
            </div>
          )}
          {selectedRoll && (
            <div className="space-y-2">
              <div className="text-[10px] uppercase tracking-wider text-fg-dim">Rollout</div>
              <KV k="index" v={`#${selectedRoll.rollout_index}`} />
              <KV k="first_action" v={selectedRoll.first_action} />
              <div className="flex items-start gap-2 text-[11.5px]">
                <span className="text-fg-dim min-w-[88px]">planned</span>
                <span className="text-fg break-words">
                  <TrajectoryWithStates actions={selectedRoll.planned_actions} states={selectedRoll.planned_states} />
                </span>
              </div>
              <KV k="final_state" v={selectedRoll.final_state} />
              <KV k="depth_completed" v={selectedRoll.depth_completed.toString()} />
              <KV k="rationality" v={selectedRoll.rationality === null ? '—' : selectedRoll.rationality.toFixed(2)} />
              <KV k="progress_bonus" v={selectedRoll.progress_bonus.toFixed(2)} />
              <KV k="reward" v={selectedRoll.reward.toFixed(3)} />
              <KV k="rollout_mode" v={selectedRoll.rollout_mode} />
              <KV k="duration_ms" v={selectedRoll.duration_ms.toString()} />
              <div className="flex items-center gap-2 mt-2">
                {selectedRoll.hit_success && <span className="chip chip-ok flex items-center gap-1"><CheckCircle2 size={11} /> success</span>}
                {selectedRoll.hit_failure && <span className="chip chip-err flex items-center gap-1"><XCircle size={11} /> failure</span>}
                {!selectedRoll.hit_success && !selectedRoll.hit_failure && <span className="chip">open</span>}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function KV({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-start gap-2 text-[11.5px]">
      <span className="text-fg-dim min-w-[88px]">{k}</span>
      <span className="text-fg break-words">{v || '—'}</span>
    </div>
  );
}

const SOURCE_STYLES: Record<PredictorSource, { bg: string; text: string; label: string }> = {
  both:      { bg: 'bg-purple-500/15 border-purple-500/40',    text: 'text-purple-300', label: 'BOTH' },
  mcts:      { bg: 'bg-sky-500/15 border-sky-500/40',          text: 'text-sky-300',    label: 'MCTS' },
  empirical: { bg: 'bg-emerald-500/15 border-emerald-500/40',  text: 'text-emerald-300',label: 'EMP'  },
  live:      { bg: 'bg-amber-500/15 border-amber-500/40',      text: 'text-amber-300',  label: 'LIVE' },
  unknown:   { bg: 'bg-bg-elevated border-bg-border',          text: 'text-fg-dim',     label: '???'  },
};

function SourceBadge({ source, count }: { source: PredictorSource; count?: number }) {
  const s = SOURCE_STYLES[source];
  return (
    <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded border text-[10px] font-mono ${s.bg} ${s.text}`}>
      {s.label}
      {count !== undefined && <span className="opacity-80">{count}</span>}
    </span>
  );
}

function PrefetchRow({ f, showHit = false }: { f: DataFetchRow; showHit?: boolean }) {
  const status: { label: string; cls: string } = f.consumed
    ? { label: 'hit',  cls: 'text-accent-ok' }
    : f.wasted
    ? { label: 'miss', cls: 'text-accent-err' }
    : f.completed_at
    ? { label: 'ready', cls: 'text-fg' }
    : { label: 'flight', cls: 'text-fg-dim' };
  return (
    <div className="text-[11px] border border-bg-border rounded p-1.5 bg-bg-elevated/40">
      <div className="flex items-center justify-between gap-2">
        <span className="text-fg truncate">{f.dependency_name}</span>
        <div className="flex items-center gap-1">
          <SourceBadge source={f.predictor_source} />
          {showHit && <span className={`text-[10px] font-mono ${status.cls}`}>{status.label}</span>}
        </div>
      </div>
      <div className="mt-0.5 text-[10.5px] text-fg-dim flex items-center justify-between">
        <span className="truncate">
          {f.action_name}
          {f.predicted_user_state && (
            <span className="ml-1 px-1 py-0.5 rounded bg-bg-elevated text-fg-muted text-[10px] font-mono">
              {f.predicted_user_state}
            </span>
          )}
        </span>
        <span className="font-mono">
          t{f.issued_at_turn}{f.predicted_turn !== null && f.predicted_turn !== f.issued_at_turn ? `→t${f.predicted_turn}` : ''}
          {' '}· conf {f.confidence.toFixed(2)}
        </span>
      </div>
      {!showHit && (
        <div className="mt-0.5 text-[10px] font-mono opacity-80">
          <span className={status.cls}>{status.label}</span>
          {f.fetch_duration_ms > 0 && <span className="ml-2 text-fg-dim">{f.fetch_duration_ms}ms</span>}
          {f.fetch_error && <span className="ml-2 text-accent-err">err</span>}
        </div>
      )}
    </div>
  );
}

function TrajectoryWithStates({ actions, states }: { actions: string[]; states: string[] }) {
  if (!actions.length) return <span>—</span>;
  return (
    <span className="break-words">
      {actions.map((a, i) => (
        <span key={i}>
          <span className="text-fg">{a}</span>
          {states[i] && (
            <span className="ml-1 px-1 py-0.5 rounded bg-bg-elevated text-fg-muted text-[10px] font-mono">
              {states[i]}
            </span>
          )}
          {i < actions.length - 1 && <span className="text-fg-dim mx-1">→</span>}
        </span>
      ))}
    </span>
  );
}
