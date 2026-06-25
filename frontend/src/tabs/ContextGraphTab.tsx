import { useEffect, useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  RefreshCw, Sparkles, Network, ScatterChart as ScatterIcon, FlaskConical,
  CheckCircle2, AlertTriangle, Loader2,
} from 'lucide-react';
import ReactFlow, {
  Background, BackgroundVariant, Controls, MarkerType, Handle, Position,
  type Node, type Edge, useNodesState, useEdgesState,
} from 'reactflow';
import 'reactflow/dist/style.css';
import dagre from '@dagrejs/dagre';
import {
  api,
  type GraphResponse, type ScatterResponse, type TraceDetail,
  type MineResponse, type SOPMeta,
} from '../lib/api';

type ViewMode = 'graph' | 'scatter';

export default function ContextGraphTab() {
  const [seeds, setSeeds] = useState<{ file: string; name: string }[]>([]);
  const [saved, setSaved] = useState<SOPMeta[]>([]);
  const [sopRef, setSopRef] = useState<string>('');

  const [view, setView] = useState<ViewMode>('graph');
  const [graph, setGraph] = useState<GraphResponse | null>(null);
  const [scatter, setScatter] = useState<ScatterResponse | null>(null);
  const [loading, setLoading] = useState(false);

  // Drill-down
  const [drill, setDrill] = useState<{ cohort?: string; action?: string; outcome?: string } | null>(null);
  const [drillTraces, setDrillTraces] = useState<TraceDetail[]>([]);

  // Learning panel
  const [mineRun, setMineRun] = useState<MineResponse | null>(null);
  const [mining, setMining] = useState(false);
  const [selectedProposals, setSelectedProposals] = useState<Set<string>>(new Set());
  const [applying, setApplying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  useEffect(() => {
    api.listSeeds().then(setSeeds).catch(() => {});
    api.listSops().then(setSaved).catch(() => {});
  }, []);
  useEffect(() => {
    if (!sopRef && seeds.length) setSopRef(`seed:${seeds[0].file}`);
  }, [seeds, sopRef]);

  async function refresh() {
    if (!sopRef) return;
    setLoading(true); setError(null);
    try {
      if (view === 'graph') {
        const r = await api.graph(sopRef);
        setGraph(r);
      } else {
        const r = await api.scatter(sopRef);
        setScatter(r);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }
  useEffect(() => { refresh(); }, [sopRef, view]);

  async function drillInto(opts: { cohort?: string; action?: string; outcome?: string }) {
    setDrill(opts);
    if (!sopRef) return;
    try {
      const t = await api.traces({ sop_ref: sopRef, ...opts });
      setDrillTraces(t);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function runMine() {
    if (!sopRef) return;
    setMining(true); setError(null); setMineRun(null); setSelectedProposals(new Set());
    try {
      const r = await api.mine(sopRef);
      setMineRun(r);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setMining(false);
    }
  }

  async function applySelected() {
    if (!mineRun || selectedProposals.size === 0) return;
    setApplying(true); setError(null);
    try {
      if (sopRef.startsWith('seed:')) {
        // Seeds are read-only files. Promote the seed to a saved SOP and apply in one step,
        // then switch the UI to the new saved id so subsequent operations target it.
        const r = await api.saveAndApply(mineRun.run_id, sopRef, Array.from(selectedProposals));
        setSuccess(`Promoted "${r.sop_name}" to a saved SOP and applied ${r.actions_updated} action update(s).`);
        // Refresh saved list and switch the dropdown to the new id
        const list = await api.listSops();
        setSaved(list);
        setSopRef(r.sop_id);
      } else {
        const r = await api.applyProposals(mineRun.run_id, sopRef, Array.from(selectedProposals));
        setSuccess(`Updated ${r.actions_updated} action(s) on this SOP.`);
      }
      setTimeout(() => setSuccess(null), 3500);
      setSelectedProposals(new Set());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setApplying(false);
    }
  }

  return (
    <div className="h-full grid grid-cols-12 gap-4">
      {/* Left: graph / scatter */}
      <div className="col-span-8 card flex flex-col overflow-hidden">
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-bg-border gap-3">
          <div className="flex items-center gap-2">
            <Sparkles size={14} className="text-accent" />
            <span className="text-sm font-medium">Context Graph</span>
            {graph && view === 'graph' && (
              <span className="chip">{graph.n_precedents} precedents</span>
            )}
            {scatter && view === 'scatter' && (
              <span className="chip">{scatter.n_precedents} points</span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <select
              value={sopRef}
              onChange={(e) => { setSopRef(e.target.value); setDrill(null); setMineRun(null); }}
              className="input py-1.5 max-w-[280px]"
            >
              <option value="" disabled>Select SOP…</option>
              {seeds.length > 0 && <optgroup label="Seeds">
                {seeds.map((s) => <option key={s.file} value={`seed:${s.file}`}>{s.name}</option>)}
              </optgroup>}
              {saved.length > 0 && <optgroup label="Saved">
                {saved.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
              </optgroup>}
            </select>
            <ViewToggle view={view} setView={setView} />
            <button className="btn" onClick={refresh} disabled={loading}>
              {loading ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />}
            </button>
          </div>
        </div>
        <div className="flex-1 overflow-hidden relative">
          {view === 'graph' ? (
            graph ? (
              <GraphCanvas data={graph} onPick={drillInto} />
            ) : <Empty loading={loading} text="No graph data yet" />
          ) : (
            scatter ? (
              <ScatterCanvas data={scatter} onPickTrace={(tid) => {
                const p = scatter.points.find((p) => p.trace_id === tid);
                if (p) drillInto({ cohort: p.cohort, action: p.action, outcome: p.outcome });
              }} />
            ) : <Empty loading={loading} text="No scatter data yet" />
          )}
        </div>
      </div>

      {/* Right: drill-down + learning */}
      <div className="col-span-4 flex flex-col gap-4 overflow-hidden">
        <div className="card flex flex-col overflow-hidden" style={{ flexBasis: '45%' }}>
          <div className="flex items-center justify-between px-4 py-2.5 border-b border-bg-border">
            <div className="flex items-center gap-2 text-sm">
              <Network size={14} className="text-accent-alt" />
              <span className="font-medium">Traces</span>
              {drill && (
                <span className="chip">
                  {drill.cohort ? `cohort=${drill.cohort}` : ''} {drill.action ? `action=${drill.action}` : ''} {drill.outcome ? `outcome=${drill.outcome}` : ''}
                </span>
              )}
            </div>
            {drill && <button className="btn btn-ghost text-xs" onClick={() => { setDrill(null); setDrillTraces([]); }}>Clear</button>}
          </div>
          <div className="flex-1 overflow-y-auto p-3 space-y-2">
            {!drill && <div className="text-fg-dim text-sm">Click a node or edge in the graph to drill down.</div>}
            {drill && drillTraces.length === 0 && <div className="text-fg-dim text-sm">No traces for this selection.</div>}
            {drillTraces.map((t) => (
              <div key={t.id} className="rounded border border-bg-border bg-bg-elevated/50 p-2.5 text-[11.5px]">
                <div className="flex justify-between text-fg-muted">
                  <span><b className="text-fg">{t.action}</b> · {t.cohort}</span>
                  <OutcomeChip outcome={t.terminal_outcome} />
                </div>
                <div className="mt-1 text-fg">{t.response_text}</div>
                {t.immediate_state && <div className="mt-1 text-fg-dim">→ {t.immediate_state}</div>}
              </div>
            ))}
          </div>
        </div>

        <div className="card flex flex-col overflow-hidden flex-1">
          <div className="flex items-center justify-between px-4 py-2.5 border-b border-bg-border">
            <div className="flex items-center gap-2 text-sm">
              <FlaskConical size={14} className="text-accent" />
              <span className="font-medium">Learning (slow loop)</span>
            </div>
            <button className="btn btn-primary text-xs" onClick={runMine} disabled={mining || !sopRef}>
              {mining ? <Loader2 size={12} className="animate-spin" /> : <Sparkles size={12} />}
              Mine
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-3 space-y-3">
            {error && (
              <div className="text-[11px] text-accent-err flex items-center gap-1">
                <AlertTriangle size={11} /> {error}
              </div>
            )}
            {success && (
              <div className="text-[11px] text-accent-ok flex items-center gap-1">
                <CheckCircle2 size={11} /> {success}
              </div>
            )}
            {!mineRun && <div className="text-fg-dim text-sm">Run Mine to compute lift and propose must_say / must_not_say updates.</div>}
            {mineRun && (
              <>
                <div className="text-[11px] text-fg-dim">
                  Analyzed {mineRun.n_precedents} precedents across {mineRun.n_sessions} sessions in {mineRun.duration_ms}ms.
                </div>
                <LiftTable rows={mineRun.lift_table} />
                <div className="mt-2">
                  <div className="text-[10px] uppercase tracking-wider text-fg-dim mb-2">Proposals</div>
                  {mineRun.proposals.length === 0 && <div className="text-fg-dim text-xs">No proposals — not enough success/failure terminals yet.</div>}
                  <AnimatePresence>
                    {mineRun.proposals.map((p) => {
                      const checked = selectedProposals.has(p.id);
                      return (
                        <motion.label
                          key={p.id}
                          initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}
                          className={`block rounded-md border ${checked ? 'border-accent/60 bg-accent/10' : 'border-bg-border'} p-2.5 cursor-pointer mb-2 text-[11.5px]`}>
                          <div className="flex items-start gap-2">
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={() => setSelectedProposals((s) => {
                                const n = new Set(s);
                                if (n.has(p.id)) n.delete(p.id); else n.add(p.id);
                                return n;
                              })}
                              className="mt-0.5"
                            />
                            <div className="flex-1">
                              <div className="font-medium text-fg">{p.action} <span className="text-fg-dim">· {p.cohort}</span></div>
                              {p.must_say_add.length > 0 && (
                                <div className="mt-1">
                                  <span className="text-accent-ok text-[10px] uppercase">must say:</span>
                                  <ul className="list-disc ml-4 text-fg">
                                    {p.must_say_add.map((s, i) => <li key={i}>{s}</li>)}
                                  </ul>
                                </div>
                              )}
                              {p.must_not_say_add.length > 0 && (
                                <div className="mt-1">
                                  <span className="text-accent-err text-[10px] uppercase">avoid:</span>
                                  <ul className="list-disc ml-4 text-fg">
                                    {p.must_not_say_add.map((s, i) => <li key={i}>{s}</li>)}
                                  </ul>
                                </div>
                              )}
                              <div className="mt-1 text-fg-dim italic">{p.rationale}</div>
                            </div>
                          </div>
                        </motion.label>
                      );
                    })}
                  </AnimatePresence>
                </div>
                {mineRun.proposals.length > 0 && (
                  <button
                    className="btn btn-primary w-full"
                    disabled={selectedProposals.size === 0 || applying}
                    onClick={applySelected}>
                    {applying ? <Loader2 size={13} className="animate-spin" /> : <CheckCircle2 size={13} />}
                    {sopRef.startsWith('seed:')
                      ? `Save copy + apply ${selectedProposals.size}`
                      : `Apply ${selectedProposals.size} to SOP`}
                  </button>
                )}
                {sopRef.startsWith('seed:') && mineRun.proposals.length > 0 && (
                  <div className="text-[11px] text-fg-dim flex items-start gap-1">
                    <AlertTriangle size={11} className="mt-0.5 flex-none" />
                    <span>Seed files are read-only. Apply will create a saved copy and switch you to it.</span>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function Empty({ loading, text }: { loading: boolean; text: string }) {
  return (
    <div className="absolute inset-0 flex items-center justify-center text-fg-dim text-sm">
      {loading ? <Loader2 size={18} className="animate-spin" /> : text}
    </div>
  );
}

function ViewToggle({ view, setView }: { view: ViewMode; setView: (v: ViewMode) => void }) {
  return (
    <div className="flex items-center rounded-md border border-bg-border bg-bg-elevated p-0.5">
      <button
        onClick={() => setView('graph')}
        className={`relative px-2.5 py-1 text-xs rounded ${view === 'graph' ? 'text-white' : 'text-fg-muted hover:text-fg'}`}
      >
        {view === 'graph' && <motion.div layoutId="cg-view" className="absolute inset-0 rounded bg-accent/20 border border-accent/40" />}
        <span className="relative flex items-center gap-1.5"><Network size={12} /> Graph</span>
      </button>
      <button
        onClick={() => setView('scatter')}
        className={`relative px-2.5 py-1 text-xs rounded ${view === 'scatter' ? 'text-white' : 'text-fg-muted hover:text-fg'}`}
      >
        {view === 'scatter' && <motion.div layoutId="cg-view" className="absolute inset-0 rounded bg-accent/20 border border-accent/40" />}
        <span className="relative flex items-center gap-1.5"><ScatterIcon size={12} /> Scatter</span>
      </button>
    </div>
  );
}

function OutcomeChip({ outcome }: { outcome: string | null }) {
  if (!outcome) return <span className="chip">open</span>;
  if (outcome === 'success') return <span className="chip chip-ok">success</span>;
  if (outcome === 'failure') return <span className="chip chip-err">failure</span>;
  return <span className="chip chip-warn">{outcome}</span>;
}

function LiftTable({ rows }: { rows: import('../lib/api').LiftRow[] }) {
  if (!rows.length) return <div className="text-fg-dim text-xs">No lift data.</div>;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[11px] border-collapse">
        <thead className="text-fg-dim">
          <tr>
            <th className="text-left py-1 pr-2">Cohort</th>
            <th className="text-left py-1 pr-2">Action</th>
            <th className="text-right py-1 pr-2">n</th>
            <th className="text-right py-1 pr-2">succ%</th>
            <th className="text-right py-1">lift</th>
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 30).map((r, i) => (
            <tr key={i} className="border-t border-bg-border">
              <td className="py-1 pr-2">{r.cohort}</td>
              <td className="py-1 pr-2 text-fg">{r.action}</td>
              <td className="py-1 pr-2 text-right font-mono">{r.n_total}</td>
              <td className="py-1 pr-2 text-right font-mono">{(r.success_rate * 100).toFixed(0)}%</td>
              <td className={`py-1 text-right font-mono ${r.lift_vs_cohort > 0 ? 'text-accent-ok' : r.lift_vs_cohort < 0 ? 'text-accent-err' : 'text-fg-muted'}`}>
                {r.lift_vs_cohort > 0 ? '+' : ''}{(r.lift_vs_cohort * 100).toFixed(0)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ----- Graph canvas (cohort -> action -> outcome) -----

const NODE_W = 170, NODE_H = 44;

function CGNode({ data }: { data: { label: string; kind: 'cohort'|'action'|'outcome'; count: number } }) {
  const palette = data.kind === 'cohort'
    ? { bg: 'bg-accent-alt/10', border: 'border-accent-alt/50', dot: '#22d3ee' }
    : data.kind === 'action'
    ? { bg: 'bg-accent/10', border: 'border-accent/50', dot: '#7c5cff' }
    : { bg: 'bg-bg-elevated', border: 'border-bg-border', dot: '#9b9ba6' };
  return (
    <div className={`relative rounded-lg border ${palette.border} ${palette.bg} px-3 py-2`} style={{ width: NODE_W, height: NODE_H }}>
      <Handle type="target" position={Position.Left} style={{ opacity: 0, width: 1, height: 1 }} />
      <div className="flex items-center justify-between text-[12px]">
        <span className="flex items-center gap-1.5 truncate">
          <span className="h-2 w-2 rounded-full" style={{ background: palette.dot, boxShadow: `0 0 6px ${palette.dot}` }} />
          <span className="truncate text-fg" title={data.label}>{data.label}</span>
        </span>
        <span className="font-mono text-[10px] text-fg-dim">{data.count}</span>
      </div>
      <div className="text-[9px] uppercase tracking-wider text-fg-dim mt-0.5">{data.kind}</div>
      <Handle type="source" position={Position.Right} style={{ opacity: 0, width: 1, height: 1 }} />
    </div>
  );
}
const cgNodeTypes = { cgNode: CGNode };

function GraphCanvas({ data, onPick }: { data: GraphResponse; onPick: (opts: { cohort?: string; action?: string; outcome?: string }) => void }) {
  const layout = useMemo(() => {
    const g = new dagre.graphlib.Graph();
    g.setDefaultEdgeLabel(() => ({}));
    g.setGraph({ rankdir: 'LR', nodesep: 30, ranksep: 110, marginx: 24, marginy: 24 });
    for (const n of data.nodes) g.setNode(n.id, { width: NODE_W, height: NODE_H });
    for (const e of data.edges) g.setEdge(e.src, e.dst);
    dagre.layout(g);
    const maxCount = Math.max(1, ...data.edges.map((e) => e.count));
    const nodes: Node[] = data.nodes.map((n) => {
      const pos = g.node(n.id);
      return {
        id: n.id,
        type: 'cgNode',
        position: { x: pos.x - NODE_W / 2, y: pos.y - NODE_H / 2 },
        data: { label: n.label, kind: n.kind, count: n.count },
      };
    });
    const edges: Edge[] = data.edges.map((e, i) => {
      const isCA = e.src.startsWith('cohort:') && e.dst.startsWith('action:');
      const liftClamp = Math.max(-0.4, Math.min(0.4, e.lift));
      const color = isCA
        ? (liftClamp >= 0
            ? `rgba(52, 211, 153, ${0.3 + liftClamp * 1.2})`
            : `rgba(248, 113, 113, ${0.3 - liftClamp * 1.2})`)
        : '#3a3a44';
      const width = 1 + (e.count / maxCount) * 4;
      return {
        id: `e-${i}-${e.src}-${e.dst}`,
        source: e.src, target: e.dst,
        type: 'smoothstep',
        style: { stroke: color, strokeWidth: width },
        markerEnd: { type: MarkerType.ArrowClosed, color, width: 14, height: 14 },
        label: isCA ? `${e.count} · ${(e.success_rate*100).toFixed(0)}%` : `${e.count}`,
        labelStyle: { fill: '#9b9ba6', fontSize: 10 },
        labelBgStyle: { fill: '#111114', stroke: '#23232a' } as never,
      };
    });
    return { nodes, edges };
  }, [data]);

  const [nodes, setNodes, onNodesChange] = useNodesState(layout.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(layout.edges);
  useEffect(() => { setNodes(layout.nodes); setEdges(layout.edges); }, [layout, setNodes, setEdges]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      nodeTypes={cgNodeTypes}
      fitView
      fitViewOptions={{ padding: 0.18 }}
      proOptions={{ hideAttribution: true }}
      minZoom={0.3}
      maxZoom={1.6}
      nodesDraggable={false}
      nodesConnectable={false}
      onNodeClick={(_, n) => {
        const id = n.id;
        if (id.startsWith('cohort:')) onPick({ cohort: id.slice('cohort:'.length) });
        else if (id.startsWith('action:')) onPick({ action: id.slice('action:'.length) });
        else if (id.startsWith('outcome:')) onPick({ outcome: id.slice('outcome:'.length) });
      }}
      onEdgeClick={(_, e) => {
        const src = (e.source as string) || '';
        const dst = (e.target as string) || '';
        const cohort = src.startsWith('cohort:') ? src.slice('cohort:'.length) : undefined;
        const action = (src.startsWith('action:') ? src.slice('action:'.length)
                      : dst.startsWith('action:') ? dst.slice('action:'.length) : undefined);
        const outcome = dst.startsWith('outcome:') ? dst.slice('outcome:'.length) : undefined;
        onPick({ cohort, action, outcome });
      }}
    >
      <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#1f1f26" />
      <Controls position="bottom-right" showInteractive={false} />
    </ReactFlow>
  );
}

// ----- Scatter canvas (PCA 2D of embeddings) -----

function ScatterCanvas({ data, onPickTrace }: { data: ScatterResponse; onPickTrace: (trace_id: string) => void }) {
  const colorFor = (outcome: string) => (
    outcome === 'success' ? '#34d399' :
    outcome === 'failure' ? '#f87171' :
    outcome === 'abandoned' ? '#fbbf24' :
    '#7c5cff'
  );
  const W = 720, H = 480, PAD = 32;
  const toX = (x: number) => PAD + ((x + 1) / 2) * (W - 2 * PAD);
  const toY = (y: number) => PAD + ((y + 1) / 2) * (H - 2 * PAD);
  return (
    <div className="relative h-full w-full overflow-auto p-3">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full">
        {/* axes faint */}
        <line x1={W/2} y1={PAD} x2={W/2} y2={H-PAD} stroke="#23232a" />
        <line x1={PAD} y1={H/2} x2={W-PAD} y2={H/2} stroke="#23232a" />
        {data.points.map((p) => (
          <circle
            key={p.trace_id}
            cx={toX(p.x)} cy={toY(p.y)} r={4}
            fill={colorFor(p.outcome)}
            opacity={0.85}
            onClick={() => onPickTrace(p.trace_id)}
            style={{ cursor: 'pointer' }}
          >
            <title>{`${p.action} · ${p.cohort} · ${p.outcome}`}</title>
          </circle>
        ))}
      </svg>
      <div className="absolute top-3 left-3 rounded-md border border-bg-border bg-bg-panel/90 backdrop-blur px-2.5 py-1.5 text-[10px] text-fg-muted space-y-0.5">
        <div className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: '#34d399' }} /> success</div>
        <div className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: '#f87171' }} /> failure</div>
        <div className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: '#fbbf24' }} /> abandoned</div>
        <div className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: '#7c5cff' }} /> open</div>
      </div>
    </div>
  );
}
