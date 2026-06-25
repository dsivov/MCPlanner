import { useMemo, useEffect } from 'react';
import ReactFlow, {
  Background, BackgroundVariant, Controls, MarkerType, Handle, Position,
  type Node, type Edge, useNodesState, useEdgesState,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { motion } from 'framer-motion';
import type { PlannerTrace } from '../lib/api';

type Props = {
  trace: PlannerTrace | null;
  className?: string;
};

type RootData = { kind: 'root'; label: string; subLabel: string };
type CandData = {
  kind: 'candidate';
  label: string;
  q: number;
  visits: number;
  isChosen: boolean;
  rationale: string;
};

const ROOT_W = 220;
const CAND_W = 200;
const CAND_H = 72;

function RootNode({ data }: { data: RootData }) {
  return (
    <div
      className="relative rounded-xl border border-accent/50 bg-bg-elevated px-4 py-3 text-center shadow-glow"
      style={{ width: ROOT_W }}
    >
      <div className="text-[10px] uppercase tracking-wider text-accent">{data.label}</div>
      <div className="mt-1 text-[12px] text-fg font-medium">{data.subLabel}</div>
      <Handle type="source" position={Position.Bottom} style={{ opacity: 0, width: 1, height: 1 }} />
    </div>
  );
}

function CandidateNode({ data }: { data: CandData }) {
  const fill = data.isChosen ? '#7c5cff' : 'rgba(124,92,255,0.45)';
  const border = data.isChosen ? 'border-accent shadow-glow' : 'border-bg-border';
  return (
    <div
      className={`relative rounded-lg border ${border} bg-bg-panel px-3 py-2`}
      style={{ width: CAND_W, height: CAND_H }}
    >
      <Handle type="target" position={Position.Top} style={{ opacity: 0, width: 1, height: 1 }} />
      <div className="flex items-center justify-between">
        <span className="text-[12px] font-medium truncate text-fg max-w-[140px]" title={data.label}>{data.label}</span>
        {data.isChosen && <span className="chip chip-accent !py-0">chosen</span>}
      </div>
      <div className="mt-1 flex items-center justify-between text-[10px] text-fg-dim font-mono">
        <span>Q {data.q.toFixed(3)}</span>
        <span>v {data.visits}</span>
      </div>
      <div className="mt-1.5 h-1 w-full rounded-full bg-bg-elevated overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(100, Math.max(2, data.q * 100))}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
          style={{ background: fill, height: '100%' }}
        />
      </div>
    </div>
  );
}

const nodeTypes = { mctsRoot: RootNode, mctsCand: CandidateNode };

function build(trace: PlannerTrace): { nodes: Node[]; edges: Edge[] } {
  const cands = trace.candidates ?? [];
  const total = cands.length;
  const startX = -((total - 1) * (CAND_W + 28)) / 2;
  const rootY = 30;
  const candY = 180;

  const nodes: Node[] = [
    {
      id: 'root',
      type: 'mctsRoot',
      position: { x: -ROOT_W / 2, y: rootY },
      data: {
        kind: 'root',
        label: 'Predicted user state',
        subLabel: trace.predicted_user_state || '—',
      } as RootData,
    },
  ];

  const edges: Edge[] = [];
  cands.forEach((c, i) => {
    const id = `c-${i}-${c.action}`;
    nodes.push({
      id,
      type: 'mctsCand',
      position: { x: startX + i * (CAND_W + 28), y: candY },
      data: {
        kind: 'candidate',
        label: c.action,
        q: c.q_value,
        visits: c.visits,
        isChosen: c.action === trace.chosen_action,
        rationale: c.rationale || '',
      } as CandData,
    });
    edges.push({
      id: `e-root-${id}`,
      source: 'root',
      target: id,
      type: 'smoothstep',
      animated: c.action === trace.chosen_action,
      style: { stroke: c.action === trace.chosen_action ? '#7c5cff' : '#3a3a44', strokeWidth: c.action === trace.chosen_action ? 1.8 : 1 },
      markerEnd: { type: MarkerType.ArrowClosed, color: c.action === trace.chosen_action ? '#7c5cff' : '#3a3a44' },
      label: c.visits ? `v${c.visits}` : undefined,
      labelStyle: { fill: '#9b9ba6', fontSize: 10 },
      labelBgStyle: { fill: '#111114', stroke: '#23232a' } as never,
    });
  });

  return { nodes, edges };
}

export default function MCTSGraph({ trace, className }: Props) {
  const data = useMemo(() => trace ? build(trace) : { nodes: [], edges: [] }, [trace]);
  const [nodes, setNodes, onNodesChange] = useNodesState(data.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(data.edges);

  useEffect(() => {
    setNodes(data.nodes);
    setEdges(data.edges);
  }, [data, setNodes, setEdges]);

  return (
    <motion.div
      className={`relative h-full w-full overflow-hidden rounded-xl border border-bg-border gradient-panel ${className ?? ''}`}
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.25 }}
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        proOptions={{ hideAttribution: true }}
        minZoom={0.4}
        maxZoom={1.6}
        nodesDraggable={false}
        nodesConnectable={false}
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#1f1f26" />
        <Controls position="bottom-right" showInteractive={false} />
      </ReactFlow>

      {trace && (
        <div className="absolute top-3 left-3 rounded-md border border-bg-border bg-bg-panel/80 backdrop-blur px-2.5 py-1.5 text-[10px] text-fg-muted">
          <div className="flex items-center gap-2">
            <span className="chip">{trace.mode === 'mcts' ? `MCTS · ${trace.rollouts} rollouts` : 'Baseline'}</span>
            {trace.nee_triggered && <span className="chip chip-warn">NEE</span>}
          </div>
        </div>
      )}
      {(!trace || trace.candidates.length === 0) && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-fg-dim text-sm">No MCTS data yet — take a turn first.</div>
        </div>
      )}
    </motion.div>
  );
}
