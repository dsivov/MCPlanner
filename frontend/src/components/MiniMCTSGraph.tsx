import { useMemo } from 'react';
import ReactFlow, {
  Background, BackgroundVariant, MarkerType, Handle, Position,
  type Node, type Edge,
} from 'reactflow';
import 'reactflow/dist/style.css';
import dagre from '@dagrejs/dagre';
import { motion } from 'framer-motion';
import type { PlannerTrace } from '../lib/api';

// Compact MCTS root + candidates graph, embedded per-turn in the chat trace panel.
// Intentionally lightweight — no rollout subtree, no animation, no interactivity.
// For full rollout-level inspection the user opens the dedicated MCTS Replay tab.

const ROOT_W = 180;
const NODE_W = 150;
const NODE_H = 52;

type MiniRootData = { cohort: string; state: string };
type MiniCandData = { action: string; q: number; visits: number; isChosen: boolean; qMax: number };

function MiniRoot({ data }: { data: MiniRootData }) {
  return (
    <div
      className="relative rounded-lg border border-accent/60 bg-bg-elevated px-2.5 py-2 text-center"
      style={{ width: ROOT_W }}
    >
      <div className="text-[9px] uppercase tracking-wider text-accent">root</div>
      <div className="mt-0.5 text-[11px] text-fg font-medium truncate" title={data.cohort}>{data.cohort || '—'}</div>
      <div className="text-[10px] text-fg-dim truncate" title={data.state}>{data.state || '—'}</div>
      <Handle type="source" position={Position.Bottom} style={{ opacity: 0, width: 1, height: 1 }} />
    </div>
  );
}

function MiniCand({ data }: { data: MiniCandData }) {
  const fill = data.isChosen ? '#7c5cff' : 'rgba(124,92,255,0.45)';
  const ratio = data.qMax > 0 ? Math.min(1, Math.max(0, data.q / data.qMax)) : 0;
  return (
    <motion.div
      initial={{ opacity: 0, y: -4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className={`relative rounded-md border ${data.isChosen ? 'border-accent shadow-glow' : 'border-bg-border'} bg-bg-panel px-2 py-1.5`}
      style={{ width: NODE_W, minHeight: NODE_H }}
    >
      <Handle type="target" position={Position.Top} style={{ opacity: 0, width: 1, height: 1 }} />
      <div className="flex items-center justify-between gap-1">
        <span className="text-[11px] font-medium truncate text-fg" title={data.action}>{data.action}</span>
        {data.isChosen && <span className="chip chip-accent !py-0 text-[9px]">★</span>}
      </div>
      <div className="mt-0.5 flex items-center justify-between text-[9.5px] text-fg-dim font-mono">
        <span>Q {data.q.toFixed(2)}</span>
        <span>v{data.visits}</span>
      </div>
      <div className="mt-0.5 h-1 w-full rounded-full bg-bg-elevated overflow-hidden">
        <div style={{ width: `${Math.round(ratio * 100)}%`, background: fill, height: '100%' }} />
      </div>
    </motion.div>
  );
}

const nodeTypes = { miniRoot: MiniRoot, miniCand: MiniCand };

export default function MiniMCTSGraph({ trace }: { trace: PlannerTrace }) {
  const { nodes, edges, height } = useMemo(() => {
    if (!trace.candidates || trace.candidates.length === 0) {
      return { nodes: [], edges: [], height: 0 };
    }
    const g = new dagre.graphlib.Graph();
    g.setDefaultEdgeLabel(() => ({}));
    g.setGraph({ rankdir: 'TB', nodesep: 8, ranksep: 28, marginx: 12, marginy: 12 });

    g.setNode('root', { width: ROOT_W, height: NODE_H });
    const qMax = Math.max(0.01, ...trace.candidates.map((c) => c.q_value));
    for (const c of trace.candidates) {
      g.setNode(`c-${c.action}`, { width: NODE_W, height: NODE_H });
      g.setEdge('root', `c-${c.action}`);
    }
    dagre.layout(g);
    const rootPos = g.node('root');

    const out_nodes: Node[] = [
      {
        id: 'root', type: 'miniRoot',
        position: { x: rootPos.x - ROOT_W / 2, y: rootPos.y - NODE_H / 2 },
        data: { cohort: trace.cohort, state: trace.predicted_user_state } as MiniRootData,
        draggable: false, selectable: false,
      },
      ...trace.candidates.map((c) => {
        const p = g.node(`c-${c.action}`);
        return {
          id: `c-${c.action}`, type: 'miniCand',
          position: { x: p.x - NODE_W / 2, y: p.y - NODE_H / 2 },
          data: {
            action: c.action, q: c.q_value, visits: c.visits,
            isChosen: c.action === trace.chosen_action, qMax,
          } as MiniCandData,
          draggable: false, selectable: false,
        };
      }),
    ];

    const out_edges: Edge[] = trace.candidates.map((c) => ({
      id: `e-${c.action}`,
      source: 'root', target: `c-${c.action}`,
      type: 'smoothstep',
      markerEnd: { type: MarkerType.ArrowClosed, width: 12, height: 12,
                   color: c.action === trace.chosen_action ? '#7c5cff' : 'rgba(124,92,255,0.35)' },
      style: { stroke: c.action === trace.chosen_action ? '#7c5cff' : 'rgba(124,92,255,0.35)',
               strokeWidth: c.action === trace.chosen_action ? 1.8 : 1 },
    }));

    const h = Math.max(...out_nodes.map((n) => n.position.y + NODE_H)) + 16;
    return { nodes: out_nodes, edges: out_edges, height: h };
  }, [trace]);

  if (!nodes.length) {
    return <div className="text-[11px] text-fg-dim italic">No MCTS candidates for this turn.</div>;
  }
  return (
    <div className="rounded-md border border-bg-border bg-bg-elevated/30 overflow-hidden" style={{ height }}>
      <ReactFlow
        nodes={nodes} edges={edges} nodeTypes={nodeTypes}
        fitView fitViewOptions={{ padding: 0.12 }}
        nodesDraggable={false} nodesConnectable={false} elementsSelectable={false}
        panOnDrag={false} panOnScroll={false} zoomOnScroll={false} zoomOnPinch={false}
        zoomOnDoubleClick={false} preventScrolling={false}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={12} size={1} color="rgba(255,255,255,0.04)" />
      </ReactFlow>
    </div>
  );
}
