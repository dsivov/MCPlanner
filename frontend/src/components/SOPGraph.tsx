import { useMemo, useEffect } from 'react';
import ReactFlow, {
  Background, BackgroundVariant, Controls, MarkerType, Handle, Position,
  type Node, type Edge, useNodesState, useEdgesState,
} from 'reactflow';
import 'reactflow/dist/style.css';
import dagre from '@dagrejs/dagre';
import { motion } from 'framer-motion';
import type { TaskDefinition } from '../lib/api';

type Props = {
  sop: TaskDefinition;
  /** The node we are currently at (e.g., the action chosen this turn). Pulses with strong glow. */
  currentNode?: string;
  /** Other nodes the planner considered this turn. Rendered with secondary highlight. */
  proposedNodes?: string[];
  /** Backward-compat alias for currentNode (still accepted by ChatTab). */
  highlightNode?: string;
  className?: string;
};

const NODE_W = 150;
const NODE_H = 44;

function layout(sop: TaskDefinition): { nodes: Node[]; edges: Edge[] } {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: 'LR', nodesep: 32, ranksep: 80, marginx: 24, marginy: 24 });

  const actions = new Set(sop.agent_actions.map((a) => a.name));
  const states = new Set(sop.user_states.map((s) => s.name));

  // Only include nodes that appear in the catalogs
  const nodeNames = new Set<string>([...actions, ...states]);
  for (const name of nodeNames) {
    g.setNode(name, { width: NODE_W, height: NODE_H });
  }

  // Edges
  const validEdges = sop.sop.edges.filter((e) => nodeNames.has(e.src) && nodeNames.has(e.dst));
  for (const e of validEdges) {
    const [from, to] =
      e.direction === 'backward' ? [e.dst, e.src] : [e.src, e.dst];
    g.setEdge(from, to);
  }

  dagre.layout(g);

  const nodes: Node[] = [...nodeNames].map((name) => {
    const n = g.node(name);
    const kind: 'action' | 'state' = actions.has(name) ? 'action' : 'state';
    return {
      id: name,
      type: 'sopNode',
      data: { label: name, kind },
      position: { x: n.x - NODE_W / 2, y: n.y - NODE_H / 2 },
    };
  });

  const edges: Edge[] = validEdges.map((e, i) => {
    const both = e.direction === 'both';
    const stroke = both ? '#22d3ee' : '#7c5cff';
    const [from, to] =
      e.direction === 'backward' ? [e.dst, e.src] : [e.src, e.dst];
    return {
      id: `e-${i}-${from}-${to}`,
      source: from,
      target: to,
      type: 'smoothstep',
      animated: true,
      style: { stroke, strokeWidth: 1.5, strokeOpacity: 0.85 },
      markerEnd: { type: MarkerType.ArrowClosed, color: stroke, width: 18, height: 18 },
      labelStyle: { fill: '#9b9ba6', fontSize: 10 },
      labelBgStyle: { fill: '#111114', stroke: '#23232a' } as never,
      label: e.note || undefined,
    };
  });

  return { nodes, edges };
}

type NodeData = {
  label: string;
  kind: 'action' | 'state';
  isCurrent?: boolean;
  isProposed?: boolean;
};

function SOPNode({ data }: { data: NodeData }) {
  const isAction = data.kind === 'action';
  const dotColor = isAction ? '#7c5cff' : '#22d3ee';
  const ring =
    data.isCurrent
      ? 'ring-2 ring-accent shadow-glow'
      : data.isProposed
      ? 'ring-1 ring-accent-alt/70 ring-offset-1 ring-offset-bg-base'
      : '';
  return (
    <div
      className={`group relative flex items-center gap-2 rounded-full border px-3.5 py-2 text-[12px] font-medium select-none
        ${isAction ? 'border-accent/50 bg-accent/12 text-white' : 'border-accent-alt/50 bg-accent-alt/10 text-white'}
        ${ring} transition-shadow`}
      style={{ minWidth: NODE_W - 8, height: NODE_H - 4 }}
    >
      <Handle type="target" position={Position.Left}  style={{ opacity: 0, width: 1, height: 1, background: dotColor }} />
      {data.isCurrent ? (
        <motion.span
          className="inline-block h-2 w-2 rounded-full flex-none"
          style={{ background: '#ffffff' }}
          animate={{ scale: [1, 1.6, 1], boxShadow: [`0 0 6px #7c5cff`, `0 0 14px #7c5cff`, `0 0 6px #7c5cff`] }}
          transition={{ duration: 1.6, repeat: Infinity, ease: 'easeInOut' }}
        />
      ) : (
        <span
          className="inline-block h-2 w-2 rounded-full flex-none"
          style={{ background: data.isProposed ? '#fbbf24' : dotColor, boxShadow: `0 0 8px ${data.isProposed ? '#fbbf24' : dotColor}` }}
        />
      )}
      <span className="truncate">{data.label}</span>
      {data.isCurrent && <span className="chip chip-accent ml-1 !py-0">now</span>}
      {data.isProposed && !data.isCurrent && <span className="chip chip-warn ml-1 !py-0">cand</span>}
      <Handle type="source" position={Position.Right} style={{ opacity: 0, width: 1, height: 1, background: dotColor }} />
    </div>
  );
}

const nodeTypes = { sopNode: SOPNode };

export default function SOPGraph({ sop, currentNode, proposedNodes, highlightNode, className }: Props) {
  const effectiveCurrent = currentNode ?? highlightNode ?? '';
  // Stable string key so we don't loop on every parent render where proposedNodes is a new array.
  const proposedKey = (proposedNodes ?? []).join('|');

  // Bake the highlight flags into the layout result so we only have one source of truth.
  const layouted = useMemo(() => {
    const base = layout(sop);
    const proposedSet = new Set(proposedKey ? proposedKey.split('|') : []);
    base.nodes = base.nodes.map((n) => ({
      ...n,
      data: {
        ...(n.data as NodeData),
        isCurrent: n.id === effectiveCurrent,
        isProposed: proposedSet.has(n.id) && n.id !== effectiveCurrent,
      },
    }));
    return base;
  }, [sop, effectiveCurrent, proposedKey]);

  const [nodes, setNodes, onNodesChange] = useNodesState(layouted.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(layouted.edges);

  useEffect(() => {
    setNodes(layouted.nodes);
    setEdges(layouted.edges);
  }, [layouted, setNodes, setEdges]);

  const empty = nodes.length === 0;

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
        fitViewOptions={{ padding: 0.18 }}
        proOptions={{ hideAttribution: true }}
        minZoom={0.25}
        maxZoom={1.8}
        nodesDraggable
        nodesConnectable={false}
        elementsSelectable
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#1f1f26" />
        <Controls position="bottom-right" showInteractive={false} />
      </ReactFlow>

      {/* Legend */}
      {!empty && (
        <div className="absolute top-3 left-3 flex items-center gap-3 rounded-md border border-bg-border bg-bg-panel/80 backdrop-blur px-2.5 py-1.5 text-[10px] text-fg-muted">
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-accent shadow-[0_0_6px_#7c5cff]" />
            action
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-accent-alt shadow-[0_0_6px_#22d3ee]" />
            user state
          </span>
        </div>
      )}

      {empty && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-fg-dim text-sm">Graph will appear here as you describe the SOP</div>
        </div>
      )}
    </motion.div>
  );
}
