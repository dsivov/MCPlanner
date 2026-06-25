import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Play, Send, Sliders, User, Bot, Brain, Activity, Loader2, Sparkles, Zap, Gauge, Network, GitBranchPlus, Square,
} from 'lucide-react';
import {
  api, type TaskDefinition, type SOPMeta, type PlannerMode, type ChatMode, type MCTSConfig,
  type PlannerTrace,
} from '../lib/api';
import SOPGraph from '../components/SOPGraph';
import MCTSGraph from '../components/MCTSGraph';
import MiniMCTSGraph from '../components/MiniMCTSGraph';

type Msg = { role: 'user' | 'assistant'; content: string; action?: string };

export default function ChatTab() {
  const [seeds, setSeeds] = useState<{ file: string; name: string }[]>([]);
  const [saved, setSaved] = useState<SOPMeta[]>([]);
  const [sopId, setSopId] = useState<string>('');     // either "seed:<file>" or sop record id
  const [sop, setSop] = useState<TaskDefinition | null>(null);

  const [plannerMode, setPlannerMode] = useState<PlannerMode>('mcts');
  const [chatMode, setChatMode] = useState<ChatMode>('human');
  const [mcts, setMcts] = useState<MCTSConfig>({
    iterations: 8, branching: 3, rollout_depth: 3, c_uct: 1.4,
    parallel_rollouts: 4, nee_threshold: 0.15, nee_min_visits: 2,
    top_k_precedents: 3, use_precedents_expand: true,
    use_precedents_score: false, use_precedents_response: true,
    pondering_enabled: true, pondering_k: 2,
    rollout_mode: 'simulate',
    router_enabled: true, tier_entropy_max_t1: 0.4,
    tier_entropy_max_t2: 1.2, tier_min_supporting_traces: 3,
    planning_granularity: 'action',
  });

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [traces, setTraces] = useState<PlannerTrace[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedTraceIdx, setSelectedTraceIdx] = useState<number | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Autopilot — in auto chat-mode only. When enabled, the next turn is automatically
  // queued after each turn completes, until the session ends (success/failure) or the
  // user disables it. The `autopilotRef` lets the loop check the *latest* toggle state
  // without forming a stale closure.
  const [autopilot, setAutopilot] = useState(false);
  const autopilotRef = useRef(false);
  useEffect(() => { autopilotRef.current = autopilot; }, [autopilot]);
  // AbortController for the in-flight /turn request, so Stop is responsive mid-turn.
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    api.listSeeds().then(setSeeds).catch(() => {});
    api.listSops().then(setSaved).catch(() => {});
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, busy]);

  // Auto-select first seed if none chosen yet
  useEffect(() => {
    if (!sopId && seeds.length) setSopId(`seed:${seeds[0].file}`);
  }, [seeds, sopId]);

  async function start() {
    if (!sopId) return;
    setError(null);
    setBusy(true);
    try {
      const r = await api.startChat(sopId, plannerMode, chatMode, mcts);
      setSessionId(r.session_id);
      setSop(r.sop);
      setMessages([]);
      setTraces([]);
      setSelectedTraceIdx(null);
      if (chatMode === 'auto') {
        await takeTurn(r.session_id, undefined);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function takeTurn(sid: string, userText?: string) {
    setBusy(true);
    setError(null);
    let ended = false;
    let aborted = false;
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      const r = await api.turn(sid, userText, ctrl.signal);
      // FUNCTIONAL updates — needed for autopilot's recursive calls. Direct closures
      // over `messages` / `traces` go stale between rapid turns and wipe history.
      setMessages((prev) => [
        ...prev,
        { role: 'user', content: r.user_message },
        { role: 'assistant', content: r.assistant_message, action: r.trace.chosen_action },
      ]);
      setTraces((prev) => {
        const next = [...prev, r.trace];
        setSelectedTraceIdx(next.length - 1);
        return next;
      });
    } catch (e: unknown) {
      // AbortError surfaces as DOMException with name === 'AbortError', or as a fetch
      // rejection with .name 'AbortError' in some browsers. Either way: ctrl.signal.aborted
      // is the reliable signal.
      if (ctrl.signal.aborted) {
        aborted = true;
      } else {
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
        // Backend signals "session already ended" via /turn rejection — stop autopilot.
        if (/already ended/i.test(msg)) ended = true;
      }
    } finally {
      setBusy(false);
      if (abortRef.current === ctrl) abortRef.current = null;
    }

    // Autopilot: schedule the next turn if still enabled and we're in auto mode and the
    // current turn wasn't aborted by Stop. Small delay yields the event loop so React
    // commits state + UI repaints between turns.
    if (autopilotRef.current && chatMode === 'auto' && !ended && !aborted) {
      setTimeout(() => {
        if (autopilotRef.current && chatMode === 'auto') {
          takeTurn(sid, undefined);
        }
      }, 120);
    }
  }

  function stopAutopilot() {
    setAutopilot(false);
    autopilotRef.current = false;             // sync immediately, don't wait for useEffect
    abortRef.current?.abort();                 // cancel in-flight /turn
  }

  async function sendUserMessage() {
    if (!sessionId) return;
    const t = input.trim();
    if (!t || busy) return;
    setInput('');
    await takeTurn(sessionId, t);
  }

  const selectedTrace = selectedTraceIdx != null ? traces[selectedTraceIdx] : traces[traces.length - 1];
  const currentNode = selectedTrace?.chosen_action;
  const proposedNodes = useMemo(
    () => selectedTrace?.candidates?.map((c) => c.action) ?? [],
    [selectedTrace],
  );
  const [graphMode, setGraphMode] = useState<'sop' | 'mcts'>('sop');

  return (
    <div className="h-full grid grid-cols-12 gap-4">
      {/* Left: setup + chat */}
      <div className="col-span-7 flex flex-col gap-4 overflow-hidden">
        <SetupBar
          seeds={seeds}
          saved={saved}
          sopId={sopId}
          setSopId={setSopId}
          plannerMode={plannerMode}
          setPlannerMode={setPlannerMode}
          chatMode={chatMode}
          setChatMode={setChatMode}
          mcts={mcts}
          setMcts={setMcts}
          onStart={start}
          onEnd={async () => {
            if (!sessionId) return;
            stopAutopilot();                            // cancel in-flight + disable loop
            try {
              await api.endSession(sessionId);
              setSessionId(null);
            } catch (e: unknown) {
              setError(e instanceof Error ? e.message : String(e));
            }
          }}
          running={!!sessionId}
          busy={busy}
        />

        <div className="card flex flex-col overflow-hidden flex-1">
          <div className="flex items-center justify-between px-4 py-2.5 border-b border-bg-border">
            <div className="flex items-center gap-2 text-sm">
              <Bot size={14} className="text-accent" />
              <span className="font-medium">{sop?.name || 'No SOP loaded'}</span>
              {plannerMode === 'mcts' ? (
                <span className="chip chip-accent">PCA-M · MCTS</span>
              ) : (
                <span className="chip">Baseline CoT+SOP</span>
              )}
              <span className="chip">{chatMode === 'human' ? 'Human user' : 'Auto user'}</span>
              {mcts.pondering_enabled && traces.length > 0 && (() => {
                const hits = traces.filter((t) => t.from_pondering).length;
                const eligible = traces.length - 1;  // turn 0 has no precomputed cache
                if (eligible <= 0) return null;
                const pct = Math.round((hits / eligible) * 100);
                return (
                  <span className={`chip ${hits > 0 ? 'chip-ok' : ''}`} title="Pondering cache hits / eligible turns">
                    Pondering {hits}/{eligible} · {pct}%
                  </span>
                );
              })()}
            </div>
            {sessionId && (
              <div className="text-[11px] font-mono text-fg-dim">sid:{sessionId.slice(0, 6)}</div>
            )}
          </div>

          <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
            {messages.length === 0 && !busy && (
              <div className="h-full flex items-center justify-center text-fg-dim text-sm">
                {sessionId
                  ? 'Send a message to start the dialogue.'
                  : 'Pick an SOP, configure the planner, and click Start.'}
              </div>
            )}
            <AnimatePresence initial={false}>
              {messages.map((m, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.2 }}
                  className={`flex items-start gap-2 ${m.role === 'user' ? 'flex-row-reverse' : ''}`}
                >
                  <div className={`mt-0.5 h-7 w-7 rounded-md flex items-center justify-center
                    ${m.role === 'user' ? 'bg-bg-elevated border border-bg-border' : 'bg-accent/15 border border-accent/40'}`}>
                    {m.role === 'user' ? <User size={13} /> : <Bot size={13} className="text-accent" />}
                  </div>
                  <div
                    className={`max-w-[80%] rounded-2xl px-3.5 py-2 text-[13px] leading-snug
                      ${m.role === 'user'
                        ? 'bg-accent/10 border border-accent/30 text-white'
                        : 'bg-bg-elevated border border-bg-border'}`}
                    onClick={() => {
                      if (m.role === 'assistant') {
                        const turnIdx = Math.floor(i / 2);
                        setSelectedTraceIdx(turnIdx);
                      }
                    }}
                  >
                    {m.action && (
                      <div className="text-[10px] uppercase tracking-wider text-accent mb-0.5 cursor-pointer">
                        {m.action}
                      </div>
                    )}
                    <div>{m.content}</div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
            {busy && (
              <div className="flex items-center gap-2 text-fg-dim text-xs px-1">
                <Loader2 size={14} className="animate-spin" />
                <span>{plannerMode === 'mcts' ? 'Running MCTS rollouts…' : 'Selecting action…'}</span>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {error && <div className="text-[11px] text-accent-err px-4 pb-1">{error}</div>}

          <div className="border-t border-bg-border p-3 flex gap-2">
            {chatMode === 'human' ? (
              <>
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), sendUserMessage())}
                  placeholder={sessionId ? "Type as the user…" : 'Start a session first'}
                  className="input"
                  disabled={!sessionId || busy}
                />
                <button className="btn btn-primary" onClick={sendUserMessage} disabled={!sessionId || busy || !input.trim()}>
                  <Send size={14} />
                </button>
              </>
            ) : (
              <div className="flex gap-2 w-full">
                <button
                  className="btn btn-primary flex-1"
                  onClick={() => sessionId && takeTurn(sessionId, undefined)}
                  disabled={!sessionId || busy || autopilot}>
                  <Play size={13} /> Step
                </button>
                {autopilot ? (
                  <button
                    className="btn flex-1 border-accent-err/60 bg-accent-err/15 text-accent-err hover:bg-accent-err/25"
                    onClick={stopAutopilot}
                    title="Stop autopilot — cancels the in-flight turn">
                    <Square size={13} fill="currentColor" /> Stop
                  </button>
                ) : (
                  <button
                    className="btn flex-1"
                    onClick={() => {
                      setAutopilot(true);
                      autopilotRef.current = true;
                      if (sessionId && !busy) takeTurn(sessionId, undefined);
                    }}
                    disabled={!sessionId}
                    title="Autopilot: keep generating turns until the session ends">
                    <Play size={13} /> Autopilot
                  </button>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Right: trace + graph */}
      <div className="col-span-5 flex flex-col gap-4 overflow-hidden">
        <div className="card flex flex-col overflow-hidden" style={{ flexBasis: '55%' }}>
          <div className="flex items-center justify-between px-4 py-2.5 border-b border-bg-border">
            <div className="flex items-center gap-2 text-sm">
              <Brain size={14} className="text-accent" />
              <span className="font-medium">Planner Trace</span>
              {selectedTrace && (
                <span className="text-[11px] text-fg-dim font-mono">
                  turn {(selectedTraceIdx ?? traces.length - 1) + 1}/{traces.length}
                </span>
              )}
            </div>
            <div className="flex items-center gap-1">
              <button
                className="btn btn-ghost text-xs"
                disabled={selectedTraceIdx == null || selectedTraceIdx === 0}
                onClick={() => setSelectedTraceIdx((s) => (s ?? 0) - 1)}
              >‹</button>
              <button
                className="btn btn-ghost text-xs"
                disabled={selectedTraceIdx == null || selectedTraceIdx >= traces.length - 1}
                onClick={() => setSelectedTraceIdx((s) => (s ?? 0) + 1)}
              >›</button>
            </div>
          </div>
          <div className="flex-1 overflow-y-auto p-4">
            {selectedTrace ? <TraceDetail trace={selectedTrace} /> : (
              <div className="text-fg-dim text-sm h-full flex items-center justify-center">
                No turns yet.
              </div>
            )}
          </div>
        </div>

        <div className="card flex flex-col overflow-hidden flex-1">
          <div className="flex items-center justify-between px-4 py-2.5 border-b border-bg-border">
            <div className="flex items-center gap-2 text-sm">
              <Sparkles size={14} className="text-accent-alt" />
              <span className="font-medium">{graphMode === 'sop' ? 'SOP graph' : 'MCTS tree'}</span>
            </div>
            <GraphModeToggle mode={graphMode} setMode={setGraphMode} />
          </div>
          <div className="flex-1 overflow-hidden">
            {graphMode === 'sop' ? (
              sop ? (
                <SOPGraph
                  sop={sop}
                  currentNode={currentNode}
                  proposedNodes={proposedNodes}
                  className="h-full"
                />
              ) : (
                <div className="h-full flex items-center justify-center text-fg-dim text-sm">No SOP loaded</div>
              )
            ) : (
              <MCTSGraph trace={selectedTrace ?? null} className="h-full" />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function TraceDetail({ trace }: { trace: PlannerTrace }) {
  const maxQ = Math.max(0.01, ...trace.candidates.map((c) => c.q_value));
  const precedents = trace.precedents ?? [];
  const tierLabel: Record<string, string> = {
    cached_playbook: 'Cached playbook',
    baseline: 'Baseline planner',
    mcts: 'Full MCTS',
  };
  const tierColors: Record<string, string> = {
    cached_playbook: 'border-accent-ok/50 bg-accent-ok/10 text-accent-ok',
    baseline: 'border-accent-warn/50 bg-accent-warn/10 text-accent-warn',
    mcts: 'border-accent/50 bg-accent/10 text-accent',
  };
  return (
    <div className="space-y-4">
      {trace.from_pondering && (
        <div className="rounded-md border border-accent-ok/50 bg-accent-ok/10 px-2.5 py-1.5 text-[11.5px] text-accent-ok flex items-center gap-2">
          <Brain size={13} />
          <span>Served from pondering — speculative MCTS hit on state <b>{trace.pondering_hit_state}</b></span>
        </div>
      )}
      {trace.tier_used && (
        <div className={`rounded-md border px-2.5 py-1.5 text-[11.5px] flex items-start gap-2 ${tierColors[trace.tier_used] || ''}`}>
          <Activity size={13} className="mt-0.5 flex-none" />
          <div className="flex-1">
            <div><b>Router → {tierLabel[trace.tier_used]}</b>
              {trace.tier_entropy !== null && <> · entropy {trace.tier_entropy.toFixed(2)}</>}
              {trace.tier_supporting_traces > 0 && <> · {trace.tier_supporting_traces} precedent{trace.tier_supporting_traces === 1 ? '' : 's'}</>}
            </div>
            {trace.tier_rationale && <div className="text-fg-dim text-[10.5px] mt-0.5">{trace.tier_rationale}</div>}
          </div>
        </div>
      )}
      <div className="grid grid-cols-2 gap-2">
        <Stat icon={<User size={11} />} label="Predicted user state" value={trace.predicted_user_state || '—'} accent />
        <Stat icon={<Zap size={11} />} label="Chosen action" value={trace.chosen_action || '—'} accent />
        {trace.chosen_strategy && <Stat icon={<GitBranchPlus size={11} />} label="Strategy" value={trace.chosen_strategy} accent />}
        {trace.cohort && <Stat icon={<Sparkles size={11} />} label="Cohort" value={trace.cohort} />}
        {trace.mood && <Stat icon={<Sparkles size={11} />} label="Mood" value={trace.mood} />}
        <Stat icon={<Activity size={11} />} label={trace.mode === 'mcts' ? 'MCTS rollouts' : 'Mode'}
          value={trace.mode === 'mcts'
            ? `${trace.rollouts} (iters ${trace.mcts_iterations})${trace.nee_triggered ? ' · NEE' : ''}${trace.from_pondering ? ' · cache' : ''}`
            : 'Baseline'} />
        <Stat icon={<Gauge size={11} />} label="Tokens (in/out)" value={`${trace.tokens_in} / ${trace.tokens_out}`} />
        <Stat
          icon={<Loader2 size={11} />}
          label="Agent latency"
          value={`${(trace.agent_duration_ms / 1000).toFixed(1)}s${trace.user_sim_ms ? ` (sim +${(trace.user_sim_ms / 1000).toFixed(1)}s)` : ''}`}
        />
      </div>

      {trace.mode === 'mcts' && trace.candidates && trace.candidates.length > 0 && (
        <div>
          <div className="text-[11px] uppercase tracking-wider text-fg-dim mb-2 flex items-center justify-between">
            <span>MCTS search ({trace.candidates.length} candidates)</span>
            <span className="text-[10px] normal-case font-normal text-fg-dim">
              full graph in MCTS Replay tab
            </span>
          </div>
          <MiniMCTSGraph trace={trace} />
        </div>
      )}

      {precedents.length > 0 && (
        <div>
          <div className="text-[11px] uppercase tracking-wider text-fg-dim mb-2 flex items-center justify-between">
            <span>Retrieved precedents ({precedents.length})</span>
            <span className="text-[10px] normal-case font-normal text-fg-dim">
              used:
              {trace.precedents_used_expand && <span className="ml-1 chip chip-accent !py-0">expand</span>}
              {trace.precedents_used_score && <span className="ml-1 chip chip-accent !py-0">score</span>}
              {trace.precedents_used_response && <span className="ml-1 chip chip-accent !py-0">response</span>}
              {!(trace.precedents_used_expand || trace.precedents_used_score || trace.precedents_used_response) && <span className="ml-1 text-fg-dim">none</span>}
            </span>
          </div>
          <div className="space-y-1.5">
            {precedents.map((p) => {
              const matchesChosen = p.action === trace.chosen_action;
              return (
                <div key={p.id} className={`rounded-md border ${matchesChosen ? 'border-accent/50 bg-accent/5' : 'border-bg-border bg-bg-elevated/50'} p-2 text-[11px]`}>
                  <div className="flex items-center justify-between">
                    <span className="text-fg"><b>{p.action}</b> <span className="text-fg-dim">· {p.cohort}</span></span>
                    <span className="font-mono text-fg-dim">sim {p.similarity.toFixed(2)}</span>
                  </div>
                  <div className="text-fg-dim mt-0.5">
                    {p.immediate_state && <>→ <span className="text-fg">{p.immediate_state}</span> </>}
                    {p.terminal_outcome && <>· terminal <span className={p.terminal_outcome==='success' ? 'text-accent-ok' : p.terminal_outcome==='failure' ? 'text-accent-err' : ''}>{p.terminal_outcome}</span></>}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {trace.state_rationale && (
        <div className="text-[12px] text-fg-muted italic border-l-2 border-accent/40 pl-2">
          {trace.state_rationale}
        </div>
      )}

      <div>
        <div className="text-[11px] uppercase tracking-wider text-fg-dim mb-2">Candidates</div>
        <div className="space-y-1.5">
          {trace.candidates.map((c) => (
            <motion.div
              key={c.action}
              initial={{ opacity: 0, x: -4 }} animate={{ opacity: 1, x: 0 }}
              className={`rounded-md border ${c.action === trace.chosen_action ? 'border-accent/60 bg-accent/10' : 'border-bg-border bg-bg-elevated/60'} p-2.5`}
            >
              <div className="flex items-center justify-between text-[12px]">
                <div className="font-medium">{c.action}</div>
                <div className="font-mono text-fg-muted text-[11px]">
                  Q {c.q_value.toFixed(3)} · v {c.visits}
                </div>
              </div>
              <div className="mt-1.5 h-1.5 w-full rounded-full bg-bg-elevated overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.max(2, (c.q_value / maxQ) * 100)}%` }}
                  transition={{ duration: 0.5, ease: 'easeOut' }}
                  className={`h-full ${c.action === trace.chosen_action ? 'bg-accent' : 'bg-accent/40'}`}
                />
              </div>
              {c.rationale && <div className="text-[11px] text-fg-dim mt-1.5">{c.rationale}</div>}
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}

function Stat({ icon, label, value, accent }: { icon: React.ReactNode; label: string; value: string; accent?: boolean }) {
  return (
    <div className="rounded-md border border-bg-border bg-bg-elevated/60 px-2.5 py-2">
      <div className="text-[10px] uppercase tracking-wider text-fg-dim flex items-center gap-1">
        {icon} {label}
      </div>
      <div className={`mt-0.5 text-[13px] font-medium ${accent ? 'text-accent' : 'text-fg'}`}>{value}</div>
    </div>
  );
}

function SetupBar(props: {
  seeds: { file: string; name: string }[];
  saved: SOPMeta[];
  sopId: string;
  setSopId: (s: string) => void;
  plannerMode: PlannerMode;
  setPlannerMode: (m: PlannerMode) => void;
  chatMode: ChatMode;
  setChatMode: (m: ChatMode) => void;
  mcts: MCTSConfig;
  setMcts: (c: MCTSConfig) => void;
  onStart: () => void;
  onEnd?: () => void;
  running: boolean;
  busy: boolean;
}) {
  const { seeds, saved, sopId, setSopId, plannerMode, setPlannerMode, chatMode, setChatMode, mcts, setMcts, onStart, running, busy } = props;
  const [openMcts, setOpenMcts] = useState(false);
  const btnRef = useRef<HTMLButtonElement>(null);
  // Compute popover position relative to the trigger button. We portal the popover to
  // document.body so it escapes the chat column's overflow-hidden clip; that means we
  // need to position it ourselves with fixed coords.
  const [popPos, setPopPos] = useState<{ top: number; left: number } | null>(null);
  useEffect(() => {
    if (!openMcts) return;
    const update = () => {
      const r = btnRef.current?.getBoundingClientRect();
      if (!r) return;
      const popW = 640;
      const margin = 8;
      // Prefer aligning popover's left to button's left; clamp to viewport.
      const desiredLeft = r.left;
      const maxLeft = window.innerWidth - popW - margin;
      const left = Math.max(margin, Math.min(desiredLeft, maxLeft));
      setPopPos({ top: r.bottom + 4, left });
    };
    update();
    window.addEventListener('resize', update);
    window.addEventListener('scroll', update, true);
    return () => {
      window.removeEventListener('resize', update);
      window.removeEventListener('scroll', update, true);
    };
  }, [openMcts]);
  return (
    <div className="card px-3 py-2.5 flex flex-wrap items-center gap-2">
      <select
        value={sopId}
        onChange={(e) => setSopId(e.target.value)}
        className="input max-w-[260px] py-1.5"
      >
        <option value="" disabled>Select SOP…</option>
        {seeds.length > 0 && <optgroup label="Seeds">
          {seeds.map((s) => <option key={s.file} value={`seed:${s.file}`}>{s.name}</option>)}
        </optgroup>}
        {saved.length > 0 && <optgroup label="Saved">
          {saved.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
        </optgroup>}
      </select>

      <Toggle
        options={[{ value: 'mcts', label: 'PCA-M' }, { value: 'baseline', label: 'Baseline' }]}
        value={plannerMode}
        onChange={(v) => setPlannerMode(v as PlannerMode)}
      />
      <Toggle
        options={[{ value: 'human', label: 'Human' }, { value: 'auto', label: 'Auto-sim' }]}
        value={chatMode}
        onChange={(v) => setChatMode(v as ChatMode)}
      />

      <div className="relative">
        <button ref={btnRef} className="btn btn-ghost text-xs" onClick={() => setOpenMcts((o) => !o)} disabled={plannerMode !== 'mcts'}>
          <Sliders size={12} /> MCTS settings
        </button>
        {openMcts && popPos && createPortal(
          <motion.div initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }}
            style={{ position: 'fixed', top: popPos.top, left: popPos.left, width: 640, maxHeight: '80vh' }}
            className="z-[1000] overflow-y-auto card p-3 space-y-3 shadow-2xl">
            {/* Full-width: Preset row */}
            <div>
              <div className="text-[10px] uppercase tracking-wider text-fg-dim mb-1.5">Preset</div>
              <div className="grid grid-cols-3 gap-1.5">
                <button className="btn btn-ghost text-xs justify-center"
                  onClick={() => setMcts({ iterations: 4, branching: 2, rollout_depth: 2, c_uct: 1.4, parallel_rollouts: 4, nee_threshold: 0.15, nee_min_visits: 2, top_k_precedents: 3, use_precedents_expand: true, use_precedents_score: false, use_precedents_response: true, pondering_enabled: true, pondering_k: 2, rollout_mode: "simulate", router_enabled: true, tier_entropy_max_t1: 0.4, tier_entropy_max_t2: 1.2, tier_min_supporting_traces: 3, planning_granularity: "action" })}>
                  Fast
                </button>
                <button className="btn btn-ghost text-xs justify-center"
                  onClick={() => setMcts({ iterations: 8, branching: 3, rollout_depth: 3, c_uct: 1.4, parallel_rollouts: 4, nee_threshold: 0.15, nee_min_visits: 2, top_k_precedents: 3, use_precedents_expand: true, use_precedents_score: false, use_precedents_response: true, pondering_enabled: true, pondering_k: 2, rollout_mode: "simulate", router_enabled: true, tier_entropy_max_t1: 0.4, tier_entropy_max_t2: 1.2, tier_min_supporting_traces: 3, planning_granularity: "action" })}>
                  Balanced
                </button>
                <button className="btn btn-ghost text-xs justify-center"
                  onClick={() => setMcts({ iterations: 16, branching: 4, rollout_depth: 4, c_uct: 1.4, parallel_rollouts: 8, nee_threshold: 0.15, nee_min_visits: 2, top_k_precedents: 3, use_precedents_expand: true, use_precedents_score: false, use_precedents_response: true, pondering_enabled: true, pondering_k: 2, rollout_mode: "simulate", router_enabled: true, tier_entropy_max_t1: 0.4, tier_entropy_max_t2: 1.2, tier_min_supporting_traces: 3, planning_granularity: "action" })}>
                  Thorough
                </button>
              </div>
            </div>

            <div className="h-px bg-bg-border" />

            {/* Two columns: core search params (left) vs components (right) */}
            <div className="grid grid-cols-2 gap-4">
              {/* LEFT: Core MCTS search parameters */}
              <div className="space-y-3">
                <div className="text-[10px] uppercase tracking-wider text-fg-dim">Search</div>
                <NumberRow label="Iterations" value={mcts.iterations} min={1} max={32} onChange={(v) => setMcts({ ...mcts, iterations: v })} />
                <NumberRow label="Branching (d)" value={mcts.branching} min={1} max={6} onChange={(v) => setMcts({ ...mcts, branching: v })} />
                <NumberRow label="Rollout depth (k)" value={mcts.rollout_depth} min={0} max={6} onChange={(v) => setMcts({ ...mcts, rollout_depth: v })} />
                <NumberRow label="Parallel rollouts" value={mcts.parallel_rollouts} min={1} max={16} onChange={(v) => setMcts({ ...mcts, parallel_rollouts: v })} />
                <NumberRow label="c (UCT)" value={mcts.c_uct} min={0.1} max={3} step={0.1} onChange={(v) => setMcts({ ...mcts, c_uct: v })} />

                <div className="h-px bg-bg-border" />
                <div className="text-[10px] uppercase tracking-wider text-fg-dim">Negative early exit</div>
                <NumberRow label="NEE threshold (0 = off)" value={mcts.nee_threshold} min={0} max={1} step={0.01} onChange={(v) => setMcts({ ...mcts, nee_threshold: v })} />
                <NumberRow label="NEE min visits" value={mcts.nee_min_visits} min={1} max={8} onChange={(v) => setMcts({ ...mcts, nee_min_visits: v })} />

                <div className="h-px bg-bg-border" />
                <div className="text-[10px] uppercase tracking-wider text-fg-dim">Planning granularity</div>
                <div className="grid grid-cols-2 gap-1.5">
                  {(['action', 'strategy'] as const).map((g) => (
                    <button
                      key={g}
                      className={`btn text-xs justify-center ${mcts.planning_granularity === g ? 'btn-primary' : 'btn-ghost'}`}
                      onClick={() => setMcts({ ...mcts, planning_granularity: g })}
                      title={g === 'action'
                        ? 'MCTS searches over individual agent actions'
                        : 'MCTS searches over Strategy groups; each strategy expands to a concrete action via SOP filtering'}
                    >
                      {g === 'action' ? 'Action' : 'Strategy'}
                    </button>
                  ))}
                </div>

                <div className="text-[10px] uppercase tracking-wider text-fg-dim">Rollout mode</div>
                <div className="grid grid-cols-3 gap-1.5">
                  {(['simulate', 'value', 'hybrid'] as const).map((m) => (
                    <button
                      key={m}
                      className={`btn text-xs justify-center ${mcts.rollout_mode === m ? 'btn-primary' : 'btn-ghost'}`}
                      onClick={() => setMcts({ ...mcts, rollout_mode: m })}
                      title={
                        m === 'simulate' ? 'Per-step user simulation + end-rationality (~7 LLM calls/rollout at depth=3)'
                        : m === 'value' ? 'Single value-scoring call per rollout (~1 LLM call/rollout)'
                        : 'Simulate first step, value-score the tail (~3 LLM calls/rollout)'
                      }
                    >
                      {m === 'simulate' ? 'Simulate' : m === 'value' ? 'Value' : 'Hybrid'}
                    </button>
                  ))}
                </div>
              </div>

              {/* RIGHT: Async/component subsystems — pondering, router, prefetch */}
              <div className="space-y-3">
                <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-fg-dim">
                  <span>Pondering (speculative MCTS)</span>
                  <label className="flex items-center gap-1.5 normal-case">
                    <input
                      type="checkbox"
                      checked={mcts.pondering_enabled}
                      onChange={(e) => setMcts({ ...mcts, pondering_enabled: e.target.checked })}
                    />
                    <span className="text-fg-muted">enabled</span>
                  </label>
                </div>
                <NumberRow label="Top-K next states" value={mcts.pondering_k} min={1} max={6} onChange={(v) => setMcts({ ...mcts, pondering_k: v })} />

                <div className="h-px bg-bg-border" />
                <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-fg-dim">
                  <span>Multi-tier router</span>
                  <label className="flex items-center gap-1.5 normal-case">
                    <input
                      type="checkbox"
                      checked={mcts.router_enabled}
                      onChange={(e) => setMcts({ ...mcts, router_enabled: e.target.checked })}
                    />
                    <span className="text-fg-muted">enabled</span>
                  </label>
                </div>
                <NumberRow label="Tier-1 entropy max (cached)" value={mcts.tier_entropy_max_t1} min={0} max={2} step={0.05} onChange={(v) => setMcts({ ...mcts, tier_entropy_max_t1: v })} />
                <NumberRow label="Tier-2 entropy max (baseline)" value={mcts.tier_entropy_max_t2} min={0} max={3} step={0.05} onChange={(v) => setMcts({ ...mcts, tier_entropy_max_t2: v })} />
                <NumberRow label="Min supporting traces" value={mcts.tier_min_supporting_traces} min={1} max={20} onChange={(v) => setMcts({ ...mcts, tier_min_supporting_traces: v })} />

                <div className="h-px bg-bg-border" />
                <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-fg-dim">
                  <span>Speculative data-prefetch</span>
                  <label className="flex items-center gap-1.5 normal-case">
                    <input
                      type="checkbox"
                      checked={mcts.data_prefetch_enabled ?? true}
                      onChange={(e) => setMcts({ ...mcts, data_prefetch_enabled: e.target.checked })}
                    />
                    <span className="text-fg-muted">enabled</span>
                  </label>
                </div>
                <div className="text-[10px] uppercase tracking-wider text-fg-dim">Predictor</div>
                <div className="grid grid-cols-4 gap-1.5">
                  {(['auto', 'mcts', 'empirical', 'union'] as const).map((p) => (
                    <button
                      key={p}
                      className={`btn text-xs justify-center ${(mcts.data_prefetch_predictor ?? 'auto') === p ? 'btn-primary' : 'btn-ghost'}`}
                      onClick={() => setMcts({ ...mcts, data_prefetch_predictor: p })}
                      title={
                        p === 'auto' ? 'MCTS rollouts when deep, else empirical-from-precedents'
                        : p === 'mcts' ? 'In-memory rollouts only (needs simulate/hybrid)'
                        : p === 'empirical' ? 'Empirical transitions from precedent_traces (works under value-mode)'
                        : 'Run both predictors and merge — source-tagged for analysis'
                      }
                    >
                      {p}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>,
          document.body,
        )}
      </div>

      <div className="grow" />

      {running && props.onEnd && (
        <button className="btn" onClick={props.onEnd} disabled={busy} title="End session and back-propagate terminal outcome">
          End
        </button>
      )}
      <button className="btn btn-primary" onClick={onStart} disabled={!sopId || busy}>
        <Play size={12} /> {running ? 'Restart' : 'Start'}
      </button>
    </div>
  );
}

function Toggle<T extends string>({
  options, value, onChange,
}: { options: { value: T; label: string }[]; value: T; onChange: (v: T) => void }) {
  return (
    <div className="flex items-center rounded-md border border-bg-border bg-bg-elevated p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          onClick={() => onChange(o.value)}
          className={`relative px-2.5 py-1 text-xs rounded ${value === o.value ? 'text-white' : 'text-fg-muted hover:text-fg'}`}
        >
          {value === o.value && <motion.div layoutId={`tog-${options[0].value}`} className="absolute inset-0 rounded bg-accent/20 border border-accent/40" />}
          <span className="relative">{o.label}</span>
        </button>
      ))}
    </div>
  );
}

function GraphModeToggle({ mode, setMode }: { mode: 'sop' | 'mcts'; setMode: (m: 'sop' | 'mcts') => void }) {
  return (
    <div className="flex items-center rounded-md border border-bg-border bg-bg-elevated p-0.5">
      <button
        onClick={() => setMode('sop')}
        className={`relative px-2.5 py-1 text-xs rounded ${mode === 'sop' ? 'text-white' : 'text-fg-muted hover:text-fg'}`}
      >
        {mode === 'sop' && <motion.div layoutId="gmode" className="absolute inset-0 rounded bg-accent/20 border border-accent/40" />}
        <span className="relative flex items-center gap-1.5"><Network size={12} /> SOP</span>
      </button>
      <button
        onClick={() => setMode('mcts')}
        className={`relative px-2.5 py-1 text-xs rounded ${mode === 'mcts' ? 'text-white' : 'text-fg-muted hover:text-fg'}`}
      >
        {mode === 'mcts' && <motion.div layoutId="gmode" className="absolute inset-0 rounded bg-accent/20 border border-accent/40" />}
        <span className="relative flex items-center gap-1.5"><GitBranchPlus size={12} /> MCTS</span>
      </button>
    </div>
  );
}

function NumberRow({ label, value, min, max, step = 1, onChange }: {
  label: string; value: number; min: number; max: number; step?: number; onChange: (v: number) => void;
}) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-[11px]"><span className="text-fg-dim">{label}</span><span className="font-mono">{value}</span></div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-violet-500"
      />
    </div>
  );
}
