import { useEffect, useRef, useState, useCallback } from 'react';
import { Mic, Square, Loader2, Radio, Database, GitBranchPlus, Gauge, Network } from 'lucide-react';
import { api, type AvatarPlan, type AvatarBlackboard, type MCTSConfig, type TaskDefinition } from '../lib/api';
import SOPGraph from '../components/SOPGraph';

// Avatar (live voice testing): a human tester talks to a GPT-Realtime voice avatar.
// The realtime model is the "weak voice agent"; our supervisor runs server-side under
// FULL SOP-action control — on each user utterance we call /plan-turn, then steer the
// model via session.update(instructions) + response.create so it speaks the SOP-
// constrained, blackboard-data-informed reply. Manual turn control (create_response:false).

const AVATAR_URL = '/avatars/brunette.glb';

// Load a bare-specifier ESM module at runtime via the browser's importmap (index.html),
// bypassing the bundler entirely. We build the dynamic import() inside `new Function` so
// Vite/rolldown's static import-analysis can't see (and try to resolve) the specifier —
// a plain `import(varName)` gets constant-folded back to the literal and fails. The
// browser then resolves "talkinghead" (and its internal `import "three"`) from the CDN.
const cdnImport: (specifier: string) => Promise<any> = new Function(
  'specifier', 'return import(specifier);',
) as any;

// Same prefetch-enabled config the autopilot uses, in human/chat_mode.
const AVATAR_MCTS: MCTSConfig = {
  iterations: 8, branching: 3, rollout_depth: 3, c_uct: 1.4,
  parallel_rollouts: 4, nee_threshold: 0.15, nee_min_visits: 2,
  top_k_precedents: 3, use_precedents_expand: true,
  use_precedents_score: false, use_precedents_response: true,
  pondering_enabled: false, pondering_k: 2,
  rollout_mode: 'simulate',
  router_enabled: true, tier_entropy_max_t1: 0.4,
  tier_entropy_max_t2: 1.2, tier_min_supporting_traces: 3,
  planning_granularity: 'action',
  // prefetch (typed loosely — MCTSConfig allows these extra fields server-side)
  ...( {
    tier3_enabled: false,
    data_prefetch_enabled: true,
    data_prefetch_predictor: 'union',
    data_prefetch_decay_lambda: 0.3,
    data_prefetch_max_outstanding: 20,
    data_prefetch_min_confidence: 0.05,
    data_prefetch_await_in_flight_ms: 2000,
  } as Partial<MCTSConfig> ),
};

// session.update REPLACES the realtime session's instructions, so each turn's
// instruction must carry the full directive framing (persona + goal + the step),
// not just "do X". We frame the chosen SOP action as a proactive move toward the
// objective so the avatar leads rather than merely answers.
function buildInstructions(plan: AvatarPlan): string {
  const role = plan.agent_role || 'a goal-driven outbound voice agent';
  const goal = plan.goal || 'complete the procedure';
  const parts: string[] = [];
  parts.push(
    `You are ${role}. Your objective: ${goal} You LEAD the conversation toward it—` +
    `proactive, not reactive.`,
  );
  parts.push(
    `Right now, take this step and drive the conversation forward with it: ` +
    `${plan.action_description || plan.chosen_action}.`,
  );
  if (plan.must_say.length) parts.push(`Make sure to convey: ${plan.must_say.join('; ')}.`);
  if (plan.must_not_say.length) parts.push(`Do NOT say: ${plan.must_not_say.join('; ')}.`);
  if (plan.data_context.length)
    parts.push(`Use this data verbatim where it fits: ${plan.data_context.map((d) => d.summary).join(' | ')}.`);
  parts.push(
    `Speak 1-2 natural sentences. Don't just answer and wait—proactively introduce the ` +
    `next step or ask the question that moves toward the objective. Never invent facts ` +
    `beyond what you were given.`,
  );
  return parts.join(' ');
}

export default function AvatarTab() {
  const [seeds, setSeeds] = useState<{ file: string; name: string }[]>([]);
  const [sopId, setSopId] = useState('');
  const [sop, setSop] = useState<TaskDefinition | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [avatarStatus, setAvatarStatus] = useState('Loading avatar…');
  const [live, setLive] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [plan, setPlan] = useState<AvatarPlan | null>(null);
  const [blackboard, setBlackboard] = useState<AvatarBlackboard | null>(null);
  const [userSubtitle, setUserSubtitle] = useState('');
  const [avatarSubtitle, setAvatarSubtitle] = useState('');
  const [terminal, setTerminal] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Imperative handles
  const headRef = useRef<any>(null);
  const avatarReadyRef = useRef(false);
  const pcRef = useRef<RTCPeerConnection | null>(null);
  const dcRef = useRef<RTCDataChannel | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const avatarDivRef = useRef<HTMLDivElement | null>(null);
  const prevAssistantRef = useRef<string | null>(null);
  const assistantBufRef = useRef('');
  const sessionIdRef = useRef<string | null>(null);
  useEffect(() => { sessionIdRef.current = sessionId; }, [sessionId]);

  // Lip-sync
  const audioCtxRef = useRef<AudioContext | null>(null);
  const lipRafRef = useRef<number | null>(null);

  useEffect(() => { api.listSeeds().then((s) => { setSeeds(s); if (s.length) setSopId(`seed:${s[0].file}`); }).catch(() => {}); }, []);

  // --- Boot the TalkingHead avatar (CDN ESM via importmap) -----------------
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const mod: any = await cdnImport('talkinghead');
        if (cancelled || !avatarDivRef.current) return;
        const head = new mod.TalkingHead(avatarDivRef.current, {
          lipsyncModules: ['en'], cameraView: 'upper', mixerGainSpeech: 3,
        });
        headRef.current = head;
        setAvatarStatus('Loading avatar model…');
        await head.showAvatar({ url: AVATAR_URL, body: 'F', avatarMood: 'neutral', lipsyncLang: 'en' });
        if (cancelled) return;
        avatarReadyRef.current = true;
        setAvatarStatus('Avatar ready. Start a session, then connect voice.');
      } catch (e: any) {
        setAvatarStatus('Avatar failed to load: ' + (e?.message || e));
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // --- Lip-sync from the model's audio energy ------------------------------
  const setJaw = (v: number) => {
    const head = headRef.current; if (!head) return;
    if (head.mtAvatar?.jawOpen) Object.assign(head.mtAvatar.jawOpen, { realtime: v, needsUpdate: true });
    else head.setFixedValue?.('jawOpen', v);
  };
  const startLipSync = (stream: MediaStream) => {
    const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
    ctx.resume?.();
    const src = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser(); analyser.fftSize = 512;
    src.connect(analyser);
    const buf = new Uint8Array(analyser.fftSize);
    audioCtxRef.current = ctx;
    let jaw = 0;
    const tick = () => {
      analyser.getByteTimeDomainData(buf);
      let sum = 0; for (let i = 0; i < buf.length; i++) { const v = (buf[i] - 128) / 128; sum += v * v; }
      const rms = Math.sqrt(sum / buf.length);
      const target = Math.min(0.7, rms * 5.5);
      jaw += (target - jaw) * (target > jaw ? 0.6 : 0.25);
      setJaw(jaw < 0.02 ? 0 : jaw);
      lipRafRef.current = requestAnimationFrame(tick);
    };
    tick();
  };
  const stopLipSync = () => {
    if (lipRafRef.current) cancelAnimationFrame(lipRafRef.current);
    lipRafRef.current = null;
    if (headRef.current?.mtAvatar?.jawOpen) Object.assign(headRef.current.mtAvatar.jawOpen, { realtime: null, needsUpdate: true });
    audioCtxRef.current?.close(); audioCtxRef.current = null;
  };

  const refreshBlackboard = useCallback(async () => {
    const sid = sessionIdRef.current; if (!sid) return;
    try { setBlackboard(await api.avatarBlackboard(sid)); } catch { /* ignore */ }
  }, []);

  // --- Steering: user finished a turn -> plan -> steer the realtime model ---
  const steer = useCallback(async (userText: string) => {
    const sid = sessionIdRef.current, dc = dcRef.current;
    if (!sid || !dc || !userText.trim()) return;
    try {
      const p = await api.avatarPlanTurn(sid, userText.trim(), prevAssistantRef.current);
      setPlan(p);
      if (p.terminal_outcome) { setTerminal(p.terminal_outcome); }
      const instr = buildInstructions(p);
      dc.send(JSON.stringify({ type: 'session.update', session: { instructions: instr } }));
      dc.send(JSON.stringify({ type: 'response.create' }));
      refreshBlackboard();
    } catch (e: any) {
      setError('plan-turn failed: ' + (e?.message || e));
    }
  }, [refreshBlackboard]);

  // --- Realtime data-channel events ---------------------------------------
  const onServerEvent = useCallback((evt: any) => {
    switch (evt.type) {
      case 'conversation.item.input_audio_transcription.completed': {
        const t = (evt.transcript || '').trim();
        if (t) { setUserSubtitle(t); steer(t); }
        break;
      }
      case 'response.audio_transcript.delta':
      case 'response.output_audio_transcript.delta':
        assistantBufRef.current += evt.delta || '';
        setAvatarSubtitle(assistantBufRef.current);
        break;
      case 'response.audio_transcript.done':
      case 'response.output_audio_transcript.done':
        prevAssistantRef.current = assistantBufRef.current.trim();
        break;
      case 'response.created':
        assistantBufRef.current = '';
        break;
      case 'error':
        setError('Realtime error: ' + (evt.error?.message || JSON.stringify(evt.error || evt)));
        break;
    }
  }, [steer]);

  async function startSession() {
    setError(null); setTerminal(null); setPlan(null); setBlackboard(null);
    prevAssistantRef.current = null;
    try {
      const r = await api.startChat(sopId, 'mcts', 'human', AVATAR_MCTS);
      setSessionId(r.session_id);
      setSop(r.sop);
      setAvatarStatus('Session started. Click “Connect voice” and start talking.');
    } catch (e: any) {
      setError('start session failed: ' + (e?.message || e));
    }
  }

  async function connect() {
    if (!sessionId) { setError('Start a session first.'); return; }
    setConnecting(true); setError(null);
    try {
      const sess = await api.mintRealtimeSession(sessionId);
      if (!sess.value) throw new Error('no realtime token');
      const pc = new RTCPeerConnection();
      pcRef.current = pc;
      const mic = await navigator.mediaDevices.getUserMedia({ audio: true });
      pc.addTrack(mic.getTracks()[0]);
      pc.ontrack = (e) => { if (audioRef.current) audioRef.current.srcObject = e.streams[0]; startLipSync(e.streams[0]); };
      const dc = pc.createDataChannel('oai-events');
      dcRef.current = dc;
      dc.addEventListener('message', (e) => { try { onServerEvent(JSON.parse(e.data)); } catch { /* ignore */ } });
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      const answer = await fetch(`https://api.openai.com/v1/realtime/calls?model=${encodeURIComponent(sess.model)}`, {
        method: 'POST', body: offer.sdp,
        headers: { Authorization: `Bearer ${sess.value}`, 'Content-Type': 'application/sdp' },
      });
      await pc.setRemoteDescription({ type: 'answer', sdp: await answer.text() });
      setLive(true);
      setAvatarStatus('Live — just talk. The avatar is under SOP control.');
    } catch (e: any) {
      setError('connect failed: ' + (e?.message || e));
    } finally {
      setConnecting(false);
    }
  }

  function disconnect() {
    setLive(false);
    stopLipSync();
    pcRef.current?.close(); pcRef.current = null; dcRef.current = null;
    if (audioRef.current) audioRef.current.srcObject = null;
    headRef.current?.setMood?.('neutral');
    setAvatarStatus('Disconnected. Connect voice to talk again.');
  }

  useEffect(() => () => { try { disconnect(); } catch { /* ignore */ } }, []);

  const trace = plan?.trace;
  const candidateActions: string[] = (trace?.candidates ?? []).map((c: any) => c.action);

  return (
    <div className="h-full flex gap-4">
      {/* LEFT: avatar + voice controls */}
      <div className="flex flex-col gap-3 w-[34%] min-w-[380px]">
        <div className="rounded-xl border border-bg-border bg-bg-panel overflow-hidden relative flex-1 min-h-[340px]">
          <div ref={avatarDivRef} className="absolute inset-0" />
          <audio ref={audioRef} autoPlay className="hidden" />
          {/* subtitles overlay */}
          <div className="absolute bottom-0 inset-x-0 p-3 bg-gradient-to-t from-black/70 to-transparent space-y-1">
            {userSubtitle && <div className="text-[12px] text-accent/90">🧑 {userSubtitle}</div>}
            {avatarSubtitle && <div className="text-[13px] text-fg-base">🤖 {avatarSubtitle}</div>}
          </div>
          {terminal && (
            <div className="absolute top-2 right-2 text-[11px] px-2 py-1 rounded bg-ok/20 text-ok border border-ok/40">
              session ended · {terminal}
            </div>
          )}
        </div>

        <div className="rounded-xl border border-bg-border bg-bg-panel p-3 space-y-2.5">
          <div className="flex items-center gap-2">
            <select value={sopId} onChange={(e) => setSopId(e.target.value)} disabled={!!sessionId}
              className="flex-1 text-[13px] bg-bg-base border border-bg-border rounded-md px-2 py-1.5">
              {seeds.map((s) => <option key={s.file} value={`seed:${s.file}`}>{s.name}</option>)}
            </select>
            {!sessionId ? (
              <button onClick={startSession} className="text-[12px] px-3 py-1.5 rounded-md bg-accent/15 text-accent border border-accent/40 hover:bg-accent/25">
                Start session
              </button>
            ) : !live ? (
              <button onClick={connect} disabled={connecting}
                className="flex items-center gap-1.5 text-[12px] px-3 py-1.5 rounded-md bg-ok/15 text-ok border border-ok/40 hover:bg-ok/25">
                {connecting ? <Loader2 size={13} className="animate-spin" /> : <Mic size={13} />} Connect voice
              </button>
            ) : (
              <button onClick={disconnect}
                className="flex items-center gap-1.5 text-[12px] px-3 py-1.5 rounded-md bg-bad/15 text-bad border border-bad/40 hover:bg-bad/25">
                <Square size={13} /> End
              </button>
            )}
          </div>
          <div className="text-[11px] text-fg-dim flex items-center gap-1.5">
            <Radio size={11} className={live ? 'text-ok' : 'text-fg-dim'} /> {avatarStatus}
          </div>
          {error && <div className="text-[11px] text-bad">{error}</div>}
        </div>
      </div>

      {/* MIDDLE: live SOP graph (current node + considered candidates) */}
      <div className="flex flex-col gap-3 w-[34%] min-w-[340px]">
        <div className="rounded-xl border border-bg-border bg-bg-panel p-3 flex-1 flex flex-col min-h-0">
          <div className="flex items-center gap-1.5 text-[11px] font-semibold text-fg-dim mb-2">
            <Network size={12} /> SOP GRAPH {plan ? `· at ${plan.chosen_action}` : ''}
          </div>
          <div className="flex-1 min-h-0 rounded-lg overflow-hidden border border-bg-border">
            {sop ? (
              <SOPGraph sop={sop} currentNode={plan?.chosen_action} proposedNodes={candidateActions} className="h-full" />
            ) : (
              <div className="h-full flex items-center justify-center text-[12px] text-fg-dim">Start a session to load the SOP graph.</div>
            )}
          </div>
        </div>
        {/* Candidates considered this turn */}
        <div className="rounded-xl border border-bg-border bg-bg-panel p-3">
          <div className="text-[11px] font-semibold text-fg-dim mb-1.5">CANDIDATES (this turn)</div>
          {candidateActions.length ? (
            <div className="flex flex-wrap gap-1.5">
              {(trace?.candidates ?? []).map((c: any, i: number) => (
                <span key={i} className={`text-[11px] px-2 py-1 rounded-md border ${i === 0 ? 'border-accent/50 bg-accent/10 text-accent' : 'border-bg-border bg-bg-base text-fg-dim'}`}
                  title={c.rationale || ''}>
                  {c.action}{i === 0 ? ' ✓' : ''}
                </span>
              ))}
            </div>
          ) : <div className="text-[12px] text-fg-dim">—</div>}
        </div>
      </div>

      {/* RIGHT: live supervisor panels */}
      <div className="flex-1 grid grid-rows-[auto_auto_1fr] gap-3 min-w-0">
        {/* Plan / classification */}
        <div className="rounded-xl border border-bg-border bg-bg-panel p-3">
          <div className="flex items-center gap-1.5 text-[11px] font-semibold text-fg-dim mb-2">
            <GitBranchPlus size={12} /> SUPERVISOR PLAN (this turn)
          </div>
          {plan ? (
            <div className="grid grid-cols-2 gap-2 text-[12px]">
              <Field label="Chosen action" value={plan.chosen_action} accent />
              <Field label="User state" value={plan.user_state} />
              <Field label="Cohort" value={plan.cohort || '—'} />
              <Field label="Mood" value={plan.mood || '—'} />
              <div className="col-span-2 text-[11px] text-fg-dim">{plan.action_description}</div>
              {plan.must_say.length > 0 && (
                <div className="col-span-2 text-[11px]"><span className="text-fg-dim">must say:</span> {plan.must_say.join('; ')}</div>
              )}
            </div>
          ) : <div className="text-[12px] text-fg-dim">Talk to the avatar — the supervisor's plan appears here each turn.</div>}
        </div>

        {/* Metrics */}
        <div className="rounded-xl border border-bg-border bg-bg-panel p-3">
          <div className="flex items-center gap-1.5 text-[11px] font-semibold text-fg-dim mb-2"><Gauge size={12} /> BENCHMARK</div>
          <div className="grid grid-cols-4 gap-2">
            <Metric label="classify ms" value={plan?.classify_ms ?? '—'} />
            <Metric label="rerank ms" value={plan?.pool_rerank_ms ?? '—'} />
            <Metric label="pool size" value={plan?.pool_size ?? '—'} />
            <Metric label="ctx items" value={plan?.data_context.length ?? '—'} />
            <Metric label="consumed" value={plan?.prefetch_consumed ?? '—'} />
            <Metric label="hidden ms" value={plan?.prefetch_latency_hidden_ms ?? '—'} />
            <Metric label="tier" value={(trace as any)?.tier_used ?? '—'} />
            <Metric label="turn" value={plan?.turn_index ?? '—'} />
          </div>
        </div>

        {/* Blackboard */}
        <div className="rounded-xl border border-bg-border bg-bg-panel p-3 overflow-auto">
          <div className="flex items-center gap-1.5 text-[11px] font-semibold text-fg-dim mb-2">
            <Database size={12} /> BLACKBOARD POOL {blackboard ? `(${blackboard.pool_size})` : ''}
          </div>
          <div className="space-y-1.5">
            {blackboard?.items.length ? blackboard.items.map((it, i) => {
              const picked = plan?.data_context.some((d) => d.dependency_name === it.dependency_name && d.source_action === it.source_action);
              return (
                <div key={i} className={`text-[11px] rounded-md px-2 py-1.5 border ${picked ? 'border-accent/50 bg-accent/10' : 'border-bg-border bg-bg-base'}`}>
                  <div className="flex items-center gap-1.5">
                    <span className={`px-1 rounded text-[9px] uppercase ${it.kind === 'instruction' ? 'bg-supervisor/20 text-supervisor' : 'bg-blackboard/20 text-blackboard'}`}>{it.kind}</span>
                    <span className="font-semibold text-fg-base">{it.dependency_name}</span>
                    <span className="text-fg-dim">· {it.source_action}</span>
                    {picked && <span className="ml-auto text-[9px] text-accent">PICKED</span>}
                  </div>
                  <div className="text-fg-dim mt-0.5 truncate">{it.summary}</div>
                </div>
              );
            }) : <div className="text-[12px] text-fg-dim">Empty — fills as the supervisor prefetches.</div>}
          </div>
        </div>
      </div>
    </div>
  );
}

function Field({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="rounded-md bg-bg-base border border-bg-border px-2 py-1">
      <div className="text-[10px] text-fg-dim">{label}</div>
      <div className={`text-[12px] font-medium ${accent ? 'text-accent' : 'text-fg-base'}`}>{value}</div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-md bg-bg-base border border-bg-border px-2 py-1 text-center">
      <div className="text-[13px] font-semibold tabular-nums text-fg-base">{value}</div>
      <div className="text-[9px] text-fg-dim">{label}</div>
    </div>
  );
}
