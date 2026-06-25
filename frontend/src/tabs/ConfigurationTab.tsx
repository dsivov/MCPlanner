import { useEffect, useMemo, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Send, Save, Copy, Sparkles, FileJson, Network, Loader2, FolderOpen, FilePlus, AlertTriangle, CheckCircle2,
} from 'lucide-react';
import { api, emptySop, type TaskDefinition, type SOPMeta } from '../lib/api';
import SOPGraph from '../components/SOPGraph';

type Msg = { role: 'user' | 'assistant'; content: string };

export default function ConfigurationTab() {
  const [sop, setSop] = useState<TaskDefinition>(emptySop());
  const [loadedId, setLoadedId] = useState<string | null>(null);   // tracks "Update" target
  const [loadedSource, setLoadedSource] = useState<string>('');     // "seed:xyz" or sop id, for display
  const [history, setHistory] = useState<Msg[]>([
    {
      role: 'assistant',
      content:
        "Hi — I'll help you build a Standard Operating Procedure (SOP) for a conversational agent. Tell me: what task should the agent handle, who is the user, and what is success?",
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [view, setView] = useState<'graph' | 'json'>('graph');
  const [saved, setSaved] = useState<SOPMeta[]>([]);
  const [seeds, setSeeds] = useState<{ file: string; name: string }[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Raw JSON editor state. Only synced from `sop` when not actively editing.
  const [jsonDraft, setJsonDraft] = useState<string>('');
  const [jsonDirty, setJsonDirty] = useState<boolean>(false);
  const [jsonError, setJsonError] = useState<string | null>(null);

  const jsonText = useMemo(() => JSON.stringify(sop, null, 2), [sop]);
  useEffect(() => {
    if (!jsonDirty) setJsonDraft(jsonText);
  }, [jsonText, jsonDirty]);

  useEffect(() => {
    api.listSops().then(setSaved).catch(() => {});
    api.listSeeds().then(setSeeds).catch(() => {});
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [history, loading]);

  function flashSuccess(msg: string) {
    setSuccess(msg);
    setTimeout(() => setSuccess(null), 2500);
  }

  async function send() {
    const text = input.trim();
    if (!text || loading) return;
    setInput('');
    setError(null);
    const newHist: Msg[] = [...history, { role: 'user', content: text }];
    setHistory(newHist);
    setLoading(true);
    try {
      const res = await api.buildTurn(newHist, sop);
      setSop(res.updated_sop);
      setJsonDirty(false);
      setHistory([...newHist, { role: 'assistant', content: res.assistant_message }]);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  async function saveAsNew() {
    setError(null);
    try {
      const meta = await api.saveSop(sop);
      setLoadedId(meta.id);
      setLoadedSource(meta.id);
      setSaved(await api.listSops());
      flashSuccess(`Saved as new SOP "${meta.name}".`);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function updateInPlace() {
    if (!loadedId) return;
    setError(null);
    try {
      const meta = await api.updateSop(loadedId, sop);
      setSaved(await api.listSops());
      flashSuccess(`Updated "${meta.name}".`);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function loadSeed(file: string) {
    const t = await api.getSeed(file);
    setSop(t);
    setJsonDirty(false);
    setLoadedId(null);
    setLoadedSource(`seed:${file}`);
    setHistory([{ role: 'assistant', content: `Loaded seed "${t.name}". Refine it via chat or edit JSON directly. Save creates a copy under your saved SOPs.` }]);
  }

  async function loadSaved(id: string) {
    const t = await api.getSop(id);
    setSop(t);
    setJsonDirty(false);
    setLoadedId(id);
    setLoadedSource(id);
    setHistory([{ role: 'assistant', content: `Loaded "${t.name}" for editing. Use Update to save changes in place, or Save as new to create a copy.` }]);
  }

  function newBlank() {
    setSop(emptySop());
    setJsonDirty(false);
    setLoadedId(null);
    setLoadedSource('');
    setHistory([{ role: 'assistant', content: "New SOP. Describe the task, user, goal, and the actions the agent should be able to take." }]);
  }

  function applyJsonDraft() {
    try {
      const parsed = JSON.parse(jsonDraft);
      // Basic sanity: required top-level shape
      if (typeof parsed !== 'object' || parsed === null) throw new Error('JSON must be an object');
      setSop(parsed);
      setJsonDirty(false);
      setJsonError(null);
      flashSuccess('JSON applied to SOP.');
    } catch (e: unknown) {
      setJsonError(e instanceof Error ? e.message : 'Invalid JSON');
    }
  }

  return (
    <div className="h-full grid grid-cols-12 gap-4">
      {/* Left: chat */}
      <div className="col-span-5 card flex flex-col overflow-hidden">
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-bg-border">
          <div className="flex items-center gap-2">
            <Sparkles size={14} className="text-accent" />
            <span className="text-sm font-medium">SOP Builder</span>
            {loadedSource && (
              <span className="chip">{loadedId ? 'editing' : 'from seed'} · {loadedSource.replace(/^seed:/, '')}</span>
            )}
          </div>
          <div className="flex items-center gap-1">
            <button className="btn btn-ghost text-xs" onClick={newBlank}>
              <FilePlus size={12} /> New
            </button>
            <SeedMenu seeds={seeds} onPick={loadSeed} />
            <SavedMenu saved={saved} onPick={loadSaved} />
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
          <AnimatePresence initial={false}>
            {history.map((m, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.18 }}
                className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[85%] rounded-2xl px-3.5 py-2 text-[13px] leading-snug
                    ${m.role === 'user'
                      ? 'bg-accent/15 border border-accent/40 text-white'
                      : 'bg-bg-elevated border border-bg-border text-fg'}`}
                >
                  {m.content}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          {loading && (
            <div className="flex items-center gap-2 text-fg-dim text-xs px-1">
              <Loader2 size={14} className="animate-spin" />
              <span>Thinking…</span>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {error && (
          <div className="text-[11px] text-accent-err px-4 pb-1 flex items-center gap-1">
            <AlertTriangle size={11} /> {error}
          </div>
        )}
        {success && (
          <div className="text-[11px] text-accent-ok px-4 pb-1 flex items-center gap-1">
            <CheckCircle2 size={11} /> {success}
          </div>
        )}

        <div className="border-t border-bg-border p-3 flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), send())}
            placeholder="Describe the task, user, goal, actions, or constraints…"
            className="input"
          />
          <button className="btn btn-primary" onClick={send} disabled={loading || !input.trim()}>
            <Send size={14} />
          </button>
        </div>
      </div>

      {/* Right: graph + json */}
      <div className="col-span-7 card flex flex-col overflow-hidden">
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-bg-border">
          <div className="flex items-center gap-2 text-sm">
            <span className="font-medium text-fg">{sop.name}</span>
            <span className="text-fg-dim">·</span>
            <span className="chip">{sop.agent_actions.length} actions</span>
            <span className="chip">{sop.user_states.length} states</span>
            <span className="chip">{sop.sop.edges.length} edges</span>
          </div>
          <div className="flex items-center gap-1.5">
            <ViewToggle view={view} setView={setView} />
            {loadedId ? (
              <>
                <button className="btn btn-primary" onClick={updateInPlace}>
                  <Save size={13} /> Update
                </button>
                <button className="btn" onClick={saveAsNew} title="Save as a new SOP">
                  <Copy size={13} /> Save as new
                </button>
              </>
            ) : (
              <button className="btn btn-primary" onClick={saveAsNew}
                disabled={!sop.agent_actions.length && !sop.user_states.length}>
                <Save size={13} /> Save
              </button>
            )}
          </div>
        </div>
        <div className="flex-1 overflow-hidden">
          {view === 'graph' ? (
            <SOPGraph sop={sop} className="h-full" />
          ) : (
            <div className="h-full flex flex-col">
              <div className="flex items-center justify-between px-3 py-2 border-b border-bg-border">
                <span className="text-[11px] text-fg-dim">Edit JSON directly. Use Apply to push edits into the SOP.</span>
                <div className="flex items-center gap-1.5">
                  {jsonDirty && <span className="chip chip-warn">unsaved edits</span>}
                  {jsonError && <span className="chip chip-err">{jsonError}</span>}
                  <button className="btn btn-ghost text-xs"
                    onClick={() => { setJsonDraft(jsonText); setJsonDirty(false); setJsonError(null); }}
                    disabled={!jsonDirty}>
                    Reset
                  </button>
                  <button className="btn btn-primary text-xs" onClick={applyJsonDraft} disabled={!jsonDirty}>
                    Apply
                  </button>
                </div>
              </div>
              <textarea
                value={jsonDraft}
                onChange={(e) => { setJsonDraft(e.target.value); setJsonDirty(true); setJsonError(null); }}
                spellCheck={false}
                className="flex-1 w-full bg-bg-base text-fg-muted p-4 text-[11.5px] font-mono outline-none border-0 resize-none"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ViewToggle({ view, setView }: { view: 'graph' | 'json'; setView: (v: 'graph' | 'json') => void }) {
  return (
    <div className="flex items-center rounded-md border border-bg-border bg-bg-elevated p-0.5">
      <button
        onClick={() => setView('graph')}
        className={`relative px-2.5 py-1 text-xs rounded ${view === 'graph' ? 'text-white' : 'text-fg-muted hover:text-fg'}`}
      >
        {view === 'graph' && <motion.div layoutId="view-active" className="absolute inset-0 rounded bg-accent/20 border border-accent/40" />}
        <span className="relative flex items-center gap-1.5"><Network size={12} /> Graph</span>
      </button>
      <button
        onClick={() => setView('json')}
        className={`relative px-2.5 py-1 text-xs rounded ${view === 'json' ? 'text-white' : 'text-fg-muted hover:text-fg'}`}
      >
        {view === 'json' && <motion.div layoutId="view-active" className="absolute inset-0 rounded bg-accent/20 border border-accent/40" />}
        <span className="relative flex items-center gap-1.5"><FileJson size={12} /> JSON</span>
      </button>
    </div>
  );
}

function SeedMenu({ seeds, onPick }: { seeds: { file: string; name: string }[]; onPick: (file: string) => void }) {
  const [open, setOpen] = useState(false);
  if (!seeds.length) return null;
  return (
    <div className="relative">
      <button className="btn btn-ghost text-xs" onClick={() => setOpen((o) => !o)}>
        <Sparkles size={12} /> Seed
      </button>
      {open && (
        <motion.div
          initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }}
          className="absolute right-0 mt-1 z-40 w-64 card p-1">
          {seeds.map((s) => (
            <button
              key={s.file}
              onClick={() => { onPick(s.file); setOpen(false); }}
              className="w-full text-left px-2.5 py-1.5 rounded hover:bg-accent/10 text-xs"
            >
              {s.name}
            </button>
          ))}
        </motion.div>
      )}
    </div>
  );
}

function SavedMenu({ saved, onPick }: { saved: SOPMeta[]; onPick: (id: string) => void }) {
  const [open, setOpen] = useState(false);
  if (!saved.length) return null;
  return (
    <div className="relative">
      <button className="btn btn-ghost text-xs" onClick={() => setOpen((o) => !o)}>
        <FolderOpen size={12} /> Saved ({saved.length})
      </button>
      {open && (
        <motion.div
          initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }}
          className="absolute right-0 mt-1 z-40 w-72 card p-1 max-h-96 overflow-y-auto">
          {saved.map((s) => (
            <button
              key={s.id}
              onClick={() => { onPick(s.id); setOpen(false); }}
              className="w-full text-left px-2.5 py-1.5 rounded hover:bg-accent/10 text-xs flex justify-between items-center"
            >
              <span className="text-fg truncate">{s.name}</span>
              <span className="text-fg-dim ml-2 text-[10px] flex-shrink-0">{new Date(s.updated_at).toLocaleString()}</span>
            </button>
          ))}
        </motion.div>
      )}
    </div>
  );
}
