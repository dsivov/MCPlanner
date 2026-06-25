import { useState } from 'react';
import { motion } from 'framer-motion';
import { Settings2, MessageSquareText, Sparkles, GitBranchPlus, Activity, Bot } from 'lucide-react';
import ConfigurationTab from './tabs/ConfigurationTab';
import ChatTab from './tabs/ChatTab';
import ContextGraphTab from './tabs/ContextGraphTab';
import MCTSReplayTab from './tabs/MCTSReplayTab';
import AvatarTab from './tabs/AvatarTab';

type Tab = 'config' | 'chat' | 'avatar' | 'context' | 'replay';

export default function App() {
  const [tab, setTab] = useState<Tab>('config');

  return (
    <div className="min-h-screen w-full flex flex-col">
      <header className="border-b border-bg-border bg-bg-base/80 backdrop-blur sticky top-0 z-30">
        <div className="mx-auto max-w-[1600px] flex items-center justify-between px-6 py-3">
          <div className="flex items-center gap-2.5">
            <div className="h-8 w-8 rounded-lg gradient-accent flex items-center justify-center shadow-glow">
              <Sparkles size={16} className="text-accent" />
            </div>
            <div className="flex flex-col leading-tight">
              <span className="text-sm font-semibold tracking-tight">PCA Planner</span>
              <span className="text-[11px] text-fg-dim">SOP-guided dialogue supervisor · PoC</span>
            </div>
          </div>

          <nav className="flex items-center gap-1 rounded-lg border border-bg-border bg-bg-panel p-1">
            <TabButton active={tab === 'config'} onClick={() => setTab('config')} icon={<Settings2 size={14} />}>
              Configuration
            </TabButton>
            <TabButton active={tab === 'chat'} onClick={() => setTab('chat')} icon={<MessageSquareText size={14} />}>
              Chat
            </TabButton>
            <TabButton active={tab === 'avatar'} onClick={() => setTab('avatar')} icon={<Bot size={14} />}>
              Avatar
            </TabButton>
            <TabButton active={tab === 'context'} onClick={() => setTab('context')} icon={<GitBranchPlus size={14} />}>
              Context Graph
            </TabButton>
            <TabButton active={tab === 'replay'} onClick={() => setTab('replay')} icon={<Activity size={14} />}>
              MCTS Replay
            </TabButton>
          </nav>

          <div className="text-[11px] text-fg-dim font-mono">v0.1.0</div>
        </div>
      </header>

      <main className="flex-1 mx-auto w-full max-w-[1600px] px-6 py-5">
        {/* Keep ALL tabs mounted so per-tab state (chat session, traces, etc.) survives
            switching. Only the active tab is visible. */}
        <div className="h-[calc(100vh-100px)] relative">
          <TabHost active={tab === 'config'}><ConfigurationTab /></TabHost>
          <TabHost active={tab === 'chat'}><ChatTab /></TabHost>
          <TabHost active={tab === 'avatar'}><AvatarTab /></TabHost>
          <TabHost active={tab === 'context'}><ContextGraphTab /></TabHost>
          <TabHost active={tab === 'replay'}><MCTSReplayTab /></TabHost>
        </div>
      </main>
    </div>
  );
}

function TabHost({ active, children }: { active: boolean; children: React.ReactNode }) {
  // Mounted but invisible when inactive — preserves component state. We use `hidden`
  // instead of remount-on-active so chat sessions, MCTS traces, drill-down state etc.
  // survive tab navigation.
  return (
    <motion.div
      initial={false}
      animate={{ opacity: active ? 1 : 0 }}
      transition={{ duration: 0.16 }}
      className="absolute inset-0"
      style={{ pointerEvents: active ? 'auto' : 'none', visibility: active ? 'visible' : 'hidden' }}
    >
      {children}
    </motion.div>
  );
}

function TabButton({
  active, onClick, icon, children,
}: { active: boolean; onClick: () => void; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`relative inline-flex items-center gap-2 rounded-md px-3 py-1.5 text-sm transition-colors
        ${active ? 'text-white' : 'text-fg-muted hover:text-fg'}`}
    >
      {active && (
        <motion.div
          layoutId="tab-active"
          className="absolute inset-0 rounded-md bg-accent/15 border border-accent/40"
          transition={{ type: 'spring', stiffness: 400, damping: 30 }}
        />
      )}
      <span className="relative flex items-center gap-2">
        {icon}
        {children}
      </span>
    </button>
  );
}
