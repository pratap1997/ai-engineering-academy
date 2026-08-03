import { useState, useMemo } from 'react';
import { Header } from './components/Header';
import { ModuleCard } from './components/ModuleCard';
import { ModuleModal } from './components/ModuleModal';
import { DiagramModal } from './components/DiagramModal';
import { RoadmapView } from './components/RoadmapView';
import { AITutorModal } from './components/AITutorModal';
import { PHASES, MODULES } from './modulesData';
import type { ModuleData, PhaseId } from './types';
import { Sparkles } from 'lucide-react';

const GithubIcon = ({ className = "w-4 h-4" }: { className?: string }) => (
  <svg className={className} fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
    <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
  </svg>
);

export function App() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedPhase, setSelectedPhase] = useState<PhaseId | 'all'>('all');
  const [viewMode, setViewMode] = useState<'grid' | 'roadmap'>('grid');
  
  const [activeModule, setActiveModule] = useState<ModuleData | null>(null);
  const [activeDiagramModule, setActiveDiagramModule] = useState<ModuleData | null>(null);

  // Filter modules based on search query and phase filter
  const filteredModules = useMemo(() => {
    return MODULES.filter((m) => {
      const matchesPhase = selectedPhase === 'all' || m.phaseId === selectedPhase;
      const query = searchQuery.toLowerCase().trim();
      const matchesSearch =
        query === '' ||
        m.id.includes(query) ||
        m.title.toLowerCase().includes(query) ||
        m.subtitle.toLowerCase().includes(query) ||
        m.overview.toLowerCase().includes(query) ||
        m.topics.some((t) => t.toLowerCase().includes(query)) ||
        m.mathHighlight.toLowerCase().includes(query);

      return matchesPhase && matchesSearch;
    });
  }, [searchQuery, selectedPhase]);

  return (
    <div className="min-h-screen flex flex-col">
      
      {/* Navigation Header */}
      <Header
        searchQuery={searchQuery}
        setSearchQuery={setSearchQuery}
        selectedPhase={selectedPhase}
        setSelectedPhase={setSelectedPhase}
        viewMode={viewMode}
        setViewMode={setViewMode}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-6 mb-16 space-y-8">
        
        {/* Hero Banner */}
        <section className="glass-panel p-8 md:p-12 relative overflow-hidden bg-gradient-to-br from-slate-900/90 via-slate-900/60 to-cyan-950/30 border-cyan-500/20">
          <div className="max-w-3xl space-y-4">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-xs font-semibold">
              <Sparkles className="w-3.5 h-3.5" /> Permanent AI Capability Curriculum
            </div>
            
            <h1 className="text-3xl md:text-5xl font-extrabold tracking-tight leading-tight">
              Master AI Engineering <br />
              <span className="gradient-text-rainbow">From Zero to Frontier Agents</span>
            </h1>

            <p className="text-sm md:text-base text-slate-300 leading-relaxed">
              50 production-grade curriculum modules built from scratch with zero framework magic. 
              From biological perceptrons and autograd engines to FlashAttention, DeepSeek MLA, MCTS Planning, and Autonomous Coding Engineers.
            </p>

            {/* Quick Stats Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-4">
              <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800">
                <div className="text-2xl font-black text-cyan-400 font-heading">50 / 50</div>
                <div className="text-[11px] text-slate-400 font-medium">Modules Complete</div>
              </div>
              <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800">
                <div className="text-2xl font-black text-emerald-400 font-heading">801 / 801</div>
                <div className="text-[11px] text-slate-400 font-medium">Tests Passing</div>
              </div>
              <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800">
                <div className="text-2xl font-black text-violet-400 font-heading">100%</div>
                <div className="text-[11px] text-slate-400 font-medium">Pure Python/NumPy</div>
              </div>
              <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800">
                <div className="text-2xl font-black text-amber-400 font-heading">650</div>
                <div className="text-[11px] text-slate-400 font-medium">Canonical Artifacts</div>
              </div>
            </div>

            {/* GitHub Repo Button */}
            <div className="pt-2 flex items-center gap-3">
              <a
                href="https://github.com/pratap1997/ai-engineering-academy"
                target="_blank"
                rel="noreferrer"
                className="btn-primary text-xs"
              >
                <GithubIcon className="w-4 h-4" /> View GitHub Repository
              </a>
            </div>
          </div>
        </section>

        {/* Search Results Summary */}
        <div className="flex items-center justify-between text-xs text-slate-400 px-1">
          <div>
            Showing <span className="font-bold text-slate-200">{filteredModules.length}</span> of {MODULES.length} modules
          </div>
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="text-cyan-400 hover:underline"
            >
              Clear search filter
            </button>
          )}
        </div>

        {/* Render Grid or Roadmap View */}
        {viewMode === 'grid' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredModules.map((module) => (
              <ModuleCard
                key={module.id}
                module={module}
                onOpenDetail={(m) => setActiveModule(m)}
                onOpenDiagram={(m) => setActiveDiagramModule(m)}
              />
            ))}
          </div>
        ) : (
          <RoadmapView
            phases={PHASES}
            modules={filteredModules}
            onOpenDetail={(m) => setActiveModule(m)}
          />
        )}

      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950 py-8 px-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div>
            © 2026 <span className="text-slate-300 font-semibold">AI Engineering Academy</span> • Built for Permanent Capability.
          </div>
          <div className="flex items-center gap-4">
              <a
                href="https://github.com/pratap1997/ai-engineering-academy"
                target="_blank"
                rel="noreferrer"
                className="text-slate-400 hover:text-cyan-400 transition-colors flex items-center gap-1"
              >
                <GithubIcon className="w-3.5 h-3.5" /> GitHub
              </a>
          </div>
        </div>
      </footer>

      {/* Modals & Drawers */}
      <ModuleModal module={activeModule} onClose={() => setActiveModule(null)} />
      <DiagramModal module={activeDiagramModule} onClose={() => setActiveDiagramModule(null)} />
      <AITutorModal onOpenModule={(m) => setActiveModule(m)} />

    </div>
  );
}

export default App;
