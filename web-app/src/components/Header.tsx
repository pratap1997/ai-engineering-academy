import React from 'react';
import { Search, CheckCircle2, Cpu, Grid, GitFork } from 'lucide-react';
import type { PhaseId } from '../types';

interface HeaderProps {
  searchQuery: string;
  setSearchQuery: (q: string) => void;
  selectedPhase: PhaseId | 'all';
  setSelectedPhase: (p: PhaseId | 'all') => void;
  viewMode: 'grid' | 'roadmap';
  setViewMode: (v: 'grid' | 'roadmap') => void;
}

export const Header: React.FC<HeaderProps> = ({
  searchQuery,
  setSearchQuery,
  selectedPhase,
  setSelectedPhase,
  viewMode,
  setViewMode
}) => {
  return (
    <header className="glass-header sticky top-0 z-40 px-6 py-4 mb-8">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        
        {/* Brand Logo & Badges */}
        <div className="flex items-center gap-4">
          <div className="w-11 h-11 rounded-xl bg-gradient-to-tr from-cyan-500 to-violet-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Cpu className="w-6 h-6 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold tracking-tight gradient-text-rainbow">
                AI Engineering Academy
              </h1>
              <span className="bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs px-2 py-0.5 rounded-full font-mono font-medium flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3" /> 801/801 Passed
              </span>
            </div>
            <p className="text-xs text-slate-400 font-medium">
              50 From-Scratch Modules • Zero-Framework Architecture • Master Curriculum
            </p>
          </div>
        </div>

        {/* Search Bar & View Mode Toggle */}
        <div className="flex items-center gap-3 w-full md:w-auto">
          <div className="relative flex-1 md:w-72">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
            <input
              type="text"
              placeholder="Search math, concepts, code..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-slate-900/80 border border-slate-700/60 rounded-xl pl-9 pr-4 py-2 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/50 transition-all"
            />
          </div>

          <div className="bg-slate-900/80 p-1 border border-slate-700/60 rounded-xl flex items-center">
            <button
              onClick={() => setViewMode('grid')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all ${
                viewMode === 'grid'
                  ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Grid className="w-3.5 h-3.5" /> Grid
            </button>
            <button
              onClick={() => setViewMode('roadmap')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all ${
                viewMode === 'roadmap'
                  ? 'bg-violet-500/20 text-violet-400 border border-violet-500/30'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <GitFork className="w-3.5 h-3.5" /> Roadmap
            </button>
          </div>
        </div>

      </div>

      {/* Phase Filter Tabs */}
      <div className="max-w-7xl mx-auto flex items-center gap-2 mt-4 pt-3 border-t border-slate-800/60 overflow-x-auto pb-1">
        <button
          onClick={() => setSelectedPhase('all')}
          className={`px-3.5 py-1.5 rounded-xl text-xs font-semibold whitespace-nowrap transition-all ${
            selectedPhase === 'all'
              ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 shadow-sm'
              : 'bg-slate-900/40 text-slate-400 border border-slate-800 hover:bg-slate-800/60'
          }`}
        >
          All 50 Modules
        </button>
        <button
          onClick={() => setSelectedPhase('phase1')}
          className={`px-3.5 py-1.5 rounded-xl text-xs font-semibold whitespace-nowrap transition-all ${
            selectedPhase === 'phase1'
              ? 'bg-sky-500/20 text-sky-400 border border-sky-500/30'
              : 'bg-slate-900/40 text-slate-400 border border-slate-800 hover:bg-slate-800/60'
          }`}
        >
          Phase 1: Foundations (001-014)
        </button>
        <button
          onClick={() => setSelectedPhase('phase2')}
          className={`px-3.5 py-1.5 rounded-xl text-xs font-semibold whitespace-nowrap transition-all ${
            selectedPhase === 'phase2'
              ? 'bg-violet-500/20 text-violet-400 border border-violet-500/30'
              : 'bg-slate-900/40 text-slate-400 border border-slate-800 hover:bg-slate-800/60'
          }`}
        >
          Phase 2: LLM Architectures (015-025)
        </button>
        <button
          onClick={() => setSelectedPhase('phase3')}
          className={`px-3.5 py-1.5 rounded-xl text-xs font-semibold whitespace-nowrap transition-all ${
            selectedPhase === 'phase3'
              ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
              : 'bg-slate-900/40 text-slate-400 border border-slate-800 hover:bg-slate-800/60'
          }`}
        >
          Phase 3: Agent Swarms (026-035)
        </button>
        <button
          onClick={() => setSelectedPhase('phase4')}
          className={`px-3.5 py-1.5 rounded-xl text-xs font-semibold whitespace-nowrap transition-all ${
            selectedPhase === 'phase4'
              ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
              : 'bg-slate-900/40 text-slate-400 border border-slate-800 hover:bg-slate-800/60'
          }`}
        >
          Phase 4: Capstone & Multimodal (036-050)
        </button>
      </div>
    </header>
  );
};
