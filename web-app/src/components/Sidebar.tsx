import React from 'react';
import {
  Compass,
  BarChart3,
  Calendar,
  FileText,
  Bookmark,
  Grid,
  Sparkles,
  Map,
  BookOpen,
  Layers,
  CheckSquare,
  Activity,
  Flame,
} from 'lucide-react';

interface SidebarProps {
  currentView: string;
  setCurrentView: (v: string) => void;
  onOpenRoadmap: () => void;
  onOpenModules: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentView,
  setCurrentView,
  onOpenRoadmap,
  onOpenModules,
}) => {
  return (
    <aside className="w-64 shrink-0 bg-[#0B0D12] border-r border-white/5 p-4 flex flex-col justify-between hidden lg:flex select-none">
      
      <div className="space-y-6">
        
        {/* LEARNING section */}
        <div className="space-y-1">
          <div className="px-3 text-[10px] font-mono font-bold uppercase tracking-wider text-slate-500 mb-2">
            LEARNING
          </div>

          <button
            onClick={() => setCurrentView('overview')}
            className={`w-full px-3 py-2 rounded-xl text-xs font-semibold flex items-center gap-2.5 transition-all ${
              currentView === 'overview'
                ? 'bg-indigo-600/20 text-indigo-400 border border-indigo-500/30 font-bold'
                : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
            }`}
          >
            <Compass className="w-4 h-4 text-indigo-400" /> Overview
          </button>

          <button
            onClick={() => setCurrentView('progress')}
            className="w-full px-3 py-2 rounded-xl text-xs font-semibold flex items-center gap-2.5 text-slate-400 hover:text-slate-200 hover:bg-white/5 transition-all"
          >
            <BarChart3 className="w-4 h-4" /> My Progress
          </button>

          <button
            onClick={() => setCurrentView('calendar')}
            className="w-full px-3 py-2 rounded-xl text-xs font-semibold flex items-center gap-2.5 text-slate-400 hover:text-slate-200 hover:bg-white/5 transition-all"
          >
            <Calendar className="w-4 h-4" /> Calendar
          </button>

          <button
            onClick={() => setCurrentView('notes')}
            className="w-full px-3 py-2 rounded-xl text-xs font-semibold flex items-center gap-2.5 text-slate-400 hover:text-slate-200 hover:bg-white/5 transition-all"
          >
            <FileText className="w-4 h-4" /> Notes
          </button>

          <button
            onClick={() => setCurrentView('bookmarks')}
            className="w-full px-3 py-2 rounded-xl text-xs font-semibold flex items-center gap-2.5 text-slate-400 hover:text-slate-200 hover:bg-white/5 transition-all"
          >
            <Bookmark className="w-4 h-4" /> Bookmarks
          </button>
        </div>

        {/* CURRICULUM section */}
        <div className="space-y-1 pt-2 border-t border-white/5">
          <div className="px-3 text-[10px] font-mono font-bold uppercase tracking-wider text-slate-500 mb-2">
            CURRICULUM
          </div>

          <button
            onClick={onOpenModules}
            className="w-full px-3 py-2 rounded-xl text-xs font-semibold flex items-center gap-2.5 text-slate-400 hover:text-slate-200 hover:bg-white/5 transition-all"
          >
            <Grid className="w-4 h-4" /> All Modules
          </button>

          <button
            onClick={() => setCurrentView('capabilities')}
            className="w-full px-3 py-2 rounded-xl text-xs font-semibold flex items-center gap-2.5 text-slate-400 hover:text-slate-200 hover:bg-white/5 transition-all"
          >
            <Sparkles className="w-4 h-4" /> By Capability
          </button>

          <button
            onClick={onOpenRoadmap}
            className="w-full px-3 py-2 rounded-xl text-xs font-semibold flex items-center gap-2.5 text-slate-400 hover:text-slate-200 hover:bg-white/5 transition-all"
          >
            <Map className="w-4 h-4" /> Roadmap View
          </button>
        </div>

        {/* PLAYBOOK section */}
        <div className="space-y-1 pt-2 border-t border-white/5">
          <div className="px-3 text-[10px] font-mono font-bold uppercase tracking-wider text-slate-500 mb-2">
            PLAYBOOK
          </div>

          <button
            onClick={() => setCurrentView('decisions')}
            className="w-full px-3 py-2 rounded-xl text-xs font-semibold flex items-center gap-2.5 text-slate-400 hover:text-slate-200 hover:bg-white/5 transition-all"
          >
            <BookOpen className="w-4 h-4" /> Decision Records
          </button>

          <button
            onClick={() => setCurrentView('patterns')}
            className="w-full px-3 py-2 rounded-xl text-xs font-semibold flex items-center gap-2.5 text-slate-400 hover:text-slate-200 hover:bg-white/5 transition-all"
          >
            <Layers className="w-4 h-4" /> Patterns
          </button>

          <button
            onClick={() => setCurrentView('checklists')}
            className="w-full px-3 py-2 rounded-xl text-xs font-semibold flex items-center gap-2.5 text-slate-400 hover:text-slate-200 hover:bg-white/5 transition-all"
          >
            <CheckSquare className="w-4 h-4" /> Checklists
          </button>

          <button
            onClick={() => setCurrentView('benchmarks')}
            className="w-full px-3 py-2 rounded-xl text-xs font-semibold flex items-center gap-2.5 text-slate-400 hover:text-slate-200 hover:bg-white/5 transition-all"
          >
            <Activity className="w-4 h-4" /> Benchmarks
          </button>
        </div>

      </div>

      {/* Streak Widget Card */}
      <div className="bg-[#11151D] border border-white/5 p-4 rounded-2xl space-y-2 mt-4">
        <div className="flex items-center gap-2 text-xs font-bold text-amber-400">
          <Flame className="w-4 h-4 fill-current" /> Streak
        </div>
        <div className="text-xl font-bold text-slate-100 font-heading">
          12 days
        </div>
        <p className="text-[11px] text-slate-400">Keep it up! 🔥</p>

        {/* M T W T F S S Activity Dots */}
        <div className="flex items-center justify-between pt-2 text-[10px] font-mono text-slate-500">
          {['M', 'T', 'W', 'T', 'F', 'S', 'S'].map((day, idx) => (
            <div key={idx} className="flex flex-col items-center gap-1">
              <span>{day}</span>
              <div
                className={`w-2.5 h-2.5 rounded-sm ${
                  idx < 5 ? 'bg-emerald-500' : 'bg-slate-800'
                }`}
              />
            </div>
          ))}
        </div>
      </div>

    </aside>
  );
};
