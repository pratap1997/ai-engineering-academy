import React from 'react';
import { Search, Bot, Compass, Map, FolderGit2, BookOpen, Brain } from 'lucide-react';

interface HeaderProps {
  activeTab: 'learn' | 'roadmap' | 'projects' | 'library';
  setActiveTab: (tab: 'learn' | 'roadmap' | 'projects' | 'library') => void;
  onOpenSearch: () => void;
  onOpenTutor: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  onOpenSearch,
  onOpenTutor,
}) => {
  return (
    <header className="h-16 shrink-0 bg-[#0D1117] border-b border-white/10 px-4 md:px-8 flex items-center justify-between z-40 sticky top-0 shadow-md">
      
      {/* Left: Brand Logo & Navigation */}
      <div className="flex items-center gap-8">
        <div
          onClick={() => setActiveTab('learn')}
          className="flex items-center gap-2.5 cursor-pointer group"
        >
          {/* Glowing Neural Logo Symbol */}
          <div className="w-8 h-8 rounded-lg bg-[#10B981]/15 border border-[#10B981]/40 flex items-center justify-center text-[#10B981] group-hover:scale-105 transition-transform">
            <Brain className="w-5 h-5" />
          </div>

          <div className="flex items-center gap-2">
            <span className="font-heading font-extrabold text-lg text-white tracking-tight">
              AI Engineering
            </span>
            <span className="font-mono text-[11px] font-bold px-2 py-0.5 bg-[#10B981]/15 border border-[#10B981]/40 text-[#10B981] rounded-md tracking-wider">
              SKOOL
            </span>
          </div>
        </div>

        {/* Top Navigation Tabs */}
        <nav className="hidden md:flex items-center gap-1">
          <button
            onClick={() => setActiveTab('learn')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
              activeTab === 'learn'
                ? 'bg-[#161B22] text-[#10B981] font-bold border border-[#10B981]/30 shadow-sm'
                : 'text-slate-400 hover:text-slate-100 hover:bg-white/5'
            }`}
          >
            <Compass className="w-4 h-4 text-[#10B981]" /> Learn
          </button>

          <button
            onClick={() => setActiveTab('roadmap')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
              activeTab === 'roadmap'
                ? 'bg-[#161B22] text-[#10B981] font-bold border border-[#10B981]/30 shadow-sm'
                : 'text-slate-400 hover:text-slate-100 hover:bg-white/5'
            }`}
          >
            <Map className="w-4 h-4 text-[#10B981]" /> Roadmap
          </button>

          <button
            onClick={() => setActiveTab('projects')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
              activeTab === 'projects'
                ? 'bg-[#161B22] text-[#10B981] font-bold border border-[#10B981]/30 shadow-sm'
                : 'text-slate-400 hover:text-slate-100 hover:bg-white/5'
            }`}
          >
            <FolderGit2 className="w-4 h-4 text-[#10B981]" /> Projects
          </button>

          <button
            onClick={() => setActiveTab('library')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
              activeTab === 'library'
                ? 'bg-[#161B22] text-[#10B981] font-bold border border-[#10B981]/30 shadow-sm'
                : 'text-slate-400 hover:text-slate-100 hover:bg-white/5'
            }`}
          >
            <BookOpen className="w-4 h-4 text-[#10B981]" /> Library
          </button>
        </nav>
      </div>

      {/* Right: Search, Mentor, Profile MP */}
      <div className="flex items-center gap-3 md:gap-4">
        
        {/* Search Command Pill */}
        <button
          onClick={onOpenSearch}
          className="w-40 sm:w-60 px-3 py-1.5 bg-[#161B22] hover:bg-[#21262D] border border-white/10 rounded-lg text-xs text-slate-400 flex items-center justify-between transition-all font-mono"
        >
          <div className="flex items-center gap-2">
            <Search className="w-3.5 h-3.5 text-slate-400" />
            <span>Search modules...</span>
          </div>
          <kbd className="hidden sm:inline-block px-1.5 py-0.5 bg-[#0D1117] text-[10px] text-slate-400 rounded border border-white/10">
            ⌘K
          </kbd>
        </button>

        {/* AI Mentor Button */}
        <button
          onClick={onOpenTutor}
          className="btn-emerald text-xs"
        >
          <Bot className="w-4 h-4" />
          <span className="hidden sm:inline">AI Mentor</span>
        </button>

        {/* Top-Right Corner User Profile (MP) */}
        <div className="relative group flex items-center gap-2 pl-1">
          <div
            title="Mahendra Pratap (AI Engineer)"
            className="w-9 h-9 rounded-full bg-gradient-to-tr from-[#10B981]/30 to-[#059669]/50 border-2 border-[#10B981] flex items-center justify-center text-white text-xs font-mono font-bold cursor-pointer shadow-md hover:scale-105 transition-all relative"
          >
            MP
            {/* Green Online Dot */}
            <span className="absolute -top-0.5 -right-0.5 w-3 h-3 bg-[#10B981] border-2 border-[#0D1117] rounded-full" />
          </div>

          {/* Hover Tooltip */}
          <div className="absolute right-0 top-11 hidden group-hover:flex flex-col items-end pointer-events-none z-50 animate-fade-in">
            <div className="bg-[#161B22] border border-[#10B981]/30 text-white px-3 py-1.5 rounded-lg shadow-xl text-xs font-mono whitespace-nowrap">
              <div className="font-bold text-[#10B981]">Mahendra Pratap</div>
              <div className="text-[10px] text-slate-400">AI Engineering Learner</div>
            </div>
          </div>
        </div>

      </div>

    </header>
  );
};
