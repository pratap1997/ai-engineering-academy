import React from 'react';
import { Search, Bot, Compass, Map, FolderGit2, BookOpen } from 'lucide-react';

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
    <header className="sticky top-0 z-40 bg-[#0B0D12]/90 backdrop-blur-md border-b border-white/5 px-6 py-3">
      <div className="max-w-7xl mx-auto flex items-center justify-between gap-4">
        
        {/* Brand & Main Destinations */}
        <div className="flex items-center gap-8">
          <div
            onClick={() => setActiveTab('learn')}
            className="flex items-center gap-2.5 cursor-pointer group"
          >
            <div className="w-8 h-8 rounded-lg bg-indigo-600/20 border border-indigo-500/30 text-indigo-400 flex items-center justify-center font-bold text-xs">
              AI
            </div>
            <span className="font-heading font-bold text-sm text-slate-100 tracking-tight group-hover:text-indigo-400 transition-colors">
              AI Engineering Academy
            </span>
          </div>

          {/* Primary Navigation Tabs */}
          <nav className="flex items-center gap-1">
            <button
              onClick={() => setActiveTab('learn')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors ${
                activeTab === 'learn'
                  ? 'bg-white/10 text-white font-bold'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
              }`}
            >
              <Compass className="w-3.5 h-3.5" /> Learn
            </button>

            <button
              onClick={() => setActiveTab('roadmap')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors ${
                activeTab === 'roadmap'
                  ? 'bg-white/10 text-white font-bold'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
              }`}
            >
              <Map className="w-3.5 h-3.5" /> Roadmap
            </button>

            <button
              onClick={() => setActiveTab('projects')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors ${
                activeTab === 'projects'
                  ? 'bg-white/10 text-white font-bold'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
              }`}
            >
              <FolderGit2 className="w-3.5 h-3.5" /> Projects
            </button>

            <button
              onClick={() => setActiveTab('library')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors ${
                activeTab === 'library'
                  ? 'bg-white/10 text-white font-bold'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
              }`}
            >
              <BookOpen className="w-3.5 h-3.5" /> Library
            </button>
          </nav>
        </div>

        {/* Right Side: Command Search, Tutor, Profile */}
        <div className="flex items-center gap-3">
          
          <button
            onClick={onOpenSearch}
            className="px-3 py-1.5 bg-[#171C26] hover:bg-[#1E2638] border border-white/5 rounded-lg text-xs text-slate-400 flex items-center gap-2 transition-colors font-mono"
          >
            <Search className="w-3.5 h-3.5 text-slate-400" />
            <span>Search...</span>
            <kbd className="px-1.5 py-0.5 bg-black/40 text-[10px] text-slate-400 rounded border border-white/10">
              ⌘K
            </kbd>
          </button>

          <button
            onClick={onOpenTutor}
            className="btn-subtle text-xs py-1.5 px-3 text-indigo-300 border-indigo-500/30 bg-indigo-500/10 hover:bg-indigo-500/20"
          >
            <Bot className="w-3.5 h-3.5 text-indigo-400" /> Mentor
          </button>

          <div className="w-8 h-8 rounded-full bg-slate-800 border border-white/10 flex items-center justify-center text-slate-300 text-xs font-bold font-mono cursor-pointer hover:border-indigo-400 transition-colors">
            MP
          </div>

        </div>

      </div>
    </header>
  );
};
