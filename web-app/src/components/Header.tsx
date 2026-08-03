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
    <header className="h-16 shrink-0 bg-[#090C10]/95 backdrop-blur-md border-b border-white/5 px-6 md:px-10 flex items-center justify-between z-40 sticky top-0">
      
      {/* Left: Brand Logo & Navigation */}
      <div className="flex items-center gap-10">
        <div
          onClick={() => setActiveTab('learn')}
          className="flex items-center gap-3 cursor-pointer group"
        >
          <div className="w-8 h-8 rounded-lg bg-indigo-600/20 border border-indigo-500/30 text-indigo-400 font-mono font-bold text-xs flex items-center justify-center">
            AI
          </div>
          <span className="font-heading font-bold text-base text-slate-100 tracking-tight group-hover:text-indigo-400 transition-colors">
            AI Engineering Academy
          </span>
        </div>

        {/* Top Navigation Tabs */}
        <nav className="hidden md:flex items-center gap-1">
          <button
            onClick={() => setActiveTab('learn')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
              activeTab === 'learn'
                ? 'bg-white/10 text-white font-bold'
                : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
            }`}
          >
            <Compass className="w-4 h-4 text-indigo-400" /> Learn
          </button>

          <button
            onClick={() => setActiveTab('roadmap')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
              activeTab === 'roadmap'
                ? 'bg-white/10 text-white font-bold'
                : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
            }`}
          >
            <Map className="w-4 h-4 text-indigo-400" /> Roadmap
          </button>

          <button
            onClick={() => setActiveTab('projects')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
              activeTab === 'projects'
                ? 'bg-white/10 text-white font-bold'
                : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
            }`}
          >
            <FolderGit2 className="w-4 h-4 text-indigo-400" /> Projects
          </button>

          <button
            onClick={() => setActiveTab('library')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
              activeTab === 'library'
                ? 'bg-white/10 text-white font-bold'
                : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
            }`}
          >
            <BookOpen className="w-4 h-4 text-indigo-400" /> Library
          </button>
        </nav>
      </div>

      {/* Right: Search, Mentor, Avatar */}
      <div className="flex items-center gap-4">
        
        {/* Search Command Pill */}
        <button
          onClick={onOpenSearch}
          className="w-48 sm:w-64 px-3.5 py-1.5 bg-[#121620] hover:bg-[#181E2B] border border-white/10 rounded-xl text-xs text-slate-400 flex items-center justify-between transition-all font-mono"
        >
          <div className="flex items-center gap-2">
            <Search className="w-3.5 h-3.5 text-slate-400" />
            <span>Search...</span>
          </div>
          <kbd className="px-1.5 py-0.5 bg-black/40 text-[10px] text-slate-400 rounded border border-white/10">
            ⌘K
          </kbd>
        </button>

        {/* Tutor Button */}
        <button
          onClick={onOpenTutor}
          className="px-3.5 py-1.5 rounded-xl bg-indigo-600/15 hover:bg-indigo-600/25 border border-indigo-500/30 text-indigo-300 text-xs font-semibold flex items-center gap-2 transition-all"
        >
          <Bot className="w-4 h-4 text-indigo-400" />
          <span className="hidden sm:inline">AI Mentor</span>
        </button>

        {/* User Profile */}
        <div className="w-8 h-8 rounded-full bg-slate-800 border border-white/10 flex items-center justify-center text-slate-200 text-xs font-mono font-bold cursor-pointer hover:border-indigo-400 transition-colors">
          MP
        </div>

      </div>

    </header>
  );
};
