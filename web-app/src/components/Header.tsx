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
    <header className="h-16 shrink-0 bg-[var(--bg-surface)] border-b border-[var(--border-subtle)] px-6 md:px-10 flex items-center justify-between z-40 sticky top-0 shadow-sm">
      
      {/* Left: Brand Logo & Navigation */}
      <div className="flex items-center gap-10">
        <div
          onClick={() => setActiveTab('learn')}
          className="flex items-center gap-3 cursor-pointer group"
        >
          {/* OpenAgentSkill parenthesised logo style: () */}
          <div className="font-mono font-extrabold text-lg text-[var(--text-primary)] flex items-center gap-1">
            <span className="text-[var(--accent-primary)]">( )</span>
            <span className="font-heading font-bold text-lg text-[var(--text-primary)] tracking-tight ml-1">
              OpenAgent<span className="font-mono text-xs font-semibold px-1.5 py-0.5 ml-1.5 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] text-[var(--text-muted)] rounded">ACADEMY</span>
            </span>
          </div>
        </div>

        {/* Top Navigation Tabs */}
        <nav className="hidden md:flex items-center gap-1">
          <button
            onClick={() => setActiveTab('learn')}
            className={`px-3.5 py-1.5 rounded-md text-xs font-semibold flex items-center gap-2 transition-all ${
              activeTab === 'learn'
                ? 'bg-[var(--bg-elevated)] text-[var(--accent-primary)] font-bold border border-[var(--border-subtle)]'
                : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]'
            }`}
          >
            <Compass className="w-4 h-4 text-[var(--accent-primary)]" /> Learn
          </button>

          <button
            onClick={() => setActiveTab('roadmap')}
            className={`px-3.5 py-1.5 rounded-md text-xs font-semibold flex items-center gap-2 transition-all ${
              activeTab === 'roadmap'
                ? 'bg-[var(--bg-elevated)] text-[var(--accent-primary)] font-bold border border-[var(--border-subtle)]'
                : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]'
            }`}
          >
            <Map className="w-4 h-4 text-[var(--accent-primary)]" /> Roadmap
          </button>

          <button
            onClick={() => setActiveTab('projects')}
            className={`px-3.5 py-1.5 rounded-md text-xs font-semibold flex items-center gap-2 transition-all ${
              activeTab === 'projects'
                ? 'bg-[var(--bg-elevated)] text-[var(--accent-primary)] font-bold border border-[var(--border-subtle)]'
                : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]'
            }`}
          >
            <FolderGit2 className="w-4 h-4 text-[var(--accent-primary)]" /> Projects
          </button>

          <button
            onClick={() => setActiveTab('library')}
            className={`px-3.5 py-1.5 rounded-md text-xs font-semibold flex items-center gap-2 transition-all ${
              activeTab === 'library'
                ? 'bg-[var(--bg-elevated)] text-[var(--accent-primary)] font-bold border border-[var(--border-subtle)]'
                : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]'
            }`}
          >
            <BookOpen className="w-4 h-4 text-[var(--accent-primary)]" /> Library
          </button>
        </nav>
      </div>

      {/* Right: Search, Mentor, Avatar */}
      <div className="flex items-center gap-4">
        
        {/* Search Command Pill */}
        <button
          onClick={onOpenSearch}
          className="w-48 sm:w-64 px-3.5 py-1.5 bg-[var(--bg-elevated)] hover:bg-[var(--bg-hover)] border border-[var(--border-subtle)] rounded-lg text-xs text-[var(--text-muted)] flex items-center justify-between transition-all font-mono"
        >
          <div className="flex items-center gap-2">
            <Search className="w-3.5 h-3.5 text-[var(--text-muted)]" />
            <span>Search modules...</span>
          </div>
          <kbd className="px-1.5 py-0.5 bg-[var(--bg-surface)] text-[10px] text-[var(--text-muted)] rounded border border-[var(--border-subtle)]">
            ⌘K
          </kbd>
        </button>

        {/* AI Mentor Emerald Button */}
        <button
          onClick={onOpenTutor}
          className="btn-emerald"
        >
          <Bot className="w-4 h-4" />
          <span className="hidden sm:inline">AI Mentor</span>
        </button>

        {/* User Profile */}
        <div className="w-8 h-8 rounded-full bg-[var(--bg-elevated)] border border-[var(--border-subtle)] flex items-center justify-center text-[var(--text-primary)] text-xs font-mono font-bold cursor-pointer hover:border-[var(--accent-primary)] transition-colors">
          MP
        </div>

      </div>

    </header>
  );
};
