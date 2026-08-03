import React, { useState, useEffect } from 'react';
import { MODULES } from '../modulesData';
import type { ModuleData } from '../types';
import { Search, X, ChevronRight } from 'lucide-react';

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectModule: (m: ModuleData) => void;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({
  isOpen,
  onClose,
  onSelectModule,
}) => {
  const [query, setQuery] = useState('');

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        if (isOpen) onClose();
        else setQuery('');
      }
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const results = MODULES.filter((m) => {
    if (!query.trim()) return true;
    const q = query.toLowerCase();
    return (
      m.id.includes(q) ||
      m.title.toLowerCase().includes(q) ||
      m.subtitle.toLowerCase().includes(q) ||
      m.topics.some((t) => t.toLowerCase().includes(q))
    );
  }).slice(0, 8);

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-20 px-4 bg-black/70 backdrop-blur-sm">
      <div className="bg-[#11151D] border border-white/10 rounded-2xl max-w-2xl w-full shadow-2xl overflow-hidden animate-fade-in">
        
        {/* Search Input Bar */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-white/10 bg-[#171C26]">
          <Search className="w-4 h-4 text-slate-400 shrink-0" />
          <input
            type="text"
            autoFocus
            placeholder="Search modules, concepts, algorithms, challenges (Press Esc to close)..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full bg-transparent text-sm text-slate-100 placeholder-slate-500 focus:outline-none"
          />
          <button onClick={onClose} className="text-slate-500 hover:text-slate-300 p-1">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Results List */}
        <div className="p-2 max-h-96 overflow-y-auto space-y-1">
          {results.length === 0 ? (
            <div className="p-6 text-center text-xs text-slate-500 font-mono">
              No matching modules or concepts found for "{query}"
            </div>
          ) : (
            results.map((module) => (
              <div
                key={module.id}
                onClick={() => {
                  onSelectModule(module);
                  onClose();
                }}
                className="p-3 rounded-xl hover:bg-[#171C26] cursor-pointer flex items-center justify-between group transition-colors border border-transparent hover:border-white/10"
              >
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 flex items-center justify-center font-mono text-xs font-bold shrink-0">
                    #{module.id}
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-slate-200 group-hover:text-indigo-400 transition-colors">
                      {module.title}
                    </h4>
                    <p className="text-[11px] text-slate-400 line-clamp-1 mt-0.5">
                      {module.subtitle}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-2 text-[11px] text-slate-500 font-mono">
                  <span>{module.estimatedMinutes}m</span>
                  <ChevronRight className="w-3.5 h-3.5 text-slate-600 group-hover:text-indigo-400 transition-colors" />
                </div>
              </div>
            ))
          )}
        </div>

        {/* Footer shortcuts */}
        <div className="px-4 py-2 bg-[#0B0D12] border-t border-white/5 flex items-center justify-between text-[11px] text-slate-500 font-mono">
          <span>Tip: Use ↑ ↓ to navigate</span>
          <span>Esc to exit</span>
        </div>

      </div>
    </div>
  );
};
