import React from 'react';
import type { ModuleData } from '../types';
import { CheckCircle2, Clock, Terminal, ChevronRight, Activity } from 'lucide-react';

interface ModuleCardProps {
  module: ModuleData;
  onOpenDetail: (m: ModuleData) => void;
  onOpenDiagram: (m: ModuleData) => void;
}

export const ModuleCard: React.FC<ModuleCardProps> = ({ module, onOpenDetail, onOpenDiagram }) => {
  const getPhaseBadgeClass = (phaseId: string) => {
    switch (phaseId) {
      case 'phase1': return 'bg-sky-500/10 text-sky-400 border-sky-500/30';
      case 'phase2': return 'bg-violet-500/10 text-violet-400 border-violet-500/30';
      case 'phase3': return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
      case 'phase4': return 'bg-amber-500/10 text-amber-400 border-amber-500/30';
      default: return 'bg-slate-500/10 text-slate-400 border-slate-500/30';
    }
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'text-emerald-400';
      case 'intermediate': return 'text-cyan-400';
      case 'advanced': return 'text-violet-400';
      case 'capstone': return 'text-amber-400 font-bold';
      default: return 'text-slate-400';
    }
  };

  return (
    <div className="glass-panel p-5 flex flex-col justify-between group relative overflow-hidden transition-all duration-300">
      
      {/* Top Header info */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <span className="font-mono text-xs font-bold text-cyan-400 bg-cyan-950/60 px-2.5 py-1 rounded-lg border border-cyan-800/40">
            MODULE {module.id}
          </span>
          <span className={`text-xs px-2.5 py-0.5 rounded-full border font-medium ${getPhaseBadgeClass(module.phaseId)}`}>
            {module.phaseId.toUpperCase()}
          </span>
        </div>

        <h3 className="text-lg font-bold text-slate-100 group-hover:text-cyan-300 transition-colors line-clamp-1">
          {module.title}
        </h3>
        <p className="text-xs text-slate-400 font-medium mb-3 line-clamp-1">
          {module.subtitle}
        </p>

        <p className="text-xs text-slate-300 line-clamp-2 mb-4 leading-relaxed">
          {module.overview}
        </p>

        {/* Math Highlight Block */}
        <div className="bg-slate-950/80 p-2.5 rounded-xl border border-slate-800/80 mb-4 font-mono text-[11px] text-cyan-200 overflow-x-auto">
          <code>{module.mathHighlight}</code>
        </div>
      </div>

      {/* Meta Footer */}
      <div>
        <div className="flex items-center justify-between text-xs text-slate-400 mb-4 pt-3 border-t border-slate-800/60">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1 font-mono text-emerald-400">
              <CheckCircle2 className="w-3.5 h-3.5" /> {module.testCount}/{module.testCount} Passed
            </span>
            <span className="flex items-center gap-1">
              <Clock className="w-3.5 h-3.5 text-slate-500" /> {module.estimatedMinutes}m
            </span>
          </div>
          <span className={`capitalize font-medium ${getDifficultyColor(module.difficulty)}`}>
            {module.difficulty}
          </span>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => onOpenDetail(module)}
            className="flex-1 btn-secondary text-xs py-2 justify-center"
          >
            <Terminal className="w-3.5 h-3.5" /> View Specs <ChevronRight className="w-3.5 h-3.5" />
          </button>
          
          {module.diagramType && (
            <button
              onClick={() => onOpenDiagram(module)}
              className="px-3 py-2 bg-violet-500/10 hover:bg-violet-500/20 text-violet-300 border border-violet-500/30 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition-all"
              title="View Interactive Diagram"
            >
              <Activity className="w-3.5 h-3.5 text-violet-400" /> Diagram
            </button>
          )}
        </div>
      </div>

    </div>
  );
};
