import React from 'react';
import type { ModuleData } from '../types';
import { Clock, ArrowRight } from 'lucide-react';

interface ModuleCardProps {
  module: ModuleData;
  onOpenDetail: (m: ModuleData) => void;
}

export const ModuleCard: React.FC<ModuleCardProps> = ({ module, onOpenDetail }) => {
  return (
    <div
      onClick={() => onOpenDetail(module)}
      className="panel-card p-5 hover:border-indigo-500/40 cursor-pointer transition-all duration-200 group flex flex-col justify-between"
    >
      <div className="space-y-3">
        
        {/* Module ID & Status */}
        <div className="flex items-center justify-between text-xs font-mono">
          <span className="text-indigo-400 font-bold">#{module.id}</span>
          <span className="text-emerald-400 bg-emerald-950/40 px-2 py-0.5 rounded border border-emerald-800/40 text-[10px]">
            Verified
          </span>
        </div>

        {/* Title & Capability Subtitle */}
        <div>
          <h4 className="text-base font-bold text-slate-100 group-hover:text-indigo-300 transition-colors">
            {module.title}
          </h4>
          <p className="text-xs text-slate-400 line-clamp-2 mt-1 leading-relaxed">
            {module.subtitle}
          </p>
        </div>

        {/* Capability Indicators */}
        <div className="flex items-center gap-3 text-[11px] font-mono text-slate-400 pt-1">
          <span className="text-emerald-400">Explain ✓</span>
          <span className="text-emerald-400">Build ✓</span>
          <span className="text-slate-500">Debug ○</span>
          <span className="text-slate-500">Teach ○</span>
        </div>

      </div>

      {/* Footer Info & Action */}
      <div className="pt-4 mt-4 border-t border-white/5 flex items-center justify-between text-xs">
        <div className="flex items-center gap-1.5 text-slate-400 font-mono text-[11px]">
          <Clock className="w-3.5 h-3.5 text-slate-500" />
          <span>{module.estimatedMinutes} min</span>
        </div>

        <button className="text-indigo-400 font-mono font-semibold group-hover:text-indigo-300 flex items-center gap-1 transition-colors">
          Continue <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
        </button>
      </div>

    </div>
  );
};
