import React from 'react';
import type { ModuleData, PhaseInfo } from '../types';
import { ChevronRight } from 'lucide-react';

interface RoadmapViewProps {
  phases: PhaseInfo[];
  modules: ModuleData[];
  onOpenDetail: (m: ModuleData) => void;
}

export const RoadmapView: React.FC<RoadmapViewProps> = ({ phases, modules, onOpenDetail }) => {
  return (
    <div className="space-y-12 py-4">
      {phases.map((phase) => {
        const phaseModules = modules.filter((m) => m.phaseId === phase.id);
        if (phaseModules.length === 0) return null;

        return (
          <div key={phase.id} className="relative pl-6 border-l-2 border-slate-800 space-y-6">
            
            {/* Phase Node Dot */}
            <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-slate-900 border-2 border-cyan-400 shadow-md shadow-cyan-500/50" />

            {/* Phase Banner */}
            <div className="bg-slate-900/80 border border-slate-800 p-6 rounded-2xl">
              <div className="flex items-center justify-between gap-4 mb-2">
                <span className={`text-xs font-bold px-3 py-1 rounded-full border ${phase.badgeClass}`}>
                  {phase.range}
                </span>
                <span className="text-xs text-slate-400 font-mono font-medium">
                  {phaseModules.length} Modules Active
                </span>
              </div>
              <h2 className="text-xl font-bold text-slate-100 mb-1">{phase.name}</h2>
              <p className="text-xs text-slate-400">{phase.description}</p>
            </div>

            {/* Module Nodes Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {phaseModules.map((module) => (
                <div
                  key={module.id}
                  onClick={() => onOpenDetail(module)}
                  className="bg-slate-900/60 hover:bg-slate-800/80 border border-slate-800/80 hover:border-cyan-500/40 p-4 rounded-xl cursor-pointer transition-all duration-200 group flex items-start justify-between"
                >
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-mono text-xs font-bold text-cyan-400">#{module.id}</span>
                      <span className="text-[10px] text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-800/60 font-mono">
                        {module.testCount}/{module.testCount} Tests
                      </span>
                    </div>
                    <h4 className="text-sm font-bold text-slate-200 group-hover:text-cyan-300 transition-colors line-clamp-1">
                      {module.title}
                    </h4>
                    <p className="text-xs text-slate-400 line-clamp-1 mt-0.5">
                      {module.subtitle}
                    </p>
                  </div>
                  <ChevronRight className="w-4 h-4 text-slate-600 group-hover:text-cyan-400 transition-colors mt-1 shrink-0" />
                </div>
              ))}
            </div>

          </div>
        );
      })}
    </div>
  );
};
