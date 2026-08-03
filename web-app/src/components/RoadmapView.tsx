import React, { useState } from 'react';
import { MODULES } from '../modulesData';
import type { ModuleData, PhaseId } from '../types';
import { ChevronDown, ChevronRight, CheckCircle2 } from 'lucide-react';

interface RoadmapViewProps {
  onOpenModule: (m: ModuleData) => void;
}

export const RoadmapView: React.FC<RoadmapViewProps> = ({ onOpenModule }) => {
  const [expandedPhases, setExpandedPhases] = useState<Record<PhaseId, boolean>>({
    phase1: true,
    phase2: false,
    phase3: false,
    phase4: false,
  });

  const togglePhase = (p: PhaseId) => {
    setExpandedPhases((prev) => ({ ...prev, [p]: !prev[p] }));
  };

  const phases: Array<{ id: PhaseId; title: string; subtitle: string; count: number; hours: number; modules: ModuleData[] }> = [
    {
      id: 'phase1',
      title: 'Phase 1 · Neural Network Foundations',
      subtitle: 'Build neural networks from first principles. Understand how models learn before using frameworks.',
      count: 14,
      hours: 18,
      modules: MODULES.filter((m) => m.phaseId === 'phase1'),
    },
    {
      id: 'phase2',
      title: 'Phase 2 · LLM Architectures & Transformers',
      subtitle: 'Attention mechanisms, KV-Cache, Quantization, MoE, FlashAttention, and DeepSeek MLA.',
      count: 11,
      hours: 24,
      modules: MODULES.filter((m) => m.phaseId === 'phase2'),
    },
    {
      id: 'phase3',
      title: 'Phase 3 · Agent Swarms & Tool Execution',
      subtitle: 'Agentic loops, MCP protocol, PEFT LoRA, RLHF/DPO alignment, evaluation, and tracing.',
      count: 10,
      hours: 20,
      modules: MODULES.filter((m) => m.phaseId === 'phase3'),
    },
    {
      id: 'phase4',
      title: 'Phase 4 · Capstones & Multimodal AI',
      subtitle: 'Graph RAG, Vision-Language Models, Long-Context RoPE, Autonomous Coding Agents, Capstone.',
      count: 15,
      hours: 32,
      modules: MODULES.filter((m) => m.phaseId === 'phase4'),
    },
  ];

  return (
    <div className="space-y-6 animate-fade-in">
      
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-slate-100 font-heading">
          Curriculum Roadmap
        </h2>
        <p className="text-xs text-slate-400 mt-1 font-medium">
          A progressive causal learning path designed to build permanent AI engineering capability.
        </p>
      </div>

      {/* Progressive Phases List */}
      <div className="space-y-4">
        {phases.map((phase) => {
          const isExpanded = expandedPhases[phase.id];

          return (
            <div
              key={phase.id}
              className="panel-card overflow-hidden border-white/5 transition-all"
            >
              {/* Phase Collapsible Header */}
              <div
                onClick={() => togglePhase(phase.id)}
                className="p-5 bg-[#11151D] hover:bg-[#171C26] cursor-pointer flex items-center justify-between transition-colors select-none"
              >
                <div className="space-y-1">
                  <div className="flex items-center gap-3">
                    <h3 className="text-base font-bold text-slate-100 font-heading">
                      {phase.title}
                    </h3>
                    <span className="text-[11px] font-mono text-slate-400 bg-white/5 px-2.5 py-0.5 rounded border border-white/5">
                      {phase.count} modules · ~{phase.hours} hours
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 font-medium">
                    {phase.subtitle}
                  </p>
                </div>

                <div className="text-slate-400 hover:text-slate-200">
                  {isExpanded ? <ChevronDown className="w-5 h-5" /> : <ChevronRight className="w-5 h-5" />}
                </div>
              </div>

              {/* Collapsible Content */}
              {isExpanded && (
                <div className="p-5 border-t border-white/5 bg-[#0B0D12]/60 space-y-3">
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                    {phase.modules.map((m, idx) => (
                      <div
                        key={m.id}
                        onClick={() => onOpenModule(m)}
                        className="panel-card p-4 hover:border-indigo-500/40 cursor-pointer transition-all flex flex-col justify-between group"
                      >
                        <div className="space-y-2">
                          <div className="flex items-center justify-between text-xs font-mono">
                            <span className="text-indigo-400 font-bold">#{m.id}</span>
                            <span className="text-slate-500">{m.estimatedMinutes}m</span>
                          </div>
                          
                          <h4 className="text-sm font-bold text-slate-100 group-hover:text-indigo-300 transition-colors">
                            {m.title}
                          </h4>
                          
                          <p className="text-xs text-slate-400 line-clamp-2 leading-relaxed">
                            {m.subtitle}
                          </p>
                        </div>

                        <div className="pt-3 mt-3 border-t border-white/5 flex items-center justify-between text-xs font-mono">
                          {idx === 0 ? (
                            <span className="text-emerald-400 font-bold flex items-center gap-1">
                              <CheckCircle2 className="w-3 h-3" /> Verified
                            </span>
                          ) : (
                            <span className="text-slate-500 flex items-center gap-1">
                              ○ Upcoming
                            </span>
                          )}
                          <span className="text-indigo-400 group-hover:translate-x-1 transition-transform">
                            View →
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

            </div>
          );
        })}
      </div>

    </div>
  );
};
