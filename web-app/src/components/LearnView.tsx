import React from 'react';
import { MODULES } from '../modulesData';
import type { ModuleData } from '../types';
import { ArrowRight, CheckCircle2, FlaskConical, BookOpen, Code, Brain, Sparkles, ShieldCheck } from 'lucide-react';
import { LossLandscape3D } from './LossLandscape3D';

interface LearnViewProps {
  onOpenModule: (m: ModuleData) => void;
  onExploreRoadmap: () => void;
}

export const LearnView: React.FC<LearnViewProps> = ({ onOpenModule, onExploreRoadmap }) => {
  const currentModule = MODULES[0]; // Module 001

  return (
    <div className="max-w-6xl mx-auto space-y-8 animate-fade-in pb-12">
      
      {/* Greeting Header - High Contrast Crisp Text */}
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-[#10B981]" />
          <h2 className="text-2xl md:text-3xl font-bold font-heading text-white tracking-tight">
            Good morning, Mahendra
          </h2>
        </div>
        <p className="text-sm text-slate-300 font-medium max-w-2xl leading-relaxed">
          Welcome to <span className="text-[#10B981] font-semibold">AI Engineering Skool</span> — 50 Verified Engineering Modules & 3D Interactive Visualizers.
        </p>
      </div>

      {/* Hero Panel: CURRENT MODULE */}
      <div className="panel-card p-6 md:p-8 relative overflow-hidden border border-white/10 bg-[#0D1117]">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
          
          {/* Left Column: Context & Actions */}
          <div className="lg:col-span-6 space-y-5">
            
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="font-bold uppercase tracking-wider text-[#10B981] bg-[#161B22] px-2.5 py-1 rounded border border-[#10B981]/30">
                PRIMARY PICK // MODULE #{currentModule.id}
              </span>
              <span className="badge-pill badge-emerald flex items-center gap-1">
                <ShieldCheck className="w-3.5 h-3.5 text-[#10B981]" /> READY TO ADOPT
              </span>
            </div>

            <div>
              <h3 className="text-2xl md:text-3xl font-bold font-heading text-white leading-tight">
                001 · Perceptron From Scratch
              </h3>

              <p className="text-sm text-slate-300 leading-relaxed mt-3 max-w-lg">
                Build Rosenblatt's 1958 binary classifier in pure NumPy. Learn geometric hyperplanes, weight updates, and Minsky's 1969 XOR proof.
              </p>
            </div>

            {/* Progress Bar */}
            <div className="space-y-1.5 pt-1 max-w-md">
              <div className="flex justify-between text-xs font-mono text-slate-400">
                <span>MODULE READINESS SCORE</span>
                <span className="text-[#10B981] font-bold">100 / 100</span>
              </div>
              <div className="w-full bg-[#161B22] rounded-full h-2 overflow-hidden border border-white/10">
                <div className="bg-[#10B981] h-full rounded-full w-[100%]" />
              </div>
            </div>

            {/* Capability Metrics */}
            <div className="flex flex-wrap items-center gap-5 text-xs font-mono pt-1">
              <span className="flex items-center gap-1.5 text-[#10B981] font-semibold">
                <CheckCircle2 className="w-3.5 h-3.5" /> Explain ✓
              </span>
              <span className="flex items-center gap-1.5 text-[#10B981] font-semibold">
                <CheckCircle2 className="w-3.5 h-3.5" /> Build ✓
              </span>
              <span className="flex items-center gap-1.5 text-[#10B981] font-semibold">
                <CheckCircle2 className="w-3.5 h-3.5" /> Debug ✓
              </span>
              <span className="flex items-center gap-1.5 text-slate-500">
                ○ Teach
              </span>
            </div>

            {/* Actions */}
            <div className="flex flex-wrap items-center gap-4 pt-2">
              <button
                onClick={() => onOpenModule(currentModule)}
                className="btn-emerald min-h-[44px] px-6"
              >
                Open Masterclass Workspace <ArrowRight className="w-4 h-4" />
              </button>
              
              <button
                onClick={() => onOpenModule(currentModule)}
                className="btn-subtle min-h-[44px] px-5"
              >
                <FlaskConical className="w-4 h-4 text-[#10B981]" /> Runnable Experiments
              </button>
            </div>

          </div>

          {/* Right Column: 3D WebGL Loss Surface Interactive Canvas */}
          <div className="lg:col-span-6">
            <LossLandscape3D learningRate={0.06} isTraining={true} />
          </div>

        </div>
      </div>

      {/* Your Learning Snapshot */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h4 className="text-base font-bold font-heading text-white">
            Curriculum Verification Snapshot
          </h4>
          <span className="text-xs text-slate-400 font-mono">50 Modules Ready</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          
          {/* Card 1: Concepts Mastered */}
          <div className="panel-card p-5 space-y-3 bg-[#0D1117] border-white/10">
            <div className="flex items-center justify-between">
              <div className="p-2 rounded-lg bg-[#161B22] text-[#10B981] border border-white/10">
                <BookOpen className="w-4 h-4" />
              </div>
              <span className="text-[11px] text-[#10B981] font-mono font-bold">100% COVERAGE</span>
            </div>
            <div className="text-3xl font-bold font-heading text-white">50</div>
            <div className="text-xs text-slate-300 font-medium">Verified Modules</div>
          </div>

          {/* Card 2: Experiments Run */}
          <div className="panel-card p-5 space-y-3 bg-[#0D1117] border-white/10">
            <div className="flex items-center justify-between">
              <div className="p-2 rounded-lg bg-[#161B22] text-[#10B981] border border-white/10">
                <FlaskConical className="w-4 h-4" />
              </div>
              <span className="text-[11px] text-[#10B981] font-mono font-bold">801 PASSING</span>
            </div>
            <div className="text-3xl font-bold font-heading text-white">801</div>
            <div className="text-xs text-slate-300 font-medium">Pytest Test Suite</div>
          </div>

          {/* Card 3: Challenges Passed */}
          <div className="panel-card p-5 space-y-3 bg-[#0D1117] border-white/10">
            <div className="flex items-center justify-between">
              <div className="p-2 rounded-lg bg-[#161B22] text-[#10B981] border border-white/10">
                <Code className="w-4 h-4" />
              </div>
              <span className="text-[11px] text-[#10B981] font-mono font-bold">PURE PYTHON</span>
            </div>
            <div className="text-3xl font-bold font-heading text-white">0</div>
            <div className="text-xs text-slate-300 font-medium">External Frameworks</div>
          </div>

          {/* Card 4: Video Masterclasses */}
          <div className="panel-card p-5 space-y-3 bg-[#0D1117] border-white/10">
            <div className="flex items-center justify-between">
              <div className="p-2 rounded-lg bg-[#161B22] text-[#10B981] border border-white/10">
                <Brain className="w-4 h-4" />
              </div>
              <span className="text-[11px] text-[#10B981] font-mono font-bold">60 FPS HD</span>
            </div>
            <div className="text-3xl font-bold font-heading text-white">Riva</div>
            <div className="text-xs text-slate-300 font-medium">NVIDIA Speech Engine</div>
          </div>

        </div>
      </div>

      {/* Upcoming Path Grid */}
      <div className="space-y-4 pt-2">
        <div className="flex items-center justify-between">
          <h4 className="text-base font-bold font-heading text-white">
            Explore All 50 Curriculum Modules
          </h4>
          <button onClick={onExploreRoadmap} className="text-xs text-[#10B981] hover:underline font-mono">
            Explore full roadmap →
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          {MODULES.slice(1, 5).map((m) => (
            <div
              key={m.id}
              onClick={() => onOpenModule(m)}
              className="panel-card p-5 cursor-pointer flex flex-col justify-between bg-[#0D1117] border-white/10 hover:border-[#10B981] transition-colors"
            >
              <div className="space-y-2">
                <div className="text-xs font-mono text-[#10B981] font-bold">MODULE #{m.id}</div>
                <h5 className="text-base font-bold font-heading text-white">{m.title}</h5>
                <p className="text-xs text-slate-300 leading-relaxed line-clamp-2">{m.subtitle}</p>
              </div>
              <div className="pt-4 mt-4 border-t border-white/10 flex items-center justify-between text-xs font-mono">
                <span className="text-[#10B981] font-bold">Ready</span>
                <span className="text-slate-400">{m.estimatedMinutes} min</span>
              </div>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
};
