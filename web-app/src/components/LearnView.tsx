import React from 'react';
import { MODULES } from '../modulesData';
import type { ModuleData } from '../types';
import { ArrowRight, CheckCircle2, FlaskConical, BookOpen, Code, Brain, Sparkles, ShieldCheck } from 'lucide-react';

interface LearnViewProps {
  onOpenModule: (m: ModuleData) => void;
  onExploreRoadmap: () => void;
}

export const LearnView: React.FC<LearnViewProps> = ({ onOpenModule, onExploreRoadmap }) => {
  const currentModule = MODULES[0]; // Module 001

  return (
    <div className="max-w-6xl mx-auto space-y-10 animate-fade-in pb-12">
      
      {/* Greeting Header */}
      <div className="space-y-1.5">
        <div className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-[var(--accent-primary)]" />
          <h2 className="text-2xl md:text-3xl font-bold font-heading text-[var(--text-primary)]">
            Good morning, Mahendra
          </h2>
        </div>
        <p className="text-sm text-[var(--text-secondary)] font-medium max-w-xl">
          Welcome to the OpenAgent AI Engineering Curriculum — 50 Verified Engineering Modules.
        </p>
      </div>

      {/* Hero Panel: CURRENT MODULE (OpenAgentSkill Decision Snapshot Style) */}
      <div className="panel-card p-6 md:p-8 relative overflow-hidden border-[var(--border-strong)] bg-[var(--bg-surface)]">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
          
          {/* Left Column: Context & Actions */}
          <div className="lg:col-span-7 space-y-5">
            
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="font-bold uppercase tracking-wider text-[var(--accent-primary)] bg-[var(--bg-elevated)] px-2.5 py-1 rounded border border-[var(--border-subtle)]">
                PRIMARY PICK // MODULE #{currentModule.id}
              </span>
              <span className="badge-pill badge-emerald flex items-center gap-1">
                <ShieldCheck className="w-3.5 h-3.5" /> READY TO ADOPT
              </span>
            </div>

            <div>
              <h3 className="text-3xl font-bold font-heading text-[var(--text-primary)] leading-tight">
                001 · Perceptron From Scratch
              </h3>

              <p className="text-sm text-[var(--text-secondary)] leading-relaxed mt-3 max-w-lg">
                Build Rosenblatt's 1958 binary classifier in pure NumPy. Learn geometric hyperplanes, weight updates, and Minsky's 1969 XOR proof.
              </p>
            </div>

            {/* Progress Bar */}
            <div className="space-y-1.5 pt-1 max-w-md">
              <div className="flex justify-between text-xs font-mono text-[var(--text-muted)]">
                <span>MODULE READINESS SCORE</span>
                <span className="text-[var(--accent-primary)] font-bold">100 / 100</span>
              </div>
              <div className="w-full bg-[var(--bg-elevated)] rounded-full h-2 overflow-hidden border border-[var(--border-subtle)]">
                <div className="bg-[var(--accent-primary)] h-full rounded-full w-[100%]" />
              </div>
            </div>

            {/* Capability Metrics */}
            <div className="flex flex-wrap items-center gap-5 text-xs font-mono pt-1">
              <span className="flex items-center gap-1.5 text-[var(--accent-primary)] font-semibold">
                <CheckCircle2 className="w-3.5 h-3.5" /> Explain ✓
              </span>
              <span className="flex items-center gap-1.5 text-[var(--accent-primary)] font-semibold">
                <CheckCircle2 className="w-3.5 h-3.5" /> Build ✓
              </span>
              <span className="flex items-center gap-1.5 text-[var(--accent-primary)] font-semibold">
                <CheckCircle2 className="w-3.5 h-3.5" /> Debug ✓
              </span>
              <span className="flex items-center gap-1.5 text-[var(--text-muted)]">
                ○ Teach
              </span>
            </div>

            {/* Actions */}
            <div className="flex items-center gap-4 pt-2">
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
                <FlaskConical className="w-4 h-4 text-[var(--accent-primary)]" /> Runnable Experiments
              </button>
            </div>

          </div>

          {/* Right Column: Clean SVG Decision Canvas */}
          <div className="lg:col-span-5 flex justify-center">
            <div className="w-full max-w-sm h-60 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] rounded-xl p-4 relative flex items-center justify-center overflow-hidden">
              <svg className="w-full h-full" viewBox="0 0 200 150">
                <line x1="0" y1="75" x2="200" y2="75" stroke="var(--border-strong)" strokeDasharray="3,3" />
                <line x1="100" y1="0" x2="100" y2="150" stroke="var(--border-strong)" strokeDasharray="3,3" />
                <line x1="20" y1="140" x2="180" y2="20" stroke="var(--text-primary)" strokeWidth="1.5" strokeDasharray="4,4" />
                <line x1="100" y1="80" x2="70" y2="40" stroke="var(--accent-primary)" strokeWidth="2" />
                <polygon points="67,37 76,41 72,48" fill="var(--accent-primary)" />
                <text x="58" y="30" fill="var(--accent-primary)" fontSize="11" fontFamily="monospace" fontWeight="bold">W</text>
                <circle cx="60" cy="40" r="3.5" fill="#EF4444" />
                <circle cx="80" cy="30" r="3.5" fill="#EF4444" />
                <circle cx="100" cy="25" r="3.5" fill="#EF4444" />
                <circle cx="70" cy="60" r="3.5" fill="#EF4444" />
                <circle cx="130" cy="110" r="3.5" fill="#10B981" />
                <circle cx="150" cy="100" r="3.5" fill="#10B981" />
                <circle cx="160" cy="120" r="3.5" fill="#10B981" />
                <circle cx="140" cy="130" r="3.5" fill="#10B981" />
              </svg>
            </div>
          </div>

        </div>
      </div>

      {/* Your Learning Snapshot */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h4 className="text-base font-bold font-heading text-[var(--text-primary)]">
            Curriculum Verification Snapshot
          </h4>
          <span className="text-xs text-[var(--text-muted)] font-mono">50 Modules Ready</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          
          {/* Card 1: Concepts Mastered */}
          <div className="panel-card p-5 space-y-3">
            <div className="flex items-center justify-between">
              <div className="p-2 rounded-lg bg-[var(--bg-elevated)] text-[var(--accent-primary)] border border-[var(--border-subtle)]">
                <BookOpen className="w-4 h-4" />
              </div>
              <span className="text-[11px] text-[var(--accent-primary)] font-mono font-bold">100% COVERAGE</span>
            </div>
            <div className="text-3xl font-bold font-heading text-[var(--text-primary)]">50</div>
            <div className="text-xs text-[var(--text-secondary)] font-medium">Verified Modules</div>
          </div>

          {/* Card 2: Experiments Run */}
          <div className="panel-card p-5 space-y-3">
            <div className="flex items-center justify-between">
              <div className="p-2 rounded-lg bg-[var(--bg-elevated)] text-[var(--accent-primary)] border border-[var(--border-subtle)]">
                <FlaskConical className="w-4 h-4" />
              </div>
              <span className="text-[11px] text-[var(--accent-primary)] font-mono font-bold">801 PASSING</span>
            </div>
            <div className="text-3xl font-bold font-heading text-[var(--text-primary)]">801</div>
            <div className="text-xs text-[var(--text-secondary)] font-medium">Pytest Test Suite</div>
          </div>

          {/* Card 3: Challenges Passed */}
          <div className="panel-card p-5 space-y-3">
            <div className="flex items-center justify-between">
              <div className="p-2 rounded-lg bg-[var(--bg-elevated)] text-[var(--accent-primary)] border border-[var(--border-subtle)]">
                <Code className="w-4 h-4" />
              </div>
              <span className="text-[11px] text-[var(--accent-primary)] font-mono font-bold">PURE PYTHON</span>
            </div>
            <div className="text-3xl font-bold font-heading text-[var(--text-primary)]">0</div>
            <div className="text-xs text-[var(--text-secondary)] font-medium">External Frameworks</div>
          </div>

          {/* Card 4: Video Masterclasses */}
          <div className="panel-card p-5 space-y-3">
            <div className="flex items-center justify-between">
              <div className="p-2 rounded-lg bg-[var(--bg-elevated)] text-[var(--accent-primary)] border border-[var(--border-subtle)]">
                <Brain className="w-4 h-4" />
              </div>
              <span className="text-[11px] text-[var(--accent-primary)] font-mono font-bold">60 FPS HD</span>
            </div>
            <div className="text-3xl font-bold font-heading text-[var(--text-primary)]">Riva</div>
            <div className="text-xs text-[var(--text-secondary)] font-medium">NVIDIA Speech Engine</div>
          </div>

        </div>
      </div>

      {/* Upcoming Path Grid */}
      <div className="space-y-4 pt-2">
        <div className="flex items-center justify-between">
          <h4 className="text-base font-bold font-heading text-[var(--text-primary)]">
            Explore All 50 Curriculum Modules
          </h4>
          <button onClick={onExploreRoadmap} className="text-xs text-[var(--accent-primary)] hover:underline font-mono">
            Explore full roadmap →
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          {MODULES.slice(1, 5).map((m) => (
            <div
              key={m.id}
              onClick={() => onOpenModule(m)}
              className="panel-card p-5 cursor-pointer flex flex-col justify-between hover:border-[var(--accent-primary)]"
            >
              <div className="space-y-2">
                <div className="text-xs font-mono text-[var(--accent-primary)] font-bold">MODULE #{m.id}</div>
                <h5 className="text-base font-bold font-heading text-[var(--text-primary)]">{m.title}</h5>
                <p className="text-xs text-[var(--text-secondary)] leading-relaxed line-clamp-2">{m.subtitle}</p>
              </div>
              <div className="pt-4 mt-4 border-t border-[var(--border-subtle)] flex items-center justify-between text-xs font-mono">
                <span className="text-[var(--accent-primary)] font-bold">Ready</span>
                <span className="text-[var(--text-muted)]">{m.estimatedMinutes} min</span>
              </div>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
};
