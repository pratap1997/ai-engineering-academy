import React from 'react';
import { MODULES } from '../modulesData';
import type { ModuleData } from '../types';
import { ArrowRight, CheckCircle2, Clock, FlaskConical, BookOpen, Code, Brain, Lock, Star } from 'lucide-react';

interface LearnViewProps {
  onOpenModule: (m: ModuleData) => void;
  onExploreRoadmap: () => void;
}

export const LearnView: React.FC<LearnViewProps> = ({ onOpenModule, onExploreRoadmap }) => {
  const currentModule = MODULES[0]; // Module 001

  return (
    <div className="space-y-8 animate-fade-in">
      
      {/* Greeting Header */}
      <div>
        <h2 className="text-2xl font-bold text-slate-100 font-heading flex items-center gap-2">
          Good morning, Mahendra 👋
        </h2>
        <p className="text-xs text-slate-400 mt-1 font-medium">
          Continue your journey to becoming an AI engineer.
        </p>
      </div>

      {/* Hero Panel: CURRENT MODULE */}
      <div className="panel-card p-6 md:p-8 relative overflow-hidden bg-gradient-to-br from-[#11151D] via-[#11151D] to-[#171C26] border-indigo-500/30">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
          
          {/* Left Text Column */}
          <div className="lg:col-span-7 space-y-4">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="font-bold uppercase tracking-wider text-slate-400">
                CURRENT MODULE
              </span>
              <span className="text-slate-400 flex items-center gap-1">
                <Clock className="w-3.5 h-3.5 text-slate-500" /> 90 min
              </span>
            </div>

            <h3 className="text-2xl font-bold text-slate-100 font-heading">
              001 · Perceptron From Scratch
            </h3>

            <p className="text-xs md:text-sm text-slate-300 leading-relaxed">
              Learn how a linear model makes binary decisions, how it learns, and why it fails on XOR.
            </p>

            {/* Progress Bar */}
            <div className="space-y-1.5 pt-2">
              <div className="flex justify-between text-xs font-mono text-slate-400">
                <span>Progress</span>
                <span className="text-slate-200 font-bold">68%</span>
              </div>
              <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-white/5">
                <div className="bg-indigo-500 h-full rounded-full w-[68%]" />
              </div>
            </div>

            {/* Capability Row */}
            <div className="flex flex-wrap items-center gap-4 text-xs font-mono pt-1">
              <span className="flex items-center gap-1.5 text-emerald-400">
                <CheckCircle2 className="w-3.5 h-3.5" /> Explain
              </span>
              <span className="flex items-center gap-1.5 text-emerald-400">
                <CheckCircle2 className="w-3.5 h-3.5" /> Implement
              </span>
              <span className="flex items-center gap-1.5 text-amber-400">
                <div className="w-3 h-3 rounded-full border-2 border-amber-400 border-t-transparent animate-spin" /> Debug
              </span>
              <span className="flex items-center gap-1.5 text-slate-500">
                ○ Teach
              </span>
            </div>

            {/* Actions */}
            <div className="flex items-center gap-3 pt-3">
              <button
                onClick={() => onOpenModule(currentModule)}
                className="btn-indigo text-xs py-2.5 px-5 shadow-lg shadow-indigo-500/20"
              >
                Continue Learning <ArrowRight className="w-4 h-4" />
              </button>
              
              <button
                onClick={() => onOpenModule(currentModule)}
                className="btn-subtle text-xs py-2.5 px-4"
              >
                <FlaskConical className="w-3.5 h-3.5 text-indigo-400" /> Go to Experiments
              </button>
            </div>

          </div>

          {/* Right SVG Canvas Decision Boundary Visualizer */}
          <div className="lg:col-span-5 flex justify-center">
            <div className="w-full max-w-sm h-56 bg-[#0B0D12] border border-white/10 rounded-2xl p-4 relative flex items-center justify-center overflow-hidden">
              <svg className="w-full h-full" viewBox="0 0 200 150">
                <line x1="0" y1="75" x2="200" y2="75" stroke="rgba(255,255,255,0.05)" strokeDasharray="3,3" />
                <line x1="100" y1="0" x2="100" y2="150" stroke="rgba(255,255,255,0.05)" strokeDasharray="3,3" />
                <line x1="20" y1="140" x2="180" y2="20" stroke="#F3F5F7" strokeWidth="1.5" strokeDasharray="4,4" />
                <line x1="100" y1="80" x2="70" y2="40" stroke="#6C8CFF" strokeWidth="2" />
                <polygon points="67,37 76,41 72,48" fill="#6C8CFF" />
                <text x="60" y="32" fill="#6C8CFF" fontSize="10" fontFamily="monospace" fontWeight="bold">W</text>
                <circle cx="60" cy="40" r="3.5" fill="#8B5CF6" />
                <circle cx="80" cy="30" r="3.5" fill="#8B5CF6" />
                <circle cx="100" cy="25" r="3.5" fill="#8B5CF6" />
                <circle cx="120" cy="20" r="3.5" fill="#8B5CF6" />
                <circle cx="70" cy="60" r="3.5" fill="#8B5CF6" />
                <circle cx="90" cy="50" r="3.5" fill="#8B5CF6" />
                <circle cx="110" cy="45" r="3.5" fill="#8B5CF6" />
                <circle cx="140" cy="30" r="3.5" fill="#8B5CF6" />
                <circle cx="130" cy="110" r="3.5" fill="#32D583" />
                <circle cx="150" cy="100" r="3.5" fill="#32D583" />
                <circle cx="160" cy="120" r="3.5" fill="#32D583" />
                <circle cx="140" cy="130" r="3.5" fill="#32D583" />
                <circle cx="170" cy="90" r="3.5" fill="#32D583" />
                <circle cx="110" cy="135" r="3.5" fill="#32D583" />
                <circle cx="120" cy="120" r="3.5" fill="#32D583" />
              </svg>
            </div>
          </div>

        </div>
      </div>

      {/* Your Learning Snapshot Grid */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-bold text-slate-200 font-heading">
            Your Learning Snapshot
          </h4>
          <span className="text-xs text-slate-500 font-mono">This Week</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="panel-card p-4 space-y-2">
            <div className="flex items-center justify-between">
              <div className="p-2 rounded-xl bg-purple-500/10 text-purple-400 border border-purple-500/20">
                <BookOpen className="w-4 h-4" />
              </div>
              <span className="text-[11px] text-emerald-400 font-mono font-bold">+2 this week</span>
            </div>
            <div className="text-2xl font-bold text-slate-100 font-heading">12</div>
            <div className="text-xs text-slate-400 font-medium">Concepts Mastered</div>
          </div>

          <div className="panel-card p-4 space-y-2">
            <div className="flex items-center justify-between">
              <div className="p-2 rounded-xl bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                <FlaskConical className="w-4 h-4" />
              </div>
              <span className="text-[11px] text-emerald-400 font-mono font-bold">+3 this week</span>
            </div>
            <div className="text-2xl font-bold text-slate-100 font-heading">7</div>
            <div className="text-xs text-slate-400 font-medium">Experiments Run</div>
          </div>

          <div className="panel-card p-4 space-y-2">
            <div className="flex items-center justify-between">
              <div className="p-2 rounded-xl bg-amber-500/10 text-amber-400 border border-amber-500/20">
                <Code className="w-4 h-4" />
              </div>
              <span className="text-[11px] text-emerald-400 font-mono font-bold">+1 this week</span>
            </div>
            <div className="text-2xl font-bold text-slate-100 font-heading">2</div>
            <div className="text-xs text-slate-400 font-medium">Challenges Passed</div>
          </div>

          <div className="panel-card p-4 space-y-2">
            <div className="flex items-center justify-between">
              <div className="p-2 rounded-xl bg-blue-500/10 text-blue-400 border border-blue-500/20">
                <Brain className="w-4 h-4" />
              </div>
              <span className="text-[11px] text-emerald-400 font-mono font-bold">+2 this week</span>
            </div>
            <div className="text-2xl font-bold text-slate-100 font-heading">4</div>
            <div className="text-xs text-slate-400 font-medium">Misconceptions Corrected</div>
          </div>
        </div>
      </div>

      {/* Upcoming Path Nodes */}
      <div className="space-y-4 pt-2">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-bold text-slate-200 font-heading">
            Upcoming Path
          </h4>
          <button onClick={onExploreRoadmap} className="text-xs text-indigo-400 hover:underline font-mono">
            Explore full roadmap →
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div
            onClick={() => onOpenModule(MODULES[1])}
            className="panel-card-active p-4 cursor-pointer flex flex-col justify-between"
          >
            <div className="space-y-2">
              <div className="text-xs font-mono text-indigo-400 font-bold">002</div>
              <h5 className="text-sm font-bold text-slate-100 font-heading">Loss Functions</h5>
              <p className="text-xs text-slate-400 line-clamp-2">Why we need a loss function to optimize.</p>
            </div>
            <div className="pt-4 mt-4 border-t border-white/5 flex items-center justify-between text-[11px] font-mono">
              <span className="text-indigo-400 font-bold">Next</span>
              <span className="text-slate-500">90 min</span>
            </div>
          </div>

          <div
            onClick={() => onOpenModule(MODULES[2])}
            className="panel-card p-4 cursor-pointer opacity-70 hover:opacity-100 transition-opacity flex flex-col justify-between"
          >
            <div className="space-y-2">
              <div className="text-xs font-mono text-slate-500 flex items-center justify-between">
                <span>003</span>
                <Lock className="w-3.5 h-3.5" />
              </div>
              <h5 className="text-sm font-bold text-slate-200 font-heading">Gradient Descent</h5>
              <p className="text-xs text-slate-400 line-clamp-2">How to minimize loss using gradients.</p>
            </div>
            <div className="pt-4 mt-4 border-t border-white/5 flex items-center justify-between text-[11px] font-mono text-slate-500">
              <span>Locked</span>
              <span>90 min</span>
            </div>
          </div>

          <div
            onClick={() => onOpenModule(MODULES[3])}
            className="panel-card p-4 cursor-pointer opacity-70 hover:opacity-100 transition-opacity flex flex-col justify-between"
          >
            <div className="space-y-2">
              <div className="text-xs font-mono text-slate-500 flex items-center justify-between">
                <span>004</span>
                <Lock className="w-3.5 h-3.5" />
              </div>
              <h5 className="text-sm font-bold text-slate-200 font-heading">NumPy Neural Net</h5>
              <p className="text-xs text-slate-400 line-clamp-2">Build a simple neural network from scratch.</p>
            </div>
            <div className="pt-4 mt-4 border-t border-white/5 flex items-center justify-between text-[11px] font-mono text-slate-500">
              <span>Locked</span>
              <span>120 min</span>
            </div>
          </div>

          <div
            onClick={() => onOpenModule(MODULES[4])}
            className="panel-card p-4 cursor-pointer border-amber-500/30 flex flex-col justify-between bg-amber-950/10"
          >
            <div className="space-y-2">
              <div className="text-xs font-mono text-amber-400 flex items-center justify-between font-bold">
                <span>Mini-capstone</span>
                <Star className="w-3.5 h-3.5 fill-current" />
              </div>
              <h5 className="text-sm font-bold text-slate-100 font-heading">Binary Classifier</h5>
              <p className="text-xs text-slate-400 line-clamp-2">Combine everything to build and evaluate a classifier.</p>
            </div>
            <div className="pt-4 mt-4 border-t border-white/5 flex items-center justify-between text-[11px] font-mono text-slate-400">
              <span className="text-amber-400 font-bold">Capstone</span>
              <span>2–4 hours</span>
            </div>
          </div>
        </div>
      </div>

    </div>
  );
};
