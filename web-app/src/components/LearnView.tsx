import React from 'react';
import { MODULES } from '../modulesData';
import type { ModuleData } from '../types';
import { ArrowRight, CheckCircle2, Clock, FlaskConical, BookOpen, Code, Brain, Lock, Star, Sparkles, Activity } from 'lucide-react';

interface LearnViewProps {
  onOpenModule: (m: ModuleData) => void;
  onExploreRoadmap: () => void;
}

export const LearnView: React.FC<LearnViewProps> = ({ onOpenModule, onExploreRoadmap }) => {
  const currentModule = MODULES[0]; // Module 001

  return (
    <div className="max-w-6xl mx-auto space-y-10 animate-fade-in pb-12">
      
      {/* Greeting Header (Clean, SVG Icon, No Unicode Emojis) */}
      <div className="space-y-1">
        <div className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-indigo-400" />
          <h2 className="text-2xl md:text-3xl font-bold text-slate-100 font-heading">
            Good morning, Mahendra
          </h2>
        </div>
        <p className="text-sm text-slate-400 font-medium max-w-xl">
          Continue your journey to becoming a frontier AI engineer.
        </p>
      </div>

      {/* Hero Panel: CURRENT MODULE */}
      <div className="panel-card p-6 md:p-8 relative overflow-hidden bg-[#121620] border-indigo-500/30 shadow-xl">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
          
          {/* Left Column: Context & Actions */}
          <div className="lg:col-span-7 space-y-5">
            
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="font-bold uppercase tracking-wider text-indigo-400">
                CURRENT MODULE // #{currentModule.id}
              </span>
              <span className="text-slate-400 flex items-center gap-1.5">
                <Clock className="w-3.5 h-3.5 text-slate-500" /> 90 min
              </span>
            </div>

            <div>
              <h3 className="text-2xl md:text-3xl font-bold text-slate-100 font-heading">
                001 · Perceptron From Scratch
              </h3>

              <p className="text-sm text-slate-300 leading-relaxed mt-2 max-w-lg">
                Learn how a linear model makes binary decisions, how it learns, and why it fails on XOR.
              </p>
            </div>

            {/* Progress Bar */}
            <div className="space-y-1.5 pt-1 max-w-md">
              <div className="flex justify-between text-xs font-mono text-slate-400">
                <span>Progress</span>
                <span className="text-slate-200 font-bold">68%</span>
              </div>
              <div className="w-full bg-[#090C10] rounded-full h-2 overflow-hidden border border-white/5">
                <div className="bg-indigo-500 h-full rounded-full w-[68%]" />
              </div>
            </div>

            {/* Capability Metrics */}
            <div className="flex flex-wrap items-center gap-5 text-xs font-mono pt-1 text-slate-400">
              <span className="flex items-center gap-1.5 text-emerald-400 font-medium">
                <CheckCircle2 className="w-3.5 h-3.5" /> Explain
              </span>
              <span className="flex items-center gap-1.5 text-emerald-400 font-medium">
                <CheckCircle2 className="w-3.5 h-3.5" /> Implement
              </span>
              <span className="flex items-center gap-1.5 text-amber-400 font-medium">
                <Activity className="w-3.5 h-3.5 text-amber-400 animate-pulse" /> Debug
              </span>
              <span className="flex items-center gap-1.5 text-slate-500">
                ○ Teach
              </span>
            </div>

            {/* Actions (Touch Target Size >= 44px) */}
            <div className="flex items-center gap-4 pt-2">
              <button
                onClick={() => onOpenModule(currentModule)}
                className="btn-indigo text-xs min-h-[44px] py-3 px-6 shadow-lg shadow-indigo-500/20"
              >
                Continue Learning <ArrowRight className="w-4 h-4" />
              </button>
              
              <button
                onClick={() => onOpenModule(currentModule)}
                className="btn-subtle text-xs min-h-[44px] py-3 px-5"
              >
                <FlaskConical className="w-4 h-4 text-indigo-400" /> Go to Experiments
              </button>
            </div>

          </div>

          {/* Right Column: Clean SVG Decision Canvas */}
          <div className="lg:col-span-5 flex justify-center">
            <div className="w-full max-w-sm h-56 bg-[#090C10] border border-white/10 rounded-2xl p-4 relative flex items-center justify-center overflow-hidden">
              <svg className="w-full h-full" viewBox="0 0 200 150">
                <line x1="0" y1="75" x2="200" y2="75" stroke="rgba(255,255,255,0.06)" strokeDasharray="3,3" />
                <line x1="100" y1="0" x2="100" y2="150" stroke="rgba(255,255,255,0.06)" strokeDasharray="3,3" />
                <line x1="20" y1="140" x2="180" y2="20" stroke="#F3F5F7" strokeWidth="1.5" strokeDasharray="4,4" />
                <line x1="100" y1="80" x2="70" y2="40" stroke="#6C8CFF" strokeWidth="2" />
                <polygon points="67,37 76,41 72,48" fill="#6C8CFF" />
                <text x="58" y="30" fill="#6C8CFF" fontSize="11" fontFamily="monospace" fontWeight="bold">W</text>
                <circle cx="60" cy="40" r="3.5" fill="#8B5CF6" />
                <circle cx="80" cy="30" r="3.5" fill="#8B5CF6" />
                <circle cx="100" cy="25" r="3.5" fill="#8B5CF6" />
                <circle cx="120" cy="20" r="3.5" fill="#8B5CF6" />
                <circle cx="70" cy="60" r="3.5" fill="#8B5CF6" />
                <circle cx="90" cy="50" r="3.5" fill="#8B5CF6" />
                <circle cx="130" cy="110" r="3.5" fill="#32D583" />
                <circle cx="150" cy="100" r="3.5" fill="#32D583" />
                <circle cx="160" cy="120" r="3.5" fill="#32D583" />
                <circle cx="140" cy="130" r="3.5" fill="#32D583" />
                <circle cx="170" cy="90" r="3.5" fill="#32D583" />
              </svg>
            </div>
          </div>

        </div>
      </div>

      {/* Your Learning Snapshot */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h4 className="text-base font-bold text-slate-100 font-heading">
            Your Learning Snapshot
          </h4>
          <span className="text-xs text-slate-500 font-mono">This Week</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          
          {/* Card 1: Concepts Mastered */}
          <div className="panel-card p-5 space-y-3">
            <div className="flex items-center justify-between">
              <div className="p-2.5 rounded-xl bg-purple-500/10 text-purple-400 border border-purple-500/20">
                <BookOpen className="w-4 h-4" />
              </div>
              <span className="text-[11px] text-emerald-400 font-mono font-bold">+2 this week</span>
            </div>
            <div className="text-3xl font-bold text-slate-100 font-heading">12</div>
            <div className="text-xs text-slate-400 font-medium">Concepts Mastered</div>
          </div>

          {/* Card 2: Experiments Run */}
          <div className="panel-card p-5 space-y-3">
            <div className="flex items-center justify-between">
              <div className="p-2.5 rounded-xl bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                <FlaskConical className="w-4 h-4" />
              </div>
              <span className="text-[11px] text-emerald-400 font-mono font-bold">+3 this week</span>
            </div>
            <div className="text-3xl font-bold text-slate-100 font-heading">7</div>
            <div className="text-xs text-slate-400 font-medium">Experiments Run</div>
          </div>

          {/* Card 3: Challenges Passed */}
          <div className="panel-card p-5 space-y-3">
            <div className="flex items-center justify-between">
              <div className="p-2.5 rounded-xl bg-amber-500/10 text-amber-400 border border-amber-500/20">
                <Code className="w-4 h-4" />
              </div>
              <span className="text-[11px] text-emerald-400 font-mono font-bold">+1 this week</span>
            </div>
            <div className="text-3xl font-bold text-slate-100 font-heading">2</div>
            <div className="text-xs text-slate-400 font-medium">Challenges Passed</div>
          </div>

          {/* Card 4: Misconceptions Corrected */}
          <div className="panel-card p-5 space-y-3">
            <div className="flex items-center justify-between">
              <div className="p-2.5 rounded-xl bg-blue-500/10 text-blue-400 border border-blue-500/20">
                <Brain className="w-4 h-4" />
              </div>
              <span className="text-[11px] text-emerald-400 font-mono font-bold">+2 this week</span>
            </div>
            <div className="text-3xl font-bold text-slate-100 font-heading">4</div>
            <div className="text-xs text-slate-400 font-medium">Misconceptions Corrected</div>
          </div>

        </div>
      </div>

      {/* Upcoming Path Grid */}
      <div className="space-y-4 pt-2">
        <div className="flex items-center justify-between">
          <h4 className="text-base font-bold text-slate-100 font-heading">
            Upcoming Path
          </h4>
          <button onClick={onExploreRoadmap} className="text-xs text-indigo-400 hover:underline font-mono">
            Explore full roadmap →
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          
          {/* Node 002 */}
          <div
            onClick={() => onOpenModule(MODULES[1])}
            className="panel-card-active p-5 cursor-pointer flex flex-col justify-between"
          >
            <div className="space-y-2">
              <div className="text-xs font-mono text-indigo-400 font-bold">002</div>
              <h5 className="text-base font-bold text-slate-100 font-heading">Loss Functions</h5>
              <p className="text-xs text-slate-400 leading-relaxed">Why we need a loss function to optimize.</p>
            </div>
            <div className="pt-4 mt-4 border-t border-white/5 flex items-center justify-between text-xs font-mono">
              <span className="text-indigo-400 font-bold">Next</span>
              <span className="text-slate-500">90 min</span>
            </div>
          </div>

          {/* Node 003 */}
          <div
            onClick={() => onOpenModule(MODULES[2])}
            className="panel-card p-5 cursor-pointer opacity-75 hover:opacity-100 transition-all flex flex-col justify-between"
          >
            <div className="space-y-2">
              <div className="text-xs font-mono text-slate-500 flex items-center justify-between">
                <span>003</span>
                <Lock className="w-3.5 h-3.5" />
              </div>
              <h5 className="text-base font-bold text-slate-200 font-heading">Gradient Descent</h5>
              <p className="text-xs text-slate-400 leading-relaxed">How to minimize loss using gradients.</p>
            </div>
            <div className="pt-4 mt-4 border-t border-white/5 flex items-center justify-between text-xs font-mono text-slate-500">
              <span>Locked</span>
              <span>90 min</span>
            </div>
          </div>

          {/* Node 004 */}
          <div
            onClick={() => onOpenModule(MODULES[3])}
            className="panel-card p-5 cursor-pointer opacity-75 hover:opacity-100 transition-all flex flex-col justify-between"
          >
            <div className="space-y-2">
              <div className="text-xs font-mono text-slate-500 flex items-center justify-between">
                <span>004</span>
                <Lock className="w-3.5 h-3.5" />
              </div>
              <h5 className="text-base font-bold text-slate-200 font-heading">NumPy Neural Net</h5>
              <p className="text-xs text-slate-400 leading-relaxed">Build a simple neural network from scratch.</p>
            </div>
            <div className="pt-4 mt-4 border-t border-white/5 flex items-center justify-between text-xs font-mono text-slate-500">
              <span>Locked</span>
              <span>120 min</span>
            </div>
          </div>

          {/* Mini-capstone Node */}
          <div
            onClick={() => onOpenModule(MODULES[4])}
            className="panel-card p-5 cursor-pointer border-amber-500/30 flex flex-col justify-between bg-amber-950/10"
          >
            <div className="space-y-2">
              <div className="text-xs font-mono text-amber-400 flex items-center justify-between font-bold">
                <span>Mini-capstone</span>
                <Star className="w-3.5 h-3.5 fill-current" />
              </div>
              <h5 className="text-base font-bold text-slate-100 font-heading">Binary Classifier</h5>
              <p className="text-xs text-slate-400 leading-relaxed">Combine everything to build and evaluate a classifier.</p>
            </div>
            <div className="pt-4 mt-4 border-t border-white/5 flex items-center justify-between text-xs font-mono text-slate-400">
              <span className="text-amber-400 font-bold">Capstone</span>
              <span>2–4 hours</span>
            </div>
          </div>

        </div>
      </div>

    </div>
  );
};
