import React, { useState } from 'react';
import type { ModuleData } from '../types';
import { CodePlayground } from './CodePlayground';
import { X, CheckCircle2, BookOpen, Code, Brain, Target, Award } from 'lucide-react';

interface ModuleModalProps {
  module: ModuleData | null;
  onClose: () => void;
}

export const ModuleModal: React.FC<ModuleModalProps> = ({ module, onClose }) => {
  if (!module) return null;
  const [activeTab, setActiveTab] = useState<'overview' | 'math' | 'code' | 'challenge'>('overview');

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md animate-fade-in">
      <div className="bg-slate-900 border border-slate-700/80 rounded-2xl max-w-3xl w-full max-h-[90vh] overflow-hidden flex flex-col shadow-2xl shadow-cyan-950/50">
        
        {/* Modal Header */}
        <div className="p-6 border-b border-slate-800 flex items-start justify-between bg-slate-900/60">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="font-mono text-xs font-bold text-cyan-400 bg-cyan-950 px-2.5 py-1 rounded-md border border-cyan-800/60">
                MODULE {module.id}
              </span>
              <span className="text-xs text-emerald-400 bg-emerald-950/60 px-2.5 py-1 rounded-md border border-emerald-800/60 font-mono font-medium flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" /> {module.testCount}/{module.testCount} Tests Passing
              </span>
            </div>
            <h2 className="text-2xl font-extrabold text-slate-100">{module.title}</h2>
            <p className="text-sm text-slate-400 font-medium">{module.subtitle}</p>
          </div>
          
          <button
            onClick={onClose}
            className="p-2 text-slate-400 hover:text-white bg-slate-800/60 hover:bg-slate-800 rounded-xl transition-all"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Navigation Tabs */}
        <div className="flex items-center gap-1 px-6 pt-3 bg-slate-950/40 border-b border-slate-800 font-medium text-sm">
          <button
            onClick={() => setActiveTab('overview')}
            className={`px-4 py-2.5 rounded-t-xl flex items-center gap-2 transition-all ${
              activeTab === 'overview'
                ? 'bg-slate-900 text-cyan-400 border-t-2 border-cyan-400 font-semibold'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <BookOpen className="w-4 h-4" /> Overview & Model
          </button>
          <button
            onClick={() => setActiveTab('math')}
            className={`px-4 py-2.5 rounded-t-xl flex items-center gap-2 transition-all ${
              activeTab === 'math'
                ? 'bg-slate-900 text-violet-400 border-t-2 border-violet-400 font-semibold'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Brain className="w-4 h-4" /> Mathematics
          </button>
          <button
            onClick={() => setActiveTab('code')}
            className={`px-4 py-2.5 rounded-t-xl flex items-center gap-2 transition-all ${
              activeTab === 'code'
                ? 'bg-slate-900 text-emerald-400 border-t-2 border-emerald-400 font-semibold'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Code className="w-4 h-4" /> Python Code
          </button>
          <button
            onClick={() => setActiveTab('challenge')}
            className={`px-4 py-2.5 rounded-t-xl flex items-center gap-2 transition-all ${
              activeTab === 'challenge'
                ? 'bg-slate-900 text-amber-400 border-t-2 border-amber-400 font-semibold'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Target className="w-4 h-4" /> Challenge
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto flex-1 space-y-6">
          
          {activeTab === 'overview' && (
            <div className="space-y-6">
              <div>
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">Executive Summary</h4>
                <p className="text-slate-300 leading-relaxed text-sm">{module.overview}</p>
              </div>

              <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800">
                <h4 className="text-xs font-bold uppercase tracking-wider text-cyan-400 mb-2 flex items-center gap-2">
                  <Brain className="w-4 h-4" /> Intuitive Mental Model
                </h4>
                <p className="text-slate-300 text-sm italic">"{module.mentalModelSummary}"</p>
              </div>

              <div>
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">Key Topics Covered</h4>
                <div className="flex flex-wrap gap-2">
                  {module.topics.map((t, idx) => (
                    <span key={idx} className="bg-slate-800/80 text-cyan-300 text-xs px-3 py-1 rounded-lg border border-slate-700">
                      {t}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'math' && (
            <div className="space-y-4">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">Formal Derivation Highlight</h4>
              <div className="bg-slate-950 p-4 rounded-xl border border-violet-900/40 text-violet-300 font-mono text-sm overflow-x-auto">
                <code>{module.mathHighlight}</code>
              </div>
              <p className="text-xs text-slate-400">
                Full derivations and proofs available in <code className="text-cyan-400">03-mathematics.md</code> inside the module directory.
              </p>
            </div>
          )}

          {activeTab === 'code' && (
            <div className="space-y-4">
              <CodePlayground initialCode={module.codeSnippet} moduleTitle={module.title} />
            </div>
          )}

          {activeTab === 'challenge' && (
            <div className="space-y-4">
              <div className="bg-amber-950/20 border border-amber-800/40 p-4 rounded-xl">
                <h4 className="text-sm font-bold text-amber-400 flex items-center gap-2 mb-2">
                  <Award className="w-4 h-4" /> {module.engineeringChallengeTitle}
                </h4>
                <p className="text-xs text-amber-200/90 leading-relaxed">
                  {module.engineeringChallengeGoal}
                </p>
              </div>
              <p className="text-xs text-slate-400">
                Complete challenge details in <code className="text-amber-400">07-engineering-challenge.md</code>. Test your solution against <code className="text-emerald-400">tests/test_{module.id}_*.py</code>.
              </p>
            </div>
          )}

        </div>

        {/* Modal Footer */}
        <div className="p-4 bg-slate-950/80 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
          <div>
            Sources: {module.sources.map(s => `${s.title} (${s.authors}, ${s.year})`).join('; ')}
          </div>
          <button
            onClick={onClose}
            className="btn-primary text-xs py-1.5 px-4"
          >
            Close Spec
          </button>
        </div>

      </div>
    </div>
  );
};
