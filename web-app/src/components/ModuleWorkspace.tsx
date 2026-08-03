import React, { useState } from 'react';
import type { ModuleData } from '../types';
import { KaTeXRenderer } from './KaTeXRenderer';
import { CodePlayground } from './CodePlayground';
import { ArrowLeft, CheckCircle2, Sparkles, AlertCircle, Send } from 'lucide-react';

interface ModuleWorkspaceProps {
  module: ModuleData;
  onBack: () => void;
}

type ArtifactKey = 'overview' | 'mentalModel' | 'math' | 'code' | 'experiments' | 'applications' | 'challenge' | 'assessment' | 'references';

export const ModuleWorkspace: React.FC<ModuleWorkspaceProps> = ({ module, onBack }) => {
  const [activeArtifact, setActiveArtifact] = useState<ArtifactKey>('overview');
  const [mentorInput, setMentorInput] = useState('');
  const [mentorMessages, setMentorMessages] = useState<Array<{ role: 'user' | 'assistant'; text: string }>>([
    {
      role: 'assistant',
      text: `Hello Mahendra! I am your AI Mentor for Module ${module.id} (${module.title}). I can explain math derivations, help debug code, or give hints for the challenge.`,
    },
  ]);

  const artifactsList: Array<{ key: ArtifactKey; label: string; num: string; status: 'complete' | 'current' | 'upcoming' }> = [
    { key: 'overview', label: 'Overview', num: '01', status: 'complete' },
    { key: 'mentalModel', label: 'Mental Model', num: '02', status: 'complete' },
    { key: 'math', label: 'Mathematics', num: '03', status: 'complete' },
    { key: 'code', label: 'Implementation', num: '04', status: 'current' },
    { key: 'experiments', label: 'Experiments', num: '05', status: 'upcoming' },
    { key: 'applications', label: 'Applications', num: '06', status: 'upcoming' },
    { key: 'challenge', label: 'Engineering Challenge', num: '07', status: 'upcoming' },
    { key: 'assessment', label: 'Assessment', num: '08', status: 'upcoming' },
    { key: 'references', label: 'References', num: '09', status: 'upcoming' },
  ];

  const handleSendMentor = (text: string) => {
    if (!text.trim()) return;
    const userMsg = text.trim();
    setMentorMessages((prev) => [...prev, { role: 'user', text: userMsg }]);
    setMentorInput('');

    setTimeout(() => {
      let reply = `Great question regarding ${module.title}. In ${activeArtifact}, the key mathematical principle is ensuring convergence through precise parameter updates without exceeding bound conditions.`;
      if (activeArtifact === 'challenge') {
        reply = `[Mentor Hint]: Observe how removing the bias term forces the decision hyper-plane through the origin (0,0).`;
      }
      setMentorMessages((prev) => [...prev, { role: 'assistant', text: reply }]);
    }, 600);
  };

  return (
    <div className="flex flex-col h-[calc(100vh-60px)] bg-[#0B0D12] overflow-hidden select-none">
      
      {/* Top Header Bar */}
      <div className="h-12 shrink-0 bg-[#11151D] border-b border-white/5 px-6 flex items-center justify-between text-xs">
        
        <div className="flex items-center gap-3">
          <button
            onClick={onBack}
            className="text-slate-400 hover:text-slate-200 flex items-center gap-1.5 font-mono font-medium transition-colors"
          >
            <ArrowLeft className="w-4 h-4" /> Back to Roadmap
          </button>

          <span className="text-slate-600">/</span>

          <span className="font-bold text-slate-200 font-heading">
            #{module.id} · {module.title}
          </span>
        </div>

        <div className="flex items-center gap-4 font-mono text-[11px]">
          <span className="text-slate-400">Progress: 68%</span>
          <div className="w-24 bg-slate-900 h-1.5 rounded-full overflow-hidden border border-white/5">
            <div className="bg-indigo-500 h-full w-[68%]" />
          </div>
        </div>

      </div>

      {/* 3-Column Workspace Main Layout */}
      <div className="flex-1 flex overflow-hidden">
        
        {/* Left Column: 9 Artifacts Navigation */}
        <aside className="w-64 shrink-0 bg-[#0B0D12] border-r border-white/5 p-4 flex flex-col justify-between overflow-y-auto hidden md:flex">
          <div className="space-y-1">
            <div className="px-3 text-[10px] font-mono font-bold uppercase tracking-wider text-slate-500 mb-3">
              MODULE PATH
            </div>

            {artifactsList.map((art) => {
              const isActive = activeArtifact === art.key;
              return (
                <button
                  key={art.key}
                  onClick={() => setActiveArtifact(art.key)}
                  className={`w-full px-3 py-2 rounded-xl text-xs font-semibold flex items-center justify-between transition-all ${
                    isActive
                      ? 'bg-indigo-600/20 text-indigo-400 border border-indigo-500/30 font-bold'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <span className="font-mono text-[10px] text-slate-500">{art.num}</span>
                    <span>{art.label}</span>
                  </div>

                  {art.status === 'complete' && <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />}
                  {art.status === 'current' && <div className="w-2 h-2 rounded-full bg-indigo-400" />}
                  {art.status === 'upcoming' && <span className="text-slate-600 font-mono text-[10px]">○</span>}
                </button>
              );
            })}
          </div>
        </aside>

        {/* Center Content Column (Max reading width 760px) */}
        <main className="flex-1 overflow-y-auto p-6 md:p-10 bg-[#0B0D12] flex justify-center">
          <div className="w-full max-w-[760px] space-y-8">
            
            {/* 01 Overview */}
            {activeArtifact === 'overview' && (
              <div className="space-y-6 animate-fade-in">
                <div>
                  <span className="text-xs font-mono font-bold text-indigo-400 uppercase tracking-wider">
                    01 // OVERVIEW & MOTIVATION
                  </span>
                  <h2 className="text-2xl font-bold text-slate-100 font-heading mt-1">
                    {module.title}
                  </h2>
                </div>

                <div className="prose prose-invert prose-slate text-sm text-slate-300 leading-relaxed space-y-4">
                  <p>{module.overview}</p>
                </div>

                {/* Math Highlight */}
                <div className="panel-card p-5 space-y-2">
                  <div className="text-xs font-mono text-slate-400 font-bold uppercase">
                    CANONICAL EQUATION
                  </div>
                  <KaTeXRenderer math={module.mathHighlight} block />
                </div>
              </div>
            )}

            {/* 02 Mental Model */}
            {activeArtifact === 'mentalModel' && (
              <div className="space-y-6 animate-fade-in">
                <div>
                  <span className="text-xs font-mono font-bold text-indigo-400 uppercase tracking-wider">
                    02 // GEOMETRIC MENTAL MODEL
                  </span>
                  <h2 className="text-2xl font-bold text-slate-100 font-heading mt-1">
                    Geometric Decision Boundaries
                  </h2>
                </div>

                <div className="prose prose-invert prose-slate text-sm text-slate-300 leading-relaxed space-y-4">
                  <p>{module.mentalModelSummary}</p>
                </div>

                {/* SVG Visual Canvas */}
                <div className="panel-card p-6 flex flex-col items-center justify-center space-y-4">
                  <div className="w-full max-w-md h-52 bg-[#0B0D12] rounded-xl border border-white/10 p-4 flex items-center justify-center">
                    <svg className="w-full h-full" viewBox="0 0 200 150">
                      <line x1="0" y1="75" x2="200" y2="75" stroke="rgba(255,255,255,0.08)" strokeDasharray="3,3" />
                      <line x1="100" y1="0" x2="100" y2="150" stroke="rgba(255,255,255,0.08)" strokeDasharray="3,3" />
                      <line x1="20" y1="130" x2="180" y2="20" stroke="#6C8CFF" strokeWidth="2" />
                      <circle cx="50" cy="40" r="4" fill="#8B5CF6" />
                      <circle cx="80" cy="30" r="4" fill="#8B5CF6" />
                      <circle cx="140" cy="110" r="4" fill="#32D583" />
                      <circle cx="160" cy="120" r="4" fill="#32D583" />
                    </svg>
                  </div>
                  <span className="text-xs text-slate-400 font-mono">
                    Figure 1: 2D linear hyper-plane dividing Class 0 and Class 1.
                  </span>
                </div>
              </div>
            )}

            {/* 03 Mathematics */}
            {activeArtifact === 'math' && (
              <div className="space-y-6 animate-fade-in">
                <div>
                  <span className="text-xs font-mono font-bold text-indigo-400 uppercase tracking-wider">
                    03 // FORMAL MATHEMATICAL DERIVATION
                  </span>
                  <h2 className="text-2xl font-bold text-slate-100 font-heading mt-1">
                    Formal Derivation & Proofs
                  </h2>
                </div>

                <div className="space-y-4 text-sm text-slate-300 leading-relaxed">
                  <p>
                    The activation output is computed by passing the linear inner product through the step function:
                  </p>
                  
                  <KaTeXRenderer math="y = \operatorname{step}(w^T x + b)" block />

                  <p>
                    Parameter updates occur when prediction error is non-zero:
                  </p>

                  <KaTeXRenderer math="w_{t+1} = w_t + \eta \cdot (y - \hat{y}) \cdot x" block />
                </div>
              </div>
            )}

            {/* 04 Implementation (Pyodide Playground) */}
            {activeArtifact === 'code' && (
              <div className="space-y-6 animate-fade-in">
                <div>
                  <span className="text-xs font-mono font-bold text-indigo-400 uppercase tracking-wider">
                    04 // FROM-SCRATCH PYTHON PLAYGROUND
                  </span>
                  <h2 className="text-2xl font-bold text-slate-100 font-heading mt-1">
                    Interactive Implementation Code
                  </h2>
                </div>

                <CodePlayground moduleTitle={module.title} initialCode={module.codeSnippet} />
              </div>
            )}

            {/* Default Fallback for other tabs */}
            {['experiments', 'applications', 'challenge', 'assessment', 'references'].includes(activeArtifact) && (
              <div className="space-y-6 animate-fade-in">
                <div>
                  <span className="text-xs font-mono font-bold text-indigo-400 uppercase tracking-wider">
                    07 // ENGINEERING CHALLENGE & EXPERIMENTS
                  </span>
                  <h2 className="text-2xl font-bold text-slate-100 font-heading mt-1 capitalize">
                    {activeArtifact}
                  </h2>
                </div>

                <div className="panel-card p-6 space-y-4">
                  <p className="text-sm text-slate-300 leading-relaxed">
                    Build a binary classifier that trains on non-linearly separable inputs to observe XOR failure.
                  </p>

                  <div className="bg-amber-950/20 border border-amber-800/40 p-4 rounded-xl text-xs text-amber-300/90 space-y-1">
                    <div className="font-bold text-amber-400 flex items-center gap-1.5">
                      <AlertCircle className="w-4 h-4" /> Mentor Mode: Hints Only Active
                    </div>
                    <div>Solutions are hidden during challenge evaluation. Use the AI mentor on the right for hints.</div>
                  </div>
                </div>
              </div>
            )}

          </div>
        </main>

        {/* Right Column: Contextual AI Mentor */}
        <aside className="w-80 shrink-0 bg-[#0B0D12] border-l border-white/5 p-4 flex flex-col justify-between hidden xl:flex">
          <div className="space-y-4 flex-1 flex flex-col justify-between">
            
            <div className="space-y-3">
              <div className="flex items-center gap-2 border-b border-white/5 pb-3">
                <Sparkles className="w-4 h-4 text-indigo-400" />
                <h3 className="text-xs font-bold text-slate-100 font-heading">
                  Contextual AI Mentor
                </h3>
              </div>

              {/* Chat Thread */}
              <div className="space-y-3 max-h-[450px] overflow-y-auto pr-1">
                {mentorMessages.map((msg, idx) => (
                  <div
                    key={idx}
                    className={`p-3 rounded-xl text-xs font-medium leading-relaxed ${
                      msg.role === 'assistant'
                        ? 'bg-[#11151D] border border-white/5 text-slate-200'
                        : 'bg-indigo-600/20 border border-indigo-500/30 text-indigo-200 ml-4'
                    }`}
                  >
                    {msg.text}
                  </div>
                ))}
              </div>
            </div>

            {/* Input Bar */}
            <div className="relative pt-2">
              <input
                type="text"
                value={mentorInput}
                onChange={(e) => setMentorInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSendMentor(mentorInput)}
                placeholder="Ask about this section..."
                className="w-full bg-[#11151D] border border-white/10 rounded-xl pl-3 pr-9 py-2 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500/50"
              />
              <button
                onClick={() => handleSendMentor(mentorInput)}
                className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-indigo-400 hover:text-indigo-300"
              >
                <Send className="w-3.5 h-3.5" />
              </button>
            </div>

          </div>
        </aside>

      </div>

    </div>
  );
};
