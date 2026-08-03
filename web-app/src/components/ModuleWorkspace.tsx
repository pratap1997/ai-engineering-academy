import React, { useState } from 'react';
import type { ModuleData } from '../types';
import { KaTeXRenderer } from './KaTeXRenderer';
import { CodePlayground } from './CodePlayground';
import { ArrowLeft, CheckCircle2, Sparkles, AlertCircle, Send, ChevronRight } from 'lucide-react';

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
      text: `Hello Mahendra! I am your AI Mentor for Module ${module.id} (${module.title}). Ask me any question, derivation hint, or code debugging question!`,
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
      let reply = `Great question regarding ${module.title}. In ${activeArtifact}, the key principle is understanding how inputs and weights combine into a decision boundary.`;
      if (userMsg.toLowerCase().includes('xor')) {
        reply = `XOR fails on a single Perceptron because positive points (0,1) & (1,0) cannot be separated from (0,0) & (1,1) by any single straight line! You need at least 2 hidden neurons (MLP) to form a non-linear boundary.`;
      } else if (userMsg.toLowerCase().includes('bias')) {
        reply = `Without a bias term b, the decision line w₁x₁ + w₂x₂ = 0 is forced to pass directly through the origin (0,0). The bias allows the hyper-plane to shift freely anywhere in space!`;
      }
      setMentorMessages((prev) => [...prev, { role: 'assistant', text: reply }]);
    }, 500);
  };

  return (
    <div className="flex flex-col h-[calc(100vh-64px)] bg-[#090C10] overflow-hidden select-none">
      
      {/* Top Breadcrumb Bar */}
      <div className="h-12 shrink-0 bg-[#0E121A] border-b border-white/5 px-6 md:px-8 flex items-center justify-between text-xs">
        
        <div className="flex items-center gap-3">
          <button
            onClick={onBack}
            className="text-slate-400 hover:text-slate-200 flex items-center gap-1.5 font-mono font-medium transition-colors"
          >
            <ArrowLeft className="w-4 h-4" /> Back to Roadmap
          </button>

          <span className="text-slate-600">/</span>

          <span className="font-bold text-slate-200 font-heading flex items-center gap-2">
            <span className="text-indigo-400 font-mono">#{module.id}</span>
            <span>{module.title}</span>
          </span>
        </div>

        <div className="flex items-center gap-4 font-mono text-[11px]">
          <span className="text-slate-400">Progress: 68%</span>
          <div className="w-24 bg-[#090C10] h-1.5 rounded-full overflow-hidden border border-white/5">
            <div className="bg-indigo-500 h-full w-[68%]" />
          </div>
        </div>

      </div>

      {/* 3-Column Workspace Main Layout */}
      <div className="flex-1 flex overflow-hidden">
        
        {/* Left Column: 9 Artifacts Navigation */}
        <aside className="w-60 shrink-0 bg-[#090C10] border-r border-white/5 p-4 flex flex-col justify-between overflow-y-auto hidden md:flex">
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
                  className={`w-full px-3 py-2.5 rounded-xl text-xs font-medium flex items-center justify-between transition-all ${
                    isActive
                      ? 'bg-indigo-600/20 text-indigo-400 border border-indigo-500/30 font-bold shadow-sm'
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

        {/* Center Content Column (Max Reading Width 760px) */}
        <main className="flex-1 overflow-y-auto p-6 md:p-10 bg-[#090C10] flex justify-center">
          <div className="w-full max-w-[760px] space-y-8 leading-relaxed">
            
            {/* 01 OVERVIEW */}
            {activeArtifact === 'overview' && (
              <div className="space-y-6 animate-fade-in text-slate-300 text-sm">
                <div>
                  <span className="text-xs font-mono font-bold text-indigo-400 uppercase tracking-wider">
                    01 // OVERVIEW & HISTORICAL MOTIVATION
                  </span>
                  <h2 className="text-2xl md:text-3xl font-bold text-slate-100 font-heading mt-1">
                    What Problem Does a Perceptron Solve?
                  </h2>
                </div>

                <div className="space-y-4">
                  <p>
                    Imagine you are trying to teach a machine to make a simple <strong>yes / no decision</strong>.
                  </p>
                  <p>
                    Not a complex one. Not <em>"is this a cat?"</em> or <em>"will this patient survive?"</em>  
                    Something simpler: <strong>"given two numbers, is their combination above a threshold?"</strong>
                  </p>
                  <p>
                    The perceptron is the simplest possible learning machine for this problem. It takes numerical inputs, weighs each one by importance, adds a bias, and outputs a binary decision: <strong>0 (no) or 1 (yes)</strong>.
                  </p>
                </div>

                {/* Canonical Math Box */}
                <div className="panel-card p-5 space-y-2 border-indigo-500/20 bg-[#121620]">
                  <div className="text-xs font-mono text-indigo-400 font-bold uppercase">
                    CANONICAL PERCEPTRON RULE
                  </div>
                  <KaTeXRenderer math="y = \operatorname{step}(w^T x + b)" block />
                  <p className="text-xs text-slate-400 font-mono pt-1">
                    Where parameter updates occur when prediction error is non-zero:
                  </p>
                  <KaTeXRenderer math="w_{t+1} = w_t + \eta \cdot (y - \hat{y}) \cdot x" block />
                </div>

                {/* History Section */}
                <div className="space-y-3 pt-2">
                  <h3 className="text-lg font-bold text-slate-100 font-heading">Historical Context: Mark I Perceptron (1957)</h3>
                  <p>
                    In 1957, Frank Rosenblatt at Cornell built the <strong>Mark I Perceptron</strong> — a physical hardware machine with 400 photocells and potentiometers adjusted by electric motors.
                  </p>
                  <p>
                    In 1969, Minsky & Papert published <em>Perceptrons</em>, proving mathematically that a single perceptron <strong>cannot solve XOR</strong> because XOR is not linearly separable. This caused the first AI Winter.
                  </p>
                </div>

                {/* Learning Outcomes */}
                <div className="panel-card p-6 space-y-3 bg-[#0E121A]">
                  <h4 className="text-xs font-mono font-bold text-indigo-400 uppercase">LEARNING OUTCOMES</h4>
                  <ul className="space-y-2 text-xs font-mono text-slate-300">
                    <li className="flex items-center gap-2"><CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" /> Explain weighted sum, bias, and step threshold decision.</li>
                    <li className="flex items-center gap-2"><CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" /> Implement a Perceptron in pure Python & NumPy without libraries.</li>
                    <li className="flex items-center gap-2"><CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" /> Train on AND/OR logic gates and prove XOR failure.</li>
                  </ul>
                </div>
              </div>
            )}

            {/* 02 MENTAL MODEL */}
            {activeArtifact === 'mentalModel' && (
              <div className="space-y-6 animate-fade-in text-slate-300 text-sm">
                <div>
                  <span className="text-xs font-mono font-bold text-indigo-400 uppercase tracking-wider">
                    02 // GEOMETRIC MENTAL MODEL
                  </span>
                  <h2 className="text-2xl font-bold text-slate-100 font-heading mt-1">
                    The Perceptron as a Weighted Voting System
                  </h2>
                </div>

                <div className="space-y-4">
                  <p>
                    Forget complex equations for a moment. Imagine a panel of three judges deciding whether to approve a loan:
                  </p>
                  <ul className="list-disc pl-5 space-y-1.5 font-mono text-xs">
                    <li><strong>Judge A (Credit Score):</strong> weight = +3</li>
                    <li><strong>Judge B (Employment History):</strong> weight = +2</li>
                    <li><strong>Judge C (Current Debt):</strong> weight = -2 (negative — high debt argues against)</li>
                  </ul>
                  <p>
                    The chairperson has a standing bias (+1). The panel sums all weighted votes plus the bias. If the sum is zero or greater, approve (1). Otherwise, reject (0).
                  </p>
                </div>

                {/* SVG Visual Canvas */}
                <div className="panel-card p-6 flex flex-col items-center justify-center space-y-4 bg-[#121620]">
                  <div className="w-full max-w-md h-52 bg-[#090C10] rounded-xl border border-white/10 p-4 flex items-center justify-center">
                    <svg className="w-full h-full" viewBox="0 0 200 150">
                      <line x1="0" y1="75" x2="200" y2="75" stroke="rgba(255,255,255,0.06)" strokeDasharray="3,3" />
                      <line x1="100" y1="0" x2="100" y2="150" stroke="rgba(255,255,255,0.06)" strokeDasharray="3,3" />
                      <line x1="20" y1="130" x2="180" y2="20" stroke="#6C8CFF" strokeWidth="2" />
                      <circle cx="50" cy="40" r="4" fill="#8B5CF6" />
                      <circle cx="80" cy="30" r="4" fill="#8B5CF6" />
                      <circle cx="140" cy="110" r="4" fill="#32D583" />
                      <circle cx="160" cy="120" r="4" fill="#32D583" />
                    </svg>
                  </div>
                  <span className="text-xs text-slate-400 font-mono">
                    Figure 1: 2D linear decision line separating Class 0 (green) and Class 1 (purple).
                  </span>
                </div>
              </div>
            )}

            {/* 03 MATHEMATICS */}
            {activeArtifact === 'math' && (
              <div className="space-y-6 animate-fade-in text-slate-300 text-sm">
                <div>
                  <span className="text-xs font-mono font-bold text-indigo-400 uppercase tracking-wider">
                    03 // FORMAL MATHEMATICAL DERIVATION
                  </span>
                  <h2 className="text-2xl font-bold text-slate-100 font-heading mt-1">
                    Formal Derivation & Proofs
                  </h2>
                </div>

                <div className="space-y-4">
                  <p>
                    The inner product of weights and input vector plus bias gives score z:
                  </p>
                  <KaTeXRenderer math="z = \sum_{i=1}^n w_i x_i + b = w^T x + b" block />

                  <p>
                    Passing z into the unit step activation produces binary prediction:
                  </p>
                  <KaTeXRenderer math="\hat{y} = \operatorname{step}(z) = \begin{cases} 1 & \text{if } z \ge 0 \\ 0 & \text{if } z < 0 \end{cases}" block />

                  <p>
                    When prediction error occurs, parameter updates are computed as:
                  </p>
                  <KaTeXRenderer math="w_{t+1} = w_t + \eta \cdot (y - \hat{y}) \cdot x" block />
                  <KaTeXRenderer math="b_{t+1} = b_t + \eta \cdot (y - \hat{y})" block />
                </div>
              </div>
            )}

            {/* 04 IMPLEMENTATION (Code Playground) */}
            {activeArtifact === 'code' && (
              <div className="space-y-6 animate-fade-in">
                <div>
                  <span className="text-xs font-mono font-bold text-indigo-400 uppercase tracking-wider">
                    04 // FROM-SCRATCH PYTHON PLAYGROUND
                  </span>
                  <h2 className="text-2xl font-bold text-slate-100 font-heading mt-1">
                    Interactive Python Implementation
                  </h2>
                  <p className="text-xs text-slate-400 mt-1">
                    Run pure Python code directly in your browser using WebAssembly.
                  </p>
                </div>

                <CodePlayground moduleTitle={module.title} initialCode={module.codeSnippet} />
              </div>
            )}

            {/* 05 EXPERIMENTS */}
            {activeArtifact === 'experiments' && (
              <div className="space-y-6 animate-fade-in text-slate-300 text-sm">
                <div>
                  <span className="text-xs font-mono font-bold text-indigo-400 uppercase tracking-wider">
                    05 // EXPERIMENTS & OBSERVATIONS
                  </span>
                  <h2 className="text-2xl font-bold text-slate-100 font-heading mt-1">
                    Runnable Empirical Experiments
                  </h2>
                </div>

                <div className="panel-card p-6 space-y-4 bg-[#121620]">
                  <h3 className="text-base font-bold text-slate-100 font-heading">Experiment 1: Removing Bias Term</h3>
                  <p>
                    What happens when bias b = 0? The decision line is locked to the origin (0,0).
                  </p>
                  <button onClick={() => setActiveArtifact('code')} className="btn-indigo text-xs py-2 px-4">
                    Run Bias Experiment in Code Playground →
                  </button>
                </div>
              </div>
            )}

            {/* Default Fallback for remaining artifacts */}
            {['applications', 'challenge', 'assessment', 'references'].includes(activeArtifact) && (
              <div className="space-y-6 animate-fade-in text-slate-300 text-sm">
                <div>
                  <span className="text-xs font-mono font-bold text-indigo-400 uppercase tracking-wider">
                    {activeArtifact.toUpperCase()}
                  </span>
                  <h2 className="text-2xl font-bold text-slate-100 font-heading mt-1 capitalize">
                    {activeArtifact}
                  </h2>
                </div>

                <div className="panel-card p-6 space-y-4 bg-[#121620]">
                  <p className="text-sm leading-relaxed">
                    Build a binary classifier that trains on non-linearly separable inputs to observe XOR failure.
                  </p>
                  <div className="bg-amber-950/20 border border-amber-800/40 p-4 rounded-xl text-xs text-amber-300/90 space-y-1">
                    <div className="font-bold text-amber-400 flex items-center gap-1.5">
                      <AlertCircle className="w-4 h-4" /> Mentor Mode: Hints Only
                    </div>
                    <div>Ask the AI Mentor on the right for step-by-step guidance!</div>
                  </div>
                </div>
              </div>
            )}

          </div>
        </main>

        {/* Right Column: Contextual AI Mentor */}
        <aside className="w-80 shrink-0 bg-[#090C10] border-l border-white/5 p-4 flex flex-col justify-between hidden xl:flex">
          <div className="space-y-4 flex-1 flex flex-col justify-between">
            
            <div className="space-y-3">
              <div className="flex items-center justify-between border-b border-white/5 pb-3">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-indigo-400" />
                  <h3 className="text-xs font-bold text-slate-100 font-heading">
                    AI Mentor
                  </h3>
                </div>
                <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                  Online
                </span>
              </div>

              {/* Chat Thread */}
              <div className="space-y-3 max-h-[450px] overflow-y-auto pr-1">
                {mentorMessages.map((msg, idx) => (
                  <div
                    key={idx}
                    className={`p-3 rounded-xl text-xs font-medium leading-relaxed ${
                      msg.role === 'assistant'
                        ? 'bg-[#121620] border border-white/5 text-slate-200'
                        : 'bg-indigo-600/20 border border-indigo-500/30 text-indigo-200 ml-4'
                    }`}
                  >
                    {msg.text}
                  </div>
                ))}
              </div>
            </div>

            {/* Quick Socratic Prompt Pills */}
            <div className="space-y-1.5 pt-2">
              <button
                onClick={() => handleSendMentor('Explain why a perceptron fails on XOR')}
                className="w-full text-left p-2 bg-[#121620] hover:bg-[#181E2B] border border-white/5 rounded-lg text-[11px] text-indigo-300 font-mono flex items-center justify-between transition-colors"
              >
                <span>Why does Perceptron fail on XOR?</span>
                <ChevronRight className="w-3.5 h-3.5 text-indigo-400" />
              </button>
              
              <button
                onClick={() => handleSendMentor('Explain what bias does')}
                className="w-full text-left p-2 bg-[#121620] hover:bg-[#181E2B] border border-white/5 rounded-lg text-[11px] text-indigo-300 font-mono flex items-center justify-between transition-colors"
              >
                <span>Why do we need a bias term?</span>
                <ChevronRight className="w-3.5 h-3.5 text-indigo-400" />
              </button>
            </div>

            {/* Input Bar */}
            <div className="relative pt-2">
              <input
                type="text"
                value={mentorInput}
                onChange={(e) => setMentorInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSendMentor(mentorInput)}
                placeholder="Ask about this section..."
                className="w-full bg-[#121620] border border-white/10 rounded-xl pl-3 pr-9 py-2 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500/50 font-mono"
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
