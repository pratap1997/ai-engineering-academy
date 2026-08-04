import React, { useState, useMemo } from 'react';
import type { ModuleData } from '../types';
import { MarkdownRenderer } from './MarkdownRenderer';
import { CodePlayground } from './CodePlayground';
import { getModuleArtifactContent } from '../utils/moduleContentLoader';
import { generateAIMentorResponse } from '../utils/aiMentorEngine';
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

  // Dynamically load markdown or python file content from disk for active artifact & module
  const diskContent = useMemo(() => {
    return getModuleArtifactContent(module.id, activeArtifact);
  }, [module.id, activeArtifact]);

  // Code snippet for implementation tab
  const pythonCode = useMemo(() => {
    if (activeArtifact === 'code' && diskContent) {
      return diskContent;
    }
    return module.codeSnippet;
  }, [activeArtifact, diskContent, module.codeSnippet]);

  const handleSendMentor = (text: string) => {
    if (!text.trim()) return;
    const userMsg = text.trim();
    setMentorMessages((prev) => [...prev, { role: 'user', text: userMsg }]);
    setMentorInput('');

    setTimeout(() => {
      const res = generateAIMentorResponse(module.id, module.title, activeArtifact, userMsg);
      setMentorMessages((prev) => [...prev, { role: 'assistant', text: res.reply }]);
    }, 400);
  };

  return (
    <div className="flex flex-col h-[calc(100vh-64px)] bg-[#090C10] overflow-hidden select-none">
      
      {/* Top Breadcrumb Bar */}
      <div className="h-14 shrink-0 bg-[#0E121A] border-b border-white/5 px-6 md:px-8 flex items-center justify-between text-sm">
        
        <div className="flex items-center gap-3">
          <button
            onClick={onBack}
            className="text-slate-300 hover:text-white flex items-center gap-1.5 font-mono font-medium transition-colors"
          >
            <ArrowLeft className="w-4 h-4" /> Back to Roadmap
          </button>

          <span className="text-slate-600">/</span>

          <span className="font-bold text-slate-100 font-heading text-base flex items-center gap-2">
            <span className="text-indigo-400 font-mono">#{module.id}</span>
            <span>{module.title}</span>
          </span>
        </div>

        <div className="flex items-center gap-4 font-mono text-xs">
          <span className="text-slate-400">Progress: 68%</span>
          <div className="w-28 bg-[#090C10] h-2 rounded-full overflow-hidden border border-white/5">
            <div className="bg-indigo-500 h-full w-[68%]" />
          </div>
        </div>

      </div>

      {/* 3-Column Workspace Main Layout */}
      <div className="flex-1 flex overflow-hidden">
        
        {/* Left Column: 9 Artifacts Navigation */}
        <aside className="w-64 shrink-0 bg-[#090C10] border-r border-white/5 p-4 flex flex-col justify-between overflow-y-auto hidden md:flex">
          <div className="space-y-1.5">
            <div className="px-3 text-xs font-mono font-bold uppercase tracking-wider text-slate-500 mb-3">
              MODULE PATH
            </div>

            {artifactsList.map((art) => {
              const isActive = activeArtifact === art.key;
              return (
                <button
                  key={art.key}
                  onClick={() => setActiveArtifact(art.key)}
                  className={`w-full px-3.5 py-3 rounded-xl text-sm font-medium flex items-center justify-between transition-all ${
                    isActive
                      ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/35 font-bold shadow-sm'
                      : 'text-slate-400 hover:text-slate-100 hover:bg-white/5'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <span className="font-mono text-xs text-slate-500">{art.num}</span>
                    <span>{art.label}</span>
                  </div>

                  {art.status === 'complete' && <CheckCircle2 className="w-4 h-4 text-emerald-400" />}
                  {art.status === 'current' && <div className="w-2.5 h-2.5 rounded-full bg-indigo-400" />}
                  {art.status === 'upcoming' && <span className="text-slate-600 font-mono text-xs">○</span>}
                </button>
              );
            })}
          </div>
        </aside>

        {/* Center Content Column (Max Reading Width 800px) */}
        <main className="flex-1 overflow-y-auto p-6 md:p-12 bg-[#090C10] flex justify-center">
          <div className="w-full max-w-[800px] space-y-8 leading-relaxed text-slate-200 text-base">
            
            {/* 04 IMPLEMENTATION (Code Playground) */}
            {activeArtifact === 'code' ? (
              <div className="space-y-6 animate-fade-in">
                <div>
                  <span className="text-xs font-mono font-bold text-indigo-400 uppercase tracking-wider">
                    04 // FROM-SCRATCH PYTHON PLAYGROUND
                  </span>
                  <h2 className="text-3xl font-bold text-slate-100 font-heading mt-1">
                    {module.title} — Python Implementation
                  </h2>
                  <p className="text-sm text-slate-400 mt-1">
                    Run pure Python code directly in your browser using Pyodide WebAssembly.
                  </p>
                </div>

                <CodePlayground moduleTitle={module.title} initialCode={pythonCode} />
              </div>
            ) : (
              /* DYNAMIC MARKDOWN RENDERER FOR ALL OTHER 8 ARTIFACTS */
              <div className="space-y-6 animate-fade-in">
                {diskContent ? (
                  <MarkdownRenderer content={diskContent} />
                ) : (
                  /* Fallback if disk content is loading or fallback mode */
                  <div className="space-y-6">
                    <div>
                      <span className="text-xs font-mono font-bold text-indigo-400 uppercase tracking-wider">
                        #{module.id} // {activeArtifact.toUpperCase()}
                      </span>
                      <h2 className="text-3xl font-bold text-slate-100 font-heading mt-1">
                        {module.title}
                      </h2>
                    </div>

                    <div className="panel-card p-6 space-y-4 bg-[#121620]">
                      <p className="text-base text-slate-200 leading-relaxed">
                        {module.overview}
                      </p>
                      <div className="bg-indigo-950/20 border border-indigo-800/40 p-4 rounded-xl text-xs font-mono text-indigo-300 space-y-1">
                        <div className="font-bold text-indigo-400 flex items-center gap-1.5 text-sm">
                          <AlertCircle className="w-4 h-4" /> Lesson Objective
                        </div>
                        <div>Master this concept from scratch with mathematical derivations & Python code.</div>
                      </div>
                    </div>
                  </div>
                )}
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
                  <h3 className="text-sm font-bold text-slate-100 font-heading">
                    AI Mentor
                  </h3>
                </div>
                <span className="text-xs font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                  Online
                </span>
              </div>

              {/* Chat Thread */}
              <div className="space-y-3 max-h-[480px] overflow-y-auto pr-1">
                {mentorMessages.map((msg, idx) => (
                  <div
                    key={idx}
                    className={`p-3.5 rounded-xl text-sm font-medium leading-relaxed ${
                      msg.role === 'assistant'
                        ? 'bg-[#121620] border border-white/5 text-slate-200'
                        : 'bg-indigo-600/20 border border-indigo-500/30 text-indigo-100 ml-4'
                    }`}
                  >
                    {msg.text}
                  </div>
                ))}
              </div>
            </div>

            {/* Quick Socratic Prompt Pills */}
            <div className="space-y-2 pt-2">
              <button
                onClick={() => handleSendMentor(`Explain key concept of ${module.title}`)}
                className="w-full text-left p-2.5 bg-[#121620] hover:bg-[#181E2B] border border-white/5 rounded-xl text-xs text-indigo-300 font-mono flex items-center justify-between transition-colors"
              >
                <span>Explain key concept</span>
                <ChevronRight className="w-4 h-4 text-indigo-400" />
              </button>
              
              <button
                onClick={() => handleSendMentor(`Give me a hint for ${activeArtifact}`)}
                className="w-full text-left p-2.5 bg-[#121620] hover:bg-[#181E2B] border border-white/5 rounded-xl text-xs text-indigo-300 font-mono flex items-center justify-between transition-colors"
              >
                <span>Give me a hint for this section</span>
                <ChevronRight className="w-4 h-4 text-indigo-400" />
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
                className="w-full bg-[#121620] border border-white/10 rounded-xl pl-3.5 pr-10 py-2.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500/50 font-mono"
              />
              <button
                onClick={() => handleSendMentor(mentorInput)}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 p-1 text-indigo-400 hover:text-indigo-300"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>

          </div>
        </aside>

      </div>

    </div>
  );
};
