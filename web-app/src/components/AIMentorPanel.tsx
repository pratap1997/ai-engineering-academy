import React, { useState } from 'react';
import { Bot, Send, Sparkles, AlertCircle, CheckCircle2, XCircle, FileText, HelpCircle, ChevronDown, ChevronUp } from 'lucide-react';
import type { ModuleData } from '../types';

interface AIMentorPanelProps {
  currentModule: ModuleData;
  onSelectPrompt: (promptText: string) => void;
}

export const AIMentorPanel: React.FC<AIMentorPanelProps> = ({ currentModule, onSelectPrompt }) => {
  const [inputMsg, setInputMsg] = useState('');
  const [isCollapsed, setIsCollapsed] = useState(false);

  const mentorActions = [
    {
      title: 'Explain this concept',
      desc: 'I need a simpler explanation',
      icon: HelpCircle,
      prompt: 'Explain the core intuition of Perceptron linear decision boundaries in plain terms.',
    },
    {
      title: 'Check my understanding',
      desc: 'Ask me a few questions',
      icon: CheckCircle2,
      prompt: 'Ask me 3 diagnostic questions about perceptron weight updates and XOR failure.',
    },
    {
      title: 'Help me debug',
      desc: 'Something isn\'t working',
      icon: Bot,
      prompt: 'My perceptron loop is not converging on OR gate. Help me debug weight updates.',
    },
    {
      title: 'Give me a hint',
      desc: 'I\'m stuck on the challenge',
      icon: Sparkles,
      prompt: 'Give me a hint on solving the Perceptron challenge without giving away the answer.',
    },
  ];

  return (
    <aside className="w-80 shrink-0 bg-[#0B0D12] border-l border-white/5 p-4 flex flex-col justify-between hidden xl:flex select-none overflow-y-auto space-y-6">
      
      <div className="space-y-5">
        
        {/* Mentor Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-indigo-400" />
            <h3 className="text-xs font-bold text-slate-100 font-heading">
              AI Mentor
            </h3>
          </div>
          <button onClick={() => setIsCollapsed(!isCollapsed)} className="text-slate-500 hover:text-slate-300">
            {isCollapsed ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
          </button>
        </div>

        {!isCollapsed && (
          <>
            {/* Active Context Card */}
            <div className="bg-[#11151D] border border-white/5 p-3 rounded-xl space-y-1 font-mono text-[11px]">
              <div className="text-slate-500 font-bold uppercase text-[10px]">You are working on</div>
              <div className="text-slate-200 font-bold">
                {currentModule.id} · {currentModule.title}
              </div>
              <div className="text-indigo-400">
                Experiment: Removing the bias
              </div>
            </div>

            {/* Quick Socratic Actions */}
            <div className="space-y-2">
              <div className="text-[11px] text-slate-400 font-medium">How can I help you?</div>
              
              <div className="space-y-1.5">
                {mentorActions.map((act, idx) => (
                  <button
                    key={idx}
                    onClick={() => onSelectPrompt(act.prompt)}
                    className="w-full p-2.5 bg-[#11151D] hover:bg-[#171C26] border border-white/5 rounded-xl text-left transition-colors group flex items-start gap-2.5"
                  >
                    <act.icon className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
                    <div>
                      <div className="text-xs font-bold text-slate-200 group-hover:text-indigo-300 transition-colors">
                        {act.title}
                      </div>
                      <div className="text-[11px] text-slate-400">
                        {act.desc}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Challenge Protection Mode Warning */}
            <div className="bg-amber-950/20 border border-amber-800/40 p-3 rounded-xl text-[11px] text-amber-300/90 space-y-1">
              <div className="font-bold flex items-center gap-1.5 text-amber-400">
                <AlertCircle className="w-3.5 h-3.5" /> Mentor mode: Hints only
              </div>
              <div>Solutions are hidden during challenge mode to protect learning.</div>
            </div>

            {/* Input Bar */}
            <div className="relative">
              <input
                type="text"
                value={inputMsg}
                onChange={(e) => setInputMsg(e.target.value)}
                placeholder="Ask anything about this module..."
                className="w-full bg-[#11151D] border border-white/10 rounded-xl pl-3 pr-9 py-2 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500/50"
              />
              <button
                onClick={() => {
                  if (inputMsg.trim()) {
                    onSelectPrompt(inputMsg);
                    setInputMsg('');
                  }
                }}
                className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-indigo-400 hover:text-indigo-300"
              >
                <Send className="w-3.5 h-3.5" />
              </button>
            </div>
          </>
        )}

        {/* Recent Activity Stream */}
        <div className="space-y-3 pt-3 border-t border-white/5">
          <div className="flex items-center justify-between text-xs">
            <span className="font-bold text-slate-300">Recent Activity</span>
            <span className="text-[11px] text-slate-500 cursor-pointer hover:underline">View all</span>
          </div>

          <div className="space-y-2.5 text-xs font-mono">
            
            <div className="flex items-start gap-2">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
              <div className="flex-1">
                <div className="text-slate-200 text-[11px] font-bold">Completed experiment</div>
                <div className="text-slate-400 text-[10px]">Learning rate comparison</div>
              </div>
              <span className="text-[10px] text-slate-500">2h ago</span>
            </div>

            <div className="flex items-start gap-2">
              <div className="w-3.5 h-3.5 rounded-full bg-indigo-500/20 text-indigo-400 flex items-center justify-center shrink-0 mt-0.5 text-[9px] font-bold">●</div>
              <div className="flex-1">
                <div className="text-slate-200 text-[11px] font-bold">Read</div>
                <div className="text-slate-400 text-[10px]">Mathematical derivation</div>
              </div>
              <span className="text-[10px] text-slate-500">Yesterday</span>
            </div>

            <div className="flex items-start gap-2">
              <XCircle className="w-3.5 h-3.5 text-rose-400 shrink-0 mt-0.5" />
              <div className="flex-1">
                <div className="text-slate-200 text-[11px] font-bold">Failed challenge attempt</div>
                <div className="text-slate-400 text-[10px]">Perceptron implementation</div>
              </div>
              <span className="text-[10px] text-slate-500">2d ago</span>
            </div>

            <div className="flex items-start gap-2">
              <FileText className="w-3.5 h-3.5 text-violet-400 shrink-0 mt-0.5" />
              <div className="flex-1">
                <div className="text-slate-200 text-[11px] font-bold">Updated note</div>
                <div className="text-slate-400 text-[10px]">Bias and decision boundary</div>
              </div>
              <span className="text-[10px] text-slate-500">3d ago</span>
            </div>

          </div>
        </div>

      </div>

    </aside>
  );
};
