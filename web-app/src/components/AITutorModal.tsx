import React, { useState } from 'react';
import { queryAITutor } from '../utils/ragTutor';
import type { RAGAnswer } from '../utils/ragTutor';
import { Bot, Send, X, Sparkles, BookOpen, ChevronRight } from 'lucide-react';
import type { ModuleData } from '../types';

interface AITutorModalProps {
  onOpenModule: (m: ModuleData) => void;
}

export const AITutorModal: React.FC<AITutorModalProps> = ({ onOpenModule }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [inputQuery, setInputQuery] = useState('');
  const [history, setHistory] = useState<RAGAnswer[]>([]);

  const sampleQueries = [
    'How does FlashAttention optimize GPU SRAM memory?',
    'Explain DPO loss vs PPO reward modeling',
    'What is Rotary Position Embedding (RoPE) scaling?',
    'How does MCTS tree search guide coding agents?',
  ];

  const handleSearch = (q: string) => {
    if (!q.trim()) return;
    const ans = queryAITutor(q);
    setHistory((prev) => [ans, ...prev]);
    setInputQuery('');
  };

  return (
    <>
      {/* Floating Trigger Button */}
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 z-40 bg-gradient-to-r from-cyan-500 to-violet-600 hover:from-cyan-400 hover:to-violet-500 text-white font-bold px-4 py-3 rounded-full shadow-2xl shadow-cyan-500/40 flex items-center gap-2 border border-cyan-300/40 transition-all duration-300 hover:scale-105"
      >
        <Bot className="w-5 h-5 animate-pulse" />
        <span className="text-xs font-mono">Ask AI RAG Tutor</span>
      </button>

      {/* RAG Tutor Chatbot Drawer */}
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-end sm:p-6 bg-slate-950/70 backdrop-blur-sm">
          <div className="bg-slate-900 border border-cyan-500/40 rounded-t-2xl sm:rounded-2xl max-w-lg w-full h-[85vh] sm:h-[650px] flex flex-col shadow-2xl shadow-cyan-950/80 overflow-hidden">
            
            {/* Header */}
            <div className="p-4 bg-slate-950 border-b border-slate-800 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-cyan-500/10 border border-cyan-500/30 rounded-xl text-cyan-400">
                  <Bot className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-slate-100 flex items-center gap-1.5">
                    Academy RAG AI Tutor <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                  </h3>
                  <p className="text-[11px] text-slate-400">BM25 + Semantic Retrieval over 50 Modules</p>
                </div>
              </div>

              <button
                onClick={() => setIsOpen(false)}
                className="p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Chat Conversation Body */}
            <div className="flex-1 p-4 overflow-y-auto space-y-4">
              
              {/* Welcome Message if history is empty */}
              {history.length === 0 && (
                <div className="space-y-4 text-xs text-slate-300">
                  <div className="p-4 bg-slate-950 border border-slate-800 rounded-xl space-y-2">
                    <p className="font-semibold text-cyan-300">
                      Welcome to the AI Engineering Academy RAG Tutor! 🤖
                    </p>
                    <p className="text-slate-400 text-[11px]">
                      Ask any question regarding our 50-module curriculum. I will search across all mathematical derivations, pure Python code implementations, and engineering challenges to give you instant answers with exact module citations.
                    </p>
                  </div>

                  <div className="space-y-2">
                    <span className="text-[11px] text-slate-400 font-mono">Suggested Questions:</span>
                    <div className="flex flex-col gap-1.5">
                      {sampleQueries.map((sq, idx) => (
                        <button
                          key={idx}
                          onClick={() => handleSearch(sq)}
                          className="text-left p-2.5 bg-slate-950 hover:bg-slate-800/80 border border-slate-800/80 rounded-lg text-slate-300 hover:text-cyan-300 transition-colors flex items-center justify-between group"
                        >
                          <span>{sq}</span>
                          <ChevronRight className="w-3.5 h-3.5 text-slate-600 group-hover:text-cyan-400" />
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Conversation History */}
              {history.map((ans: RAGAnswer, idx: number) => (
                <div key={idx} className="space-y-2 text-xs">
                  <div className="bg-cyan-950/40 border border-cyan-800/40 p-3 rounded-xl text-cyan-200 font-semibold align-right max-w-[85%] ml-auto">
                    {ans.query}
                  </div>

                  <div className="bg-slate-950 border border-slate-800 p-4 rounded-xl space-y-3 text-slate-300">
                    <pre className="whitespace-pre-wrap font-sans text-xs leading-relaxed text-slate-200">
                      {ans.response}
                    </pre>

                    {/* Matched Module Action Buttons */}
                    <div className="pt-2 border-t border-slate-800 flex flex-wrap gap-2">
                      {ans.matchedModules.map((m: ModuleData) => (
                        <button
                          key={m.id}
                          onClick={() => {
                            setIsOpen(false);
                            onOpenModule(m);
                          }}
                          className="btn-secondary text-[11px] py-1 px-2.5 text-cyan-300 border-cyan-800/60 hover:bg-cyan-950/60"
                        >
                          <BookOpen className="w-3 h-3 text-cyan-400" /> Open #{m.id} Spec
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              ))}

            </div>

            {/* Input Bar */}
            <div className="p-3 bg-slate-950 border-t border-slate-800">
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  handleSearch(inputQuery);
                }}
                className="flex items-center gap-2"
              >
                <input
                  type="text"
                  value={inputQuery}
                  onChange={(e) => setInputQuery(e.target.value)}
                  placeholder="Ask a technical question about any module..."
                  className="flex-1 bg-slate-900 border border-slate-800 focus:border-cyan-500/60 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none"
                />
                <button
                  type="submit"
                  disabled={!inputQuery.trim()}
                  className="btn-primary py-2 px-3 text-xs bg-cyan-600 hover:bg-cyan-500"
                >
                  <Send className="w-3.5 h-3.5" />
                </button>
              </form>
            </div>

          </div>
        </div>
      )}
    </>
  );
};
