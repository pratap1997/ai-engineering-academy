import { useState } from 'react';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { LearnView } from './components/LearnView';
import { RoadmapView } from './components/RoadmapView';
import { ModuleWorkspace } from './components/ModuleWorkspace';
import { AIMentorPanel } from './components/AIMentorPanel';
import { CommandPalette } from './components/CommandPalette';
import { MODULES } from './modulesData';
import type { ModuleData } from './types';
import { Bot, X } from 'lucide-react';

export function App() {
  const [activeTab, setActiveTab] = useState<'learn' | 'roadmap' | 'projects' | 'library'>('learn');
  const [currentView, setCurrentView] = useState<string>('overview');
  const [selectedModule, setSelectedModule] = useState<ModuleData | null>(null);
  const [isSearchOpen, setIsSearchOpen] = useState<boolean>(false);
  const [isMentorDrawerOpen, setIsMentorDrawerOpen] = useState<boolean>(false);

  const handleOpenModule = (module: ModuleData) => {
    setSelectedModule(module);
  };

  const handleSelectMentorPrompt = () => {
    if (!selectedModule) {
      setSelectedModule(MODULES[0]);
    }
  };

  return (
    <div className="min-h-screen bg-[#090C10] text-slate-100 flex flex-col font-sans antialiased">
      
      {/* Top Header Navigation */}
      <Header
        activeTab={activeTab}
        setActiveTab={(tab) => {
          setActiveTab(tab);
          setSelectedModule(null);
        }}
        onOpenSearch={() => setIsSearchOpen(true)}
        onOpenTutor={() => setIsMentorDrawerOpen(true)}
      />

      {/* Main Page Layout */}
      {selectedModule ? (
        // 3-Column Engineering Module Workspace
        <ModuleWorkspace
          module={selectedModule}
          onBack={() => setSelectedModule(null)}
        />
      ) : (
        // Command Center Layout with Left Sidebar + 100% Spacious Main Center View
        <div className="flex-1 flex overflow-hidden w-full">
          
          {/* Left Navigation Sidebar */}
          <Sidebar
            currentView={currentView}
            setCurrentView={setCurrentView}
            onOpenRoadmap={() => setActiveTab('roadmap')}
            onOpenModules={() => setActiveTab('roadmap')}
          />

          {/* Main Central View (Spacious, Un-crowded) */}
          <main className="flex-1 p-6 md:p-10 overflow-y-auto">
            {activeTab === 'learn' && (
              <LearnView
                onOpenModule={handleOpenModule}
                onExploreRoadmap={() => setActiveTab('roadmap')}
              />
            )}

            {activeTab === 'roadmap' && (
              <RoadmapView onOpenModule={handleOpenModule} />
            )}

            {activeTab === 'projects' && (
              <div className="max-w-5xl mx-auto space-y-6 animate-fade-in">
                <h2 className="text-2xl font-bold font-heading">Capstones & Projects</h2>
                <p className="text-xs text-slate-400">
                  Production engineering projects built from first principles.
                </p>
                <div className="panel-card p-6">
                  <h3 className="text-base font-bold text-slate-200">Autonomous Coding Engineer Capstone</h3>
                  <p className="text-xs text-slate-400 mt-1">Integrates autograd, RAG retrieval, MCTS planning, and WASM execution.</p>
                  <button onClick={() => handleOpenModule(MODULES[MODULES.length - 1])} className="btn-indigo text-xs mt-4">
                    Open Capstone Workspace →
                  </button>
                </div>
              </div>
            )}

            {activeTab === 'library' && (
              <div className="max-w-5xl mx-auto space-y-6 animate-fade-in">
                <h2 className="text-2xl font-bold font-heading">Knowledge Library</h2>
                <p className="text-xs text-slate-400">
                  Deep references, mathematical proofs, and architectural decision records.
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="panel-card p-5 space-y-2">
                    <h4 className="text-sm font-bold text-indigo-400">Attention Memory Matrix Optimization</h4>
                    <p className="text-xs text-slate-400">Standard self-attention vs FlashAttention memory tiling proofs.</p>
                  </div>
                  <div className="panel-card p-5 space-y-2">
                    <h4 className="text-sm font-bold text-indigo-400">Rotary Position Embedding Math</h4>
                    <p className="text-xs text-slate-400">RoPE rotation matrix derivations and YaRN scaling.</p>
                  </div>
                </div>
              </div>
            )}
          </main>

        </div>
      )}

      {/* Floating AI Mentor Button (Bottom Right) */}
      {!selectedModule && !isMentorDrawerOpen && (
        <button
          onClick={() => setIsMentorDrawerOpen(true)}
          className="fixed bottom-6 right-6 z-40 bg-indigo-600 hover:bg-indigo-500 text-white p-3.5 rounded-full shadow-2xl flex items-center gap-2 font-semibold text-xs transition-all hover:scale-105"
        >
          <Bot className="w-5 h-5" />
          <span>Ask AI Mentor</span>
        </button>
      )}

      {/* Floating AI Mentor Modal Drawer */}
      {isMentorDrawerOpen && (
        <div className="fixed inset-y-0 right-0 z-50 w-96 bg-[#090C10] border-l border-white/10 shadow-2xl flex flex-col animate-slide-left">
          <div className="p-4 border-b border-white/5 flex items-center justify-between">
            <span className="font-heading font-bold text-sm text-slate-100 flex items-center gap-2">
              <Bot className="w-4 h-4 text-indigo-400" /> AI Mentor Drawer
            </span>
            <button onClick={() => setIsMentorDrawerOpen(false)} className="text-slate-500 hover:text-slate-200">
              <X className="w-4 h-4" />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto">
            <AIMentorPanel
              currentModule={MODULES[0]}
              onSelectPrompt={handleSelectMentorPrompt}
            />
          </div>
        </div>
      )}

      {/* Cmd+K Search Command Palette */}
      <CommandPalette
        isOpen={isSearchOpen}
        onClose={() => setIsSearchOpen(false)}
        onSelectModule={handleOpenModule}
      />

    </div>
  );
}

export default App;
