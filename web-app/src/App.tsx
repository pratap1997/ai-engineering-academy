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

export function App() {
  const [activeTab, setActiveTab] = useState<'learn' | 'roadmap' | 'projects' | 'library'>('learn');
  const [currentView, setCurrentView] = useState<string>('overview');
  const [selectedModule, setSelectedModule] = useState<ModuleData | null>(null);
  const [isSearchOpen, setIsSearchOpen] = useState<boolean>(false);

  const handleOpenModule = (module: ModuleData) => {
    setSelectedModule(module);
  };

  const handleSelectMentorPrompt = () => {
    if (!selectedModule) {
      setSelectedModule(MODULES[0]);
    }
  };

  return (
    <div className="min-h-screen bg-[#0B0D12] text-slate-100 flex flex-col font-sans antialiased">
      
      {/* Top Header Navigation */}
      <Header
        activeTab={activeTab}
        setActiveTab={(tab) => {
          setActiveTab(tab);
          setSelectedModule(null);
        }}
        onOpenSearch={() => setIsSearchOpen(true)}
        onOpenTutor={() => setIsSearchOpen(true)}
      />

      {/* Main Page Layout */}
      {selectedModule ? (
        // 3-Column Engineering Module Workspace
        <ModuleWorkspace
          module={selectedModule}
          onBack={() => setSelectedModule(null)}
        />
      ) : (
        // Command Center Layout with Left Sidebar + Center View + Right AI Mentor
        <div className="flex-1 flex overflow-hidden max-w-[1700px] w-full mx-auto">
          
          {/* Left Navigation Sidebar */}
          <Sidebar
            currentView={currentView}
            setCurrentView={setCurrentView}
            onOpenRoadmap={() => setActiveTab('roadmap')}
            onOpenModules={() => setActiveTab('roadmap')}
          />

          {/* Main Central View */}
          <main className="flex-1 p-6 md:p-8 overflow-y-auto">
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
              <div className="space-y-6 animate-fade-in">
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
              <div className="space-y-6 animate-fade-in">
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

          {/* Right AI Mentor Panel (Visible in Learn Command Center) */}
          {activeTab === 'learn' && (
            <AIMentorPanel
              currentModule={MODULES[0]}
              onSelectPrompt={handleSelectMentorPrompt}
            />
          )}

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
