import React, { useState } from 'react';
import type { ModuleData } from '../types';
import { X, Activity, Play, RotateCcw } from 'lucide-react';

interface DiagramModalProps {
  module: ModuleData | null;
  onClose: () => void;
}

export const DiagramModal: React.FC<DiagramModalProps> = ({ module, onClose }) => {
  if (!module) return null;
  const [activeStep, setActiveStep] = useState(0);

  const steps = [
    { title: 'Step 1: Input Projections', desc: 'Queries, Keys, and Values projected through weight matrices.' },
    { title: 'Step 2: Tiled Memory Partitioning', desc: 'Partition Q, K, V into SRAM blocks for O(1) memory complexity.' },
    { title: 'Step 3: Online Softmax Normalization', desc: 'Compute running max and scaling factor without materializing N x N matrix.' },
    { title: 'Step 4: Output Accumulation', desc: 'Accumulate weighted outputs into high-bandwidth memory output matrix.' }
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/85 backdrop-blur-md">
      <div className="bg-slate-900 border border-violet-500/40 rounded-2xl max-w-4xl w-full p-6 flex flex-col gap-4 shadow-2xl shadow-violet-950/50">
        
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-violet-500/10 border border-violet-500/30 rounded-xl text-violet-400">
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-lg font-bold text-slate-100">{module.title} Diagram</h3>
                <span className="bg-violet-500/20 text-violet-300 border border-violet-500/30 text-[10px] uppercase font-bold px-2 py-0.5 rounded-full">
                  Archify Interactive Motion
                </span>
              </div>
              <p className="text-xs text-slate-400">Runtime Flow Architecture Spec</p>
            </div>
          </div>

          <button onClick={onClose} className="p-2 text-slate-400 hover:text-white bg-slate-800 rounded-xl">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Diagram Canvas Simulation */}
        <div className="bg-slate-950 border border-slate-800 rounded-xl p-8 min-h-[320px] flex flex-col items-center justify-center relative overflow-hidden">
          
          {/* SVG Diagram Nodes */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 w-full relative z-10">
            {steps.map((s, idx) => (
              <div
                key={idx}
                onClick={() => setActiveStep(idx)}
                className={`p-4 rounded-xl border cursor-pointer transition-all duration-300 ${
                  activeStep === idx
                    ? 'bg-violet-950/60 border-violet-500 text-violet-200 shadow-lg shadow-violet-500/20 scale-105'
                    : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:bg-slate-800/80'
                }`}
              >
                <div className="font-mono text-[10px] uppercase font-bold text-violet-400 mb-1">
                  0{idx + 1} // Phase
                </div>
                <h4 className="text-xs font-bold mb-1 text-slate-200">{s.title}</h4>
                <p className="text-[11px] text-slate-400 leading-tight">{s.desc}</p>
              </div>
            ))}
          </div>

          {/* Interactive Flow Indicator */}
          <div className="mt-8 p-4 bg-slate-900/80 border border-slate-800 rounded-xl max-w-lg w-full text-center">
            <div className="text-xs font-bold text-cyan-400 font-mono mb-1">
              CURRENT STEP STATE: {steps[activeStep].title}
            </div>
            <div className="text-xs text-slate-300">
              {steps[activeStep].desc}
            </div>
          </div>

        </div>

        {/* Footer Actions */}
        <div className="flex items-center justify-between pt-2 text-xs text-slate-400">
          <div className="flex items-center gap-2">
            <button
              onClick={() => setActiveStep((prev) => (prev + 1) % steps.length)}
              className="btn-primary text-xs py-1.5 px-3"
            >
              <Play className="w-3.5 h-3.5" /> Next Step
            </button>
            <button
              onClick={() => setActiveStep(0)}
              className="btn-secondary text-xs py-1.5 px-3"
            >
              <RotateCcw className="w-3.5 h-3.5" /> Reset
            </button>
          </div>

          <span className="font-mono text-[11px] text-slate-500">
            Powered by Archify v2.13 Interactive Renderer
          </span>
        </div>

      </div>
    </div>
  );
};
