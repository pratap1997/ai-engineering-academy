import React, { useState } from 'react';
import { Sliders, Sparkles, RefreshCw } from 'lucide-react';

interface InteractiveVisualizerProps {
  moduleId: string;
}

export const InteractiveVisualizer: React.FC<InteractiveVisualizerProps> = ({ moduleId }) => {
  const num = parseInt(moduleId, 10);

  // Module 010/011: Self-Attention Matrix State
  const [seqLength, setSeqLength] = useState<number>(4);
  const words = ['The', 'agent', 'solved', 'task', 'smoothly', 'today'].slice(0, seqLength);

  // Module 003: Gradient Descent Learning Rate
  const [lr, setLr] = useState<number>(0.1);
  const [step, setStep] = useState<number>(0);

  // Module 017: MoE Gating Router State
  const [activeExpert, setActiveExpert] = useState<number>(1);

  return (
    <div className="panel-card p-6 bg-[#121620] border-indigo-500/30 space-y-5 my-4">
      <div className="flex items-center justify-between border-b border-white/5 pb-3">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-indigo-400" />
          <h3 className="text-sm font-bold text-slate-100 font-heading">
            Interactive Architecture Visualizer
          </h3>
        </div>
        <span className="text-xs font-mono text-indigo-400 bg-indigo-500/10 px-2.5 py-0.5 rounded border border-indigo-500/20">
          Live Model
        </span>
      </div>

      {/* VISUALIZER FOR MODULE 010 / 011 (ATTENTION & TRANSFORMERS) */}
      {(num === 10 || num === 11) && (
        <div className="space-y-4">
          <div className="flex items-center justify-between text-xs font-mono text-slate-300">
            <span>Self-Attention Matrix: Softmax(QK^T / √d_k)</span>
            <div className="flex items-center gap-2">
              <Sliders className="w-3.5 h-3.5 text-slate-400" />
              <span>Tokens: {seqLength}</span>
              <input
                type="range"
                min="2"
                max="6"
                value={seqLength}
                onChange={(e) => setSeqLength(Number(e.target.value))}
                className="w-20 accent-indigo-500"
              />
            </div>
          </div>

          {/* Attention Heatmap Grid */}
          <div className="bg-[#090C10] p-4 rounded-xl border border-white/10 overflow-x-auto">
            <div className="grid gap-1 font-mono text-xs" style={{ gridTemplateColumns: `repeat(${seqLength + 1}, minmax(0, 1fr))` }}>
              {/* Header Row */}
              <div className="text-slate-500 font-bold p-1">Q \ K</div>
              {words.map((w, i) => (
                <div key={i} className="text-indigo-300 font-bold text-center p-1 truncate">
                  {w}
                </div>
              ))}

              {/* Rows */}
              {words.map((rowWord, r) => (
                <React.Fragment key={r}>
                  <div className="text-indigo-300 font-bold flex items-center p-1 truncate">
                    {rowWord}
                  </div>
                  {words.map((_, c) => {
                    const weight = r === c ? 0.85 : Math.abs(r - c) === 1 ? 0.45 : 0.15;
                    const opacity = Math.min(1, weight);
                    return (
                      <div
                        key={c}
                        className="h-9 rounded flex items-center justify-center text-[10px] text-white font-bold transition-all hover:scale-105"
                        style={{ backgroundColor: `rgba(99, 102, 241, ${opacity})` }}
                      >
                        {weight.toFixed(2)}
                      </div>
                    );
                  })}
                </React.Fragment>
              ))}
            </div>
          </div>
          <p className="text-xs text-slate-400 font-mono text-center">
            Darker indigo indicates higher attention score alignment between Query and Key tokens.
          </p>
        </div>
      )}

      {/* VISUALIZER FOR MODULE 003 / 004 (BACKPROP & GRADIENT DESCENT) */}
      {(num <= 5 && num !== 1) && (
        <div className="space-y-4">
          <div className="flex items-center justify-between text-xs font-mono text-slate-300">
            <span>Loss Surface Optimization: L(w) = (w - 2)^2</span>
            <div className="flex items-center gap-3">
              <span>LR: {lr}</span>
              <input
                type="range"
                min="0.05"
                max="0.3"
                step="0.05"
                value={lr}
                onChange={(e) => setLr(Number(e.target.value))}
                className="w-16 accent-indigo-500"
              />
              <button
                onClick={() => setStep((s) => s + 1)}
                className="btn-indigo text-xs py-1 px-3 min-h-0 flex items-center gap-1"
              >
                <RefreshCw className="w-3 h-3" /> Step
              </button>
            </div>
          </div>

          <div className="h-44 bg-[#090C10] rounded-xl border border-white/10 p-3 relative flex items-center justify-center">
            <svg className="w-full h-full" viewBox="0 0 200 120">
              <path d="M 10 20 Q 100 110 190 20" stroke="rgba(255,255,255,0.2)" strokeWidth="2" fill="none" />
              {/* Ball trajectory */}
              {(() => {
                const wVal = Math.max(0.2, 2.5 * Math.pow(1 - lr, step));
                const cx = 100 - wVal * 30;
                const cy = 100 - Math.pow(wVal, 2) * 12;
                return <circle cx={cx} cy={cy} r="6" fill="#32D583" className="transition-all duration-300" />;
              })()}
            </svg>
          </div>
          <p className="text-xs text-slate-400 font-mono text-center">
            Step {step}: Parameter w descending along negative gradient -∇L(w).
          </p>
        </div>
      )}

      {/* VISUALIZER FOR MODULE 017 (MIXTURE OF EXPERTS) */}
      {num >= 17 && (
        <div className="space-y-4">
          <div className="flex items-center justify-between text-xs font-mono text-slate-300">
            <span>MoE Router Gating Top-K Selection</span>
            <div className="flex gap-2">
              {[1, 2, 3, 4].map((e) => (
                <button
                  key={e}
                  onClick={() => setActiveExpert(e)}
                  className={`px-2.5 py-1 rounded text-xs font-mono ${
                    activeExpert === e ? 'bg-indigo-600 text-white font-bold' : 'bg-white/5 text-slate-400'
                  }`}
                >
                  Expert #{e}
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-4 gap-3 font-mono text-xs">
            {[1, 2, 3, 4].map((exp) => (
              <div
                key={exp}
                className={`p-4 rounded-xl border transition-all text-center space-y-1 ${
                  activeExpert === exp
                    ? 'bg-indigo-600/20 border-indigo-500 text-indigo-200 font-bold shadow-lg'
                    : 'bg-[#090C10] border-white/5 text-slate-500 opacity-60'
                }`}
              >
                <div>Expert {exp}</div>
                <div className="text-[10px]">{activeExpert === exp ? 'ACTIVE (Top-1)' : 'Bypassed'}</div>
              </div>
            ))}
          </div>
        </div>
      )}

    </div>
  );
};
