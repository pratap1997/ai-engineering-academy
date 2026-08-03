import React, { useState } from 'react';
import { runPythonCode } from '../utils/pyodideRunner';
import type { PyodideResult } from '../utils/pyodideRunner';
import { Play, RotateCcw, Terminal as TerminalIcon, Clock, AlertTriangle, CheckCircle2 } from 'lucide-react';

interface CodePlaygroundProps {
  initialCode: string;
  moduleTitle: string;
}

export const CodePlayground: React.FC<CodePlaygroundProps> = ({ initialCode, moduleTitle }) => {
  const [code, setCode] = useState(initialCode);
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<PyodideResult | null>(null);

  const handleRun = async () => {
    setIsRunning(true);
    setResult(null);
    try {
      const res = await runPythonCode(code);
      setResult(res);
    } catch (err: any) {
      setResult({
        output: '',
        error: err.message || String(err),
        executionTimeMs: 0,
      });
    } finally {
      setIsRunning(false);
    }
  };

  const handleReset = () => {
    setCode(initialCode);
    setResult(null);
  };

  return (
    <div className="space-y-4">
      
      {/* Code Editor Panel */}
      <div className="bg-slate-950 border border-slate-800 rounded-xl overflow-hidden shadow-inner">
        <div className="flex items-center justify-between px-4 py-2 bg-slate-900 border-b border-slate-800 text-xs text-slate-400">
          <div className="flex items-center gap-2 font-mono">
            <TerminalIcon className="w-3.5 h-3.5 text-cyan-400" />
            <span>Interactive Python WASM Editor ({moduleTitle})</span>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleReset}
              className="btn-secondary text-[11px] py-1 px-2.5"
              disabled={isRunning}
            >
              <RotateCcw className="w-3 h-3" /> Reset Code
            </button>
            <button
              onClick={handleRun}
              disabled={isRunning}
              className="btn-primary text-[11px] py-1 px-3 bg-emerald-600 hover:bg-emerald-500 border-emerald-400 text-white"
            >
              <Play className="w-3 h-3 fill-current" />
              {isRunning ? 'Running WASM...' : 'Execute Code'}
            </button>
          </div>
        </div>

        {/* Text Area Code Editor */}
        <textarea
          value={code}
          onChange={(e) => setCode(e.target.value)}
          spellCheck={false}
          className="w-full h-64 p-4 bg-slate-950 font-mono text-xs text-slate-200 focus:outline-none focus:ring-1 focus:ring-cyan-500/50 resize-y leading-relaxed"
        />
      </div>

      {/* Execution Output Console */}
      {isRunning && (
        <div className="p-4 bg-slate-900/90 border border-cyan-500/40 rounded-xl flex items-center gap-3 text-xs text-cyan-300 animate-pulse">
          <div className="w-4 h-4 rounded-full border-2 border-cyan-400 border-t-transparent animate-spin" />
          <span>Initialising Pyodide WASM Runtime & Executing Code...</span>
        </div>
      )}

      {result && (
        <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-2 font-mono text-xs">
          <div className="flex items-center justify-between text-slate-400 border-b border-slate-800 pb-2">
            <div className="flex items-center gap-2">
              {result.error ? (
                <span className="flex items-center gap-1 text-rose-400 font-bold">
                  <AlertTriangle className="w-3.5 h-3.5" /> Execution Error
                </span>
              ) : (
                <span className="flex items-center gap-1 text-emerald-400 font-bold">
                  <CheckCircle2 className="w-3.5 h-3.5" /> Execution Successful
                </span>
              )}
            </div>

            <div className="flex items-center gap-1 text-[11px] text-slate-500">
              <Clock className="w-3 h-3" /> {result.executionTimeMs} ms
            </div>
          </div>

          {result.output && (
            <pre className="text-slate-200 whitespace-pre-wrap overflow-x-auto max-h-48 pt-1">
              {result.output}
            </pre>
          )}

          {result.error && (
            <pre className="text-rose-400 whitespace-pre-wrap overflow-x-auto max-h-48 pt-1 bg-rose-950/20 p-3 rounded border border-rose-900/40">
              {result.error}
            </pre>
          )}
        </div>
      )}

    </div>
  );
};
