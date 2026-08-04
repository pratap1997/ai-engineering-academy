import React, { useState, useEffect } from 'react';
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

  // Sync internal code state when switching modules
  useEffect(() => {
    setCode(initialCode);
    setResult(null);
  }, [initialCode]);

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
      <div className="bg-[#090C10] border border-white/10 rounded-xl overflow-hidden shadow-xl">
        <div className="flex items-center justify-between px-4 py-2.5 bg-[#121620] border-b border-white/10 text-xs text-slate-300">
          <div className="flex items-center gap-2 font-mono">
            <TerminalIcon className="w-4 h-4 text-indigo-400" />
            <span className="font-bold">Python WASM Editor ({moduleTitle})</span>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleReset}
              className="btn-subtle text-xs py-1.5 px-3 min-h-0"
              disabled={isRunning}
            >
              <RotateCcw className="w-3.5 h-3.5" /> Reset
            </button>
            <button
              onClick={handleRun}
              disabled={isRunning}
              className="btn-indigo text-xs py-1.5 px-4 min-h-0 bg-emerald-600 hover:bg-emerald-500 text-white font-bold flex items-center gap-1.5"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              {isRunning ? 'Running WASM...' : 'Execute Code'}
            </button>
          </div>
        </div>

        {/* Text Area Code Editor */}
        <textarea
          value={code}
          onChange={(e) => setCode(e.target.value)}
          spellCheck={false}
          className="w-full h-80 p-4 bg-[#090C10] font-mono text-xs text-indigo-200 focus:outline-none focus:border-indigo-500/50 resize-y leading-relaxed"
        />
      </div>

      {/* Execution Output Console */}
      {isRunning && (
        <div className="p-4 bg-indigo-950/20 border border-indigo-500/40 rounded-xl flex items-center gap-3 text-xs text-indigo-300 font-mono animate-pulse">
          <div className="w-4 h-4 rounded-full border-2 border-indigo-400 border-t-transparent animate-spin" />
          <span>Initialising Pyodide WASM Runtime & Executing Code...</span>
        </div>
      )}

      {result && (
        <div className="bg-[#090C10] border border-white/10 rounded-xl p-4 space-y-2 font-mono text-xs shadow-lg">
          <div className="flex items-center justify-between text-slate-400 border-b border-white/5 pb-2">
            <div className="flex items-center gap-2">
              {result.error ? (
                <span className="flex items-center gap-1.5 text-rose-400 font-bold">
                  <AlertTriangle className="w-4 h-4" /> Execution Error
                </span>
              ) : (
                <span className="flex items-center gap-1.5 text-emerald-400 font-bold">
                  <CheckCircle2 className="w-4 h-4" /> Execution Successful
                </span>
              )}
            </div>

            <div className="flex items-center gap-1.5 text-xs text-slate-400">
              <Clock className="w-3.5 h-3.5" /> {result.executionTimeMs} ms
            </div>
          </div>

          {result.output && (
            <pre className="text-slate-200 whitespace-pre-wrap overflow-x-auto max-h-56 pt-1 leading-relaxed">
              {result.output}
            </pre>
          )}

          {result.error && (
            <pre className="text-rose-300 whitespace-pre-wrap overflow-x-auto max-h-56 pt-1 bg-rose-950/20 p-3 rounded-lg border border-rose-900/40 leading-relaxed">
              {result.error}
            </pre>
          )}
        </div>
      )}

    </div>
  );
};
