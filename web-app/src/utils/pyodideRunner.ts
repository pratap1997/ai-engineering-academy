declare global {
  interface Window {
    loadPyodide?: (config: { indexURL: string }) => Promise<any>;
    pyodide?: any;
  }
}

let pyodideInstance: any = null;
let isLoadingPyodide = false;

export async function getPyodideInstance(): Promise<any> {
  if (pyodideInstance) return pyodideInstance;

  if (isLoadingPyodide) {
    while (isLoadingPyodide) {
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
    return pyodideInstance;
  }

  isLoadingPyodide = true;

  try {
    if (!window.loadPyodide) {
      await new Promise<void>((resolve, reject) => {
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js';
        script.onload = () => resolve();
        script.onerror = () => reject(new Error('Failed to load Pyodide WebAssembly script.'));
        document.head.appendChild(script);
      });
    }

    if (window.loadPyodide) {
      pyodideInstance = await window.loadPyodide({
        indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/',
      });
    }
    isLoadingPyodide = false;
    return pyodideInstance;
  } catch (err) {
    isLoadingPyodide = false;
    throw err;
  }
}

export interface PyodideResult {
  output: string;
  error?: string;
  executionTimeMs: number;
}

export async function runPythonCode(code: string): Promise<PyodideResult> {
  const startTime = performance.now();
  try {
    const pyodide = await getPyodideInstance();

    // Redirect stdout & stderr
    const setupOutputBuffer = `
import sys
import io
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()
`;
    await pyodide.runPythonAsync(setupOutputBuffer);

    // Execute user code
    let error: string | undefined = undefined;
    try {
      await pyodide.runPythonAsync(code);
    } catch (e: any) {
      error = e.message || String(e);
    }

    // Retrieve captured stdout & stderr
    const retrieveOutput = `
out = sys.stdout.getvalue()
err = sys.stderr.getvalue()
out + ("\\n--- ERRORS ---\\n" + err if err else "")
`;
    const output = await pyodide.runPythonAsync(retrieveOutput);
    const endTime = performance.now();

    return {
      output: output.trim() || '(Executed successfully with no output)',
      error,
      executionTimeMs: Math.round(endTime - startTime),
    };
  } catch (err: any) {
    const endTime = performance.now();
    return {
      output: '',
      error: err.message || String(err),
      executionTimeMs: Math.round(endTime - startTime),
    };
  }
}
