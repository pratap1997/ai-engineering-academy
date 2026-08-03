import React, { useMemo } from 'react';
import katex from 'katex';

interface KaTeXRendererProps {
  math: string;
  block?: boolean;
  className?: string;
}

export const KaTeXRenderer: React.FC<KaTeXRendererProps> = ({
  math,
  block = false,
  className = '',
}) => {
  const html = useMemo(() => {
    try {
      // Clean raw text formatting if any
      const cleaned = math
        .replace(/\\text\{step\}/g, '\\operatorname{step}')
        .replace(/\\text\{where\}/g, '\\quad\\text{where}')
        .replace(/\\text\{Children\}/g, '\\text{Children}');

      return katex.renderToString(cleaned, {
        displayMode: block,
        throwOnError: false,
      });
    } catch (err) {
      return math;
    }
  }, [math, block]);

  if (block) {
    return (
      <div
        className={`katex-block p-4 my-3 bg-slate-900/90 border border-slate-800 rounded-xl overflow-x-auto text-slate-100 font-mono text-sm ${className}`}
        dangerouslySetInnerHTML={{ __html: html }}
      />
    );
  }

  return (
    <span
      className={`katex-inline inline-block px-1 text-indigo-300 font-mono text-xs ${className}`}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
};
