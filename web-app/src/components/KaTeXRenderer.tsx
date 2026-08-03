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
      return katex.renderToString(math, {
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
        className={`katex-block p-4 my-3 bg-[#121620] border border-white/10 rounded-xl overflow-x-auto text-slate-100 font-mono text-sm ${className}`}
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
