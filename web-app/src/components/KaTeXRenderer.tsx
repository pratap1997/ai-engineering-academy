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
        output: 'html', // Output clean HTML only without duplicating MathML string
      });
    } catch (err) {
      return math;
    }
  }, [math, block]);

  if (block) {
    return (
      <div
        className={`katex-block p-5 my-4 bg-[#121620] border border-white/10 rounded-xl overflow-x-auto text-slate-100 font-mono text-base md:text-lg flex justify-center ${className}`}
        dangerouslySetInnerHTML={{ __html: html }}
      />
    );
  }

  return (
    <span
      className={`katex-inline inline-block px-1 text-indigo-300 font-mono text-sm ${className}`}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
};
