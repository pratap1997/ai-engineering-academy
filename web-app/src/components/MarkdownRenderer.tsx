import React, { useMemo } from 'react';
import { parseMarkdownToHtml } from '../utils/markdownParser';

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content, className = '' }) => {
  const htmlContent = useMemo(() => {
    return parseMarkdownToHtml(content);
  }, [content]);

  return (
    <div
      className={`markdown-body space-y-4 font-sans text-base leading-relaxed text-slate-200 ${className}`}
      dangerouslySetInnerHTML={{ __html: htmlContent }}
    />
  );
};
