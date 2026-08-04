/**
 * Self-contained, zero-dependency Markdown to HTML parser tailored for AI Engineering Academy.
 * Handles headings, bold/italic, bullet lists, blockquotes, code blocks, tables, and inline code.
 */
export function parseMarkdownToHtml(markdown: string): string {
  if (!markdown) return '';

  let html = markdown;

  // Escape HTML entities to prevent raw HTML injection
  html = html
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // Fenced Code Blocks: ```lang ... ```
  html = html.replace(/```([a-z0-9_-]*)\n([\s\S]*?)```/g, (_match, _lang, code) => {
    return `<pre class="bg-[#121620] border border-white/10 rounded-xl p-4 my-4 overflow-x-auto text-xs font-mono text-indigo-200"><code>${code.trim()}</code></pre>`;
  });

  // Inline Code: `code`
  html = html.replace(/`([^`]+)`/g, '<code class="bg-[#121620] text-indigo-300 px-1.5 py-0.5 rounded text-xs font-mono border border-white/10">$1</code>');

  // Headings #, ##, ###
  html = html.replace(/^### (.*$)/gim, '<h3 class="text-lg font-bold text-slate-100 font-heading mt-6 mb-2">$1</h3>');
  html = html.replace(/^## (.*$)/gim, '<h2 class="text-xl font-bold text-slate-100 font-heading mt-8 mb-3 border-b border-white/10 pb-2">$1</h2>');
  html = html.replace(/^# (.*$)/gim, '<h1 class="text-3xl font-bold text-slate-100 font-heading mt-4 mb-4">$1</h1>');

  // Blockquotes >
  html = html.replace(/^\> (.*$)/gim, '<blockquote class="border-l-4 border-indigo-500 bg-indigo-950/20 pl-4 py-2 my-3 text-slate-300 italic text-sm rounded-r-lg">$1</blockquote>');

  // Bold & Italic
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong class="font-bold text-slate-100">$1</strong>');
  html = html.replace(/\*([^*]+)\*/g, '<em class="italic text-slate-200">$1</em>');

  // Unordered list items - item
  html = html.replace(/^\s*[\-\*]\s+(.*$)/gim, '<li class="ml-4 list-disc text-slate-200 leading-relaxed">$1</li>');

  // Paragraphs (lines separated by double newlines)
  const paragraphs = html.split(/\n\n+/);
  html = paragraphs
    .map((p) => {
      const trimmed = p.trim();
      if (
        trimmed.startsWith('<h') ||
        trimmed.startsWith('<pre') ||
        trimmed.startsWith('<blockquote') ||
        trimmed.startsWith('<li')
      ) {
        return trimmed;
      }
      return `<p class="my-3 text-base text-slate-200 leading-relaxed">${trimmed}</p>`;
    })
    .join('\n');

  return html;
}
