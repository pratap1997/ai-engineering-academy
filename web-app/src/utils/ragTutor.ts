import { MODULES } from '../modulesData';
import type { ModuleData } from '../types';

export interface RAGAnswer {
  query: string;
  response: string;
  matchedModules: ModuleData[];
}

export function queryAITutor(query: string): RAGAnswer {
  const cleanQuery = query.toLowerCase().trim();
  const queryTokens = cleanQuery.split(/\s+/).filter(Boolean);

  // Score each module based on keyword overlap in title, overview, math, code, topics
  const scoredModules = MODULES.map((m) => {
    let score = 0;

    const fullText = [
      m.id,
      m.title,
      m.subtitle,
      m.overview,
      m.mentalModelSummary,
      m.mathHighlight,
      m.codeSnippet,
      m.topics.join(' '),
      m.engineeringChallengeTitle,
      m.engineeringChallengeGoal,
    ]
      .join(' ')
      .toLowerCase();

    for (const token of queryTokens) {
      if (token.length <= 2) continue;
      if (m.title.toLowerCase().includes(token)) score += 10;
      if (m.topics.some((t) => t.toLowerCase().includes(token))) score += 8;
      if (m.mathHighlight.toLowerCase().includes(token)) score += 6;
      if (fullText.includes(token)) score += 2;
    }

    return { module: m, score };
  });

  // Sort modules by relevance score descending
  scoredModules.sort((a, b) => b.score - a.score);
  const matched = scoredModules
    .filter((sm) => sm.score > 0)
    .slice(0, 3)
    .map((sm) => sm.module);

  if (matched.length === 0) {
    return {
      query,
      response: `I searched across all 50 AI Engineering Academy modules, but I couldn't find a direct match for "${query}". Try asking about specific topics like "FlashAttention", "RoPE scaling", "DPO vs PPO", "Vector Databases", or "MCTS Planning".`,
      matchedModules: [MODULES[0]],
    };
  }

  const primary = matched[0];

  const responseText = `
### 💡 Academy AI Tutor Response for: "${query}"

**Primary Module Match:** Module **#${primary.id} — ${primary.title}**

#### 🧠 Key Concept & Intuition
${primary.overview}

#### 📐 Core Mathematical Derivation
\`\`\`latex
${primary.mathHighlight}
\`\`\`

#### 💻 Pure Python Implementation Teaser
\`\`\`python
${primary.codeSnippet.split('\n').slice(0, 8).join('\n')}
...
\`\`\`

#### 🛠️ Engineering Challenge
**${primary.engineeringChallengeTitle}**: ${primary.engineeringChallengeGoal}

---
*Referenced Modules:* ${matched.map((m) => `#${m.id} (${m.title})`).join(', ')}
`.trim();

  return {
    query,
    response: responseText,
    matchedModules: matched,
  };
}
