---
name: omni-route
description: >
  OmniRoute AI Gateway — Unified token-efficient routing across 290+ AI providers (GPT, Claude, Gemini, DeepSeek, Kimi, 90+ free options).
  Eliminates token quota concerns via RTK+Caveman 15-95% prompt compression, quota-aware auto-fallback, and a local OpenAI-compatible API proxy.
  Use when: running long Academy module generation sessions, hitting rate limits, or wanting to save tokens on research tasks.
triggers:
  - "omni route"
  - "omniroute"
  - "token limit"
  - "token saving"
  - "route tokens"
  - "free model"
  - "model fallback"
  - "quota exceeded"
---

# OmniRoute Skill — Token-Efficient AI Routing

## Quick Start

```powershell
# 1. Navigate to OmniRoute
cd c:\Users\Mahendra Pratap\Desktop\ai Learning\ai-engineering-academy\tools\omni-route

# 2. Install dependencies (first time only)
npm install

# 3. Start the gateway
npm start
# Gateway runs at http://localhost:4000/v1
# Dashboard at http://localhost:4001
```

## Using OmniRoute in Academy Sessions

Set your AI tool's API base URL to `http://localhost:4000/v1` and use any API key string.

### Model Priority for Academy Sessions
1. **gemini-2.5-pro** — Primary (large context, fast, free tier)
2. **claude-3-5-haiku** — Fallback (fast, cheap, great for code)
3. **deepseek-chat** — Free fallback (strong reasoning)
4. **kimi-k1.5** — Free fallback (long context)

## Token Compression

OmniRoute uses **RTK+Caveman compression** to reduce prompt size 15–95% before sending:
- Removes redundant whitespace, comments
- Compresses repetitive patterns
- Preserves semantic meaning

## Config File

See `tools/omni-route.config` for Academy-specific routing configuration.
