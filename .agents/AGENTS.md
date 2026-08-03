# AI Engineering Academy — Workspace Agent Rules

## Project Identity

This workspace is the **AI Engineering Academy** — a structured, curriculum-first learning system
that turns AI concepts into permanent, compounding capabilities. The root of all decisions is:

> "People don't pay for knowledge. People pay to become capable."

Every module, every artifact, every line of code must answer:
**"Does this make me — or a future learner — a better AI engineer?"**

---

## The Four Permanent Assets

All work should build one of these:

| Asset | What it includes |
|---|---|
| **Knowledge** | Concepts, math, papers, architectures, history |
| **Skills** | Implement, debug, optimize, evaluate, deploy |
| **Projects** | Real applications that demonstrate capability |
| **Judgment** | When to use what, and why, under constraints |

---

## Module Standard (Nine Artifacts)

Every module MUST produce exactly 9 artifacts in this sequence:

```
01-overview.md          — Context, motivation, prerequisites
02-mental-model.md      — Geometric / intuitive understanding
03-mathematics.md       — Formal derivation (LaTeX)
04-implementation.py    — From-scratch Python (no ML frameworks)
05-experiments.py       — Runnable experiments with observations
06-real-applications.md — 3+ production use cases
07-engineering-challenge.md — Capability test (no hints)
08-assessment.md        — Readiness check + debrief
09-references.md        — Provenance-tracked sources
```

Plus mandatory supporting files:
- `module.yaml` — Metadata (id, title, difficulty, prerequisites, estimated_minutes)
- `tests/test_<module>.py` — Full pytest suite (≥16 tests covering 4 categories)
- `ERRATA.md` — Known errors and corrections
- `CHANGELOG.md` — Version history

---

## Active Skills and When to Use Them

### Visual & Motion Design
- **`motion-design`** — Apply Disney animation principles (anticipation, follow-through, squash/stretch) to all interactive diagrams. Use `Energetic` archetype for neural signals.
- **`greensock/gsap-skills`** — Official GSAP animation skills for 60fps web animations, React timelines, and ScrollTrigger in module visualizers.
- **`nexu-io/html-anything`** — Agent-native HTML layout generator (75 template skills) for YouTube tutorial slide decks, posters, and lab dashboards.
- **`ui-ux-pro-max`** — Design system decisions: palettes, typography, component patterns, accessibility. Run `python skills/ui-ux-pro-max/scripts/search.py` for lookups.

### Agentic Loop Engineering & Multi-Agent Architecture
- **`cobusgreyling/loop-engineering`** — Loop primitives (Schedule, Triage, State, Worktree, Verifier) for autonomous agent loops (triage, daily progress logging, automated testing).
- **`omnigent-ai/omnigent`** — Meta-harness architecture for multi-model debates and sandboxed evaluation of learner challenge submissions.
- **`hesreallyhim/awesome-claude-code`** — Canonical index for high-quality slash commands, CLAUDE.md patterns, and vetted agent skills.
- **`NVIDIA/skills`** — Official NVIDIA instruction sets for CUDA-X, PyTorch GPU acceleration (Module 004+), and SkillSpector threat auditing.

### Tier 2 — Available After `npm install -g @claude-flow/cli` (see SETUP_SKILLS.md)

- **`sparc-methodology`** — Structured planning: Spec → Pseudocode → Arch → Refine → Complete. Use for every new module.
- **`swarm-orchestration`** — Hierarchical multi-agent task execution. Use when generating 3+ module files simultaneously.
- **`workflow-automation`** — Reusable YAML-defined multi-step pipelines. Use for automated curriculum generation.
- **`security-audit`** — Input validation, path traversal, CVE scanning. Run before publishing Python code.
- **`pair-programming`** — TDD collaborative mode. Use for all 04-implementation.py files. Run with `--mode tdd`.
- **`github-automation`** — PR creation, issue tracking, CI/CD. Needs `gh` CLI + `claude-flow` CLI.
- **`github-workflow-automation`** — GitHub Actions pipeline management.
- **`verification-quality`** — 0.95 truth-score threshold, auto-rollback. Needs `ruflo@alpha` CLI.
- **`neural-training`** — SONA, MoE, EWC++ pattern training. Reserved for Module 003+.

### Tier 3 — Future Phase (RAG Knowledge OS, 5+ modules complete)
*Do not install until 5 modules are finished — rule: "never build infrastructure before it saves you time"*

- **`agentdb-vector-search`** — Semantic search across all module content (RAG tutor). Needs `npx agentdb@latest`.
- **`agentdb-memory-patterns`** — Learner progress tracking, misconception logging. Needs `agentdb` + `agentic-flow`.
- **`agentdb-optimization`** — HNSW indexing at scale (1000+ knowledge entries). Needs `agentdb`.
- **`embeddings`** — Module concept similarity. Needs `claude-flow` CLI + ONNX models.
- **`graphify`** — AI concept knowledge graph. Needs `pip install graphifyy` + optional Gemini API key.
- **`defuddle`** — Web content extraction. Needs `npm install -g defuddle`.

### Global Skills from Repository Integration (2026-08-02)
*Added from 17-repo analysis — all installed as global skills in `~/.gemini/config/skills/`*

- **`firecrawl`** — Convert any website to LLM-ready Markdown; build RAG datasets, enrich module knowledge. (firecrawl.dev API or self-hosted Docker)
- **`langchain-reference`** — Production LangChain code patterns: RAG pipelines, ReAct agents, LangGraph multi-agent workflows. Reference for module "Use It" sections.
- **`docling`** — IBM document parser (PDF/DOCX/PPTX → Markdown/JSON with table + equation preservation). Preferred over markitdown for complex academic papers.
- **`scrapling`** — Self-healing, adaptive web scraping with anti-bot bypass. Use for dataset acquisition and training data pipelines.
- **`taste-skill`** — AI design quality rules (3 configurable dials) for premium, non-generic frontend UI. Use when building module visualizers.
- **`meilisearch`** — Hybrid search (BM25 + vector/semantic, sub-50ms). Use when building searchable knowledge bases or hybrid RAG retrieval.
- **`voicebox`** — Local voice AI studio (TTS + voice cloning + STT via Whisper). 100% private, REST API for agent integration.
- **`system-prompts-reference`** — Archive of leaked/disclosed system prompts from Claude, GPT-4, Gemini, Cursor. Use for prompt engineering modules 026+.

### Future Module Content Sources (from 17-repo analysis)

| Module | Source Repo | Topic |
|---|---|---|
| 026 | `microsoft/autogen` + `openclaw/openclaw` | Agentic Loop Primitives (ReAct, Tool Use) |
| 027 | `microsoft/autogen` + `langchain-ai/langchain` | Multi-Agent Orchestration |
| 028 | `mem0ai/mem0` | Agent Memory (Vector + Graph persistent) |
| 030 | `infiniflow/ragflow` | Production RAG Systems |
| 033 | `hiyouga/LlamaFactory` | Unified LLM Fine-Tuning (LoRA, QLoRA, DPO) |
| 034+ | `punkpeye/awesome-mcp-servers` | MCP Tool Integration for Agents |

*Reference repos: `rasbt/LLMs-from-scratch` (Modules 003-014 enrichment), `microsoft/generative-ai-for-beginners` (real-world applications)*

---

## Non-Negotiable Rules

1. **Never build infrastructure before it saves you time.** No RAG system until 5+ modules exist.
2. **Every module must pass all tests before merging.** `C:\Python314\python.exe -m pytest modules/ -v --tb=short` must show 0 failures.
3. **All sources must be provenance-tracked.** Every claim in a module needs a `module.yaml` source entry.
4. **Labels are always 0/1 binary.** Consistent across all perceptron and classification modules.
5. **XOR is a geometric limitation.** Always treat it as proof of single-layer failure, not just an example.
6. **The curriculum is the competitive advantage.** Code quality matters, but explanation quality matters more.
7. **One learner must complete a module before the next one is built.** Validate before expanding.

---

## Current Module Status

| Module | Status | Tests |
|---|---|---|
| `001-perceptron` | ✅ Complete | 16/16 passing |
| `002-multilayer` | ✅ Complete | 16/16 passing |
| `003-backprop` | ✅ Complete | 16/16 passing |
| `004-training-loops` | ✅ Complete | 16/16 passing |
| `005-regularization` | ✅ Complete | 16/16 passing |
| `006-cnn-basics` | ✅ Complete | 16/16 passing |
| `007-modern-cnns` | ✅ Complete | 16/16 passing |
| `008-rnn-basics` | ✅ Complete | 16/16 passing |
| `009-lstm-gru` | ✅ Complete | 16/16 passing |
| `010-attention` | ✅ Complete | 16/16 passing |
| `011-transformer` | ✅ Complete | 17/17 passing |
| `012-bert-gpt-pretraining` | ✅ Complete | 16/16 passing |
| `013-tokenization` | ✅ Complete | 16/16 passing |
| `014-positional-encodings` | ✅ Complete | 16/16 passing |
| `015-kv-cache-gqa` | ✅ Complete | 16/16 passing |
| `016-quantization` | ✅ Complete | 16/16 passing |
| `017-moe` | ✅ Complete | 16/16 passing |
| `018-flash-attention` | ✅ Complete | 16/16 passing |
| `020-multi-head-latent-attention` | ✅ Complete | 16/16 passing |
| `021-dpo-preference-optimization` | ✅ Complete | 16/16 passing |
| `022-model-distillation` | ✅ Complete | 16/16 passing |
| `023-speculative-decoding` | ✅ Complete | 16/16 passing |
| `024-vector-databases-indexing` | ✅ Complete | 16/16 passing |
| `025-advanced-rag-reranking` | ✅ Complete | 16/16 passing |
| `026-agentic-loop-primitives` | ✅ Complete | 16/16 passing |
| `027-multi-agent-orchestration` | ✅ Complete | 16/16 passing |
| `028-agent-memory-systems` | ✅ Complete | 16/16 passing |
| `029-structured-outputs` | ✅ Complete | 16/16 passing |
| `030-production-rag-systems` | ✅ Complete | 16/16 passing |
| `031-tool-use-function-calling-runtime` | ✅ Complete | 16/16 passing |
| `032-mcp-model-context-protocol` | ✅ Complete | 16/16 passing |
| `033-fine-tuning-peft-lora` | ✅ Complete | 16/16 passing |
| `034-preference-alignment-rlhf-ppo` | ✅ Complete | 16/16 passing |
| `035-agent-evaluation-benchmarks` | ✅ Complete | 16/16 passing |
| `036-graph-rag-knowledge-graphs` | ✅ Complete | 16/16 passing |
| `037-multimodal-llms-vision-language` | ✅ Complete | 16/16 passing |
| `038-long-context-scaling-rope-yarn` | ✅ Complete | 16/16 passing |
| `039-agentic-workflow-automation-dag` | ✅ Complete | 16/16 passing |
| `040-production-agent-observability-tracing` | ✅ Complete | 16/16 passing |
| `041-autonomous-coding-agents-swe` | ✅ Complete | 16/16 passing |
| `042-agentic-search-web-scraping-crawling` | ✅ Complete | 16/16 passing |
| `043-voice-audio-agents-streaming-stt-tts` | ✅ Complete | 16/16 passing |
| `044-hybrid-search-sparse-dense-retrieval` | ✅ Complete | 16/16 passing |
| `045-system-prompt-engineering-jailbreak-defense` | ✅ Complete | 16/16 passing |
| `046-agentic-planning-tree-search-mcts` | ✅ Complete | 16/16 passing |
| `047-multi-modal-rag-documents-charts-tables` | ✅ Complete | 16/16 passing |
| `048-distributed-agent-swarms-p2p` | ✅ Complete | 16/16 passing |
| `049-ai-gateway-routing-caching-compression` | ✅ Complete | 16/16 passing |
| `050-capstone-autonomous-ai-engineer` | ✅ Complete | 16/16 passing |

> **Total: 801/801 tests passing — 50 of 50 modules complete (100%)**

---

## Test Command

```powershell
# Run ALL modules (001-050): earlier modules in ai-engineering-academy/modules/, newer in modules/
C:\Python314\python.exe -m pytest ai-engineering-academy/modules/ modules/ -v --tb=short

# Run only new modules (026-050):
C:\Python314\python.exe -m pytest modules/ -v --tb=short

# Run only legacy modules (001-025):
C:\Python314\python.exe -m pytest ai-engineering-academy/modules/ -v --tb=short
```

---

## Skill Decision Tree

```
New module needed?
  → sparc-methodology (plan first) [Tier 2]
  → swarm-orchestration (3+ files) [Tier 2]
  → agent-coder + agent-tester (implementations) [Tier 1]

Implementation file (04-*.py)?
  → pair-programming TDD mode [Tier 2]
  → agent-reviewer before commit [Tier 1]
  → security-audit before publish [Tier 2]

Visual/UI needed?
  → motion-design (animation principles) [Tier 1]
  → ui-ux-pro-max (design system) [Tier 1]

Research/sourcing needed?
  → agent-researcher (papers + synthesis) [Tier 1]
  → markitdown (PDF → markdown) [Tier 1, installed]
  → defuddle (scrape docs) [Tier 3]

Knowledge base query (5+ modules)?
  → agentdb-vector-search (RAG) [Tier 3]
  → graphify (concept graph) [Tier 3]
```

---

## File Structure

```
ai-engineering-academy/
├── modules/
│   ├── 001-perceptron/   ✅ 16 tests
│   ├── 002-multilayer/   ✅ 16 tests
│   ├── 003-backprop/     ✅ 16 tests
│   ├── 004-training-loops/ ✅ 16 tests
│   ├── 005-regularization/ ✅ 16 tests
│   ├── 006-cnn-basics/   ✅ 16 tests
│   ├── 007-modern-cnns/  ✅ 16 tests
│   ├── 008-rnn-basics/   ✅ 16 tests
│   └── 009-lstm-gru/     ✅ 16 tests
├── tools/
│   ├── omni-route/       ← OmniRoute AI Gateway (cloned, 10,563 files)
│   └── omni-route.config ← Academy routing config (token compression ON)
├── .github/workflows/test.yml
├── SOURCES.md
├── CONTRIBUTING.md
└── pyproject.toml

.agents/
├── AGENTS.md         ← this file
├── skills/
│   └── omni-route/SKILL.md ← OmniRoute gateway skill
├── skills.json       ← 24 selected skills
└── SETUP_SKILLS.md   ← install guide for Tier 2 & 3
```

## Token Efficiency Infrastructure

### OmniRoute AI Gateway
- **Location**: `tools/omni-route/` (cloned from github.com/diegosouzapw/OmniRoute)
- **Purpose**: Routes AI requests across 290+ providers, 90+ free models, with 15-95% RTK+Caveman prompt compression
- **Start**: `cd tools/omni-route && npm install && npm start`
- **Endpoint**: `http://localhost:4000/v1` (OpenAI-compatible)
- **Config**: `tools/omni-route.config` (Academy model priority: gemini-2.5-pro → claude-3-5-haiku → deepseek-chat → kimi-k1.5)
- **Skill**: `.agents/skills/omni-route/SKILL.md`
