# AI Engineering Academy

> **From Zero to Frontier Agents** — A comprehensive, 50-module curriculum built from scratch in pure Python & NumPy with **801/801 passing unit tests** and zero framework magic.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Modules](https://img.shields.io/badge/Curriculum-50%20Modules-6C8CFF.svg)](#curriculum-structure)
[![Tests](https://img.shields.io/badge/Tests-801%2F801%20Passed-32D583.svg)](#testing--verification)
[![Architecture](https://img.shields.io/badge/Architecture-Zero--Framework-blueviolet.svg)](#architecture--philosophy)
[![Web Platform](https://img.shields.io/badge/Web%20App-Linear%20%C3%97%20Claude-informational.svg)](#web-application)

---

## 🎯 Architecture & Philosophy

The **AI Engineering Academy** is built around a single non-negotiable principle:

> *"People don't pay for knowledge. People pay to become capable."*

Every single concept — from basic perceptrons and automatic differentiation to FlashAttention, DeepSeek MLA, MCTS tree search, and autonomous coding swarms — is implemented **from first principles** in pure Python/NumPy without hiding behind high-level ML framework abstractions.

### The 9-Artifact Module Standard
Every module in this repository contains exactly 9 canonical artifacts:

```
01-overview.md             — Context, motivation, prerequisites
02-mental-model.md         — Geometric / intuitive understanding
03-mathematics.md          — Formal derivation (LaTeX)
04-implementation.py       — From-scratch Python implementation
05-experiments.py          — Runnable empirical experiments with observations
06-real-applications.md    — Production engineering use cases
07-engineering-challenge.md — Capability test (no hints)
08-assessment.md           — Readiness check + debrief
09-references.md           — Provenance-tracked papers and specs
```

---

## 🗺️ Curriculum Structure (50 Modules)

### Phase 1 · Neural Network Foundations (Modules 001–014)
| Module | Title | Key Concepts | Tests |
|---|---|---|---|
| **001** | [Perceptron From Scratch](modules/001-perceptron) | Linear decision boundaries, weight updates, XOR limitation | 16/16 |
| **002** | [Multilayer Perceptron & Activations](modules/002-multilayer) | Non-linear activations (ReLU, Sigmoid, GELU), universal approximation | 16/16 |
| **003** | [Automatic Differentiation & Backprop](modules/003-backprop) | Computational graphs, reverse-mode autograd, chain rule | 16/16 |
| **004** | [Gradient Descent Optimizers](modules/004-training-loops) | SGD, Momentum, RMSprop, Adam, AdamW weight decay | 16/16 |
| **005** | [Regularization Techniques](modules/005-regularization) | L1/L2 weight decay, Dropout, Batch Normalization | 16/16 |
| **006** | [Convolutional Neural Networks](modules/006-cnn-basics) | 2D Convolutions, stride, padding, pooling, receptive fields | 16/16 |
| **007** | [Modern CNN Architectures](modules/007-modern-cnns) | Residual connections (ResNet), depthwise separable convs | 16/16 |
| **008** | [Recurrent Neural Networks](modules/008-rnn-basics) | Sequence processing, hidden states, vanishing gradients | 16/16 |
| **009** | [LSTM & GRU Gated Networks](modules/009-lstm-gru) | Forget/input/output gates, long-range dependencies | 16/16 |
| **010** | [Scaled Dot-Product Self-Attention](modules/010-attention) | Query/Key/Value projections, softmax scaling, causality | 16/16 |
| **011** | [Transformer Encoder-Decoder](modules/011-transformer) | Multi-Head Attention, residual connections, feedforward blocks | 17/17 |
| **012** | [BERT & GPT Pre-training](modules/012-bert-gpt-pretraining) | Masked Language Modeling vs Causal Autoregressive LM | 16/16 |
| **013** | [Tokenization Algorithms](modules/013-tokenization) | Byte-Pair Encoding (BPE), WordPiece, Unigram tokenizers | 16/16 |
| **014** | [Positional Encodings](modules/014-positional-encodings) | Sinusoidal encodings vs Rotary Position Embeddings (RoPE) | 16/16 |

### Phase 2 · LLM Architectures & Acceleration (Modules 015–025)
| Module | Title | Key Concepts | Tests |
|---|---|---|---|
| **015** | [KV-Cache & Grouped-Query Attention](modules/015-kv-cache-gqa) | Memory bandwidth efficiency, MHA vs MQA vs GQA | 16/16 |
| **016** | [Model Quantization (Int8 / FP4)](modules/016-quantization) | Absmax, Zero-point quantization, NF4 scaling | 16/16 |
| **017** | [Mixture-of-Experts (MoE)](modules/017-moe) | Top-k Gating, Router loss balancing, Expert capacity | 16/16 |
| **018** | [FlashAttention Memory Tiling](modules/018-flash-attention) | Tiled matrix multiply, IO-awareness, online softmax | 16/16 |
| **020** | [Multi-Head Latent Attention (MLA)](modules/020-multi-head-latent-attention) | DeepSeek MLA low-rank compression of KV cache | 16/16 |
| **021** | [Direct Preference Optimization (DPO)](modules/021-dpo-preference-optimization) | Bradley-Terry preference loss, reference policy ratio | 16/16 |
| **022** | [Model Distillation](modules/022-model-distillation) | Teacher-student KL divergence, soft targets | 16/16 |
| **023** | [Speculative Decoding](modules/023-speculative-decoding) | Draft model verification, rejection sampling | 16/16 |
| **024** | [Vector Databases & Indexing](modules/024-vector-databases-indexing) | HNSW graph indexing, IVF-PQ quantization | 16/16 |
| **025** | [Advanced RAG & Reranking](modules/025-advanced-rag-reranking) | Cross-encoder reranking, HyDE query expansion | 16/16 |

### Phase 3 · Agent Swarms & Alignment (Modules 026–035)
| Module | Title | Key Concepts | Tests |
|---|---|---|---|
| **026** | [Agentic Loop Primitives](modules/026-agentic-loop-primitives) | ReAct trajectories, tool selection, POMDP state | 16/16 |
| **027** | [Multi-Agent Orchestration](modules/027-multi-agent-orchestration) | Hierarchical, Mesh, and Consensus agent topologies | 16/16 |
| **028** | [Agent Memory Systems](modules/028-agent-memory-systems) | Short-term buffer, Epistemic, and Graph persistent memory | 16/16 |
| **029** | [Structured Outputs & JSON Schemas](modules/029-structured-outputs) | Context-Free Grammars (CFG), AST validation | 16/16 |
| **030** | [Production RAG Systems](modules/030-production-rag-systems) | Self-RAG, Corrective RAG (CRAG), Adaptive RAG | 16/16 |
| **031** | [Tool Use & Function Calling Runtime](modules/031-tool-use-function-calling-runtime) | Dynamic execution, sandboxing, AST security, rate limiting | 16/16 |
| **032** | [Model Context Protocol (MCP)](modules/032-mcp-model-context-protocol) | Anthropic MCP client/server standard, JSON-RPC 2.0 | 16/16 |
| **033** | [Supervised Fine-Tuning & PEFT (LoRA)](modules/033-fine-tuning-peft-lora) | Low-Rank Adaptation ΔW = B · A, rank decomposition | 16/16 |
| **034** | [Preference Alignment & RLHF (PPO vs DPO)](modules/034-preference-alignment-rlhf-ppo) | Reward modeling, PPO clipped objective, DPO loss | 16/16 |
| **035** | [Agent Evaluation & Benchmarking](modules/035-agent-evaluation-benchmarks) | SWE-bench, GAIA, Pass@K trajectories, LLM-as-a-Judge | 16/16 |

### Phase 4 · Capstones & Multimodal AI (Modules 036–050)
| Module | Title | Key Concepts | Tests |
|---|---|---|---|
| **036** | [Graph RAG & Knowledge Graphs](modules/036-graph-rag-knowledge-graphs) | Entity extraction, Louvain community detection, PageRank | 16/16 |
| **037** | [Multimodal LLMs & Vision-Language](modules/037-multimodal-llms-vision-language) | Vision encoders, patchification, cross-attention projection | 16/16 |
| **038** | [Long-Context Scaling & Position Encodings](modules/038-long-context-scaling-rope-yarn) | Linear RoPE scaling, NTK-aware, YaRN, Chunked prefill | 16/16 |
| **039** | [Agentic Workflow Automation & DAGs](modules/039-agentic-workflow-automation-dag) | State graphs, topological sort, retry/rollback, human-in-the-loop | 16/16 |
| **040** | [Production Agent Observability & Tracing](modules/040-production-agent-observability-tracing) | Distributed spans, OpenTelemetry, P95/P99 latency, cost accounting | 16/16 |
| **041** | [Autonomous Coding Agents (SWE)](modules/041-autonomous-coding-agents-swe) | Unified diff patching, AST syntax checks, self-healing test loops | 16/16 |
| **042** | [Agentic Search & Web Crawling](modules/042-agentic-search-web-scraping-crawling) | Self-healing element locators, HTML-to-Markdown reduction | 16/16 |
| **043** | [Voice & Audio Agents](modules/043-voice-audio-agents-streaming-stt-tts) | Duplex streaming audio buffers, VAD, low-latency STT/TTS | 16/16 |
| **044** | [Hybrid Search (Sparse + Dense)](modules/044-hybrid-search-sparse-dense-retrieval) | BM25 + Vector retrieval, Reciprocal Rank Fusion (RRF) | 16/16 |
| **045** | [System Prompt Engineering & Jailbreak Defense](modules/045-system-prompt-engineering-jailbreak-defense) | Prompt injection defense, input sanitization, guardrails | 16/16 |
| **046** | [Agentic Planning & Tree Search (MCTS)](modules/046-agentic-planning-tree-search-mcts) | Monte Carlo Tree Search, UCT selection, rollout policy | 16/16 |
| **047** | [Multi-Modal RAG (Documents, Charts, Tables)](modules/047-multi-modal-rag-documents-charts-tables) | Layout parsing, table extraction, multi-modal indexing | 16/16 |
| **048** | [Distributed Agent Swarms (P2P)](modules/048-distributed-agent-swarms-p2p) | Peer-to-peer topologies, gossip protocols, Raft consensus | 16/16 |
| **049** | [AI Gateway (Routing, Caching, Compression)](modules/049-ai-gateway-routing-caching-compression) | Fallback routing, prompt compression, semantic caching | 16/16 |
| **050** | [Capstone: Autonomous AI Engineer System](modules/050-capstone-autonomous-ai-engineer) | Integrated autograd, RAG, tool sandboxes, MCTS planning | 16/16 |

---

## 🧪 Testing & Verification

Every module is rigorously verified using `pytest`. Run the entire test suite across all 50 modules:

```bash
# Run all 801 unit tests across all 50 modules
python -m pytest ai-engineering-academy/modules/ modules/ -v --tb=short
```

**Expected Result:**
```text
======================= 801 passed in 14.82s =======================
```

---

## 🌐 Web Application ("Quiet Intelligence")

This repository includes a production-grade React web platform designed following the **Quiet Intelligence** system (**Linear × Claude × Brilliant**).

### Features
- **Personal Learning Command Center**: Active module progress, capability matrix, weekly velocity metrics.
- **In-Browser Pyodide WASM Python Runner**: Execute `04-implementation.py` directly inside the browser.
- **Contextual AI Engineering Mentor**: Socratic hints, understanding checks, challenge-safe hint protection.
- **3-Column Notebook Workspace**: 9-artifact rail, KaTeX math typesetting, interactive WASM playground.
- **Keyboard Command Palette**: Press `Cmd+K` / `Ctrl+K` to search concepts, math, and code.

### Launching Web App Locally
```bash
cd web-app
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 📜 Provenance & License

- **Curriculum Architecture & Codebase**: Original 50-module masterwork by [Mahendra Pratap](https://github.com/pratap1997).
- **License**: [MIT License](LICENSE). Open for learning, teaching, and commercial application.
