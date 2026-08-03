# Source and Licence Governance

This file records every external source consulted in building the AI Engineering Academy.
It is updated whenever a new source is consulted, adapted, or cited.

**Safe default for all sources:**
Read broadly. Cite precisely. Write independently. Create original diagrams.
Implement from first principles. Never assume a public repository means all content is reusable.

---

## Format

Each source entry records:

| Field | Description |
|---|---|
| Title | Full name of the source |
| Author / Org | Author or maintaining organization |
| Repository / Publication | Canonical location |
| Path | Exact file or section consulted |
| Commit / Version | Pinned when consulted |
| Access Date | ISO date (YYYY-MM-DD) |
| Licence | Exact licence type |
| Use Type | `concept_seed` / `reference` / `cited` / `adapted` |
| Adapted Material | Describe exactly what was adapted, if anything |
| Attribution Location | Where attribution appears in the academy |
| Notes | Restrictions, warnings, or special considerations |

---

## Active Sources

### Microsoft AI For Beginners

| Field | Value |
|---|---|
| **Title** | Microsoft AI For Beginners |
| **Author / Org** | Microsoft |
| **Repository** | https://github.com/microsoft/AI-For-Beginners |
| **Path** | `lessons/3-NeuralNetworks/03-Perceptron/README.md` |
| **Commit** | *pin when consulted* |
| **Access Date** | 2026-08-01 |
| **Licence** | MIT |
| **Use Type** | `concept_seed` |
| **Adapted Material** | None. Module 01 is independently written using this as a concept seed. |
| **Attribution** | `modules/001-perceptron/09-references.md`, `modules/001-perceptron/module.yaml` |
| **Notes** | MIT licence permits use with attribution. The module structure, experiments, challenge, and assessment are original. |

---

## Reference Sources (Private Study Only)

### fast.ai / fastbook

| Field | Value |
|---|---|
| **Title** | Practical Deep Learning for Coders (fastbook) |
| **Author / Org** | Jeremy Howard, Sylvain Gugger / fast.ai |
| **Repository** | https://github.com/fastai/fastbook |
| **Licence** | Code: GPLv3. Prose and notebook narrative: significant redistribution and commercial-use restrictions. |
| **Use Type** | `reference` — private study only |
| **Adapted Material** | None. Not to be copied or closely adapted. |
| **Notes** | Do not incorporate fastbook prose, diagrams, or notebook text into academy modules. GPL code is not compatible with academy MIT licence without deliberate acceptance of licence consequences. |

### Dive into Deep Learning

| Field | Value |
|---|---|
| **Title** | Dive into Deep Learning |
| **Author / Org** | Aston Zhang, Zachary Lipton, Mu Li, Alexander Smola |
| **Repository** | https://github.com/d2l-ai/d2l-en |
| **Licence** | Book content: CC BY-SA 4.0. Sample code: modified MIT-style. |
| **Use Type** | `reference` |
| **Adapted Material** | None yet. If book prose or figures are adapted, share-alike obligations may apply. |
| **Notes** | Prefer learning from it, citing it, and writing independent explanations with original diagrams. Confirm exact licence on any specific file before adapting. |

### Sebastian Raschka — Machine Learning Book Repo

| Field | Value |
|---|---|
| **Title** | Machine Learning with PyTorch and Scikit-Learn |
| **Author / Org** | Sebastian Raschka |
| **Repository** | https://github.com/rasbt/machine-learning-book |
| **Licence** | Code: MIT (repository). Book prose: separately copyrighted. |
| **Use Type** | `reference` |
| **Adapted Material** | None. Treat code and book prose separately. Do not reproduce book prose. |
| **Notes** | Code examples under MIT are usable with attribution. Book text is not. |

### Made With ML

| Field | Value |
|---|---|
| **Title** | Made With ML |
| **Author / Org** | Goku Mohandas |
| **Repository** | https://github.com/GokuMohandas/Made-With-ML |
| **Licence** | Check repository for exact terms |
| **Use Type** | `reference` — engineering and MLOps patterns |
| **Adapted Material** | None |
| **Notes** | Relevant for later engineering modules on deployment, testing, and production systems. |

---

## Deferred Sources (Not Yet Consulted)

These sources are recorded for future reference once the relevant modules are planned.

| Source | Intended Use | Deferred Until |
|---|---|---|
| Raschka — testing-ml | ML testing patterns | Module on evaluation |
| Official PyTorch docs | Framework reference | Module 004+ |
| Official NumPy docs | Array operations | Module 001 implementation |
| Rosenblatt (1958) original paper | Historical / authoritative | Module 001 references |
| FAISS documentation | Vector search | RAG modules |
| LangGraph documentation | Agent orchestration | Agent modules |

---

## Licence Rules Summary

| Source | Permitted | Not Permitted |
|---|---|---|
| MIT repositories | Use, modify, redistribute with attribution | Remove licence notice |
| CC BY-SA 4.0 | Use, adapt, redistribute with attribution | Use under incompatible licence |
| GPLv3 | Use privately, study | Incorporate into MIT project without GPL acceptance |
| Restrictive prose | Read privately | Copy, adapt, or publish |
| Public repository | Read and cite | Assume reusability without checking licence |


---

### hesreallyhim/awesome-claude-code

| Field | Value |
|---|---|
| **Title** | Awesome Claude Code |
| **Author / Org** | @hesreallyhim |
| **Repository** | https://github.com/hesreallyhim/awesome-claude-code |
| **Access Date** | 2026-08-01 |
| **Licence** | MIT |
| **Use Type** | `reference` |
| **Adapted Material** | Curated index for agent skills, slash commands, `CLAUDE.md` patterns, and hooks integration. |
| **Attribution** | `SOURCES.md`, `.agents/AGENTS.md` |
| **Notes** | Canonical index for upgrading workspace agent capabilities and CLI slash commands. |

---

### cobusgreyling/loop-engineering

| Field | Value |
|---|---|
| **Title** | Loop Engineering |
| **Author / Org** | Cobus Greyling |
| **Repository** | https://github.com/cobusgreyling/loop-engineering |
| **Access Date** | 2026-08-01 |
| **Licence** | MIT |
| **Use Type** | `concept_seed` |
| **Adapted Material** | Autonomous loop primitives (Schedule, Triage, State, Worktree, Verifier) & CLI tooling pattern (`loop-init`, `loop-audit`). |
| **Attribution** | `SOURCES.md`, `.agents/AGENTS.md` |
| **Notes** | Governs autonomous background agent loops for course tutoring, issue triage, and continuous evaluation. |

---

### nexu-io/html-anything

| Field | Value |
|---|---|
| **Title** | HTML Anything |
| **Author / Org** | nexu-io |
| **Repository** | https://github.com/nexu-io/html-anything |
| **Access Date** | 2026-08-01 |
| **Licence** | Apache-2.0 |
| **Use Type** | `reference` |
| **Adapted Material** | Agent-native HTML layout generator & 75 visual output skill templates (slide decks, posters, lab dashboards). |
| **Attribution** | `SOURCES.md`, `.agents/AGENTS.md` |
| **Notes** | Used to generate visual presentation slides for YouTube tutorials and interactive lab interfaces. |

---

### omnigent-ai/omnigent

| Field | Value |
|---|---|
| **Title** | Omnigent Meta-Harness |
| **Author / Org** | omnigent-ai |
| **Repository** | https://github.com/omnigent-ai/omnigent |
| **Access Date** | 2026-08-01 |
| **Licence** | Apache-2.0 |
| **Use Type** | `concept_seed` |
| **Adapted Material** | Multi-agent meta-harness, multi-model debate architecture, and disposable cloud sandbox execution (E2B / Modal). |
| **Attribution** | `SOURCES.md`, `.agents/AGENTS.md` |
| **Notes** | Informs the architecture for sandboxed evaluation of learner challenge submissions. |

---

### NVIDIA/skills

| Field | Value |
|---|---|
| **Title** | NVIDIA Agent Skills & SkillSpector |
| **Author / Org** | NVIDIA |
| **Repository** | https://github.com/NVIDIA/skills |
| **Access Date** | 2026-08-01 |
| **Licence** | Apache-2.0 / NVIDIA Open License |
| **Use Type** | `reference` |
| **Adapted Material** | Instruction sets for CUDA-X, PyTorch acceleration, RAG blueprints, and SkillSpector threat scanner. |
| **Attribution** | `SOURCES.md`, `.agents/AGENTS.md` |
| **Notes** | Used for Module 004+ GPU training acceleration, PyTorch optimization, and skill security auditing. |

---

### greensock/gsap-skills

| Field | Value |
|---|---|
| **Title** | GSAP Official Agent Skills |
| **Author / Org** | GreenSock |
| **Repository** | https://github.com/greensock/gsap-skills |
| **Access Date** | 2026-08-01 |
| **Licence** | MIT |
| **Use Type** | `reference` |
| **Adapted Material** | 60fps web animation instruction skills for React, Timelines, ScrollTrigger, and HTML canvas. |
| **Attribution** | `SOURCES.md`, `.agents/AGENTS.md` |
| **Notes** | Integrates with `motion-design` for building high-performance interactive module visualizations. |

---

---

### pixel-agents-hq/pixel-agents

| Field | Value |
|---|---|
| **Title** | Pixel Agents |
| **Author / Org** | pixel-agents-hq |
| **Repository** | https://github.com/pixel-agents-hq/pixel-agents |
| **Access Date** | 2026-08-01 |
| **Licence** | MIT |
| **Use Type** | `reference` |
| **Adapted Material** | Visual agent monitoring & status tracking UI representation for background CLI agents. |
| **Attribution** | `SOURCES.md`, `.agents/AGENTS.md` |
| **Notes** | Used for real-time visual monitoring of background subagents and swarm workflows. |

---

### diegosouzapw/OmniRoute

| Field | Value |
|---|---|
| **Title** | OmniRoute AI Gateway |
| **Author / Org** | diegosouzapw |
| **Repository** | https://github.com/diegosouzapw/OmniRoute |
| **Access Date** | 2026-08-01 |
| **Licence** | MIT |
| **Use Type** | `reference` |
| **Adapted Material** | Unified AI Routing Gateway, 15-95% RTK+Caveman token compression, and quota-aware multi-provider fallback. |
| **Attribution** | `SOURCES.md`, `.agents/AGENTS.md` |
| **Notes** | Used for AI infrastructure routing, multi-provider model orchestration, and token efficiency optimization. |

---

*Last updated: 2026-08-01*
