# 01 - Overview: Agent Memory Systems

## Introduction

As AI agents become increasingly autonomous and long-lived, the most pressing limitation they face is the **session amnesia problem**. Standard Large Language Models are stateless—they have no inherent memory between API calls. Each interaction starts with a blank slate unless context is explicitly provided. While simply appending history to the context window works for brief interactions, the strict **context window limits** and degradation in attention over long texts require more sophisticated memory architectures.

This module introduces the conceptual and practical frameworks for endowing AI agents with persistent, structured memory, enabling them to learn from past interactions, recall facts, and execute complex procedural tasks across multiple sessions.

## 5 Memory Types from Cognitive Science

Human memory isn't a single monolithic database; it is a highly specialized, multi-tiered system. Cognitive science generally categorizes memory into five distinct types:

1. **Working Memory**: The small, immediate scratchpad of consciousness. It holds the information currently being processed and typically has a very limited capacity (e.g., "7 ± 2" items).
2. **Episodic Memory**: The autobiographical record of experiences, tied to specific times, places, and emotions. "What happened yesterday?"
3. **Semantic Memory**: The structured knowledge of facts, concepts, and meanings, independent of the context in which they were learned. "What is the capital of France?"
4. **Procedural Memory**: The unconscious memory of how to do things—motor skills and habits. "How to ride a bike."
5. **Prospective Memory**: The memory of intentions and planned actions in the future. "Remember to buy milk at 6 PM."

## Mapping to AI Agent Implementation

In AI engineering, we map these cognitive concepts to concrete technical architectures:

- **Working Memory → Context Window / In-Context Messages**: Implemented as a FIFO queue of recent messages or a summarization buffer passed directly into the LLM context.
- **Episodic Memory → Conversation Logs / Vector Embeddings**: Stored as time-stamped embeddings in a vector database, retrieved via similarity search to give agents a sense of past interactions with specific users.
- **Semantic Memory → External Knowledge Bases / RAG**: Structured databases or specialized vector stores containing factual documents, ground truth, and domain knowledge (Retrieval-Augmented Generation).
- **Procedural Memory → Tool Schemas / Action Libraries**: Function definitions, few-shot examples of tool usage, and prompt templates that tell the agent *how* to execute a capability.
- **Prospective Memory → Scheduled Tasks / Reminders**: Cron jobs, event queues, and time-triggered prompts injected into the agent's context to initiate future actions.

## The Mem0 Architecture

A modern paradigm for implementing robust agent memory is the **Mem0 Architecture**, which treats memory not as a static database, but as an active, self-maintaining system. It operates on a continuous **Extract → Update → Retrieve** cycle:
1. **Extract**: The agent parses the current interaction to identify new facts, preferences, or important events.
2. **Update**: The memory system consolidates this information—adding new memories, decaying older ones, and merging duplicates.
3. **Retrieve**: When faced with a new query, the agent performs a similarity search over the memory store to pull relevant context back into working memory.

## Prerequisites
- **Module 024**: Vector Databases & Indexing (Core requirement for Episodic/Semantic retrieval)
- **Module 026**: Agentic Loops (Requirement for understanding the Extract-Update-Retrieve lifecycle)
