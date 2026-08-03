# 02 - Mental Model: The AI Hippocampus

## The "AI Hippocampus" Metaphor

In the human brain, the hippocampus acts as the critical routing and indexing center for memory. It doesn't permanently store all information; instead, it holds onto recent experiences (working and short-term memory) and gradually consolidates important ones into the cerebral cortex for long-term storage (episodic and semantic memory). 

When designing an agent memory system, you are essentially building an **AI Hippocampus**. This system acts as a write/read database with intelligent middleware. It listens to the agent's working memory (the context window), extracts salient facts, scores them for importance, and writes them to a persistent vector store. Later, it reads from that store and injects relevant context back into the agent's prompt.

## Memory as a Write/Read Database

Agent memory is not a passive log file; it is an active database with strict lifecycle management:
- **Write**: Memories are encoded into dense vectors. Metadata (timestamps, access counts, importance scores) is attached.
- **Expiry**: Unlike traditional databases where records live forever, agent memory must emulate biological forgetting. Irrelevant memories decay and are eventually garbage collected.
- **Importance Scoring**: Not all memories are equal. "User is allergic to peanuts" is infinitely more important than "User asked about the weather on Tuesday."

## The Recency vs Importance Tradeoff

A core challenge in memory systems is deciding what to retrieve. 
- **Recency**: A memory from 5 minutes ago is highly relevant to the current conversational flow.
- **Importance**: A memory from 5 months ago ("User is allergic to peanuts") might be structurally critical, even if it hasn't been accessed recently.

Retrieval algorithms must balance these factors: retrieving a highly recent but trivial memory vs an old but critical memory. 

## The Forgetting Curve (Ebbinghaus)

In 1885, Hermann Ebbinghaus discovered the **Forgetting Curve**, demonstrating that memory retention drops exponentially over time unless the memory is actively recalled or possesses high inherent strength. 

In AI agents, we implement this mathematically. A memory's retention decays exponentially over time. However, every time the memory is retrieved (accessed), its "Memory Strength" increases, flattening the decay curve. This ensures that frequently used facts stick around, while single-use trivia fades away, keeping the memory database clean and fast.

## Visual: Memory Hierarchy Pyramid

```text
                  /\
                 /  \
                /    \
               /      \
              / Working\  <- Immediate context window (Tokens, fast, limited)
             /  Memory  \
            /------------\
           /   Episodic   \  <- Past interactions (Vector DB, similarity search)
          /     Memory     \
         /------------------\
        / Semantic & Proced.\ <- Knowledge bases, Tool schemas (Structured DBs)
       /       Memory        \
      ------------------------
```

## How Retrieval Works: Similarity Search in Embedding Space

Retrieval relies heavily on embedding models. When a user asks a question, the query is embedded into a dense vector space. The memory system then performs a k-Nearest Neighbors (k-NN) or Cosine Similarity search against the memory database.

### Memory Write & Retrieval Flow

**Write Flow (Consolidation):**
```text
[User Input] --> (Working Memory) --> [Extract Facts LLM] --> (Score Importance) 
                                                                    |
                                                                    v
                                                             [Vector Store]
```

**Retrieve Flow (Recall):**
```text
[User Query] --> (Embed Query) --> [Similarity Search in Vector Store]
                                               |
                                               v
[LLM Context Window] <--- (Inject Top-K Memories + Boost by Importance/Recency)
```
