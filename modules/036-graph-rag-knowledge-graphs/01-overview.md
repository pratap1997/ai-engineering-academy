# 01 - Overview: Graph RAG & Knowledge Graphs

## Introduction
Limitations of vector-only RAG for multi-hop queries. Vector similarity struggles when the answer requires synthesizing information scattered across disparate documents.

## Knowledge Graph Structure
- **Nodes (Entities):** Nouns or objects (e.g., people, places, concepts).
- **Edges (Relations):** Verbs or connections (e.g., works-at, born-in, causes).
- **Properties:** Attributes attached to nodes or edges.

## Graph RAG Architecture
1. **Extraction:** LLM extracts entities and relations from source text.
2. **Community Detection:** Grouping related entities into communities.
3. **Summarization:** Generating summaries for each community.
4. **Querying:** Using local (entity-specific) or global (community-based) graph search.
