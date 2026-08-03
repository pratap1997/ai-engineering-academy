# 07 - Engineering Challenge: Personalized Agent with Memory Decay

## The Challenge

You are tasked with building a **Personalized Agent Memory System** that manages user preferences and facts over a simulated long-term period. Your system must autonomously manage the lifecycle of memories without running out of bounds.

### Requirements

1. **Memory Importance Scoring**:
   Implement a dynamic scoring algorithm for retrieval:
   $Score = w_1 \cdot \text{Recency} + w_2 \cdot \text{Frequency} + w_3 \cdot \text{Similarity}$
   You must tune the weights so that an important fact from a year ago can still beat a trivial fact from yesterday.

2. **Automatic Forgetting**:
   Implement a `tick_day()` function that simulates the passage of 24 hours. After every tick, apply the Ebbinghaus forgetting curve. Any memory whose retention drops below a threshold (e.g., 0.1) must be automatically deleted.

3. **Memory Consolidation**:
   Implement a method to merge similar memories. If a new memory has >0.95 cosine similarity with an existing memory, instead of adding a new vector, you must increment the access frequency of the existing memory and update its timestamp.

### Success Criteria

1. **High Recall**: Over a test set of 50 queries spanning a simulated 100-day period, your agent must retrieve the top-1 most relevant memory >90% of the time.
2. **Bounded Capacity**: After 1000 interactions and 365 simulated days, your memory store must contain fewer than 100 items (demonstrating successful forgetting and consolidation).

### Starter Code
```python
class PersonalizedAgent:
    def __init__(self):
        # Your memory initialization here
        pass
        
    def observe(self, fact: str):
        # Consolidate and store
        pass
        
    def query(self, question: str) -> str:
        # Retrieve and return best fact
        pass
        
    def tick_day(self):
        # Apply forgetting curve and prune
        pass
```
*No external ML frameworks are allowed. Use the provided bag-of-words embedder or build your own from scratch.*
