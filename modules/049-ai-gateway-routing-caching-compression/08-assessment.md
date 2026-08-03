# 08 - Assessment

1. **What is the primary purpose of an AI Gateway?**
   - A) To train new Large Language Models.
   - B) To act as a centralized proxy for routing, caching, and observability of LLM API calls.
   - C) To provide a user interface for ChatGPT.
   - D) To generate embeddings for vector databases.
   - *Answer: B*

2. **Which of the following best describes "Token Compression" in an AI Gateway?**
   - A) Zipping the JSON payload to reduce network transfer time.
   - B) Removing non-semantic conversational fluff from a prompt to reduce LLM input cost.
   - C) Downsampling the weights of an LLM.
   - D) Truncating the LLM's response to fit the screen.
   - *Answer: B*

3. **In the provider priority cost function $Cost(p) = w_1 \cdot \text{Latency} + w_2 \cdot \text{Price} + w_3 \cdot \text{ErrorRate}$, how should the weights be adjusted for an offline batch-processing job where speed doesn't matter?**
   - A) High $w_1$, Low $w_2$
   - B) High $w_1$, High $w_3$
   - C) Low $w_1$, High $w_2$
   - D) Low $w_1$, Low $w_2$
   - *Answer: C (Low weight on latency, high weight on price)*

4. **What is the difference between an Exact Cache and a Semantic Cache?**
   - A) Exact cache uses embeddings; semantic cache uses strings.
   - B) Exact cache requires a 100% string match; semantic cache uses vector similarity to find conceptually identical queries.
   - C) Exact cache is for OpenAI; semantic cache is for Anthropic.
   - D) Semantic cache is always faster than an exact cache.
   - *Answer: B*

5. **If an AI Gateway implements a fallback queue, what happens when the primary provider (e.g., OpenAI) returns a 429 Rate Limit error?**
   - A) The application crashes immediately.
   - B) The gateway returns a 429 error to the user.
   - C) The gateway intercepts the error and automatically retries the request against a secondary provider (e.g., Anthropic).
   - D) The gateway trains a new model.
   - *Answer: C*

6. **What is a potential downside of setting the semantic cache similarity threshold ($\theta$) too low (e.g., 0.50)?**
   - A) The cache will never get a hit.
   - B) The cache will return irrelevant answers for distinct queries.
   - C) The vector database will crash.
   - D) Token costs will increase.
   - *Answer: B*

7. **How does an AI Gateway prevent "vendor lock-in"?**
   - A) By signing multi-year contracts with OpenAI.
   - B) By forcing the application to use proprietary SDKs.
   - C) By exposing a standard API (often OpenAI-compatible) to the application, while handling the translation to different provider APIs internally.
   - D) By running models exclusively on local hardware.
   - *Answer: C*

8. **If an original prompt has 100 tokens, and the AI Gateway compresses it to 65 tokens before sending it to the LLM, what is the compression ratio?**
   - A) 1.53
   - B) 0.65
   - C) 0.35
   - D) 35
   - *Answer: C (1 - 65/100 = 0.35)*

9. **Why is it recommended to use a Two-Tier caching strategy (Exact + Semantic)?**
   - A) Because Exact matching (like Redis) is extremely fast and computationally cheap, while Semantic matching catches variations but requires expensive embedding generation.
   - B) Because Semantic caching is faster.
   - C) Because Exact caching works for images, and Semantic for text.
   - D) It is not recommended; only Semantic caching should be used.
   - *Answer: A*

10. **What is the "Caveman" technique in the context of prompt compression?**
    - A) A hardware acceleration technique.
    - B) Stripping polite filler words and grammar to create a raw, keyword-dense prompt that the LLM still understands.
    - C) Using outdated, legacy LLM models to save money.
    - D) Encrypting prompts before sending them.
    - *Answer: B*
