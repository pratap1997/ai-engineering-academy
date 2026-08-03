# 07 - Engineering Challenge: Build a High-Throughput AI Gateway

## Scenario
You are the lead platform engineer for a fast-growing AI startup. Your core product relies on LLMs to summarize articles. Currently, your backend directly calls the OpenAI API. 
Over the last week, you've experienced:
1. Two major OpenAI outages causing total downtime.
2. An unexpected $5,000 API bill due to users submitting highly redundant text with lots of conversational fluff.
3. Slow response times because users frequently ask to summarize the exact same viral news articles.

## The Challenge
Build a local AI Gateway (Reverse Proxy) in Python (using FastAPI or a similar framework) that implements the following features to solve these problems:

1. **Two-Tier Caching (Local)**
   - Implement an Exact Cache using a dictionary or Redis.
   - Implement a Mock Semantic Cache: Given a query, if the Jaccard similarity of words between the new query and a cached query is $> 0.8$, return the cached response.

2. **Basic Prompt Compression**
   - Implement a simple "Caveman" text compressor that strips out a predefined list of conversational stop-words ("please", "can you", "I would like you to", "thank you") before processing the prompt.

3. **Routing and Fallback**
   - The gateway must support two mock providers: `ProviderA` (Primary, fails 30% of the time) and `ProviderB` (Fallback).
   - If `ProviderA` throws an exception, seamlessly route the request to `ProviderB` and return its response to the user.

## Requirements
- Do not use external LLM APIs for the test; create mock functions that simulate LLM latency (e.g., `asyncio.sleep(1)`) and occasionally throw random errors to test the fallback.
- The client should only interact with your Gateway's `/v1/chat/completions` endpoint.
- Track and print the "Token Compression Ratio" for every request.

## Success Criteria
- The client receives a successful response even when Provider A fails.
- Identical and highly similar requests return instantly from the cache.
- The length of the prompt sent to the mock provider is verifiably smaller than the prompt received from the client.
