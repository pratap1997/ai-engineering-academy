# Engineering Challenge: Secure Multi-Tenant Tool Gateway

## Problem Statement

You are tasked with building a tool execution gateway for a platform that hosts thousands of autonomous agents. Different users own different agents, and agents must not be able to access data or execute tools outside of their granted permissions.

## Requirements

1.  **Multi-Tenancy:** The runtime must identify which user/agent is requesting the tool execution.
2.  **Resource Quotas:** Implement a rate limiter that tracks quotas per tenant, not globally.
3.  **Argument Level Security:** Implement a rule engine that can block specific arguments for specific tenants (e.g., Tenant A can use `read_file` on `/tmp/a/*` but not `/etc/passwd`).
4.  **Fallback Mechanisms:** If a tool call fails or is blocked, the runtime must return a structured error message to the LLM so the LLM can adjust its strategy.

## Constraints
- Do not use external libraries (no `pydantic`, `ratelimit`, etc.).
- Ensure execution overhead is under 5ms per tool call validation.
- All actions must be logged in a centralized, append-only structure.

*(No hints provided. Use the concepts from 04-implementation.py to design your solution.)*
