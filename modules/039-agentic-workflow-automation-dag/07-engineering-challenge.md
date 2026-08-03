# 07 - Engineering Challenge: Resilient Multi-Step Financial Transaction DAG

## The Challenge
You are tasked with building a stateful DAG engine to process a multi-step financial transaction (e.g., transferring funds, verifying fraud, applying taxes). If any step fails, the workflow should retry. If it completely fails, it should rollback to the last stable state. Certain large transactions require Human-in-the-Loop (HITL) approval.

## Requirements
1. Define a `StateGraph` with at least 5 nodes (e.g., Init, FraudCheck, CalculateTax, HITL_Approval, TransferFunds).
2. `TransferFunds` should simulate a flaky external API and use exponential backoff for retries.
3. If the transfer amount is > $10,000, `HITL_Approval` must pause the graph and wait for external input.
4. Implement checkpointing so that if `TransferFunds` fails repeatedly, the state rolls back to before the attempt, ensuring money isn't lost in limbo.

## Constraints
- Use only standard Python libraries.
- The state must be fully serializable to JSON at every checkpoint.
- Must cleanly raise an exception if a cycle is introduced in the graph edges.
