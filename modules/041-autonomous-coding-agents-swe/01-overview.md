# Module 041: Autonomous Coding Agents & SWE

## Context & Motivation
Software Engineering (SWE) tasks require reasoning across multiple files, executing tools, and evaluating outcomes. Autonomous coding agents use ReAct (Reason + Act) loops to read a codebase, edit files, and run tests.

## Prerequisites
- Module 026: Agentic Loop Primitives
- Module 031: Tool Use & Function Calling

## Core Concepts
1. **Workspace Environment**: A virtual file tree mimicking a terminal.
2. **Unified Diff Patching**: Creating targeted edits rather than rewriting entire files.
3. **Syntax & Test Validation**: The feedback loop to ensure correctness.
4. **Self-Correction**: Iteratively fixing code after test failures.
