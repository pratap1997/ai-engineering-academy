# Module 032: Model Context Protocol (MCP)

## Overview
The Model Context Protocol (MCP) is an open standard introduced by Anthropic that enables developers to build secure, two-way connections between AI models and local or remote data sources and tools.

## Architecture
MCP follows a Client-Server architecture:
- **MCP Hosts**: Programs like Claude Desktop or Cursor that host AI models.
- **MCP Clients**: Components inside the host that initiate requests.
- **MCP Servers**: Lightweight programs that expose specific capabilities (Resources, Prompts, Tools) to clients.

## Core Primitives
1. **Resources**: Data sources that the AI can read (e.g., local files, database tables, API responses).
2. **Prompts**: Reusable prompt templates and workflows provided by the server.
3. **Tools**: Executable functions that the AI can invoke to take actions or fetch dynamic data.

## Why MCP?
Before MCP, every AI integration required custom, one-off code. MCP standardizes the integration layer, much like USB-C standardized hardware connections.
