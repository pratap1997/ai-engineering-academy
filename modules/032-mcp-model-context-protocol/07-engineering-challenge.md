# Engineering Challenge: Multi-Server MCP Aggregator

## Objective
Build a "Router" MCP Server that acts as an aggregator. It should implement the MCP Server protocol but internally act as an MCP Client connecting to multiple downstream MCP Servers.

## Requirements
1. **Dynamic Registration**: The router must be able to dynamically connect to new downstream servers.
2. **Namespace Prefixing**: When advertising tools and resources, the router must prefix them with the downstream server's ID to avoid collisions (e.g., `serverA_toolName`).
3. **Transparent Forwarding**: When a client calls a tool on the router, the router must parse the prefix, forward the call to the appropriate downstream server, and return the response.

## Constraints
- No external libraries.
- Must handle scenarios where a downstream server is offline gracefully.
