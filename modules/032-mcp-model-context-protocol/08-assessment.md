# Module 032 Assessment

1. **What does MCP stand for?**
   - Model Context Protocol.

2. **Who created the MCP standard?**
   - Anthropic.

3. **What is the primary architectural pattern of MCP?**
   - Client-Server architecture.

4. **What are the three core capability primitives in MCP?**
   - Resources, Prompts, and Tools.

5. **Which protocol does MCP use for message formatting?**
   - JSON-RPC 2.0.

6. **What is the role of an MCP Host?**
   - The application (like Claude Desktop) that initiates the connection and hosts the AI model.

7. **How does an MCP Client learn what a Server can do?**
   - Through dynamic discovery using capability listing methods (e.g., `tools/list`, `resources/list`).

8. **What does an MCP Resource represent?**
   - A data source or contextual information the AI can read (e.g., a file or database table).

9. **Why is MCP compared to USB-C?**
   - Because it provides a universal, standardized connection layer between AI models and data sources, eliminating custom integrations.

10. **What must an MCP server do if a requested tool does not exist?**
    - Return a JSON-RPC error response indicating "Method not found".
