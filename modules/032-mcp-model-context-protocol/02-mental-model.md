# Mental Model: The USB-C for AI

Think of the Model Context Protocol (MCP) as the **USB-C standard for AI models**.

## The Pre-MCP Era (The Cable Chaos)
Historically, if you wanted an AI model to talk to a database, you had to write a custom integration. If you wanted it to search the web, you wrote another custom integration. This is analogous to the era when every mobile phone had a different, proprietary charging cable.

## The MCP Era (The Universal Standard)
MCP introduces a universal plug-and-play standard:
- The **MCP Client** (in the AI host) is the USB-C port on your laptop.
- The **MCP Server** (connecting to your data) is the USB-C peripheral (like a hard drive or monitor).

Once a data source or tool is wrapped in an MCP server, ANY AI model or host that supports the MCP standard can instantly connect to it, discover its capabilities, and interact with it safely.

## Dynamic Discovery
Just as plugging in a USB device tells your OS what the device can do (print, store data, play audio), connecting an MCP client to a server initiates a handshake where the server advertises its capabilities (Resources, Prompts, Tools).
