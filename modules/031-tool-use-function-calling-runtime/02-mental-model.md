# The Mental Model: OS Kernel & Syscalls

To understand how a Tool Runtime operates, look at how an Operating System (OS) kernel manages user-space applications.

## The Operating System Metaphor

In a traditional OS:
1.  **User-Space Application:** A program (like a web browser) runs in an unprivileged mode. It cannot directly read files from disk or send packets over the network.
2.  **Syscalls:** When the application needs to interact with the world, it makes a "system call" (syscall) to the kernel.
3.  **The Kernel:** The kernel receives the syscall, checks permissions, executes the hardware operation safely, and returns the result.

**In an Agentic System:**
1.  **The LLM (User-Space):** The LLM is an untrusted, unprivileged intelligence. It generates text (intent) but cannot directly "do" anything.
2.  **Function Calling (Syscalls):** The LLM outputs a structured payload (JSON) specifying a tool name and arguments. This is the equivalent of a syscall.
3.  **The Tool Runtime (Kernel):** The runtime intercepts this structured output. It is the trusted environment. It validates permissions, applies rate limits, executes the tool (often in a further isolated sandbox), and returns the observation to the LLM.

## The Defense-in-Depth Principle

A secure tool runtime applies the principle of "Defense-in-Depth," assuming the LLM *will* eventually output a malicious or destructive command (either due to hallucination or prompt injection).

1.  **Layer 1: Prompt/Instructions:** Telling the model what it shouldn't do (Weakest).
2.  **Layer 2: Output Parsing:** Rejecting malformed or invalid JSON.
3.  **Layer 3: Risk Scoring & Heuristics:** Blocking known bad patterns (e.g., `rm -rf`).
4.  **Layer 4: Permission Sandbox:** Checking if the agent is authorized for this specific tool/resource.
5.  **Layer 5: Execution Sandbox:** Running the tool in an isolated environment (Docker, Firecracker microVM, or restricted AST execution for Python) (Strongest).
