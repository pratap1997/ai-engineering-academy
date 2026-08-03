# Mental Model: The Walkie-Talkie with an Instant Interpreter

To understand real-time voice agents, discard the "email" metaphor (write a complete message, hit send, wait for a complete reply). Instead, adopt the **"Walkie-talkie with an Instant Interpreter"** metaphor.

## The Traditional Pipeline (The Translator Email)
In a traditional voice app (like older smart assistants):
1. You record a complete voice memo.
2. You send the file.
3. The server transcribes the whole file.
4. The server thinks and writes a whole reply.
5. The server synthesizes the whole reply into an audio file.
6. You download and listen to the file.
This feels sluggish and non-conversational.

## The Streaming Pipeline (The Instant Interpreter)
Imagine you are speaking into a walkie-talkie to a real-time interpreter:
- As you speak, the interpreter is already translating your words in their head (Streaming STT).
- The moment you pause and say "over" (VAD detecting the end of speech), the interpreter immediately starts speaking the first part of their response.
- They are formulating the end of their sentence while speaking the beginning of it (LLM Streaming + Streaming TTS).

## Audio Frames and Conversational Turns
In this model, audio is not a file; it is an infinite river of **frames** (tiny chunks of sound).
- **The River Flows In:** The agent constantly drinks from this river, analyzing frames for speech (VAD).
- **The Turn:** When the agent detects silence (an endpoint), the conversational turn flips.
- **The River Flows Out:** The agent immediately starts pouring its own audio frames back into the river, chunk by chunk, keeping the flow continuous.

The key to low latency is never waiting for a complete thought or complete audio file before acting on the pieces you already have.
