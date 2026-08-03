# Module 043: Voice & Audio Agents (Streaming STT, TTS Pipelines, Audio Chunking, Low-Latency Voice I/O)

## Context and Motivation
Voice is the most natural interface for human-computer interaction. However, building voice-native AI agents requires fundamentally different architectures than text-based chatbots. Standard API calls (record -> transcribe -> generate -> synthesize -> play) introduce multi-second latency, breaking conversational flow.

To build conversational agents that feel human, we must transition to **streaming architectures**.

## The Real-Time Voice Agent Architecture
A low-latency voice agent operates concurrently across three main stages, known as the STT → LLM → TTS pipeline.

1. **Speech-to-Text (STT) / Automatic Speech Recognition (ASR):**
   - Continuously receives chunks of audio data.
   - Detects when the user starts and stops speaking (Voice Activity Detection).
   - Streams partial transcripts as the user speaks.

2. **Large Language Model (LLM):**
   - Receives partial or complete transcripts from the STT engine.
   - Begins generating a text response via token streaming.

3. **Text-to-Speech (TTS):**
   - Consumes the LLM's streamed tokens (often sentence by sentence or phrase by phrase).
   - Generates and streams audio chunks back to the user before the full LLM response is complete.

## Voice Activity Detection (VAD)
Voice Activity Detection (VAD) is the critical component that determines "who has the floor." It distinguishes human speech from background noise. Proper VAD prevents the agent from interrupting the user and ensures it knows exactly when the user has finished their turn.

## Latency Minimization (Voice-to-Voice < 500ms)
To achieve natural conversation, end-to-end latency (from the moment the user stops speaking to the moment the agent starts speaking) must be under 500 milliseconds.

This requires:
- **Audio Buffer Streaming:** Processing audio in small frames (e.g., 10-20ms) instead of waiting for a complete recording.
- **Concurrent Execution:** The LLM must start predicting before the TTS finishes its previous sentence; TTS must start synthesizing the first sentence while the LLM is still generating the second.
- **Endpointing Optimization:** Rapidly detecting the end of a user's speech using predictive VAD.

## Prerequisites
- Understanding of token streaming (Module 029)
- Basic knowledge of concurrency and asynchronous programming
