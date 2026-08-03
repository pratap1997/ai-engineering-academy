# 07. Engineering Challenge: Build a Low-Latency Streaming Voice Chatbot

## The Challenge

Your task is to build a low-latency, voice-to-voice conversational agent that can handle streaming audio input and output. The agent must transcribe the user's speech in real-time, generate a textual response from an LLM, and stream the synthesized voice back to the user—ideally keeping the end-to-end latency under 1000ms.

## Requirements

1. **Audio Input Streaming:** Capture microphone audio in chunks (e.g., 20ms or 50ms frames) using PyAudio or WebRTC.
2. **Voice Activity Detection (VAD):** Implement a VAD module (such as WebRTC VAD or Silero VAD) to detect when the user starts and stops speaking.
3. **Streaming STT:** Send audio chunks to a fast STT engine (e.g., Deepgram, Faster-Whisper, or OpenAI Realtime API).
4. **LLM Generation:** Stream the transcribed text into an LLM and capture the generated text stream.
5. **Streaming TTS:** Send generated text chunks to a TTS engine (e.g., ElevenLabs or a local FastSpeech model) and play the audio buffer immediately as it arrives.
6. **Interruption Handling:** If the VAD detects user speech while the TTS is playing, instantly halt the TTS playback and cancel the LLM generation.

## Constraints

- Do not use pre-built orchestration frameworks (like LiveKit or Daily) for the core pipeline; you must wire the STT -> LLM -> TTS streams yourself.
- You can use external APIs for the individual components (e.g., OpenAI, ElevenLabs).

## Evaluation Criteria

- **Latency:** Time to first audio byte (TTFAB) after the user finishes speaking must be less than 1.5 seconds.
- **Turn-taking:** The agent must accurately detect the end of a user's utterance without cutting them off prematurely.
- **Interruption:** The system must immediately stop speaking when the user interrupts.
