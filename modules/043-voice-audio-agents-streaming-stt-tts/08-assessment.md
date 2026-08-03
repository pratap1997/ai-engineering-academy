# 08. Assessment & Debrief

## Quiz

1. **What is the primary cause of latency in a traditional sequential Voice AI pipeline (STT -> LLM -> TTS)?**
   - A. Network bandwidth limitations
   - B. Waiting for the user to finish speaking before starting STT, and waiting for the full LLM response before starting TTS
   - C. The speed of the microphone hardware
   - D. Inefficient audio codecs
   *(Answer: B)*

2. **Which technology is commonly used to detect if a user is actively speaking versus background noise?**
   - A. Mel-Frequency Cepstral Coefficients (MFCC)
   - B. Voice Activity Detection (VAD)
   - C. Text-to-Speech (TTS)
   - D. Dynamic Time Warping (DTW)
   *(Answer: B)*

3. **In the context of audio streaming, what does TTFAB stand for?**
   - A. Time To First Audio Byte
   - B. Text To Frequency Audio Buffer
   - C. Time To Fast Audio Broadcast
   - D. Token To Frame Audio Byte
   *(Answer: A)*

4. **Why are WebSockets preferred over REST APIs for streaming voice agents?**
   - A. WebSockets are inherently more secure
   - B. WebSockets allow persistent, bidirectional, full-duplex communication with lower overhead per message
   - C. REST APIs cannot transmit binary data
   - D. WebSockets automatically compress audio data
   *(Answer: B)*

5. **What is a major advantage of native multimodal audio models (like GPT-4o Audio) over the STT -> LLM -> TTS cascade?**
   - A. They require less RAM to run
   - B. They preserve non-textual information like tone, emotion, and speaker identity
   - C. They only support English, making them faster
   - D. They do not require an internet connection
   *(Answer: B)*

6. **What role does "chunking" play in Text-To-Speech streaming?**
   - A. Breaking the audio file into multiple languages
   - B. Sending the LLM text output to the TTS engine in small segments (like sentences or phrases) so audio generation can begin before the LLM finishes
   - C. Compressing the audio to save disk space
   - D. Combining multiple users' voices into one
   *(Answer: B)*

7. **Silero VAD is widely used because:**
   - A. It is a cloud-only enterprise solution
   - B. It is a lightweight, fast neural VAD model that runs locally and handles noise well
   - C. It provides the highest quality Text-To-Speech available
   - D. It automatically translates languages
   *(Answer: B)*

8. **When handling user interruptions (barge-in), what must the system do?**
   - A. Ignore the user until the TTS finishes playing
   - B. Immediately stop the TTS audio playback, cancel ongoing LLM generation, and flush audio buffers
   - C. Lower the volume of the TTS but keep playing it
   - D. Store the user's audio and play it back to them later
   *(Answer: B)*

9. **Which audio format/codec is commonly used for raw, uncompressed low-latency streaming?**
   - A. MP3
   - B. FLAC
   - C. PCM (Pulse-Code Modulation)
   - D. OGG
   *(Answer: C)*

10. **How does WebRTC help in Voice AI applications?**
    - A. It generates human-like synthetic voices
    - B. It provides a standardized protocol for real-time peer-to-peer audio/video streaming with built-in echo cancellation and noise suppression
    - C. It translates speech between different languages offline
    - D. It acts as an LLM for reasoning
    *(Answer: B)*

## Debrief

If you scored less than 8/10, review the concepts of VAD and streaming architectures. Understanding the distinction between chunked pipelines and native multimodal models is crucial for modern AI engineering.
