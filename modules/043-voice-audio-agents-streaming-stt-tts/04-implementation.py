import math
import struct
import wave
import time

class WAVEAudioBuffer:
    """Manages PCM audio frames in a continuous buffer."""
    def __init__(self, sample_rate=16000, channels=1, sample_width=2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.buffer = bytearray()
    
    def add_frames(self, data: bytes):
        """Append raw PCM bytes to the buffer."""
        self.buffer.extend(data)
        
    def get_frames(self, num_bytes=None) -> bytes:
        """Retrieve and remove bytes from the buffer."""
        if num_bytes is None:
            num_bytes = len(self.buffer)
        num_bytes = min(num_bytes, len(self.buffer))
        data = self.buffer[:num_bytes]
        self.buffer = self.buffer[num_bytes:]
        return bytes(data)
        
    def get_samples(self) -> list:
        """Return the buffer content as a list of 16-bit integer samples."""
        num_samples = len(self.buffer) // 2
        if num_samples == 0:
            return []
        return list(struct.unpack(f'<{num_samples}h', self.buffer[:num_samples*2]))
        
    def clear(self):
        """Empty the buffer."""
        self.buffer = bytearray()
        
    @property
    def duration_seconds(self) -> float:
        """Calculate the duration of the audio in the buffer in seconds."""
        if self.sample_rate == 0 or self.channels == 0 or self.sample_width == 0:
            return 0.0
        return len(self.buffer) / (self.sample_rate * self.channels * self.sample_width)


class VoiceActivityDetector:
    """Energy-based Voice Activity Detection (VAD)."""
    def __init__(self, energy_threshold=500.0, required_active_frames=3):
        self.energy_threshold = energy_threshold
        self.required_active_frames = required_active_frames
        
    def compute_energy(self, pcm_bytes: bytes) -> float:
        """Compute the Root Mean Square (RMS) energy of PCM audio."""
        num_samples = len(pcm_bytes) // 2
        if num_samples == 0: 
            return 0.0
        samples = struct.unpack(f'<{num_samples}h', pcm_bytes[:num_samples*2])
        sq_sum = sum(float(s) * float(s) for s in samples)
        return math.sqrt(sq_sum / num_samples)
        
    def is_speech(self, pcm_bytes: bytes) -> bool:
        """Determine if a chunk of audio contains speech."""
        energy = self.compute_energy(pcm_bytes)
        return energy > self.energy_threshold


class MockSTTEngine:
    """Mock Streaming Speech-to-Text Engine."""
    def __init__(self, latency_sec=0.0):
        self.latency_sec = latency_sec
        self.internal_buffer = bytearray()
        
    def stream_audio(self, pcm_bytes: bytes):
        """Simulate streaming audio into the STT engine."""
        self.internal_buffer.extend(pcm_bytes)
        
    def get_transcript(self, finish=False) -> str:
        """Simulate returning transcription results."""
        if self.latency_sec > 0:
            time.sleep(self.latency_sec)
            
        num_samples = len(self.internal_buffer) // 2
        
        # Very simple mock logic: return words based on the audio length
        if num_samples < 8000:
            transcript = ""
        elif num_samples < 16000:
            transcript = "hello"
        else:
            transcript = "hello world"
            
        if finish:
            self.internal_buffer = bytearray()
            
        return transcript


class MockTTSEngine:
    """Mock Text-to-Speech Engine."""
    def __init__(self, sample_rate=16000, latency_sec=0.0):
        self.sample_rate = sample_rate
        self.latency_sec = latency_sec
        
    def synthesize(self, text: str) -> bytes:
        """Synthesize PCM audio bytes for the given text."""
        if self.latency_sec > 0:
            time.sleep(self.latency_sec)
            
        if not text:
            return b""
            
        # Generate dummy sine wave for a duration proportional to text length
        # 0.05 seconds per character
        duration = len(text) * 0.05 
        num_samples = int(duration * self.sample_rate)
        
        samples = []
        for i in range(num_samples):
            # 440 Hz A tone
            val = int(math.sin(2 * math.pi * 440.0 * i / self.sample_rate) * 8000)
            samples.append(val)
            
        return struct.pack(f'<{num_samples}h', *samples)


class VoiceAgentPipeline:
    """End-to-End Duplex Voice Pipeline managing VAD, STT, LLM-mock, and TTS."""
    def __init__(self, vad=None, stt=None, tts=None):
        self.vad = vad or VoiceActivityDetector()
        self.stt = stt or MockSTTEngine()
        self.tts = tts or MockTTSEngine()
        
        self.state = "IDLE"
        self.history = []
        self.silence_frames = 0
        self.max_silence_frames = 2
        
    def process_chunk(self, chunk: bytes) -> bytes:
        """
        Process an incoming audio chunk and return synthesized audio if a turn is complete.
        Returns empty bytes if the agent is still listening or idle.
        """
        response_audio = b""
        
        is_speech = self.vad.is_speech(chunk)
        
        if is_speech:
            self.silence_frames = 0
            if self.state == "IDLE":
                self.state = "LISTENING"
            
            if self.state == "LISTENING":
                self.stt.stream_audio(chunk)
        else:
            if self.state == "LISTENING":
                self.silence_frames += 1
                
                # If enough silence has passed, process the turn
                if self.silence_frames >= self.max_silence_frames:
                    transcript = self.stt.get_transcript(finish=True)
                    
                    if transcript:
                        self.history.append({"role": "user", "content": transcript})
                        
                        # Mock LLM generation
                        response_text = f"Acknowledged: {transcript}"
                        self.history.append({"role": "agent", "content": response_text})
                        
                        response_audio = self.tts.synthesize(response_text)
                    
                    self.state = "IDLE"
                    self.silence_frames = 0
                else:
                    # Still in listening state, but buffering trailing silence
                    self.stt.stream_audio(chunk)
                
        return response_audio
