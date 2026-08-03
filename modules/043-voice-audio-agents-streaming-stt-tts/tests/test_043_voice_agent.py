import unittest
import importlib.util
import os
import struct

def load_module():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(os.path.dirname(current_dir), "04-implementation.py")
    spec = importlib.util.spec_from_file_location("module_043_impl", module_path)
    impl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(impl)
    return impl

impl = load_module()

class TestAudioBufferManagement(unittest.TestCase):
    def setUp(self):
        self.buffer = impl.WAVEAudioBuffer(sample_rate=16000, channels=1, sample_width=2)
        
    def test_buffer_initialization(self):
        self.assertEqual(self.buffer.sample_rate, 16000)
        self.assertEqual(self.buffer.channels, 1)
        self.assertEqual(len(self.buffer.buffer), 0)
        
    def test_add_and_get_frames(self):
        data = b'\x00\x01\x02\x03'
        self.buffer.add_frames(data)
        self.assertEqual(len(self.buffer.buffer), 4)
        
        retrieved = self.buffer.get_frames(2)
        self.assertEqual(retrieved, b'\x00\x01')
        self.assertEqual(len(self.buffer.buffer), 2)
        
        retrieved_rest = self.buffer.get_frames()
        self.assertEqual(retrieved_rest, b'\x02\x03')
        
    def test_duration_calculation(self):
        # 16000 samples * 2 bytes = 32000 bytes = 1 second
        data = bytes(32000)
        self.buffer.add_frames(data)
        self.assertAlmostEqual(self.buffer.duration_seconds, 1.0)
        
    def test_get_samples(self):
        # Add two 16-bit samples: 256 (0x0100) and 512 (0x0200) -> bytes: 00 01, 00 02
        data = struct.pack('<hh', 256, 512)
        self.buffer.add_frames(data)
        samples = self.buffer.get_samples()
        self.assertEqual(samples, [256, 512])

class TestVoiceActivityDetector(unittest.TestCase):
    def setUp(self):
        self.vad = impl.VoiceActivityDetector(energy_threshold=500.0)
        
    def test_silence_detection(self):
        # Silence
        silence = struct.pack('<160h', *[0]*160)
        self.assertFalse(self.vad.is_speech(silence))
        
    def test_speech_detection(self):
        # High energy sine-like mock data
        speech = struct.pack('<160h', *[1000]*160)
        self.assertTrue(self.vad.is_speech(speech))
        
    def test_energy_computation(self):
        data = struct.pack('<4h', 10, 10, -10, -10)
        energy = self.vad.compute_energy(data)
        # RMS of 10, 10, -10, -10 is exactly 10.0
        self.assertAlmostEqual(energy, 10.0)
        
    def test_empty_buffer(self):
        energy = self.vad.compute_energy(b'')
        self.assertEqual(energy, 0.0)
        self.assertFalse(self.vad.is_speech(b''))

class TestMockEngines(unittest.TestCase):
    def test_stt_initialization(self):
        stt = impl.MockSTTEngine(latency_sec=0)
        self.assertEqual(stt.latency_sec, 0)
        self.assertEqual(len(stt.internal_buffer), 0)
        
    def test_stt_transcription(self):
        stt = impl.MockSTTEngine(latency_sec=0)
        stt.stream_audio(bytes(20000)) # 10000 samples -> "hello"
        self.assertEqual(stt.get_transcript(), "hello")
        
        stt.stream_audio(bytes(20000)) # Now 20000 samples -> "hello world"
        self.assertEqual(stt.get_transcript(finish=True), "hello world")
        self.assertEqual(len(stt.internal_buffer), 0)
        
    def test_tts_synthesis(self):
        tts = impl.MockTTSEngine(sample_rate=16000, latency_sec=0)
        text = "hi"
        # "hi" is 2 chars. 2 * 0.05s = 0.1s. 16000 * 0.1 = 1600 samples
        # 1600 samples = 3200 bytes
        audio = tts.synthesize(text)
        self.assertEqual(len(audio), 3200)
        
    def test_tts_empty_text(self):
        tts = impl.MockTTSEngine(latency_sec=0)
        audio = tts.synthesize("")
        self.assertEqual(len(audio), 0)

class TestEndToEndPipeline(unittest.TestCase):
    def setUp(self):
        self.stt = impl.MockSTTEngine(latency_sec=0)
        self.tts = impl.MockTTSEngine(latency_sec=0, sample_rate=16000)
        self.vad = impl.VoiceActivityDetector(energy_threshold=100.0)
        self.pipeline = impl.VoiceAgentPipeline(vad=self.vad, stt=self.stt, tts=self.tts)
        
    def test_initial_state(self):
        self.assertEqual(self.pipeline.state, "IDLE")
        self.assertEqual(len(self.pipeline.history), 0)
        
    def test_idle_silence(self):
        silence = struct.pack('<160h', *[0]*160)
        resp = self.pipeline.process_chunk(silence)
        self.assertEqual(resp, b"")
        self.assertEqual(self.pipeline.state, "IDLE")
        
    def test_speech_transition(self):
        speech = struct.pack('<16000h', *[1000]*16000) # 1s of speech
        resp = self.pipeline.process_chunk(speech)
        self.assertEqual(resp, b"")
        self.assertEqual(self.pipeline.state, "LISTENING")
        self.assertEqual(len(self.stt.internal_buffer), 32000)
        
    def test_full_turn(self):
        # 1. Inject enough speech to get "hello world" (>16000 samples)
        speech = struct.pack('<18000h', *[1000]*18000)
        self.pipeline.process_chunk(speech)
        
        # 2. Inject silence twice to trigger the response
        silence = struct.pack('<160h', *[0]*160)
        resp1 = self.pipeline.process_chunk(silence)
        self.assertEqual(resp1, b"")
        
        resp2 = self.pipeline.process_chunk(silence)
        self.assertTrue(len(resp2) > 0) # Should have TTS audio
        self.assertEqual(self.pipeline.state, "IDLE")
        
        # Check history
        self.assertEqual(len(self.pipeline.history), 2)
        self.assertEqual(self.pipeline.history[0]["role"], "user")
        self.assertEqual(self.pipeline.history[0]["content"], "hello world")
        self.assertEqual(self.pipeline.history[1]["role"], "agent")
        self.assertIn("hello world", self.pipeline.history[1]["content"])

if __name__ == "__main__":
    unittest.main()
