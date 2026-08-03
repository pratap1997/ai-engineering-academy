import time
import struct
import math

try:
    from . import implementation as impl
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("module_043_impl", "04-implementation.py")
    impl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(impl)

def run_vad_sensitivity_experiment():
    print("--- Experiment 1: VAD Threshold Sensitivity ---")
    thresholds = [100, 500, 2000]
    
    # Generate 1 second of silence + noise
    silence = struct.pack('<16000h', *[0]*16000)
    noise = struct.pack('<16000h', *[int(math.sin(i)*200) for i in range(16000)])
    speech = struct.pack('<16000h', *[int(math.sin(i)*5000) for i in range(16000)])
    
    for threshold in thresholds:
        vad = impl.VoiceActivityDetector(energy_threshold=threshold)
        s_res = vad.is_speech(silence)
        n_res = vad.is_speech(noise)
        sp_res = vad.is_speech(speech)
        print(f"Threshold {threshold}: Silence={s_res}, Low Noise={n_res}, Speech={sp_res}")
    print()

def run_audio_chunk_size_experiment():
    print("--- Experiment 2: Audio Chunk Size vs Latency ---")
    chunk_sizes = [160, 480, 1600, 8000] # 10ms, 30ms, 100ms, 500ms at 16kHz
    
    for size in chunk_sizes:
        buffer = impl.WAVEAudioBuffer()
        start = time.perf_counter()
        
        # Simulate processing 1 second of audio in chunks
        for _ in range(16000 // size):
            chunk = bytes([0] * (size * 2))
            buffer.add_frames(chunk)
            _ = buffer.get_frames(size * 2)
            
        elapsed = time.perf_counter() - start
        print(f"Chunk size {size} samples ({size/16} ms): Buffer ops took {elapsed:.5f}s")
    print()

def run_pipeline_latency_experiment():
    print("--- Experiment 3: Pipeline End-to-End Latency Breakdown ---")
    # Using engines with defined synthetic latencies
    stt = impl.MockSTTEngine(latency_sec=0.05)
    tts = impl.MockTTSEngine(latency_sec=0.1)
    pipeline = impl.VoiceAgentPipeline(stt=stt, tts=tts)
    
    speech_chunk = struct.pack('<16000h', *[3000]*16000)
    silence_chunk = struct.pack('<16000h', *[0]*16000)
    
    print("Injecting speech...")
    start_time = time.perf_counter()
    pipeline.process_chunk(speech_chunk)
    
    # Requires 2 silence frames to trigger response
    print("Injecting silence 1...")
    pipeline.process_chunk(silence_chunk)
    
    print("Injecting silence 2 (triggers response)...")
    resp_audio = pipeline.process_chunk(silence_chunk)
    
    total_time = time.perf_counter() - start_time
    print(f"Total Turnaround Latency: {total_time:.3f}s")
    print(f"Generated response audio bytes: {len(resp_audio)}")
    print()

def run_simulated_wer_experiment():
    print("--- Experiment 4: Simulated WER Calculation ---")
    print("Word Error Rate (WER) = (Substitutions + Insertions + Deletions) / Reference_Length")
    
    reference = "the quick brown fox jumps over the lazy dog"
    hypothesis = "the fast brown fox jump over the lazy dogs"
    
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # Simplified simulation (not a true Levenshtein distance)
    matches = sum(1 for r, h in zip(ref_words, hyp_words) if r == h)
    
    ref_len = len(ref_words)
    errors = ref_len - matches
    wer = errors / ref_len
    
    print(f"Reference:  {reference}")
    print(f"Hypothesis: {hypothesis}")
    print(f"Approx WER: {wer:.2%} ({errors} errors / {ref_len} words)")
    print()

if __name__ == "__main__":
    run_vad_sensitivity_experiment()
    run_audio_chunk_size_experiment()
    run_pipeline_latency_experiment()
    run_simulated_wer_experiment()
