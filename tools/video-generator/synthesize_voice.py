import sys
import re
import math
import riva.client
from format_audio import convert_to_pcm_wav

def split_text_into_chunks(text, max_chars=120):
    """
    Splits long text into sentence chunks under max_chars to stay within Riva's ~20s token limit.
    """
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk = (current_chunk + " " + sentence).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def synthesize_audio(
    text: str,
    output_wav_path: str = "tools/video-generator/public/audio.wav",
    server: str = "grpc.nvcf.nvidia.com:443",
    function_id: str = "ddacc747-1269-4fab-bfd9-8f593dead106",
    auth_token: str = "Bearer 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",
    voice: str = "Chatterbox-Multilingual.en-US.Male",
    language_code: str = "en-US"
):
    """
    Synthesizes speech using NVIDIA Riva TTS gRPC API with Chatterbox Male voice.
    Automatically chunks long text into <=20s API requests and concatenates audio.
    Returns (success: bool, chunk_timings: list)
    """
    print(f"[RIVA TTS] Connecting to NVIDIA Riva gRPC server at {server}...")
    
    auth = riva.client.Auth(
        ssl_root_cert=None,
        use_ssl=True,
        uri=server,
        metadata_args=[
            ["function-id", function_id],
            ["authorization", auth_token]
        ]
    )

    service = riva.client.SpeechSynthesisService(auth)

    chunks = split_text_into_chunks(text)
    print(f"[RIVA TTS] Split narration text into {len(chunks)} chunks for Riva synthesis:")
    for idx, c in enumerate(chunks, 1):
        print(f"  Chunk {idx}: \"{c}\"")

    combined_audio_bytes = bytearray()
    chunk_timings = []
    current_frame = 0

    for idx, chunk in enumerate(chunks, 1):
        try:
            print(f"[RIVA TTS] Synthesizing chunk {idx}/{len(chunks)}...")
            response = service.synthesize(
                text=chunk,
                voice_name=voice,
                language_code=language_code,
                sample_rate_hz=44100
            )
            audio_data = response.audio
            combined_audio_bytes.extend(audio_data)

            # 44100 Hz, 16-bit (2 bytes per sample) mono
            chunk_duration_sec = len(audio_data) / (44100.0 * 2.0)
            chunk_frames = math.ceil(chunk_duration_sec * 60)

            start_frame = current_frame
            end_frame = current_frame + chunk_frames
            chunk_timings.append({
                "text": chunk,
                "startFrame": start_frame,
                "endFrame": end_frame,
            })
            current_frame = end_frame + 5 # 5 frame subtle gap
        except Exception as err:
            print(f"[ERROR] Riva synthesis failed on chunk {idx}: {err}")
            return False, []

    # Formats raw audio stream into valid RIFF PCM WAV file
    convert_to_pcm_wav(bytes(combined_audio_bytes), output_wav_path, sample_rate=44100)

    print(f"[SUCCESS] Generated Full TTS Audio WAV ({current_frame} total frames): {output_wav_path}")
    return True, chunk_timings

if __name__ == "__main__":
    sample_text = "Welcome to the AI Engineering Skool. Today we build autonomous agents from scratch."
    if len(sys.argv) > 1:
        sample_text = sys.argv[1]
    
    synthesize_audio(sample_text)
