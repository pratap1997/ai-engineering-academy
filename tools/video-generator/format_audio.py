import wave

def convert_to_pcm_wav(raw_audio_bytes, output_path, sample_rate=44100):
    """
    Ensures raw audio data is formatted as a valid PCM 16-bit WAV file with RIFF headers.
    """
    with wave.open(output_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono audio
        wav_file.setsampwidth(2)  # 16-bit PCM (2 bytes)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(raw_audio_bytes)
    print(f"[AUDIO CONVERTER] Wrote valid PCM 16-bit WAV: {output_path}")
