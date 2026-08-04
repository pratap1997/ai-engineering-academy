import os
import sys
import subprocess

def synthesize_audio(text, output_wav="audio.wav", voice="Chatterbox-Multilingual.en-US.Male"):
    """
    Synthesizes text to speech using NVIDIA Riva gRPC Chatterbox Multilingual model.
    """
    cmd = [
        sys.executable,
        "tools/python-clients/scripts/tts/talk.py",
        "--server", "grpc.nvcf.nvidia.com:443",
        "--use-ssl",
        "--metadata", "function-id", "ddacc747-1269-4fab-bfd9-8f593dead106",
        "--metadata", "authorization", "Bearer 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",
        "--language-code", "en-US",
        "--text", text,
        "--voice", voice,
        "--output", output_wav
    ]

    print(f"🎙️ Synthesizing audio via NVIDIA Riva gRPC (Voice: {voice})...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0 and os.path.exists(output_wav):
        print(f"✅ Audio generated successfully: {output_wav} ({os.path.getsize(output_wav)} bytes)")
        return True
    else:
        print(f"❌ Error synthesizing audio: {result.stderr or result.stdout}")
        return False

if __name__ == "__main__":
    test_text = "Welcome to the AI Engineering Academy. Today we build multi-agent autonomous swarms from first principles."
    if len(sys.argv) > 1:
        test_text = sys.argv[1]
    synthesize_audio(test_text, "tools/video-generator/audio.wav")
