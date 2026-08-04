import os
import sys
import shutil
import subprocess
from synthesize_voice import synthesize_audio

def generate_full_video(
    text="Welcome to the AI Engineering Academy masterclass. Today we build autonomous AI agents from first principles.",
    title="AI ENGINEERING ACADEMY",
    subtitle="Masterclass: Building Autonomous AI Agents From First Principles",
    output_mp4="out/masterclass_video.mp4"
):
    """
    End-to-End Automated Video Generation Pipeline:
    1. NVIDIA Riva gRPC TTS Speech Synthesis (Male Voice)
    2. Remotion React Video Composition (60fps Kinetic Animations)
    3. Headless Video Render -> Output MP4
    """
    print("==========================================================")
    print("🎬 STARTING END-TO-END AUTOMATED VIDEO GENERATION PIPELINE")
    print("==========================================================")

    # 1. Ensure public directory exists
    public_dir = os.path.join(os.path.dirname(__file__), "public")
    os.makedirs(public_dir, exist_ok=True)
    audio_wav = os.path.join(public_dir, "audio.wav")

    # 2. Synthesize Speech Audio via NVIDIA Riva gRPC
    success = synthesize_audio(text, audio_wav)
    if not success:
        print("❌ Speech synthesis failed. Aborting pipeline.")
        return False

    # 3. Render Remotion React Video Composition
    print("🎥 Rendering Remotion 60fps React Video Composition...")
    render_cmd = [
        "npx.cmd" if os.name == "nt" else "npx",
        "remotion",
        "render",
        "MasterclassVideo",
        output_mp4,
        "--props", f'{{"title":"{title}","subtitle":"{subtitle}","audioSrc":"audio.wav"}}'
    ]

    try:
        res = subprocess.run(render_cmd, cwd=os.path.dirname(__file__), capture_output=True, text=True)
        if res.returncode == 0:
            print("==========================================================")
            print(f"🎉 VIDEO GENERATED SUCCESSFULLY: {output_mp4}")
            print("==========================================================")
            return True
        else:
            print(f"❌ Video render failed: {res.stderr or res.stdout}")
            return False
    except Exception as err:
        print(f"❌ Execution error: {err}")
        return False

if __name__ == "__main__":
    text_input = "Welcome to the AI Engineering Academy masterclass. Today we build autonomous AI agents from first principles."
    if len(sys.argv) > 1:
        text_input = sys.argv[1]
    
    generate_full_video(text=text_input)
