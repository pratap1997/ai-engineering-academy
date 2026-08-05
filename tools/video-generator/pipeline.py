import os
import sys
import json
import math
import wave
import subprocess
from synthesize_voice import synthesize_audio

def get_wav_duration_seconds(wav_path):
    """
    Returns exact duration of WAV file in seconds using Python standard library wave module.
    """
    try:
        with wave.open(wav_path, 'r') as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        print(f"[WARNING] Could not read WAV duration: {e}")
        return 66.0

def generate_full_video(
    text="Welcome to the AI Engineering Academy masterclass. Today we build autonomous AI agents from first principles.",
    title="MODULE 001: THE PERCEPTRON",
    subtitle="Complete Masterclass: Mathematical Derivation, NumPy Code & XOR Limits",
    output_mp4="out/module_001_perceptron.mp4"
):
    """
    End-to-End Automated Video Generation Pipeline:
    1. NVIDIA Riva gRPC TTS Speech Synthesis (Male Voice, Chunked)
    2. Exact Audio Duration Calculation (Dynamic Frame Sync)
    3. Remotion Multi-Scene React Video Composition (60fps Kinetic Animations)
    4. Headless Video Render -> Output MP4
    """
    print("==========================================================")
    print("STARTING MULTI-SCENE ANIMATED VIDEO GENERATION PIPELINE")
    print("==========================================================")

    # 1. Ensure public directory exists
    public_dir = os.path.join(os.path.dirname(__file__), "public")
    os.makedirs(public_dir, exist_ok=True)
    audio_wav = os.path.join(public_dir, "audio.wav")

    # 2. Synthesize Speech Audio via NVIDIA Riva gRPC
    success, chunk_timings = synthesize_audio(text, audio_wav)
    if not success:
        print("[ERROR] Speech synthesis failed. Aborting pipeline.")
        return False

    # 3. Calculate exact audio duration for frame synchronization
    audio_duration = get_wav_duration_seconds(audio_wav)
    duration_in_frames = math.ceil(audio_duration * 60)
    print(f"[SYNC] Audio Duration: {audio_duration:.2f}s -> Video Frames @ 60fps: {duration_in_frames}")

    # 4. Ensure output directory exists
    out_dir = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out_dir, exist_ok=True)

    # 5. Save props to props.json for clean Remotion execution
    props_data = {
        "title": title,
        "subtitle": subtitle,
        "codeSnippet": "class Perceptron:\n    def __init__(self, d_in, lr=0.01):\n        self.w = np.zeros(d_in)\n        self.b = 0.0\n        self.lr = lr\n\n    def predict(self, x):\n        z = np.dot(x, self.w) + self.b\n        return np.where(z >= 0, 1, 0)\n\n    def fit(self, X, y, epochs=100):\n        for _ in range(epochs):\n            for xi, target in zip(X, y):\n                update = self.lr * (target - self.predict(xi))\n                self.w += update * xi\n                self.b += update",
        "audioSrc": "audio.wav",
        "durationInFrames": duration_in_frames,
        "subtitleChunks": chunk_timings,
    }
    props_json_path = os.path.join(os.path.dirname(__file__), "props.json")
    with open(props_json_path, "w", encoding="utf-8") as f:
        json.dump(props_data, f, indent=2)

    # 6. Render Remotion React Video Composition (with retry on Chrome timeout)
    print("Rendering Multi-Scene Remotion 60fps React Video Composition...")
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

    # Env vars to help Chromium start correctly in headless mode
    render_env = os.environ.copy()
    render_env["REMOTION_CHROME_PATH"] = chrome_path

    render_cmd = [
        "npx.cmd" if os.name == "nt" else "npx",
        "remotion",
        "render",
        "MasterclassVideo",
        output_mp4,
        "--props=props.json",
        "--timeout=180000",
        "--concurrency=1",
        "--chromium-disable-web-security",
    ]

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        print(f"[RENDER] Attempt {attempt}/{max_retries}...")
        try:
            res = subprocess.run(
                render_cmd,
                cwd=os.path.dirname(__file__),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors='ignore',
                env=render_env,
            )
            if res.returncode == 0:
                print("==========================================================")
                print(f"SUCCESS: MULTI-SCENE ANIMATED VIDEO GENERATED: {output_mp4}")
                print("==========================================================")
                return True
            else:
                err_msg = (res.stderr or res.stdout or "")[:500]
                print(f"[ERROR] Attempt {attempt} failed: {err_msg}")
                if attempt < max_retries:
                    import time
                    wait = attempt * 5
                    print(f"[RETRY] Waiting {wait}s before next attempt...")
                    time.sleep(wait)
        except Exception as err:
            print(f"[ERROR] Execution error on attempt {attempt}: {err}")

    print("[FATAL] All render attempts failed.")
    return False

if __name__ == "__main__":
    text_input = "Welcome to the AI Engineering Academy masterclass. Today we build autonomous AI agents from first principles."
    if len(sys.argv) > 1:
        text_input = sys.argv[1]
    
    generate_full_video(text=text_input)
