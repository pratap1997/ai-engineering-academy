# 🎬 AI Engineering Academy — Video Studio Architecture & Production Blueprint

> **Mission**: Produce automated, publication-ready, 60fps high-definition masterclass videos and short-form media for all 50 modules in the AI Engineering Academy curriculum with 3Blue1Brown-quality math animations, cinematic code reveals, and zero manual editing.

---

## 🏛️ End-to-End System Architecture

```
                  ┌────────────────────────────────────────────────────────┐
                  │              Lesson Content & Script Engine            │
                  │   ai-engineering-academy/modules/001-perceptron/       │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │        1. NVIDIA Riva gRPC Speech Synthesis            │
                  │        - Voice: Chatterbox-Multilingual.en-US.Male      │
                  │        - Output: RIFF PCM 44.1kHz 16-bit WAV           │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │        2. Manim Mathematical Animation Engine          │
                  │        - 3Blue1Brown-style 2D/3D geometry & vectors     │
                  │        - Render: 60fps MP4 math scenes                 │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │        3. Remotion React Video Composition             │
                  │        - Scene 1: Intro & Historical Badge             │
                  │        - Scene 2: Geometric Hyperplane (React)         │
                  │        - Scene 3: Formal Mathematics Cards             │
                  │        - Scene 4: Matrix Digital Rain Code Reveal      │
                  │        - Scene 5: Live Weight Learning (SVG)           │
                  │        - Scene 6: XOR Impossibility Proof              │
                  │        - Transitions: VHS Glitch & RGB Split           │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │        4. MoviePy & OpenCV Post-Processing             │
                  │        - Audio Ducking & Ambient Music Layering         │
                  │        - Multi-Scene Stitching & Subtitle Overlays     │
                  │        - Watermarking & Dynamic Equalizer              │
                  └───────────────────────────┬────────────────────────────┘
                                              │
       ┌──────────────────────────────────────┴──────────────────────────────────────┐
       ▼                                                                             ▼
┌─────────────────────────────┐                               ┌─────────────────────────────┐
│ 16:9 Masterclass Video MP4 │                               │  9:16 Shorts & Reels Video  │
│  (YouTube & Web App Player) │                               │  (TikTok / Shorts / Reels)  │
└─────────────────────────────┘                               └─────────────────────────────┘
```

---

## 📦 Repositories & Open Source Ecosystem

Here is the complete registry of open-source repositories integrated into our video production pipeline:

| Repository Name | GitHub URL | Purpose & Integration |
|---|---|---|
| **Remotion** | [github.com/remotion-dev/remotion](https://github.com/remotion-dev/remotion) | Core React 60fps video rendering engine, frame-accurate animation hooks (`useCurrentFrame`, `spring`, `interpolate`). |
| **Manim** | [github.com/ManimCommunity/manim](https://github.com/ManimCommunity/manim) | 3Blue1Brown's mathematical animation engine. Renders vector dot products, loss landscapes, and coordinate hyperplanes. |
| **MoviePy** | [github.com/Zulko/moviepy](https://github.com/Zulko/moviepy) | Multi-scene stitching, audio ducking, background music mixing, and animated GIF creation. |
| **OpenCV** | [github.com/opencv/opencv-python](https://github.com/opencv/opencv-python) | Computer vision video frame processing, bounding box overlays, decision boundaries, and frame resizing. |
| **HandAnim** | [github.com/subroy13/handanim](https://github.com/subroy13/handanim.git) | Whiteboard hand-drawn sketch animations (`sketch_animator.py`) character-by-character line writing. |
| **NVIDIA Riva Python Clients** | [github.com/nvidia-riva/python-clients](https://github.com/nvidia-riva/python-clients) | Speech synthesis client calling NVIDIA Riva gRPC API with Chatterbox Male voice (`grpc.nvcf.nvidia.com:443`). |
| **Vox AI Motion Graphics** | [github.com/Anil-matcha/vox-ai-motion-graphics-generator](https://github.com/Anil-matcha/vox-ai-motion-graphics-generator) | Reference 6-stage pipeline architecture (Beat-Map → Keyframes → Motion → Voice → Assembly). |
| **OpenAgentSkill** | [github.com/Leon-Drq/openagentskill](https://github.com/Leon-Drq/openagentskill) | Agent skill registry model for packaging rendering & synthesis pipeline commands as CLI skills. |
| **Short Video Creator** | [github.com/sw-aka/Short-Video-Creator](https://github.com/sw-aka/Short-Video-Creator) | Automated 9:16 vertical video & word-level subtitle generation blueprint. |

---

## 🛠️ Installed Skills & Helper Libraries

| Skill / Tool | Path / Location | Functionality |
|---|---|---|
| **`motion-design`** | Global Skill (`.agents/skills/motion-design`) | Disney animation principles (anticipation, squash/stretch, follow-through) for neural signals. |
| **`greensock/gsap-skills`** | Global Skill (`.agents/skills/gsap-skills`) | Official GSAP timelines and smooth easing functions in Remotion scenes. |
| **`ui-ux-pro-max`** | Global Skill (`.agents/skills/ui-ux-pro-max`) | High-contrast editorial color palette design system (`#FBFBFA` cream, `#059669` emerald). |
| **`Pillow (PIL)`** | `pip install Pillow` | Image watermarking, slide badge rendering, transparent overlay manipulation. |
| **`PyAV / Imageio`** | `pip install imageio` | High-speed FFmpeg frame stream reading, encoding, and GIF exports for GitHub READMEs. |

---

## 💻 Complete Technology Stack Breakdown

### 1. Audio Synthesis Engine (`synthesize_voice.py`)
- **Server Endpoint**: `grpc.nvcf.nvidia.com:443`
- **Function ID**: `ddacc747-1269-4fab-bfd9-8f593dead106`
- **Voice Model**: `Chatterbox-Multilingual.en-US.Male` (Sample rate: 44,100 Hz)
- **Automatic Text Chunking**: Splits script into $\le 120$ character sentences to respect Riva's 500-token per-call inference limit and concatenates audio byte buffers smoothly.

### 2. React Video Composition (`MasterclassVideo.tsx`)
- **Frame Rate**: 60 FPS
- **Resolution**: 1920x1080 (16:9 Widescreen Masterclass)
- **Scenes & Components**:
  - `IntroScene.tsx`: Title card, glowing particle background, historical Rosenblatt 1958 badge.
  - `MentalModelScene.tsx`: Geometric coordinate plane with animated hyperplane rotation separating class 0 and 1.
  - `MathematicsScene.tsx`: Formal mathematical derivation cards ($z = \mathbf{w}^T \mathbf{x} + b$, Heaviside step, $\Delta \mathbf{w} = \eta (y - \hat{y}) \mathbf{x}$).
  - `MatrixCodeReveal.tsx`: Digital Matrix falling code rain that solidifies column-by-column into clean Python code.
  - `WeightLearningAnimation.tsx`: Real-time SVG scatter plot where decision line physically moves frame-by-frame during training.
  - `XorLimitScene.tsx`: Geometric proof of XOR non-separability (Minsky & Papert 1969).
  - `GlitchTransition.tsx`: VHS-style RGB channel split + CRT scanline distortion between chapters.

### 3. Pipeline Automation (`pipeline.py`)
- Automatically measures WAV duration using Python's `wave` module.
- Calculates exact `durationInFrames = math.ceil(audio_duration * 60)`.
- Passes dynamic props via `props.json` to Remotion CLI.
- Executes headless Chromium render with `--gl=swiftshader` and auto-retry logic for 100% reliable builds on Windows.

---

## ⚡ Execution Command Quick Reference

```powershell
# 1. Synthesize Speech & Render 16:9 Masterclass Video
python tools/video-generator/generate_module_001_video.py

# 2. Render Whiteboard Hand-Drawn Math Animation
python tools/video-generator/sketch_animator.py

# 3. Process Multi-Scene Video & Audio Ducking with MoviePy
python tools/video-generator/video_editor.py

# 4. Preview Remotion Studio interactively in browser
cd tools/video-generator
npm run dev
```
