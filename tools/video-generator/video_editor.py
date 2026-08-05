import os
import sys

try:
    from moviepy import VideoFileClip, concatenate_videoclips, AudioFileClip, CompositeAudioClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

def stitch_video_scenes(video_paths, output_path="out/full_masterclass.mp4"):
    """
    Uses MoviePy to concatenate multiple individual scene clips into a complete video.
    """
    if not MOVIEPY_AVAILABLE:
        print("[ERROR] MoviePy is not installed.")
        return False

    print(f"[MOVIEPY] Stitching {len(video_paths)} video scenes...")
    clips = []
    for path in video_paths:
        if os.path.exists(path):
            clips.append(VideoFileClip(path))
        else:
            print(f"[WARN] Scene path not found: {path}")

    if not clips:
        print("[ERROR] No valid video clips found to stitch.")
        return False

    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    print(f"[SUCCESS] Stitched full masterclass video saved: {output_path}")
    return True

def convert_clip_to_gif(mp4_path, gif_output_path="out/demo.gif", fps=15):
    """
    Converts a section of MP4 video clip to animated GIF for GitHub README documentation.
    """
    if not MOVIEPY_AVAILABLE:
        print("[ERROR] MoviePy is not installed.")
        return False

    print(f"[MOVIEPY] Converting MP4 to GIF: {mp4_path} -> {gif_output_path}")
    clip = VideoFileClip(mp4_path)
    clip.write_gif(gif_output_path, fps=fps)
    print(f"[SUCCESS] Generated GitHub README GIF: {gif_output_path}")
    return True

if __name__ == "__main__":
    print(f"MoviePy Available: {MOVIEPY_AVAILABLE}")
