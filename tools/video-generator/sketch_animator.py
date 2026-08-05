import os
import cv2
import numpy as np

def generate_hand_sketch_video(
    lines_text=["w1*x1 + w2*x2 + b = 0", "Step Activation: f(z) = 1 if z >= 0 else 0"],
    output_mp4="out/sketch_animation.mp4",
    fps=30,
    duration_sec=4
):
    """
    Simulates HandAnim-style whiteboard hand-drawn writing animation using OpenCV.
    Draws mathematical equations character-by-character on a clean whiteboard.
    """
    width, height = 1280, 720
    total_frames = fps * duration_sec
    os.makedirs(os.path.dirname(output_mp4), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_mp4, fourcc, fps, (width, height))

    # Clean whiteboard background
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 248  # Off-white

    all_chars = " ".join(lines_text)
    total_chars = len(all_chars)

    print(f"[HANDANIM] Rendering Whiteboard Sketch Animation: {output_mp4}")

    for frame_idx in range(total_frames):
        frame = canvas.copy()
        
        # Calculate how many characters to reveal based on frame progress
        progress = frame_idx / total_frames
        chars_to_show = int(progress * total_chars)

        y_offset = 250
        char_counter = 0

        for line in lines_text:
            visible_line = ""
            for char in line:
                if char_counter < chars_to_show:
                    visible_line += char
                    char_counter += 1
                else:
                    break

            # Draw text with hand-drawn style dark ink
            cv2.putText(
                frame,
                visible_line,
                (150, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (30, 27, 24), # Dark charcoal ink
                3,
                cv2.LINE_AA
            )
            y_offset += 100

        out.write(frame)

    out.release()
    print(f"[SUCCESS] HandAnim Sketch Animation generated: {output_mp4}")
    return True

if __name__ == "__main__":
    generate_hand_sketch_video()
