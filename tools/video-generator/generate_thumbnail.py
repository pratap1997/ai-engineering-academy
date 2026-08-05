"""
generate_thumbnail.py — Automated 1280x720 High-CTR YouTube Thumbnail Generator.

Uses OpenAgent Editorial Styling:
  - Clean charcoal/emerald gradient background
  - Large bold white title + emerald module badge
  - High-contrast code snippet preview card
  - 1280x720 PNG output ready for YouTube Studio upload
"""

import os
import argparse
from PIL import Image, ImageDraw, ImageFont

def generate_module_thumbnail(module_id="001", title="PERCEPTRON", subtitle="From Scratch in Pure Python", output_png=None):
    if output_png is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_png = os.path.join(base_dir, "out", f"module_{module_id}_thumbnail.png")
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    # 1280x720 Canvas (YouTube Standard)
    width, height = 1280, 720
    img = Image.new("RGB", (width, height), color="#090C10")
    draw = ImageDraw.Draw(img)

    # Background Radial Glow
    for r in range(400, 0, -5):
        alpha = int((1 - r / 400) * 40)
        draw.ellipse([800 - r, 360 - r, 800 + r, 360 + r], fill=(16, 185, 129, alpha))

    # Top Brand Bar: () AI ENGINEERING SKOOL
    draw.rectangle([60, 50, 480, 100], fill="#121620", outline="#10B981", width=2)
    try:
        font_badge = ImageFont.truetype("arial.ttf", 22)
        font_title = ImageFont.truetype("arialbd.ttf", 64)
        font_sub = ImageFont.truetype("arial.ttf", 32)
        font_code = ImageFont.truetype("cour.ttf", 20)
    except IOError:
        font_badge = font_title = font_sub = font_code = ImageFont.load_default()

    draw.text((80, 62), f"() AI ENGINEERING SKOOL  •  #{module_id}", fill="#10B981", font=font_badge)

    # Main Title
    draw.text((60, 150), title.upper(), fill="#FFFFFF", font=font_title)

    # Subtitle
    draw.text((60, 240), subtitle, fill="#94A3B8", font=font_sub)

    # Code / Visual Preview Card (Right side 640x480)
    card_box = [620, 150, 1220, 630]
    draw.rectangle(card_box, fill="#0D1117", outline="#10B981", width=3)

    # Window bar
    draw.ellipse([640, 170, 656, 186], fill="#FF5F57")
    draw.ellipse([666, 170, 682, 186], fill="#FEBC2E")
    draw.ellipse([692, 170, 708, 186], fill="#28C840")
    draw.text((730, 168), "04-implementation.py", fill="#10B981", font=font_code)

    # Python snippet
    code_lines = [
        "class Perceptron:",
        "  def __init__(self, lr=0.01):",
        "    self.w = np.zeros(d)",
        "    self.b = 0.0",
        "",
        "  def fit(self, X, y):",
        "    for _ in range(epochs):",
        "      for xi, yi in zip(X, y):",
        "        y_hat = step(np.dot(self.w, xi) + self.b)",
        "        err = yi - y_hat",
        "        self.w += lr * err * xi",
        "        self.b += lr * err",
    ]

    y_offset = 210
    for line in code_lines:
        color = "#A5B4FC" if "def" in line or "class" in line else "#F3F5F7"
        if "err" in line or "self.w +=" in line:
            color = "#10B981"
        draw.text((640, y_offset), line, fill=color, font=font_code)
        y_offset += 32

    # Bottom Pill Badges
    draw.rectangle([60, 630, 320, 675], fill="#121620", outline="#10B981", width=1)
    draw.text((80, 642), "100% PURE PYTHON", fill="#10B981", font=font_badge)

    draw.rectangle([340, 630, 580, 675], fill="#121620", outline="#475569", width=1)
    draw.text((360, 642), "801 TESTS PASSING", fill="#F3F5F7", font=font_badge)

    # Save PNG
    img.save(output_png)
    print("==========================================================")
    print(f"[SUCCESS] Generated YouTube Thumbnail: {output_png}")
    print("==========================================================")
    return output_png

if __name__ == "__main__":
    generate_module_thumbnail()
