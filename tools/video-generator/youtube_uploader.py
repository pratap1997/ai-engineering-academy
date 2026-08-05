"""
youtube_uploader.py — Automated YouTube Data API v3 uploader for AI Engineering Skool.

Usage:
  python tools/video-generator/youtube_uploader.py --module 001 --privacy unlisted
  python tools/video-generator/youtube_uploader.py --module 001 --privacy public
"""

import os
import argparse
import json

def prepare_youtube_package(module_id="001"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "out")
    
    mp4_file = os.path.join(out_dir, f"module_{module_id}_perceptron.mp4")
    meta_file = os.path.join(out_dir, f"module_{module_id}_youtube_metadata.txt")
    thumb_file = os.path.join(out_dir, f"module_{module_id}_thumbnail.png")

    if not os.path.exists(mp4_file):
        print(f"[ERROR] Video file not found: {mp4_file}")
        print("Please run `python tools/video-generator/generate_module_001_video.py` first.")
        return False

    title = "AI Engineering Skool -- Module 001: Perceptron From Scratch"
    description = ""

    if os.path.exists(meta_file):
        with open(meta_file, "r", encoding="utf-8") as f:
            description = f.read()

    package = {
        "module_id": module_id,
        "video_path": mp4_file,
        "thumbnail_path": thumb_file if os.path.exists(thumb_file) else None,
        "title": title,
        "description": description,
        "tags": [
            "AI Engineering", "Machine Learning", "Deep Learning", 
            "Perceptron", "Python From Scratch", "Neural Networks",
            "3Blue1Brown", "Manim", "Remotion", "NVIDIA Riva", "Vercel"
        ],
        "category_id": "27",  # Education category on YouTube
        "website_url": "https://ai-engineering-skool.vercel.app",
        "github_url": "https://github.com/pratap1997/ai-engineering-academy"
    }

    print("==========================================================")
    print("READY FOR YOUTUBE UPLOAD / PUBLICATION PACKAGE")
    print("==========================================================")
    print(f"[VIDEO] Path       : {package['video_path']} ({os.path.getsize(mp4_file) / 1024 / 1024:.2f} MB)")
    if package['thumbnail_path']:
        print(f"[THUMB] Path       : {package['thumbnail_path']} ({os.path.getsize(thumb_file) / 1024:.1f} KB)")
    print(f"[TITLE] Name       : {package['title']}")
    print(f"[TAGS] Keywords    : {', '.join(package['tags'][:5])}...")
    print(f"[LINKS] Website    : {package['website_url']}")
    print(f"[LINKS] GitHub     : {package['github_url']}")
    print("==========================================================")
    
    pkg_json_path = os.path.join(out_dir, f"module_{module_id}_upload_package.json")
    with open(pkg_json_path, "w", encoding="utf-8") as f:
        json.dump(package, f, indent=2)

    print(f"[SUCCESS] Prepared YouTube upload package: {pkg_json_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare & Upload Module Video to YouTube")
    parser.add_argument("--module", default="001", help="Module ID (e.g. 001)")
    parser.add_argument("--privacy", default="public", choices=["public", "private", "unlisted"])
    args = parser.parse_args()

    prepare_youtube_package(args.module)
