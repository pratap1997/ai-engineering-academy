"""
youtube_metadata_exporter.py — Exports formatted YouTube Title, Description & Timestamps.

Includes:
  - Open-Source GitHub Repository Link
  - Live Vercel / Netlify Web App Link
  - Word-level burned-in subtitles note
  - Module timestamps & syllabus topics
"""

import os

def export_youtube_metadata(module_id="001", title="Perceptron From Scratch", output_txt="out/module_001_youtube_metadata.txt", **kwargs):
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)

    metadata = f"""AI Engineering Skool -- Module {module_id}: {title} (Complete Mathematical Derivation & Pure Python Impl)

In this masterclass, we build {title} from scratch in pure Python and NumPy with ZERO high-level framework dependencies. Learn the underlying mathematics, geometric decision hyperplanes, loss landscapes, and edge-case failure modes.

CHAPTER TIMESTAMPS:
00:00 - Introduction & Historical Context (Rosenblatt 1958)
00:10 - Geometric Intuition & Decision Hyperplanes
00:21 - Formal Mathematical Derivation & Step Activation
00:35 - Pure Python & NumPy Implementation (Matrix Code Reveal)
00:47 - Live Weight Update Learning Animation (Scatter Plot)
00:57 - XOR Limitation & Minsky & Papert 1969 Proof

LIVE WEB APP & OPEN SOURCE REPOSITORY:
- Live Interactive Web App: https://ai-engineering-skool.vercel.app
- GitHub Open Source Codebase: https://github.com/pratap1997/ai-engineering-academy

KEY CONCEPTS COVERED:
1. Linear Decision Boundaries z = W^T X + b
2. Heaviside Step Activation Function
3. Perceptron Weight Update Rule Delta W = eta * (y - y_hat) * X
4. Convergence Theorem on Linearly Separable Datasets
5. XOR Non-Linear Impossibility & Multi-Layer Perceptron Motivation

#AI #MachineLearning #DeepLearning #Python #AIEngineering #NeuralNetworks #FromScratch #Vercel #OpenSource
"""

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(metadata)

    print("----------------------------------------------------------")
    print("YOUTUBE VIDEO METADATA & DESCRIPTION:")
    print("----------------------------------------------------------")
    print(metadata)
    print(f"[SUCCESS] Saved YouTube metadata to {output_txt}")

# Alias for backwards compatibility
generate_youtube_metadata = export_youtube_metadata

if __name__ == "__main__":
    export_youtube_metadata()
