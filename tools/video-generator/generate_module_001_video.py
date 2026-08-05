import os
import sys
from pipeline import generate_full_video
from youtube_metadata_exporter import generate_youtube_metadata

def build_comprehensive_module_001_masterclass():
    print("==========================================================")
    print("GENERATING COMPREHENSIVE MASTERCLASS VIDEO FOR MODULE 001: PERCEPTRON")
    print("==========================================================")

    narration_text = (
        "Welcome to Module 1 of the AI Engineering Academy: The Perceptron. "
        "Introduced by Frank Rosenblatt in 1958, the Perceptron is the foundational building block of modern deep learning. "
        "It models a biological neuron by calculating a weighted sum of inputs and applying a step activation function for binary classification. "
        "Geometrically, a Perceptron represents a linear decision boundary or hyperplane separating positive and negative classes in feature space. "
        "Mathematically, the pre-activation value z equals the dot product of weight vector W and input vector X plus bias b. "
        "When a prediction error occurs, weights update using the rule: delta W equals learning rate eta times target y minus prediction y-hat, multiplied by input X. "
        "Here is our clean Python implementation initializing weights to zero and iterating through epochs to adjust boundaries. "
        "While the Perceptron converges perfectly on linearly separable data like AND and OR logic gates, Marvin Minsky and Seymour Papert proved in 1969 that a single layer fails completely on non-linearly separable XOR problems, motivating Multi-Layer Perceptrons."
    )

    title = "MODULE 001: THE PERCEPTRON"
    subtitle = "Complete Masterclass: Mathematical Derivation, NumPy Code & XOR Limits"
    output_mp4 = "out/module_001_perceptron.mp4"

    # 1. Run automated speech synthesis + Remotion video render
    success = generate_full_video(
        text=narration_text,
        title=title,
        subtitle=subtitle,
        output_mp4=output_mp4
    )

    if success:
        # 2. Export YouTube Title, Description & Chapter Timestamps
        print("\n----------------------------------------------------------")
        print("YOUTUBE VIDEO METADATA & DESCRIPTION:")
        print("----------------------------------------------------------")
        metadata = generate_youtube_metadata(
            module_id="001",
            title="Perceptron From Scratch (Complete Mathematical Derivation & Python Impl)",
            repo_url="https://github.com/pratap1997/ai-engineering-academy"
        )
        print(metadata)
        
        # Save metadata file
        meta_file = "out/module_001_youtube_metadata.txt"
        with open(meta_file, "w", encoding="utf-8") as f:
            f.write(metadata)
        print(f"[SUCCESS] Saved YouTube metadata to {meta_file}")

if __name__ == "__main__":
    build_comprehensive_module_001_masterclass()
