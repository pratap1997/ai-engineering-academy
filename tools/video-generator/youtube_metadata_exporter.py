import sys

def generate_youtube_metadata(module_id="001", title="Perceptron From Scratch", repo_url="https://github.com/pratap1997/ai-engineering-academy"):
    """
    Generates YouTube video title, description, chapter timestamps, and tags for any module.
    """
    description = f"""🚀 AI Engineering Academy — Module {module_id}: {title}

In this masterclass, we implement {title} from scratch in pure Python and NumPy with ZERO high-level framework dependencies. Learn the underlying mathematics, geometric mental models, and edge-case failure modes.

📌 CHAPTER TIMESTAMPS:
00:00 - Introduction & Motivation
01:15 - Intuitive Mental Model & Decision Boundaries
03:45 - Formal Mathematical Derivation
06:30 - Pure Python & NumPy Implementation
09:15 - Runnable Empirical Experiments
11:45 - Real-World Engineering Applications & XOR Limitations
14:20 - Self-Assessment & Engineering Challenge

🔗 REPOSITORY & LIVE WEB APP:
- GitHub Repository: {repo_url}
- Interactive Web App Workspace: http://localhost:5173

#AI #MachineLearning #DeepLearning #Python #AIEngineering #NeuralNetworks #FromScratch
"""
    return description

if __name__ == "__main__":
    meta = generate_youtube_metadata()
    print(meta)
