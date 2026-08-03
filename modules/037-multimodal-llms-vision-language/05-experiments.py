import numpy as np
from importlib.util import spec_from_file_location, module_from_spec
import sys
from pathlib import Path

# Load implementation
current_dir = Path(__file__).parent
module_path = current_dir / "04-implementation.py"
spec = spec_from_file_location("module_037_impl", module_path)
impl = module_from_spec(spec)
sys.modules["module_037_impl"] = impl
spec.loader.exec_module(impl)

def experiment_1_patch_size():
    print("--- Experiment 1: Patch Size vs Token Count ---")
    image = np.zeros((224, 224, 3))
    print(f"Image Resolution: 224x224x3")
    
    for patch_size in [16, 32, 56]:
        patchifier = impl.ImagePatchifier(patch_size)
        patches = patchifier.patchify(image)
        print(f"Patch Size {patch_size}x{patch_size} -> {patches.shape[0]} visual tokens")

def experiment_2_projection_alignment():
    print("\n--- Experiment 2: Projection Layer Alignment ---")
    vision_dim = 768
    text_dim = 4096
    proj = impl.MultimodalProjectionLayer(vision_dim, text_dim)
    
    mock_vision_feature = np.random.randn(1, vision_dim)
    output = proj.forward(mock_vision_feature)
    
    print(f"Input Vision Dim: {mock_vision_feature.shape}")
    print(f"Output Text Dim: {output.shape}")
    print(f"Alignment Success: {output.shape[1] == text_dim}")

def experiment_3_attention_distribution():
    print("\n--- Experiment 3: Vision Attention Weight Distribution ---")
    enc = impl.VisionEncoder(input_dim=192, hidden_dim=64)
    mock_patches = np.random.randn(4, 192)  # 4 patches
    out, attn = enc.forward(mock_patches)
    
    print("Attention Weights Matrix (4x4):")
    print(np.round(attn, 2))
    print(f"Sum of weights per row: {np.sum(attn, axis=1)}")

def experiment_4_synthetic_interleaving():
    print("\n--- Experiment 4: Synthetic Interleaving Pipeline ---")
    vlm = impl.SimpleVLM(patch_size=16, vision_hidden_dim=256, text_dim=512, vocab_size=1000)
    
    image = np.zeros((32, 32, 3)) # 4 patches
    prompt = "Describe this <IMAGE> in detail"
    
    print(f"Prompt: '{prompt}'")
    print(f"Image patches expected: { (32//16)**2 }")
    
    output_seq = vlm.forward(prompt, image)
    print(f"Final sequence length: {output_seq.shape[0]} tokens")
    print(f"Final sequence embedding dim: {output_seq.shape[1]}")
    
if __name__ == "__main__":
    experiment_1_patch_size()
    experiment_2_projection_alignment()
    experiment_3_attention_distribution()
    experiment_4_synthetic_interleaving()
