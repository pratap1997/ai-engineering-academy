import pytest
import numpy as np
import sys
import importlib.util
from pathlib import Path

# Load implementation securely following instructions
def load_impl():
    current_dir = Path(__file__).parent
    module_path = current_dir.parent / "04-implementation.py"
    spec = importlib.util.spec_from_file_location("module_037_impl", module_path)
    impl = importlib.util.module_from_spec(spec)
    sys.modules["module_037_impl"] = impl
    spec.loader.exec_module(impl)
    return impl

impl = load_impl()

class TestPatchification:
    def test_patch_shape_correctness(self):
        img = np.zeros((32, 32, 3))
        patchifier = impl.ImagePatchifier(patch_size=8)
        patches = patchifier.patchify(img)
        assert patches.shape == (16, 8*8*3), "Patch shape mismatch"
        
    def test_total_patch_count(self):
        img = np.zeros((16, 32, 3))
        patchifier = impl.ImagePatchifier(patch_size=16)
        patches = patchifier.patchify(img)
        assert patches.shape[0] == 2, "Incorrect total number of patches"
        
    def test_element_conservation(self):
        img = np.zeros((8, 8, 3))
        patchifier = impl.ImagePatchifier(patch_size=4)
        patches = patchifier.patchify(img)
        assert patches.size == 8 * 8 * 3, "Total element count must remain constant"
        
    def test_indivisible_dimensions_rejection(self):
        img = np.zeros((10, 10, 3))
        patchifier = impl.ImagePatchifier(patch_size=3)
        with pytest.raises(AssertionError):
            patchifier.patchify(img)

class TestProjectionLayer:
    def test_output_dimension_alignment(self):
        proj = impl.MultimodalProjectionLayer(vision_dim=64, text_dim=128)
        vis_feat = np.zeros((10, 64))
        out = proj.forward(vis_feat)
        assert out.shape == (10, 128), "Projection did not align to text dimension"
        
    def test_zero_input_bias_addition(self):
        proj = impl.MultimodalProjectionLayer(vision_dim=32, text_dim=16)
        proj.b_proj = np.ones((16,))
        vis_feat = np.zeros((5, 32))
        out = proj.forward(vis_feat)
        np.testing.assert_array_equal(out, np.ones((5, 16)), "Bias addition failed")
        
    def test_deterministic_weight_multiplication(self):
        proj = impl.MultimodalProjectionLayer(vision_dim=2, text_dim=2)
        proj.W_proj = np.array([[1.0, 2.0], [3.0, 4.0]])
        proj.b_proj = np.zeros((2,))
        vis_feat = np.array([[1.0, 1.0]])
        out = proj.forward(vis_feat)
        np.testing.assert_array_equal(out, np.array([[4.0, 6.0]]), "Matrix multiplication failed")
        
    def test_batch_independence(self):
        proj = impl.MultimodalProjectionLayer(vision_dim=10, text_dim=20)
        vis_feat = np.ones((100, 10))
        out = proj.forward(vis_feat)
        assert out.shape == (100, 20), "Batch dimension corrupted"

class TestMultimodalTokenizer:
    def test_pure_text_processing(self):
        tok = impl.MultimodalTokenizer(text_vocab_size=100)
        tokens = tok.tokenize("hello world")
        assert len(tokens) == 2, "Length mismatch"
        assert tok.image_token_id not in tokens, "Image token erroneously found"
        
    def test_single_image_tag_interleaving(self):
        tok = impl.MultimodalTokenizer(text_vocab_size=100)
        tokens = tok.tokenize("hello <IMAGE> world")
        assert len(tokens) == 3, "Length mismatch"
        assert tokens[1] == tok.image_token_id, "Image token misplaced"
        
    def test_multiple_images_handling(self):
        tok = impl.MultimodalTokenizer(text_vocab_size=100)
        tokens = tok.tokenize("<IMAGE> <IMAGE>")
        assert len(tokens) == 2, "Length mismatch"
        assert tokens[0] == tok.image_token_id, "Missing image token 1"
        assert tokens[1] == tok.image_token_id, "Missing image token 2"
        
    def test_hashing_bounds(self):
        tok = impl.MultimodalTokenizer(text_vocab_size=50)
        tokens = tok.tokenize("a b c d e f g h i j")
        for t in tokens:
            assert 0 <= t < 50, "Hash tokenization out of bounds"

class TestForwardPassPipeline:
    def test_combined_sequence_total_length(self):
        vlm = impl.SimpleVLM(patch_size=4, vision_hidden_dim=32, text_dim=64, vocab_size=100)
        img = np.zeros((8, 8, 3)) # 4 patches
        text = "This is <IMAGE>" # 2 text + 4 image patches = 6
        out = vlm.forward(text, img)
        assert out.shape == (6, 64), "Final sequence output shape mismatch"
        
    def test_vision_encoder_attention_matrix(self):
        enc = impl.VisionEncoder(input_dim=16, hidden_dim=32)
        x = np.zeros((5, 16))
        out, attn = enc.forward(x)
        assert attn.shape == (5, 5), "Attention weights matrix dimension incorrect"
        
    def test_vision_encoder_output_dimensionality(self):
        enc = impl.VisionEncoder(input_dim=16, hidden_dim=32)
        x = np.zeros((5, 16))
        out, attn = enc.forward(x)
        assert out.shape == (5, 32), "Encoder output hidden dim incorrect"
        
    def test_graceful_ignore_without_image_tag(self):
        vlm = impl.SimpleVLM(patch_size=4, vision_hidden_dim=32, text_dim=64, vocab_size=100)
        img = np.zeros((4, 4, 3)) # 1 patch (ignored)
        text = "hello world" # 2 tokens
        out = vlm.forward(text, img)
        assert out.shape == (2, 64), "Pipeline incorrectly interleaved image without tag"
