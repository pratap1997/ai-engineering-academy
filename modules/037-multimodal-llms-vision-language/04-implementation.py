import numpy as np
import math

class ImagePatchifier:
    """Converts a 2D image into a flattened sequence of patches."""
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def patchify(self, image):
        # image shape: (height, width, channels)
        h, w, c = image.shape
        p = self.patch_size
        assert h % p == 0 and w % p == 0, "Image dimensions must be divisible by patch size"
        
        num_patches_h = h // p
        num_patches_w = w // p
        
        patches = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                patch = image[i*p:(i+1)*p, j*p:(j+1)*p, :]
                patches.append(patch.flatten())
                
        return np.array(patches) # shape: (num_patches, p*p*c)


class VisionEncoder:
    """Simplified Vision Transformer (ViT) Self-Attention Block."""
    def __init__(self, input_dim, hidden_dim, num_heads=1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Initialize with small random weights for demonstration
        np.random.seed(42)
        self.W_q = np.random.randn(input_dim, hidden_dim) * 0.02
        self.W_k = np.random.randn(input_dim, hidden_dim) * 0.02
        self.W_v = np.random.randn(input_dim, hidden_dim) * 0.02
        
    def forward(self, x):
        # x shape: (seq_len, input_dim)
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        
        # Scaled dot-product attention
        scores = Q @ K.T / math.sqrt(self.hidden_dim)
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        out = attn_weights @ V
        return out, attn_weights


class MultimodalProjectionLayer:
    """Aligns visual feature dimension to text embedding dimension."""
    def __init__(self, vision_dim, text_dim):
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        
        np.random.seed(43)
        self.W_proj = np.random.randn(vision_dim, text_dim) * 0.02
        self.b_proj = np.zeros((text_dim,))
        
    def forward(self, vision_features):
        # Linear projection: H_vision * W + b
        return vision_features @ self.W_proj + self.b_proj


class MultimodalTokenizer:
    """Interleaves standard text tokens with a special <IMAGE> token."""
    def __init__(self, text_vocab_size):
        self.text_vocab_size = text_vocab_size
        self.image_token = "<IMAGE>"
        self.image_token_id = -1 # Special ID for images
        
    def tokenize(self, text):
        tokens = text.split()
        token_ids = []
        for t in tokens:
            if t == self.image_token:
                token_ids.append(self.image_token_id)
            else:
                # Mock tokenization by hashing
                token_ids.append(abs(hash(t)) % self.text_vocab_size)
        return token_ids


class SimpleVLM:
    """Full Vision-Language Model Pipeline Forward Pass."""
    def __init__(self, patch_size, vision_hidden_dim, text_dim, vocab_size):
        self.patchifier = ImagePatchifier(patch_size)
        
        # Assume 3 color channels
        patch_dim = patch_size * patch_size * 3 
        self.vision_encoder = VisionEncoder(patch_dim, vision_hidden_dim)
        self.projection = MultimodalProjectionLayer(vision_hidden_dim, text_dim)
        self.tokenizer = MultimodalTokenizer(vocab_size)
        
        # Mock text embedding layer
        np.random.seed(44)
        self.text_embeddings = np.random.randn(vocab_size, text_dim) * 0.02
        
    def forward(self, text, image):
        # 1. Process image into projected visual tokens
        patches = self.patchifier.patchify(image)
        vision_features, attn = self.vision_encoder.forward(patches)
        projected_vision = self.projection.forward(vision_features)
        
        # 2. Process text into token IDs
        token_ids = self.tokenizer.tokenize(text)
        
        # 3. Interleave text and vision tokens
        combined_sequence = []
        for tid in token_ids:
            if tid == self.tokenizer.image_token_id:
                # Expand <IMAGE> token into actual projected visual sequence
                for v in projected_vision:
                    combined_sequence.append(v)
            else:
                combined_sequence.append(self.text_embeddings[tid])
                
        return np.array(combined_sequence)
