# Implementation of LoRA
import numpy as np

class LinearLayer:
    def __init__(self, in_features, out_features, seed=42):
        np.random.seed(seed)
        self.weight = np.random.randn(out_features, in_features) * 0.01
        self.bias = np.zeros((out_features,))
        self.in_features = in_features
        self.out_features = out_features
        
    def forward(self, x):
        return x @ self.weight.T + self.bias
        
class LoRALayer:
    def __init__(self, base_layer: LinearLayer, rank=4, alpha=8):
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.A = np.random.randn(rank, base_layer.in_features) * 0.01
        self.B = np.zeros((base_layer.out_features, rank))
        
    def forward(self, x):
        base_out = self.base_layer.forward(x)
        lora_out = (x @ self.A.T @ self.B.T) * self.scaling
        return base_out + lora_out
        
    def merge(self):
        return self.base_layer.weight + (self.B @ self.A) * self.scaling

class QuantizedWeights:
    def __init__(self, weight_matrix, bits=8):
        self.original_shape = weight_matrix.shape
        self.bits = bits
        bins = 127.0 if bits == 8 else 7.0
        self.scale = np.max(np.abs(weight_matrix)) / bins
        self.quantized = np.round(weight_matrix / self.scale).astype(np.int8)
            
    def dequantize(self):
        return self.quantized.astype(np.float32) * self.scale

class SFTTrainer:
    def __init__(self, model, learning_rate=1e-3):
        self.model = model
        self.lr = learning_rate
        
    def compute_loss(self, logits, targets):
        return np.mean((logits - targets)**2)
        
    def format_sft_data(self, instruction, response):
        return f"Instruction:\n{instruction}\nResponse:\n{response}"

class LoRAModel:
    def __init__(self, layers):
        self.layers = layers
