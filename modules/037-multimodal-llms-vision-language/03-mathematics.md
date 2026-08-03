# Mathematics of Vision-Language Models

## 1. Image Patchification

Given an image $I \in \mathbb{R}^{H \times W \times C}$ (Height, Width, Channels), we divide it into patches of size $P \times P$.

The number of patches $N$ is:
$$ N = \frac{H \cdot W}{P^2} $$

Each patch is flattened into a vector $x_p \in \mathbb{R}^{P^2 \cdot C}$. The sequence of patches is represented as:
$$ X_{patches} = [x_{p_1}, x_{p_2}, \dots, x_{p_N}] \in \mathbb{R}^{N \times (P^2 \cdot C)} $$

## 2. Vision Transformer (ViT) Encoding

The patches are linearly projected and added to positional embeddings. They are then passed through the Vision Transformer. The core operation is self-attention:

$$ \text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

Where $Q$ (Query), $K$ (Key), and $V$ (Value) are linear transformations of the input sequence, and $d_k$ is the hidden dimension size of the vision encoder.

The output of the vision encoder is a sequence of visual features:
$$ H_{vision} \in \mathbb{R}^{N \times d_{vision}} $$

## 3. Multimodal Projection Layer

To allow the LLM to understand $H_{vision}$, we must project it into the text embedding space dimension ($d_{text}$). In the simplest form (e.g., LLaVA-1.5), this is a linear transformation:

$$ H_{projected} = H_{vision} \cdot W_{proj} + b_{proj} $$

Where:
- $W_{proj} \in \mathbb{R}^{d_{vision} \times d_{text}}$ is the learned projection weight matrix.
- $b_{proj} \in \mathbb{R}^{d_{text}}$ is the learned bias vector.
- $H_{projected} \in \mathbb{R}^{N \times d_{text}}$ is the aligned visual tokens.

## 4. Sequence Concatenation

Let the text prompt tokens be embedded as $H_{text} \in \mathbb{R}^{M \times d_{text}}$, where $M$ is the number of text tokens.

The final sequence fed into the LLM backbone is the concatenation of the text embeddings and the projected vision embeddings:

$$ H_{input} = [H_{text}^{(1 \dots i)}, H_{projected}, H_{text}^{(i+1 \dots M)}] $$

The LLM then applies its standard causal self-attention over this joint sequence to generate the next token.
