# Module 037: Multimodal LLMs & Vision-Language Models (VLMs)

## Context
Traditional Large Language Models (LLMs) operate strictly on text. They read text tokens and predict text tokens. However, the world is multimodal—humans perceive the world through vision, sound, touch, and text simultaneously. Vision-Language Models (VLMs) bridge this gap by enabling LLMs to "see" and understand images, documents, and videos alongside text.

## Core Architecture
A typical Vision-Language Model (like LLaVA or GPT-4o's vision components) consists of three primary components:

1. **Vision Encoder:** Extracts meaningful visual features from an image. Usually, this is a Vision Transformer (ViT) or a ResNet variant. CLIP (Contrastive Language-Image Pretraining) is a popular choice for the vision encoder because it aligns visual features with semantic concepts.
2. **Projection Layer (Cross-Modality Alignment):** A neural network layer (often a simple Linear layer or a multi-layer perceptron/MLP) that maps the visual features from the Vision Encoder's dimensional space into the LLM's text embedding space.
3. **LLM Backbone:** A standard pre-trained LLM (like Llama, Mistral, or GPT) that processes the combined sequence of text tokens and projected image tokens to generate a response.

## Patchification
Images cannot be directly fed into an LLM. They are 2D grids of pixels, while transformers expect 1D sequences of tokens. The process of converting an image into a sequence of tokens is called **Patchification**.
- An image of size $H \times W \times C$ is divided into a grid of non-overlapping patches of size $P \times P$.
- Each patch is flattened into a 1D vector and passed through a linear layer to create a sequence of "visual tokens."

## Joint Vision-Text Sequences
Once the image is patchified, encoded, and projected, it acts just like a sequence of text words. If a user asks "What is in this image? [IMAGE_DATA]", the model sees a sequence like:
`[TEXT_TOKEN_1] [TEXT_TOKEN_2] ... [VISION_TOKEN_1] [VISION_TOKEN_2] ... [VISION_TOKEN_N]`

The LLM applies its standard self-attention mechanism across both text and vision tokens simultaneously, allowing it to reason about the image in the context of the user's prompt.
