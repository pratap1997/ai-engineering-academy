# Assessment

1. **What is the primary role of the Projection Layer in a VLM?**
   - *Answer:* It aligns the dimensional space of the vision encoder's output with the text embedding space of the LLM backbone, acting as a translator between modalities.

2. **If an image is 224x224 and the patch size is 16x16, how many visual patches are generated?**
   - *Answer:* $(224 \times 224) / (16 \times 16) = 196$ patches.

3. **Why do we use Patchification instead of feeding individual pixels to a transformer?**
   - *Answer:* Transformers have $O(N^2)$ complexity with sequence length. An image has too many pixels to treat each as a token; patchification reduces the sequence length to a manageable size while retaining local spatial information.

4. **What is the difference between early fusion and late fusion in multimodal architectures?**
   - *Answer:* Early fusion combines modalities at the input level before deep processing, while late fusion processes them independently and combines their high-level outputs at the end. VLMs like LLaVA use early-to-mid fusion by interleaving tokens at the LLM input layer.

5. **In the self-attention formula $Attn(Q, K, V)$, what does dividing by $\sqrt{d_k}$ accomplish?**
   - *Answer:* It scales the dot products to prevent the softmax function from entering regions with extremely small gradients, which slows down or prevents learning.

6. **How does the LLM know the order of image patches?**
   - *Answer:* Positional embeddings are added to the patches before they are fed into the Vision Transformer, ensuring spatial relationships are retained.

7. **What is CLIP and why is it frequently used as a Vision Encoder?**
   - *Answer:* CLIP is a model trained to predict which caption goes with which image. It provides rich visual embeddings that are already highly semantically aligned with human language concepts.

8. **Describe the sequence passed to the LLM if the prompt is "Describe <IMAGE> now." (Assume 4 patches).**
   - *Answer:* `[Describe token] [Patch 1] [Patch 2] [Patch 3] [Patch 4] [now token] [.]`

9. **What happens to the VLM's memory usage as image resolution increases?**
   - *Answer:* The number of patches increases quadratically, which in turn causes the self-attention mechanism's memory usage to increase quadratically relative to the patch count. High-res images require massive memory.

10. **Why do we not retrain the entire LLM from scratch when building a model like LLaVA?**
    - *Answer:* Pre-training an LLM requires massive resources and vast text knowledge. By freezing the LLM and Vision Encoder and only training the Projection Layer, we efficiently align modalities while preserving the LLM's vast prior knowledge.
