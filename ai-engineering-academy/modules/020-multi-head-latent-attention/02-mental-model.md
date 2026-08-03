# Module 020: Mental Model — The Zip Archive & Pre-Combined Lens

## 1. The Zip Archive Analogy

Imagine a high-resolution 100-page photo album (Full Key/Value Matrices):

- **MHA**: Keeping 100 uncompressed 4K photo prints in your backpack for every user. Backpack explodes (VRAM overflow).
- **GQA**: Throwing away 90 out of 100 photos and keeping only 10 shared low-res copies. Low memory, but lost critical detail.
- **MLA**: Compressing all 100 photos into a single `.zip` file ($\mathbf{c}_{KV}$). You keep only the single tiny `.zip` file in your pocket.

---

## 2. Matrix Absorption (The Pre-Combined Lens)

Usually, to inspect a photo inside a `.zip` file, you must unzip it back onto the table (decompressing Keys/Values in VRAM).

DeepSeek's **Matrix Absorption** is like looking at the `.zip` file through a special camera lens ($\mathbf{W}_{UQ} \mathbf{W}_{UK}^T$) that lets you view the similarity directly **without ever unzipping the file**!

$$\mathbf{Q} \mathbf{K}^T = \mathbf{x} (\mathbf{W}_Q \mathbf{W}_K^T) \mathbf{x}^T = \mathbf{c}_Q (\mathbf{W}_{UQ} \mathbf{W}_{UK}^T) \mathbf{c}_{KV}^T$$

By pre-multiplying $\mathbf{W}_{absorbed} = \mathbf{W}_{UQ} \mathbf{W}_{UK}^T$ during model loading, the inference GPU performs a single matrix multiplication between query latent vectors $\mathbf{c}_Q$ and compressed KV latent vectors $\mathbf{c}_{KV}$.
