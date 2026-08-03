# Module 033: Supervised Fine-Tuning & PEFT (LoRA / QLoRA)
## Overview
Full fine-tuning updates all weights of a pre-trained model, which is memory-intensive. PEFT (Parameter-Efficient Fine-Tuning) methods like LoRA freeze the base model and train a small set of injected parameters.
LoRA uses rank decomposition: $ \Delta W = B \cdot A $ to save memory.
SFT (Supervised Fine-Tuning) datasets must be formatted carefully (e.g. Instruction-Response pairs).
