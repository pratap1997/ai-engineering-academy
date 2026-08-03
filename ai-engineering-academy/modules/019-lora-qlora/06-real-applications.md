# Module 019: Real Applications — Production PEFT & QLoRA

## 1. HuggingFace `peft` Library

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# -> "trainable params: 41,943,040 || all params: 8,072,204,288 || trainable%: 0.5195%"
```

---

## 2. Merging and Saving LoRA Adapters for Deployment

```python
# Save adapter weights only (small ~50MB file!)
model.save_pretrained("./my_custom_adapter")

# Merge adapter weights into base model and save full 16-bit checkpoint
model_merged = model.merge_and_unload()
model_merged.save_pretrained("./my_merged_llama3")
```

---

## 3. Multi-Adapter Serving (Adapter Fusion / Dynamic Switching)

Because base model weights are static and frozen:
- A single base model server (e.g. LLaMA 3 70B) can serve **hundreds of specialized customer adapters simultaneously**!
- When Request A arrives (Code Assistant) → apply Adapter 1 weights.
- When Request B arrives (Legal Assistant) → apply Adapter 2 weights.
- Swapping adapters requires loading only a $50\text{ MB}$ matrix instead of reloading a $140\text{ GB}$ model!
