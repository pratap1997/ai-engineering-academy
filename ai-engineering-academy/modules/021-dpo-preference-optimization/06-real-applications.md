# Module 021: Real Applications — Production DPO Pipelines & Model Alignment

## 1. HuggingFace `trl` DPOTrainer

In production alignment pipelines (e.g. Zephyr-7B, LLaMA 3 Instruct):

```python
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
ref_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

dpo_config = DPOConfig(
    beta=0.1,
    learning_rate=5e-7,
    batch_size=4,
    gradient_accumulation_steps=8,
    max_length=2048,
    max_prompt_length=1024,
    remove_unused_columns=False
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=preference_dataset, # Contains 'prompt', 'chosen', 'rejected'
    tokenizer=tokenizer
)

dpo_trainer.train()
```

---

## 2. Preference Loss Variants Comparison

| Method | Pair Requirements | Loss Formulation | Primary Use Case |
|---|---|---|---|
| **DPO** | Paired $(y_w, y_l)$ | $-\log \sigma(\hat{r}_w - \hat{r}_l)$ | Standard LLM Preference Alignment |
| **KTO** | Unpaired (Binary Pass/Fail) | Kahneman-Tversky Prospect Theory | Unpaired User Feedback / Binary Upvotes |
| **IPO** | Paired $(y_w, y_l)$ | $(\hat{r}_w - \hat{r}_l - \frac{1}{2\beta})^2$ | Prevents overconfidence / regularization |
| **ORPO** | Monolithic SFT + DPO | Cross-Entropy + Odds Ratio Penalty | Single-step alignment without reference model |
