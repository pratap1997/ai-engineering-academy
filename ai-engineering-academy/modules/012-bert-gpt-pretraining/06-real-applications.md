# Module 012: Real Applications

## 1. HuggingFace BERT Fine-tuning (Sentiment Classification)

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

inputs = tokenizer("The movie was fantastic!", return_tensors="pt", padding=True)
outputs = model(**inputs)
logits = outputs.logits  # (1, 2) -- positive/negative
```

## 2. GPT-2 Text Generation

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

input_ids = tokenizer.encode("The Eiffel Tower is in", return_tensors="pt")
output = model.generate(input_ids, max_new_tokens=20, do_sample=True, top_k=50)
print(tokenizer.decode(output[0]))
# -> "The Eiffel Tower is in Paris, France, and attracts millions..."
```

## 3. BERT vs RoBERTa vs GPT-2

| Model | Pre-training Tasks | Data | Key Change |
|---|---|---|---|
| **BERT** | MLM + NSP | 16GB | Original |
| **RoBERTa** | MLM only | 160GB | No NSP, larger batches, more data |
| **ALBERT** | MLM + SOP | 16GB | Parameter sharing, sentence ordering |
| **GPT-2** | CLM | 40GB (WebText) | Decoder-only, zero-shot tasks |
| **GPT-3** | CLM | 300B tokens | Scale: 175B parameters |

## 4. The Pre-train → Fine-tune Paradigm

```
Phase 1: Pre-training (days, on massive unlabeled text)
  └── Learns: syntax, semantics, facts, reasoning

Phase 2: Fine-tuning (minutes, on small labeled dataset)
  └── Adapts: task-specific head + final layers
  └── Examples: 1,000 labeled emails → spam classifier
```
