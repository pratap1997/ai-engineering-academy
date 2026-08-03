# Module 013: Real Applications — Production Tokenizers

## 1. OpenAI `tiktoken` (Byte-level BPE)

GPT-2, GPT-3.4, and GPT-4 use Byte-level BPE via the high-performance `tiktoken` library (written in Rust):

```python
import tiktoken

# Load tokenizer for GPT-4
enc = tiktoken.encoding_for_model("gpt-4")

text = "Antigravity AI Engineering Academy 🚀"
tokens = enc.encode(text)
decoded = enc.decode(tokens)

print("Token IDs:", tokens)
print("Token Count:", len(tokens))
print("Decoded:", decoded)
```

**Why Byte-level?**
By using 256 byte values as the base vocabulary instead of Unicode characters, any arbitrary string of bytes (UTF-8) can be tokenized with zero OOV errors without needing `[UNK]` tokens.

---

## 2. HuggingFace `tokenizers` Library

```python
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

# Train a fast BPE Tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(vocab_size=30000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
files = ["corpus.txt"]
tokenizer.train(files, trainer)

output = tokenizer.encode("Hello world!")
print(output.tokens)
```

---

## 3. Comparison of Production Tokenizers

| Model | Tokenizer Type | Vocab Size | Special Tokens |
|---|---|---|---|
| **BERT** | WordPiece | 30,522 | `[PAD]=0`, `[UNK]=100`, `[CLS]=101`, `[SEP]=102`, `[MASK]=103` |
| **GPT-2** | Byte-level BPE | 50,257 | `<|endoftext|>=50256` |
| **GPT-4 (cl100k_base)** | Byte-level BPE | 100,000 | `<|endoftext|>`, `<|endofprompt|>` |
| **LLaMA 3 (tiktoken)** | Byte-level BPE | 128,000 | `<|begin_of_text|>`, `<|end_of_text|>` |
| **T5 / ALBERT** | SentencePiece (Unigram) | 32,000 | `<pad>`, `</s>`, `<unk>` |
