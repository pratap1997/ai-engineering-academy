# Module 013: Engineering Challenge — Byte-Fallback BPE Tokenizer

## 1. Challenge Task

Construct a self-contained `ByteFallbackBPETokenizer` in pure Python that:
1. Operates at the raw **UTF-8 byte level** (0..255).
2. Merges byte pairs based on corpus frequencies.
3. Guarantees **0% OOV error rate** (any UTF-8 string, including emojis and non-English scripts, can be encoded and decoded losslessy).
4. Includes round-trip assertion: `decode(encode(text)) == text` for arbitrary strings.

---

## 2. Validation Criteria

1. Handles ASCII, multi-byte UTF-8 (emojis, CJK), and rare symbols cleanly.
2. Perfect round-trip restoration for any input string.
3. Zero reliance on `[UNK]` tokens.
