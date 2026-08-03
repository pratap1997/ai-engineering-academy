"""
AI ENGINEERING ACADEMY -- MODULE 013 ENGINEERING CHALLENGE SOLUTION
Byte-Fallback BPE Tokenizer with 100% Roundtrip Lossless Guarantee
"""

from collections import defaultdict
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class ByteFallbackBPETokenizer:
    """
    Byte-level BPE tokenizer operating directly on UTF-8 bytes.
    Base vocabulary size = 256 (all raw byte values 0..255).
    Guarantees 0% OOV error rate and perfect roundtrip decoding.
    """

    def __init__(self):
        # Base vocab is 256 bytes
        self.vocab = {bytes([i]): i for i in range(256)}
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        self.merges = []  # list of (bytes1, bytes2) -> merged_bytes

    def train(self, texts, vocab_size=300):
        # Convert texts to tuples of 1-byte objects
        word_freqs = defaultdict(int)
        for text in texts:
            raw_bytes = text.encode("utf-8")
            # Word represented as tuple of single-byte objects
            byte_tuple = tuple(bytes([b]) for b in raw_bytes)
            word_freqs[byte_tuple] += 1

        num_merges = vocab_size - 256
        for _ in range(max(0, num_merges)):
            pairs = defaultdict(int)
            for w_tuple, freq in word_freqs.items():
                for i in range(len(w_tuple) - 1):
                    pairs[(w_tuple[i], w_tuple[i + 1])] += freq
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            merged_bytes = best_pair[0] + best_pair[1]

            new_id = len(self.vocab)
            self.vocab[merged_bytes] = new_id
            self.id_to_bytes[new_id] = merged_bytes
            self.merges.append(best_pair)

            # Update word_freqs
            new_word_freqs = {}
            for w_tuple, freq in word_freqs.items():
                new_w = []
                i = 0
                while i < len(w_tuple):
                    if i < len(w_tuple) - 1 and (w_tuple[i], w_tuple[i + 1]) == best_pair:
                        new_w.append(merged_bytes)
                        i += 2
                    else:
                        new_w.append(w_tuple[i])
                        i += 1
                new_word_freqs[tuple(new_w)] = freq
            word_freqs = new_word_freqs

    def encode(self, text):
        raw_bytes = text.encode("utf-8")
        byte_tuple = tuple(bytes([b]) for b in raw_bytes)

        for b1, b2 in self.merges:
            if len(byte_tuple) <= 1:
                break
            merged = b1 + b2
            new_tuple = []
            i = 0
            while i < len(byte_tuple):
                if i < len(byte_tuple) - 1 and (byte_tuple[i], byte_tuple[i + 1]) == (b1, b2):
                    new_tuple.append(merged)
                    i += 2
                else:
                    new_tuple.append(byte_tuple[i])
                    i += 1
            byte_tuple = tuple(new_tuple)

        return [self.vocab[b] for b in byte_tuple]

    def decode(self, ids):
        byte_chunks = [self.id_to_bytes[i] for i in ids]
        all_bytes = b"".join(byte_chunks)
        return all_bytes.decode("utf-8", errors="replace")


def verify_byte_fallback_tokenizer():
    print("=" * 65)
    print("MODULE 013 CHALLENGE SOLUTION: BYTE-FALLBACK BPE TOKENIZER")
    print("=" * 65)

    corpus = [
        "AI Engineering Academy 🚀 Hello World!",
        "Neural networks learn representations from data.",
        "Deep learning and natural language processing."
    ]

    tokenizer = ByteFallbackBPETokenizer()
    tokenizer.train(corpus, vocab_size=300)

    test_cases = [
        "AI Engineering Academy 🚀",
        "Unseen text with non-English: こんにちは, 世界! 🌍",
        "Special characters: @#$%^&*()_+~`",
    ]

    for text in test_cases:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        print(f"\nOriginal: '{text}'")
        print(f"Encoded IDs ({len(ids)} tokens): {ids[:8]}...")
        print(f"Decoded:  '{decoded}'")
        assert decoded == text, f"Mismatch: expected '{text}', got '{decoded}'"
        print("Lossless roundtrip verified [OK]")

    print("\n" + "=" * 65)
    print("ALL CHALLENGE VERIFICATIONS PASSED [OK]")
    print("=" * 65)


if __name__ == "__main__":
    verify_byte_fallback_tokenizer()
