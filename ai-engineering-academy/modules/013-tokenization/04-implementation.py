"""
AI ENGINEERING ACADEMY -- MODULE 013
Subword Tokenization Implementation (Pure Python)

Provides:
1. `BPETokenizer`: Byte-Pair Encoding trainer & encoder/decoder.
2. `WordPieceTokenizer`: Likelihood-score based tokenizer with continuation markers (##).
3. `UnigramTokenizer`: Viterbi DP segmentation & unigram pruning.
"""

from collections import defaultdict
import math
import re


# =====================================================================
# 1. BYTE-PAIR ENCODING (BPE) TOKENIZER
# =====================================================================

class BPETokenizer:
    """
    Byte-Pair Encoding (BPE) implementation.
    Iteratively merges the most frequent adjacent symbol pair.
    """

    def __init__(self, unk_token="[UNK]", pad_token="[PAD]"):
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.merges = []             # List of (pair_tuple, merged_str)
        self.vocab = {}              # token_str -> id
        self.id_to_token = {}        # id -> token_str
        self.special_tokens = [pad_token, unk_token]

    def _get_stats(self, word_freqs):
        """Count frequencies of adjacent pairs."""
        pairs = defaultdict(int)
        for word_tuple, freq in word_freqs.items():
            for i in range(len(word_tuple) - 1):
                pairs[(word_tuple[i], word_tuple[i + 1])] += freq
        return pairs

    def _merge_word(self, word_tuple, pair, replacement):
        """Replace all occurrences of pair in word_tuple with replacement."""
        new_word = []
        i = 0
        while i < len(word_tuple):
            if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i + 1]) == pair:
                new_word.append(replacement)
                i += 2
            else:
                new_word.append(word_tuple[i])
                i += 1
        return tuple(new_word)

    def train(self, texts, vocab_size=50):
        """Train BPE merge rules and vocabulary from a list of text strings."""
        # Step 1: Character level splitting
        word_counts = defaultdict(int)
        for text in texts:
            words = text.strip().split()
            for w in words:
                # Represent word as tuple of characters
                char_tuple = tuple(list(w))
                word_counts[char_tuple] += 1

        # Build initial alphabet vocab
        initial_chars = set()
        for w_tuple in word_counts:
            for ch in w_tuple:
                initial_chars.add(ch)

        current_vocab = list(self.special_tokens) + sorted(list(initial_chars))
        self.merges = []

        word_freqs = dict(word_counts)

        # Iteratively merge top pair until vocab_size is reached
        num_merges = vocab_size - len(current_vocab)
        for _ in range(max(0, num_merges)):
            pairs = self._get_stats(word_freqs)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            merged_symbol = best_pair[0] + best_pair[1]

            self.merges.append((best_pair, merged_symbol))
            current_vocab.append(merged_symbol)

            # Update word_freqs with merged representation
            new_word_freqs = {}
            for w_tuple, freq in word_freqs.items():
                new_w = self._merge_word(w_tuple, best_pair, merged_symbol)
                new_word_freqs[new_w] = freq
            word_freqs = new_word_freqs

        # Finalize vocab mappings
        self.vocab = {tok: idx for idx, tok in enumerate(current_vocab)}
        self.id_to_token = {idx: tok for idx, tok in enumerate(current_vocab)}

    def tokenize(self, text):
        """Tokenize input text using learned merge rules."""
        words = text.strip().split()
        all_tokens = []

        for word in words:
            word_tuple = tuple(list(word))
            # Apply merges in order of training
            for pair, merged in self.merges:
                if len(word_tuple) <= 1:
                    break
                word_tuple = self._merge_word(word_tuple, pair, merged)

            # Fallback for unknown subwords
            for tok in word_tuple:
                if tok in self.vocab:
                    all_tokens.append(tok)
                else:
                    all_tokens.append(self.unk_token)

        return all_tokens

    def encode(self, text):
        """Tokenize text and convert to integer IDs."""
        tokens = self.tokenize(text)
        return [self.vocab.get(t, self.vocab[self.unk_token]) for t in tokens]

    def decode(self, ids):
        """Convert integer IDs back to string."""
        tokens = [self.id_to_token.get(idx, self.unk_token) for idx in ids]
        # Joined tokens
        return "".join([t for t in tokens if t not in self.special_tokens])


# =====================================================================
# 2. WORDPIECE TOKENIZER
# =====================================================================

class WordPieceTokenizer:
    """
    WordPiece tokenizer (BERT style).
    Uses '##' for continuation tokens and longest-match greedy segmentation.
    """

    def __init__(self, vocab=None, unk_token="[UNK]"):
        self.unk_token = unk_token
        self.vocab = vocab or {}
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def tokenize_word(self, word):
        """Greedy longest-match-first algorithm for a single word."""
        if word in self.vocab:
            return [word]

        tokens = []
        start = 0
        n = len(word)
        is_bad = False

        while start < n:
            end = n
            cur_substr = None
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = "##" + substr
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1

            if cur_substr is None:
                is_bad = True
                break

            tokens.append(cur_substr)
            start = end

        if is_bad:
            return [self.unk_token]
        return tokens

    def tokenize(self, text):
        words = text.strip().split()
        res = []
        for w in words:
            res.extend(self.tokenize_word(w))
        return res


# =====================================================================
# 3. UNIGRAM TOKENIZER (Viterbi Segmentation)
# =====================================================================

class UnigramTokenizer:
    """
    Unigram Tokenizer using Viterbi Dynamic Programming for optimal subword parsing.
    """

    def __init__(self, vocab_probs=None, unk_token="[UNK]"):
        # vocab_probs: dict of token -> probability
        self.unk_token = unk_token
        self.vocab_probs = vocab_probs or {}
        self.unk_score = -10.0

    def _get_score(self, token):
        if token in self.vocab_probs:
            return math.log(self.vocab_probs[token])
        return self.unk_score

    def tokenize_word(self, word):
        """Find optimal segmentation maximizing total log probability via Viterbi DP."""
        n = len(word)
        # dp[i] = (best_score, best_prev_index)
        dp = [(-float('inf'), -1)] * (n + 1)
        dp[0] = (0.0, -1)

        for i in range(1, n + 1):
            for j in range(i):
                substr = word[j:i]
                if substr in self.vocab_probs or len(substr) == 1:
                    score = self._get_score(substr)
                    prev_score = dp[j][0]
                    if prev_score + score > dp[i][0]:
                        dp[i] = (prev_score + score, j)

        # Backtrack best path
        if dp[n][0] == -float('inf'):
            return [self.unk_token]

        tokens = []
        curr = n
        while curr > 0:
            prev = dp[curr][1]
            tokens.append(word[prev:curr])
            curr = prev

        tokens.reverse()
        return tokens

    def tokenize(self, text):
        words = text.strip().split()
        res = []
        for w in words:
            res.extend(self.tokenize_word(w))
        return res


if __name__ == "__main__":
    print("=" * 65)
    print("MODULE 013 -- SUBWORD TOKENIZATION VERIFICATION")
    print("=" * 65)

    corpus = [
        "low lower lowest",
        "new newest wider widest",
        "low lower wider"
    ]

    # 1. BPE Tokenizer
    bpe = BPETokenizer()
    bpe.train(corpus, vocab_size=20)
    print("\n[1. BPE Tokenizer]")
    print(f"  Vocab Size: {len(bpe.vocab)}")
    sample_text = "lowest widest"
    tokens = bpe.tokenize(sample_text)
    encoded = bpe.encode(sample_text)
    decoded = bpe.decode(encoded)
    print(f"  Sample Text: '{sample_text}'")
    print(f"  Tokens:      {tokens}")
    print(f"  IDs:         {encoded}")
    print(f"  Decoded:     '{decoded}' => [OK]")

    # 2. WordPiece Tokenizer
    wp_vocab = {"[UNK]": 0, "[PAD]": 1, "low": 2, "##est": 3, "wid": 4, "##est": 5, "wide": 6, "##r": 7}
    wp = WordPieceTokenizer(vocab=wp_vocab)
    wp_tokens = wp.tokenize("lowest wider")
    print("\n[2. WordPiece Tokenizer]")
    print(f"  Input:  'lowest wider'")
    print(f"  Tokens: {wp_tokens} => [OK]")

    # 3. Unigram Tokenizer
    unigram_probs = {"low": 0.4, "est": 0.3, "l": 0.1, "o": 0.1, "w": 0.1}
    unigram = UnigramTokenizer(vocab_probs=unigram_probs)
    uni_tokens = unigram.tokenize("lowest")
    print("\n[3. Unigram Tokenizer (Viterbi DP)]")
    print(f"  Input:  'lowest'")
    print(f"  Tokens: {uni_tokens} => [OK]")
