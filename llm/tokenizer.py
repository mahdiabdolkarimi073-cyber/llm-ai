"""Trainable Byte-Pair Encoding (BPE) tokenizer implemented from scratch.

No external NLP libraries used. Supports training on a text corpus,
encoding text to token IDs, and decoding IDs back to text.
"""

import json
import os
from collections import Counter


class BPETokenizer:
    """Minimal BPE tokenizer with byte-level pre-tokenization.

    Supports optional special tokens (e.g. <|user|>, <|assistant|>, <|end|>)
    that are stored as single token IDs and matched before BPE encoding.
    """

    def __init__(self, vocab_size: int = 8000):
        self.target_vocab_size = vocab_size
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}
        self.special_tokens: dict[str, int] = {}
        self._init_byte_vocab()

    def _init_byte_vocab(self):
        """Initialize vocabulary with single bytes (0-255)."""
        for i in range(256):
            self.vocab[i] = bytes([i])

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _text_to_byte_ids(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def _get_pairs(self, ids: list[int]) -> Counter:
        pairs = Counter()
        for i in range(len(ids) - 1):
            pairs[(ids[i], ids[i + 1])] += 1
        return pairs

    def _merge(self, ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        result = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                result.append(new_id)
                i += 2
            else:
                result.append(ids[i])
                i += 1
        return result

    def add_special_token(self, token: str, token_id: int) -> None:
        """Register a special token with a fixed ID.

        The token bytes are stored directly so decode works correctly.
        Must be called after train() and before save().
        """
        self.special_tokens[token] = token_id
        self.vocab[token_id] = token.encode("utf-8")

    def add_special_tokens(self, tokens: dict[str, int]) -> None:
        """Register multiple special tokens at once."""
        for token, token_id in tokens.items():
            self.add_special_token(token, token_id)

    def special_token_id(self, token: str) -> int | None:
        """Return the ID of a special token, or None if not registered."""
        return self.special_tokens.get(token)

    def _encode_special(self, text: str) -> list[int] | None:
        """If text is exactly a special token, return its ID. Otherwise None."""
        if text in self.special_tokens:
            return [self.special_tokens[text]]
        return None

    def train(self, texts: list[str], verbose: bool = False):
        """Train BPE merges on a corpus of texts.

        Args:
            texts: List of training strings.
            verbose: Print progress.
        """
        all_ids: list[list[int]] = [self._text_to_byte_ids(t) for t in texts]

        num_merges = self.target_vocab_size - 256 - len(self.special_tokens)
        if num_merges <= 0:
            return

        next_id_start = 256
        for merge_idx in range(num_merges):
            pair_counts = Counter()
            for ids in all_ids:
                if len(ids) < 2:
                    continue
                pairs = self._get_pairs(ids)
                pair_counts.update(pairs)

            if not pair_counts:
                break

            best_pair = pair_counts.most_common(1)[0][0]
            new_id = next_id_start + merge_idx
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            all_ids = [self._merge(ids, best_pair, new_id) for ids in all_ids]

            if verbose and (merge_idx + 1) % 500 == 0:
                print(f"  BPE merge {merge_idx + 1}/{num_merges}, vocab size: {self.vocab_size}")

        if verbose:
            print(f"  BPE training complete. Final vocab size: {self.vocab_size}")

    def encode(self, text: str) -> list[int]:
        """Encode text to a list of token IDs using learned merges.

        Special tokens are matched first (longest match), then the
        remaining text segments are BPE-encoded separately.
        """
        if not self.special_tokens:
            return self._encode_bpe(text)

        sorted_specials = sorted(self.special_tokens.keys(), key=len, reverse=True)
        result: list[int] = []
        remaining = text
        while remaining:
            matched = False
            for special in sorted_specials:
                if remaining.startswith(special):
                    result.append(self.special_tokens[special])
                    remaining = remaining[len(special):]
                    matched = True
                    break
            if not matched:
                next_special_pos = len(remaining)
                for special in sorted_specials:
                    pos = remaining.find(special)
                    if pos != -1 and pos < next_special_pos:
                        next_special_pos = pos
                segment = remaining[:next_special_pos]
                result.extend(self._encode_bpe(segment))
                remaining = remaining[next_special_pos:]
        return result

    def _encode_bpe(self, text: str) -> list[int]:
        """Standard BPE encode without special token handling."""
        ids = self._text_to_byte_ids(text)
        merge_ranks = {pair: rank for rank, pair in enumerate(self.merges.keys())}

        while len(ids) >= 2:
            pairs = self._get_pairs(ids)
            best_pair = None
            best_rank = float('inf')
            for pair in pairs:
                if pair in merge_ranks and merge_ranks[pair] < best_rank:
                    best_rank = merge_ranks[pair]
                    best_pair = pair
            if best_pair is None:
                break
            ids = self._merge(ids, best_pair, self.merges[best_pair])

        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        byte_stream = b""
        for token_id in ids:
            if token_id in self.vocab:
                byte_stream += self.vocab[token_id]
            else:
                byte_stream += b"\x00"
        return byte_stream.decode("utf-8", errors="replace")

    def save(self, path: str):
        """Save tokenizer to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "target_vocab_size": self.target_vocab_size,
            "merges": {f"{k[0]},{k[1]}": v for k, v in self.merges.items()},
            "vocab_size": self.vocab_size,
        }
        if self.special_tokens:
            data["special_tokens"] = dict(self.special_tokens)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, path: str):
        """Load tokenizer from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.target_vocab_size = data["target_vocab_size"]
        self.merges = {}
        self.vocab = {}
        self.special_tokens = {}
        self._init_byte_vocab()
        for key, val in data["merges"].items():
            parts = key.split(",")
            pair = (int(parts[0]), int(parts[1]))
            self.merges[pair] = val
            self.vocab[val] = self.vocab[pair[0]] + self.vocab[pair[1]]
        for token, token_id in data.get("special_tokens", {}).items():
            self.special_tokens[token] = token_id
            self.vocab[token_id] = token.encode("utf-8")
