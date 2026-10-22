"""Trainable Byte-Pair Encoding (BPE) tokenizer implemented from scratch.

No external NLP libraries used. Supports training on a text corpus,
encoding text to token IDs, and decoding IDs back to text.
"""

import json
import os
from collections import Counter


class BPETokenizer:
    """Minimal BPE tokenizer with byte-level pre-tokenization."""

    def __init__(self, vocab_size: int = 8000):
        self.target_vocab_size = vocab_size
        # Start with 256 byte-level tokens
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}
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

    def train(self, texts: list[str], verbose: bool = False):
        """Train BPE merges on a corpus of texts.

        Args:
            texts: List of training strings.
            verbose: Print progress.
        """
        # Convert all texts to byte-ID sequences
        all_ids: list[list[int]] = [self._text_to_byte_ids(t) for t in texts]

        num_merges = self.target_vocab_size - 256
        if num_merges <= 0:
            return

        for merge_idx in range(num_merges):
            # Count pairs across all sequences
            pair_counts = Counter()
            for ids in all_ids:
                if len(ids) < 2:
                    continue
                pairs = self._get_pairs(ids)
                pair_counts.update(pairs)

            if not pair_counts:
                break

            # Most frequent pair
            best_pair = pair_counts.most_common(1)[0][0]
            new_id = 256 + merge_idx
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # Apply merge to all sequences
            all_ids = [self._merge(ids, best_pair, new_id) for ids in all_ids]

            if verbose and (merge_idx + 1) % 500 == 0:
                print(f"  BPE merge {merge_idx + 1}/{num_merges}, vocab size: {self.vocab_size}")

        if verbose:
            print(f"  BPE training complete. Final vocab size: {self.vocab_size}")

    def encode(self, text: str) -> list[int]:
        """Encode text to a list of token IDs using learned merges."""
        ids = self._text_to_byte_ids(text)
        # Apply merges in order of learned rank (lower rank = higher priority)
        # We need to apply merges greedily by rank order
        merge_ranks = {pair: rank for rank, pair in enumerate(self.merges.keys())}

        while len(ids) >= 2:
            pairs = self._get_pairs(ids)
            # Find the pair with the lowest merge rank
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
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, path: str):
        """Load tokenizer from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.target_vocab_size = data["target_vocab_size"]
        self.merges = {}
        self.vocab = {}
        self._init_byte_vocab()
        for key, val in data["merges"].items():
            parts = key.split(",")
            pair = (int(parts[0]), int(parts[1]))
            self.merges[pair] = val
            self.vocab[val] = self.vocab[pair[0]] + self.vocab[pair[1]]
