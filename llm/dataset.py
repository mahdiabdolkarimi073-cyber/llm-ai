"""Dataset utilities for loading and batching text data."""

import torch
from torch.utils.data import Dataset

from config import ModelConfig


class TextDataset(Dataset):
    """Character/token-level dataset that produces (x, y) pairs for next-token prediction.

    Each sample is a contiguous slice of the token stream of length context_length + 1.
    x = tokens[i : i + context_length]
    y = tokens[i+1 : i+1 + context_length]
    """

    def __init__(self, token_ids: list[int], config: ModelConfig):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = config.context_length

    def __len__(self) -> int:
        if len(self.data) <= self.context_length:
            return 1
        return len(self.data) - self.context_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        T = self.context_length
        if len(self.data) <= T:
            # Pad if data is shorter than context length
            padded = torch.zeros(T + 1, dtype=torch.long)
            padded[:len(self.data)] = self.data
            x = padded[:T]
            y = padded[1:T + 1]
        else:
            x = self.data[idx:idx + T]
            y = self.data[idx + 1:idx + 1 + T]
        return x, y


def encode_corpus(texts: list[str], tokenizer, max_tokens: int | None = None) -> list[int]:
    """Encode a list of texts into a single flat list of token IDs."""
    all_ids: list[int] = []
    for text in texts:
        ids = tokenizer.encode(text)
        all_ids.extend(ids)
        if max_tokens is not None and len(all_ids) >= max_tokens:
            all_ids = all_ids[:max_tokens]
            break
    return all_ids


def split_data(token_ids: list[int], val_frac: float = 0.1) -> tuple[list[int], list[int]]:
    """Split token IDs into train and validation sets."""
    split_idx = int(len(token_ids) * (1 - val_frac))
    return token_ids[:split_idx], token_ids[split_idx:]
