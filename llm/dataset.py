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


# ---------------------------------------------------------------------------
# Conversation dataset with loss masking
# ---------------------------------------------------------------------------

LOSS_IGNORE = -100


class ConversationDataset(Dataset):
    """Dataset for conversation training with loss masking.

    Only assistant response tokens contribute to the loss; user tokens
    and special-token markers are masked with LOSS_IGNORE (-100) so the
    model learns to generate responses, not memorize prompts.

    Each conversation is serialized as:
       <|user|>msg<|end|><|assistant|>response<|end|>

    Token positions belonging to the assistant response (including the
    <|end|> that terminates it) are kept as targets; all others are set
    to LOSS_IGNORE.
    """

    def __init__(
        self,
        conversations: list[str],
        tokenizer,
        config: ModelConfig,
        assistant_token: str = "<|assistant|>",
        end_token: str = "<|end|>",
        user_token: str = "<|user|>",
    ):
        self.context_length = config.context_length
        self.samples: list[tuple[list[int], list[int]]] = []

        for text in conversations:
            ids = tokenizer.encode(text)
            mask = self._build_loss_mask(ids, tokenizer, assistant_token, end_token, user_token)
            self._chunk(ids, mask, config.context_length)

    def _build_loss_mask(
        self,
        ids: list[int],
        tokenizer,
        assistant_token: str,
        end_token: str,
        user_token: str,
    ) -> list[bool]:
        """Return a per-position boolean list: True where the token is part of an assistant response."""
        ass_id = tokenizer.special_token_id(assistant_token)
        end_id = tokenizer.special_token_id(end_token)
        usr_id = tokenizer.special_token_id(user_token)

        mask = [False] * len(ids)
        in_assistant = False
        for i, tid in enumerate(ids):
            if tid == ass_id:
                in_assistant = True
                continue
            if tid == usr_id:
                in_assistant = False
                continue
            if tid == end_id:
                if in_assistant:
                    mask[i] = True
                in_assistant = False
                continue
            if in_assistant:
                mask[i] = True
        return mask

    def _chunk(self, ids: list[int], mask: list[bool], ctx_len: int) -> None:
        """Slice the token stream into context-length windows with (x, y, loss_mask)."""
        T = ctx_len
        if len(ids) <= T:
            padded_ids = ids + [0] * (T + 1 - len(ids))
            padded_mask = mask + [False] * (T + 1 - len(mask))
            x = padded_ids[:T]
            y = padded_ids[1:T + 1]
            y_mask = padded_mask[1:T + 1]
            y_target = [y[j] if y_mask[j] else LOSS_IGNORE for j in range(T)]
            self.samples.append((x, y_target))
            return

        stride = T
        for start in range(0, len(ids) - T - 1, stride):
            x = ids[start:start + T]
            y = ids[start + 1:start + 1 + T]
            y_mask = mask[start + 1:start + 1 + T]
            y_target = [y[j] if y_mask[j] else LOSS_IGNORE for j in range(T)]
            self.samples.append((x, y_target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
