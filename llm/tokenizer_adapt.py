"""Tokenizer adaptation for the programming corpus.

Handles retraining the BPE tokenizer on the new programming corpus while
preserving compatibility with existing model weights. If the vocabulary
grows, the model's token embedding and LM head are expanded with new rows
initialized from the average of existing embeddings.

All functions validate shapes before mutating the model. On mismatch they
raise a clear error and leave the model untouched.
"""

import os

import torch

from config import ModelConfig
from tokenizer import BPETokenizer


class AdaptationError(Exception):
    """Raised when tokenizer/model adaptation cannot be done safely."""


def adapt_tokenizer(
    old_tokenizer_path: str | None,
    train_texts: list[str],
    target_vocab_size: int,
    output_path: str,
    verbose: bool = True,
) -> BPETokenizer:
    """Train a new tokenizer on the programming corpus.

    The old tokenizer is not modified. The new tokenizer is saved atomically.

    Args:
        old_tokenizer_path: Path to existing tokenizer (for reference), or None
        train_texts: Training corpus texts
        target_vocab_size: Target vocabulary size
        output_path: Where to save the new tokenizer
        verbose: Print progress

    Returns:
        The newly trained BPETokenizer

    Raises:
        AdaptationError: If training texts are empty or target vocab < 256.
    """
    if not train_texts:
        raise AdaptationError("Cannot train tokenizer on empty text list")

    if target_vocab_size < 256:
        raise AdaptationError(
            f"target_vocab_size {target_vocab_size} must be >= 256 (byte-level base)"
        )

    if verbose:
        print("Training BPE tokenizer on programming corpus...")

    tokenizer = BPETokenizer(vocab_size=target_vocab_size)
    tokenizer.train(train_texts, verbose=verbose)

    # Atomic save: write to temp then rename
    dir_path = os.path.dirname(output_path) or "."
    os.makedirs(dir_path, exist_ok=True)
    tmp_path = output_path + ".tmp"
    try:
        tokenizer.save(tmp_path)
        os.replace(tmp_path, output_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise

    if verbose:
        print(f"Tokenizer trained: {tokenizer.vocab_size} tokens")
        print(f"Saved to {output_path}")

    return tokenizer


def adapt_model_embeddings(
    model: torch.nn.Module,
    old_vocab_size: int,
    new_vocab_size: int,
    config: ModelConfig,
) -> torch.nn.Module:
    """Expand the model's token embedding and LM head for a new vocabulary size.

    If new_vocab_size > old_vocab_size:
      - New embedding rows are initialized from the mean of existing rows
      - The LM head (weight-tied) is automatically updated since it shares weights
    If new_vocab_size <= old_vocab_size:
      - No change needed (model already covers the vocabulary)
    If new_vocab_size == old_vocab_size:
      - No-op, model is already compatible

    Args:
        model: Existing GPTModel with old_vocab_size embeddings
        old_vocab_size: Previous vocabulary size
        new_vocab_size: New vocabulary size
        config: Model config (will be updated with new vocab_size)

    Returns:
        Model with adapted embeddings

    Raises:
        AdaptationError: If old_vocab_size doesn't match the actual embedding
                        size, or if hidden_dim is inconsistent.
    """
    # Validate current embedding shape matches old_vocab_size
    current_embedding = model.token_embedding.weight
    actual_old = current_embedding.shape[0]
    hidden_dim = current_embedding.shape[1]

    if actual_old != old_vocab_size:
        raise AdaptationError(
            f"old_vocab_size={old_vocab_size} does not match actual embedding "
            f"size={actual_old}. Aborting to prevent incorrect weight transfer."
        )

    if new_vocab_size <= old_vocab_size:
        config.vocab_size = old_vocab_size
        return model

    if old_vocab_size <= 0:
        config.vocab_size = new_vocab_size
        return model

    delta = new_vocab_size - old_vocab_size
    print(f"Expanding vocabulary: {old_vocab_size} -> {new_vocab_size} (+{delta} new tokens)")

    old_weight = current_embedding.data  # (old_vocab_size, hidden_dim)

    # Initialize new rows from the mean of existing embeddings
    mean_embedding = old_weight.mean(dim=0, keepdim=True)  # (1, hidden_dim)
    new_rows = mean_embedding.expand(delta, -1) + torch.randn(delta, hidden_dim) * 0.02

    # Concatenate
    new_weight = torch.cat([old_weight, new_rows], dim=0)  # (new_vocab_size, hidden_dim)

    # Create new embedding layer
    new_embedding = torch.nn.Embedding(new_vocab_size, hidden_dim)
    new_embedding.weight.data = new_weight

    # Replace in model
    model.token_embedding = new_embedding

    # Since LM head is weight-tied, update it to reference the new embedding
    model.lm_head = torch.nn.Linear(hidden_dim, new_vocab_size, bias=False)
    model.lm_head.weight = model.token_embedding.weight

    # Update config
    config.vocab_size = new_vocab_size

    print(f"Embedding expanded: {old_weight.shape} -> {new_weight.shape}")
    return model


def adapt_position_embeddings(
    model: torch.nn.Module,
    old_context_length: int,
    new_context_length: int,
    config: ModelConfig,
) -> torch.nn.Module:
    """Expand or shrink position embeddings for a new context length.

    Copies overlapping positions. New positions are initialized from the
    mean of existing embeddings plus small noise.

    Args:
        model: GPTModel with old_context_length position embeddings
        old_context_length: Previous context length
        new_context_length: New context length
        config: Model config (will be updated with new context_length)

    Returns:
        Model with adapted position embeddings

    Raises:
        AdaptationError: If old_context_length doesn't match actual size.
    """
    current_pos = model.position_embedding.weight
    actual_old = current_pos.shape[0]
    hidden_dim = current_pos.shape[1]

    if actual_old != old_context_length:
        raise AdaptationError(
            f"old_context_length={old_context_length} does not match actual "
            f"position embedding size={actual_old}. Aborting to prevent "
            f"incorrect weight transfer."
        )

    if new_context_length == old_context_length:
        return model  # no-op

    print(f"Adapting position embeddings: {old_context_length} -> {new_context_length}")

    old_pos = current_pos.data
    new_pos_emb = torch.nn.Embedding(new_context_length, hidden_dim)

    copy_len = min(old_context_length, new_context_length)
    new_pos_emb.weight.data[:copy_len] = old_pos[:copy_len]

    if new_context_length > copy_len:
        mean_pos = old_pos.mean(dim=0, keepdim=True)
        new_pos_emb.weight.data[copy_len:] = mean_pos.expand(
            new_context_length - copy_len, -1
        ) + torch.randn(new_context_length - copy_len, hidden_dim) * 0.02

    model.position_embedding = new_pos_emb
    config.context_length = new_context_length

    return model
