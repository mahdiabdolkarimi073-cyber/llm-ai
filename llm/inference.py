"""Autoregressive text generation with temperature and top-k sampling."""

import torch

from config import ModelConfig
from model import GPTModel


def generate_text(
    model: GPTModel,
    tokenizer,
    prompt: str,
    config: ModelConfig,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
) -> str:
    """Generate text from a prompt using the trained model.

    Args:
        model: Trained GPTModel
        tokenizer: Trained BPETokenizer
        prompt: Input text to continue
        config: Model config (uses defaults for temperature/top_k if not given)
        max_new_tokens: Override max tokens to generate
        temperature: Override sampling temperature
        top_k: Override top-k sampling

    Returns:
        Generated text (prompt + continuation)
    """
    device = "cpu"
    model = model.to(device)
    model.eval()

    max_new = max_new_tokens or config.max_new_tokens
    temp = temperature if temperature is not None else config.temperature
    k = top_k if top_k is not None else config.top_k

    # Encode prompt
    ids = tokenizer.encode(prompt)
    if len(ids) == 0:
        ids = [0]

    idx = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        generated = model.generate(idx, max_new_tokens=max_new, temperature=temp, top_k=k)

    # Decode full output
    all_ids = generated[0].tolist()
    text = tokenizer.decode(all_ids)
    return text


def generate_multiple(
    model: GPTModel,
    tokenizer,
    prompt: str,
    config: ModelConfig,
    n_samples: int = 3,
    **kwargs,
) -> list[str]:
    """Generate multiple samples from the same prompt."""
    results = []
    for _ in range(n_samples):
        text = generate_text(model, tokenizer, prompt, config, **kwargs)
        results.append(text)
    return results
