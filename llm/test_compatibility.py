"""Compatibility test: verify tokenizer vocab matches checkpoint config.

This test is WRITTEN but NOT RUN, per project requirements.
It checks:
  1. Each checkpoint's saved config vocab_size matches its paired tokenizer
  2. Tokenizer can encode/decode a sample string round-trip
  3. Model architecture built from checkpoint config has correct output dims

To run manually:
    python llm/test_compatibility.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checkpoint import load_config_from_checkpoint, CheckpointError
from tokenizer import BPETokenizer
from config import ModelConfig
from model import GPTModel


CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

PAIRS = [
    ("specialized_best", "specialized_best.pt", "tokenizer_v2.json"),
    ("specialized", "specialized.pt", "tokenizer_v2.json"),
    ("phase1", "model.pt", "tokenizer.json"),
]


def test_vocab_compatibility():
    """Each checkpoint's vocab_size must match its tokenizer's vocab_size."""
    for label, ckpt_name, tok_name in PAIRS:
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
        tok_path = os.path.join(CHECKPOINT_DIR, tok_name)

        if not os.path.exists(ckpt_path) or not os.path.exists(tok_path):
            print(f"SKIP: {label} — files missing")
            continue

        try:
            saved_config = load_config_from_checkpoint(ckpt_path)
        except CheckpointError as e:
            print(f"FAIL: {label} — could not load config: {e}")
            continue

        tokenizer = BPETokenizer(vocab_size=saved_config.vocab_size)
        tokenizer.load(tok_path)

        assert tokenizer.vocab_size == saved_config.vocab_size, (
            f"{label}: tokenizer vocab ({tokenizer.vocab_size}) != "
            f"checkpoint vocab ({saved_config.vocab_size})"
        )
        print(f"PASS: {label} — vocab match ({tokenizer.vocab_size})")


def test_tokenizer_roundtrip():
    """Tokenizer encode/decode should round-trip on ASCII text."""
    tok_path = os.path.join(CHECKPOINT_DIR, "tokenizer_v2.json")
    if not os.path.exists(tok_path):
        print("SKIP: tokenizer_v2.json missing")
        return

    tokenizer = BPETokenizer(vocab_size=1500)
    tokenizer.load(tok_path)

    sample = "def hello():\n    print('hello')\n"
    ids = tokenizer.encode(sample)
    decoded = tokenizer.decode(ids)
    assert decoded == sample, f"Round-trip failed: {repr(decoded)} != {repr(sample)}"
    print(f"PASS: tokenizer round-trip ({len(ids)} tokens)")


def test_model_output_dims():
    """Model built from checkpoint config should produce correct output shape."""
    import torch

    ckpt_path = os.path.join(CHECKPOINT_DIR, "specialized_best.pt")
    if not os.path.exists(ckpt_path):
        print("SKIP: specialized_best.pt missing")
        return

    try:
        config = load_config_from_checkpoint(ckpt_path)
    except CheckpointError as e:
        print(f"FAIL: could not load config: {e}")
        return

    model = GPTModel(config)
    x = torch.zeros(1, 10, dtype=torch.long)
    logits, _ = model(x)
    assert logits.shape == (1, 10, config.vocab_size), (
        f"Output shape {logits.shape} != (1, 10, {config.vocab_size})"
    )
    print(f"PASS: model output dims correct ({tuple(logits.shape)})")


def run_all_tests():
    tests = [
        test_vocab_compatibility,
        test_tokenizer_roundtrip,
        test_model_output_dims,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            failed += 1
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    run_all_tests()
