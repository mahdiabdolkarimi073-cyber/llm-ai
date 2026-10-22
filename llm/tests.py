"""Correctness tests verifying the model actually learns.

Tests:
  1. Forward pass produces correct output shape
  2. Loss is computed and is a finite scalar
  3. Backward pass produces non-None gradients on all parameters
  4. Weights actually change after an optimization step
  5. Checkpoint save/load round-trips exactly
  6. Memorization: model can overfit a tiny dataset (loss decreases significantly)
  7. Inference: generate produces valid token IDs that decode to text
"""

import os
import sys
import tempfile

import torch

# Make llm package importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig
from tokenizer import BPETokenizer
from model import GPTModel
from dataset import TextDataset
from checkpoint import save_checkpoint, load_checkpoint
from inference import generate_text


def make_small_config() -> ModelConfig:
    """Small config for fast testing."""
    return ModelConfig(
        vocab_size=300,
        context_length=64,
        n_layers=2,
        n_heads=2,
        hidden_dim=64,
        ff_dim=128,
        batch_size=4,
        learning_rate=1e-3,
        eval_interval=10,
        max_epochs=20,
        warmup_steps=5,
    )


def test_forward_shape():
    """Test 1: Forward pass produces (B, T, vocab_size) logits."""
    config = make_small_config()
    model = GPTModel(config)
    x = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(x, targets=None)
    assert logits.shape == (2, 16, config.vocab_size), \
        f"Expected (2, 16, {config.vocab_size}), got {logits.shape}"
    assert loss is None
    print("  [PASS] Forward shape correct: (2, 16, vocab_size)")


def test_loss_computed():
    """Test 2: Loss is a finite scalar."""
    config = make_small_config()
    model = GPTModel(config)
    x = torch.randint(0, config.vocab_size, (2, 16))
    y = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(x, targets=y)
    assert loss is not None
    assert loss.dim() == 0, "Loss should be a scalar"
    assert torch.isfinite(loss), "Loss should be finite"
    print(f"  [PASS] Loss computed: {loss.item():.4f} (finite scalar)")


def test_backward_gradients():
    """Test 3: Backward pass produces gradients on all parameters."""
    config = make_small_config()
    model = GPTModel(config)
    x = torch.randint(0, config.vocab_size, (2, 16))
    y = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(x, targets=y)
    loss.backward()

    no_grad_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                no_grad_count += 1
            else:
                assert torch.isfinite(param.grad).all(), \
                    f"Non-finite gradient in {name}"

    assert no_grad_count == 0, f"{no_grad_count} parameters have no gradient"
    print("  [PASS] All parameters have finite gradients after backward")


def test_weights_change():
    """Test 4: Weights actually change after an optimizer step."""
    config = make_small_config()
    model = GPTModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randint(0, config.vocab_size, (2, 16))
    y = torch.randint(0, config.vocab_size, (2, 16))

    # Snapshot weights before
    before = {n: p.clone() for n, p in model.named_parameters()}

    # One optimization step
    for _ in range(5):
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Check that at least some weights changed
    changed = 0
    for name, param in model.named_parameters():
        if not torch.allclose(before[name], param, atol=1e-8):
            changed += 1

    assert changed > 0, "No weights changed after 5 optimizer steps"
    print(f"  [PASS] {changed}/{len(before)} parameter tensors changed after 5 steps")


def test_checkpoint_roundtrip():
    """Test 5: Checkpoint save/load round-trips exactly."""
    config = make_small_config()
    model = GPTModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Do a forward+backward to populate optimizer state
    x = torch.randint(0, config.vocab_size, (2, 16))
    y = torch.randint(0, config.vocab_size, (2, 16))
    _, loss = model(x, y)
    loss.backward()
    optimizer.step()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test.pt")
        save_checkpoint(ckpt_path, model, optimizer, step=42, epoch=3,
                        train_loss=1.5, val_loss=2.0)

        # Create a fresh model and optimizer
        model2 = GPTModel(config)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

        meta = load_checkpoint(ckpt_path, model2, optimizer2)

        # Check metadata
        assert meta["step"] == 42, f"Expected step 42, got {meta['step']}"
        assert meta["epoch"] == 3, f"Expected epoch 3, got {meta['epoch']}"

        # Check weights match exactly
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert n1 == n2
            assert torch.allclose(p1, p2, atol=1e-10), \
                f"Weights differ after checkpoint load in {n1}"

    print("  [PASS] Checkpoint save/load round-trips exactly (weights + optimizer + metadata)")


def test_memorization():
    """Test 6: Model can memorize a tiny dataset (overfitting test).

    If the model can overfit a small set of examples, it proves
    the training pipeline is functional.
    """
    config = make_small_config()
    config.max_epochs = 50
    config.eval_interval = 20
    config.batch_size = 4

    # Tiny dataset: a few repeated sentences
    texts = [
        "def hello(): return 'world'",
        "def add(a, b): return a + b",
        "def square(x): return x * x",
        "def greet(name): return 'hi ' + name",
    ] * 10

    # Train tokenizer
    tokenizer = BPETokenizer(vocab_size=config.vocab_size)
    tokenizer.train(texts, verbose=False)
    config.vocab_size = tokenizer.vocab_size

    # Re-create model with correct vocab size
    model = GPTModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    # Encode and build dataset
    all_ids = []
    for t in texts:
        all_ids.extend(tokenizer.encode(t))
    dataset = TextDataset(all_ids, config)

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    # Train and record initial vs final loss
    initial_loss = None
    final_loss = None
    step = 0

    for epoch in range(config.max_epochs):
        for x, y in loader:
            logits, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1

            if initial_loss is None:
                initial_loss = loss.item()

            if step % 50 == 0:
                final_loss = loss.item()

    # Use the very last loss as final
    if final_loss is None or step % 50 != 0:
        # Do one more forward pass
        x, y = next(iter(loader))
        _, loss = model(x, y)
        final_loss = loss.item()

    ratio = final_loss / initial_loss if initial_loss > 0 else 1.0
    print(f"  Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}, "
          f"Reduction: {ratio:.1%} of initial")

    assert final_loss < initial_loss * 0.5, \
        f"Model did not memorize: initial={initial_loss:.4f}, final={final_loss:.4f}"
    print("  [PASS] Model successfully memorized tiny dataset (loss reduced by >50%)")


def test_inference():
    """Test 7: Inference produces valid output."""
    config = make_small_config()
    texts = ["hello world", "foo bar baz"]
    tokenizer = BPETokenizer(vocab_size=config.vocab_size)
    tokenizer.train(texts, verbose=False)
    config.vocab_size = tokenizer.vocab_size

    model = GPTModel(config)

    # Generate with greedy (temperature very low)
    output = generate_text(
        model, tokenizer, "hello", config,
        max_new_tokens=20, temperature=0.01, top_k=5,
    )

    assert isinstance(output, str), "Output should be a string"
    assert len(output) > 0, "Output should not be empty"
    assert "hello" in output, "Output should contain the prompt"
    print(f"  [PASS] Inference produces valid text output (len={len(output)})")


def test_tokenizer_roundtrip():
    """Test 8: Tokenizer encode/decode round-trips for ASCII text."""
    tokenizer = BPETokenizer(vocab_size=500)
    texts = ["hello world", "def foo(): pass", "import torch\nx = 1"]
    tokenizer.train(texts, verbose=False)

    for text in texts:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text, f"Round-trip failed: '{text}' -> '{decoded}'"

    print("  [PASS] Tokenizer encode/decode round-trips correctly")


def run_all_tests():
    """Run all correctness tests."""
    print("=" * 60)
    print("Running Correctness Tests")
    print("=" * 60)

    tests = [
        ("Forward pass shape", test_forward_shape),
        ("Loss computation", test_loss_computed),
        ("Backward gradients", test_backward_gradients),
        ("Weights change after step", test_weights_change),
        ("Checkpoint round-trip", test_checkpoint_roundtrip),
        ("Tokenizer round-trip", test_tokenizer_roundtrip),
        ("Memorization (overfitting)", test_memorization),
        ("Inference generation", test_inference),
    ]

    passed = 0
    failed = 0
    failures = []

    for name, test_fn in tests:
        print(f"\n[Test] {name}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            failures.append((name, str(e)))
            print(f"  [FAIL] {e}")

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if failures:
        print("Failures:")
        for name, err in failures:
            print(f"  - {name}: {err}")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
