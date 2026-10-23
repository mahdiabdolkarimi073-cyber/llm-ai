"""Main entry point: train the model or run tests.

Usage:
    python llm/main.py test       # Run correctness tests
    python llm/main.py train      # Train on sample data
    python llm/main.py generate   # Generate text from checkpoint
    python llm/main.py info       # Print model architecture and parameter count
"""

import os
import sys

# Make package imports work when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig
from tokenizer import BPETokenizer
from model import GPTModel
from dataset import TextDataset, encode_corpus, split_data
from training import train, evaluate_loss
from inference import generate_text, generate_multiple
from checkpoint import load_checkpoint
from sample_data import SAMPLE_DATA


def run_tests():
    from tests import run_all_tests
    run_all_tests()


def run_train():
    # Use a smaller config for the demo so it finishes in reasonable time on CPU
    config = ModelConfig(
        vocab_size=2000,
        context_length=128,
        n_layers=4,
        n_heads=4,
        hidden_dim=256,
        ff_dim=1024,
        batch_size=16,
        max_epochs=10,
        eval_interval=20,
        eval_iters=10,
        learning_rate=1e-3,
        warmup_steps=10,
    )
    print(f"Model config (demo):")
    print(f"  Layers: {config.n_layers}, Heads: {config.n_heads}")
    print(f"  Hidden dim: {config.hidden_dim}, FF dim: {config.ff_dim}")
    print(f"  Context length: {config.context_length}")
    print(f"  Vocab size (target): {config.vocab_size}")
    print(f"  Estimated params: {config.total_params_estimate:,}")
    print()

    # Step 1: Train tokenizer
    print("Training BPE tokenizer...")
    tokenizer = BPETokenizer(vocab_size=config.bpe_vocab_size)
    tokenizer.train(SAMPLE_DATA, verbose=True)

    # Update config with actual vocab size
    config.vocab_size = tokenizer.vocab_size
    print(f"Actual vocab size: {config.vocab_size}")

    # Save tokenizer
    tok_path = os.path.join(config.checkpoint_dir, "tokenizer.json")
    tokenizer.save(tok_path)
    print(f"Tokenizer saved to {tok_path}")

    # Step 2: Encode corpus
    print("\nEncoding corpus...")
    # Repeat data to get more training signal for the demo
    repeated_data = SAMPLE_DATA * 10
    all_ids = encode_corpus(repeated_data, tokenizer)
    print(f"Total tokens: {len(all_ids)}")

    # Adapt context length to data size for the small demo
    if len(all_ids) < config.context_length * 10:
        config.context_length = min(128, max(32, len(all_ids) // 20))
        print(f"Adjusted context length to {config.context_length} for small dataset")

    train_ids, val_ids = split_data(all_ids, val_frac=0.1)
    print(f"Train tokens: {len(train_ids)}, Val tokens: {len(val_ids)}")

    # Step 3: Create model
    model = GPTModel(config)
    print(f"\nModel parameters: {model.count_parameters():,}")

    # Step 4: Create datasets
    train_dataset = TextDataset(train_ids, config)
    val_dataset = TextDataset(val_ids, config)

    # Step 5: Train
    print()
    history = train(model, train_dataset, val_dataset, config, resume=False)

    print(f"\nTraining history ({len(history)} eval points):")
    if history:
        print(f"  First: train_loss={history[0]['train_loss']:.4f}, val_loss={history[0]['val_loss']:.4f}")
        print(f"  Last:  train_loss={history[-1]['train_loss']:.4f}, val_loss={history[-1]['val_loss']:.4f}")


def run_generate():
    config = ModelConfig()
    ckpt_path = os.path.join(config.checkpoint_dir, "model.pt")
    tok_path = os.path.join(config.checkpoint_dir, "tokenizer.json")

    if not os.path.exists(ckpt_path):
        print("No checkpoint found. Run 'python llm/main.py train' first.")
        return

    # Load config from checkpoint if available
    from checkpoint import load_config_from_checkpoint, load_checkpoint, CheckpointError
    try:
        saved_config = load_config_from_checkpoint(ckpt_path)
    except CheckpointError as e:
        print(f"ERROR: Could not read checkpoint config: {e}")
        return
    if saved_config is not None:
        config = saved_config

    # Load tokenizer
    tokenizer = BPETokenizer(vocab_size=config.vocab_size)
    tokenizer.load(tok_path)

    # Validate vocab compatibility
    if tokenizer.vocab_size != config.vocab_size:
        print(f"ERROR: Tokenizer vocab ({tokenizer.vocab_size}) does not match "
              f"checkpoint vocab ({config.vocab_size}).")
        print(f"  Tokenizer: {tok_path}")
        print(f"  Checkpoint: {ckpt_path}")
        return

    config.vocab_size = tokenizer.vocab_size

    # Load model
    model = GPTModel(config)
    try:
        meta = load_checkpoint(ckpt_path, model)
    except CheckpointError as e:
        print(f"ERROR: Could not load checkpoint: {e}")
        return
    print(f"Loaded checkpoint: step={meta['step']}, val_loss={meta['val_loss']:.4f}")

    # Generate
    prompts = [
        "def ",
        "function ",
        "const ",
        "import ",
    ]

    for prompt in prompts:
        print(f"\n--- Prompt: '{prompt}' ---")
        samples = generate_multiple(
            model, tokenizer, prompt, config,
            n_samples=2, max_new_tokens=100, temperature=0.8, top_k=40,
        )
        for i, s in enumerate(samples):
            print(f"  Sample {i+1}: {repr(s)}")


def run_info():
    config = ModelConfig()
    print("Model Configuration:")
    print(f"  Architecture: Decoder-only Transformer")
    print(f"  Layers: {config.n_layers}")
    print(f"  Attention heads: {config.n_heads}")
    print(f"  Head dimension: {config.head_dim}")
    print(f"  Hidden dimension: {config.hidden_dim}")
    print(f"  Feed-forward dimension: {config.ff_dim}")
    print(f"  Context length: {config.context_length}")
    print(f"  Vocab size (target): {config.vocab_size}")
    print(f"  Normalization: Pre-LayerNorm")
    print(f"  Activation: GELU")
    print(f"  Loss: Cross-Entropy")
    print(f"  Optimizer: AdamW (lr={config.learning_rate}, wd={config.weight_decay})")
    print(f"  Gradient clipping: {config.grad_clip}")
    print()

    model = GPTModel(config)
    print(f"Total trainable parameters: {model.count_parameters():,}")
    print(f"Estimated parameters: {config.total_params_estimate:,}")
    print()

    print("Parameter breakdown:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {list(param.shape)} = {param.numel():,}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "info"

    if cmd == "test":
        run_tests()
    elif cmd == "train":
        run_train()
    elif cmd == "generate":
        run_generate()
    elif cmd == "info":
        run_info()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python llm/main.py [test|train|generate|info]")
        sys.exit(1)
