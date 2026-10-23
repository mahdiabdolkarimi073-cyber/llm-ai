"""Specialized training script for the programming code corpus.

Features:
  - Resumes from existing checkpoint (continues training, does NOT overwrite)
  - Tracks best checkpoint by validation loss (saves separately)
  - Logs progress with train/val loss, learning rate, and timing
  - Adapts tokenizer and model embeddings when vocabulary changes
  - Writes a training log file for later analysis
  - CPU-only, with configurable batch size and context length
  - Atomic checkpoint saves (temp file + rename) so killed processes
    cannot corrupt existing checkpoints
  - Safe checkpoint loading with shape validation before mutating model
  - On error or interruption, existing valid checkpoints are preserved
  - Clear error messages on any incompatibility — no silent weight transfer

Usage:
    python llm/train_specialized.py              # Full training
    python llm/train_specialized.py --resume     # Resume from checkpoint
    python llm/train_specialized.py --info       # Show corpus stats only
"""

import argparse
import hashlib
import json
import os
import signal
import sys
import time

# Make package imports work when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader

from config import ModelConfig
from tokenizer import BPETokenizer
from model import GPTModel
from dataset import TextDataset
from checkpoint import (
    save_checkpoint,
    load_checkpoint,
    load_config_from_checkpoint,
    is_checkpoint_compatible,
    CheckpointError,
)
from data_pipeline import DataPipeline, CodeDocument
from corpus import build_corpus
from tokenizer_adapt import (
    adapt_tokenizer,
    adapt_model_embeddings,
    adapt_position_embeddings,
    AdaptationError,
)


# Directories
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

# File paths
EXISTING_CKPT = os.path.join(CHECKPOINT_DIR, "model.pt")        # Phase 1 checkpoint
EXISTING_TOK = os.path.join(CHECKPOINT_DIR, "tokenizer.json")   # Phase 1 tokenizer
SPECIALIZED_CKPT = os.path.join(CHECKPOINT_DIR, "specialized.pt")
BEST_CKPT = os.path.join(CHECKPOINT_DIR, "specialized_best.pt")
NEW_TOKENIZER = os.path.join(CHECKPOINT_DIR, "tokenizer_v2.json")
TRAIN_LOG = os.path.join(LOG_DIR, "training_log.json")


# ---------------------------------------------------------------------------
# Graceful shutdown flag
# ---------------------------------------------------------------------------
_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    print(f"\n  [SIGNAL] Shutdown requested (signal {signum}). "
          f"Will stop at next safe checkpoint boundary.")
    _shutdown_requested = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def get_specialized_config() -> ModelConfig:
    """Config optimized for CPU training on the programming corpus.

    Uses a smaller context length to fit more batches in memory,
    and a moderate learning rate for fine-tuning from existing weights.
    """
    return ModelConfig(
        bpe_vocab_size=1500,
        vocab_size=1500,
        context_length=128,
        n_layers=2,
        n_heads=4,
        hidden_dim=128,
        ff_dim=512,
        batch_size=32,
        max_epochs=10,
        eval_interval=50,
        eval_iters=10,
        learning_rate=1e-3,
        weight_decay=0.1,
        dropout=0.1,
        warmup_steps=10,
        grad_clip=1.0,
        early_stopping_patience=5,
        checkpoint_dir=CHECKPOINT_DIR,
        temperature=0.7,
        top_k=40,
        max_new_tokens=150,
    )


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(config: ModelConfig) -> dict:
    """Build corpus, run pipeline, train tokenizer, encode and split data.

    Returns a dict with all processed data and stats.

    Raises:
        RuntimeError: If corpus is empty or data pipeline fails.
    """
    # Build corpus
    print("\n[1/5] Building programming corpus...")
    documents = build_corpus()
    if not documents:
        raise RuntimeError("Corpus is empty — build_corpus() returned no documents")
    print(f"  Raw documents: {len(documents)}")

    # Run data pipeline
    print("\n[2/5] Running data pipeline (clean, validate, deduplicate, split)...")
    pipeline = DataPipeline(
        min_chars=20,
        max_chars=10000,
        train_frac=0.8,
        val_frac=0.1,
        test_frac=0.1,
        seed=42,
    )
    stats = pipeline.process(documents)
    print(pipeline.format_report(stats))

    train_docs, val_docs, test_docs = stats["splits"]
    if not train_docs:
        raise RuntimeError("Data pipeline produced 0 training documents")
    if not val_docs:
        raise RuntimeError("Data pipeline produced 0 validation documents")

    # Extract texts
    train_texts = [d.text for d in train_docs]
    val_texts = [d.text for d in val_docs]
    test_texts = [d.text for d in test_docs]

    # Train or load tokenizer (cache only if training data hash matches)
    print("\n[3/5] Preparing BPE tokenizer...")
    train_data_hash = hashlib.md5(
        "".join(train_texts).encode("utf-8")
    ).hexdigest()
    tokenizer_meta_path = NEW_TOKENIZER + ".meta.json"
    tokenizer = None
    if os.path.exists(NEW_TOKENIZER) and os.path.exists(tokenizer_meta_path):
        try:
            with open(tokenizer_meta_path, "r") as f:
                meta = json.load(f)
            if meta.get("train_data_hash") == train_data_hash and \
               meta.get("target_vocab_size") == config.bpe_vocab_size:
                cached = BPETokenizer(vocab_size=config.bpe_vocab_size)
                cached.load(NEW_TOKENIZER)
                if cached.vocab_size == meta.get("actual_vocab_size"):
                    print(f"  Reusing cached tokenizer from {NEW_TOKENIZER} "
                          f"(vocab={cached.vocab_size}, hash={train_data_hash[:8]})")
                    tokenizer = cached
                else:
                    print(f"  Cached tokenizer vocab mismatch, retraining...")
            else:
                print(f"  Training data changed (hash differs), retraining tokenizer...")
        except Exception as e:
            print(f"  Could not load cached tokenizer ({e}), retraining...")

    if tokenizer is None:
        print("  Training new BPE tokenizer on training corpus...")
        try:
            tokenizer = adapt_tokenizer(
                old_tokenizer_path=EXISTING_TOK if os.path.exists(EXISTING_TOK) else None,
                train_texts=train_texts,
                target_vocab_size=config.bpe_vocab_size,
                output_path=NEW_TOKENIZER,
                verbose=True,
            )
        except AdaptationError as e:
            raise RuntimeError(f"Tokenizer training failed: {e}") from e
        # Save metadata for future cache validation
        tok_meta = {
            "train_data_hash": train_data_hash,
            "target_vocab_size": config.bpe_vocab_size,
            "actual_vocab_size": tokenizer.vocab_size,
        }
        try:
            with open(tokenizer_meta_path, "w") as f:
                json.dump(tok_meta, f)
        except Exception:
            pass

    actual_vocab = tokenizer.vocab_size
    config.vocab_size = actual_vocab
    print(f"  Actual vocab size: {actual_vocab}")

    # Encode each split separately
    print("\n[4/5] Encoding corpus to token IDs...")
    def encode_batch(texts, label):
        ids = []
        for t in texts:
            ids.extend(tokenizer.encode(t))
        print(f"  {label}: {len(texts)} docs -> {len(ids):,} tokens")
        return ids

    train_ids = encode_batch(train_texts, "Train")
    val_ids = encode_batch(val_texts, "Val")
    test_ids = encode_batch(test_texts, "Test")

    if not train_ids:
        raise RuntimeError("Tokenizer produced 0 training tokens — check tokenizer config")

    # Save test set metadata for evaluation
    os.makedirs(DATA_DIR, exist_ok=True)
    test_meta = {
        "test_texts": test_texts,
        "test_languages": [d.language for d in test_docs],
        "test_categories": [d.category for d in test_docs],
    }
    with open(os.path.join(DATA_DIR, "test_set.json"), "w", encoding="utf-8") as f:
        json.dump(test_meta, f, ensure_ascii=False, indent=2)

    return {
        "stats": stats,
        "tokenizer": tokenizer,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "train_docs": train_docs,
        "val_docs": val_docs,
        "test_docs": test_docs,
        "actual_vocab": actual_vocab,
    }


# ---------------------------------------------------------------------------
# Optimizer with proper weight decay groups
# ---------------------------------------------------------------------------

def _create_optimizer(model: GPTModel, config: ModelConfig) -> torch.optim.AdamW:
    """Create AdamW with weight decay only on weight matrices (dim >= 2),
    not on biases or LayerNorm parameters."""
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=config.learning_rate, betas=(0.9, 0.95))


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model, loader, device="cpu", max_iters=20) -> float:
    """Compute average loss on a data loader."""
    model.eval()
    total_loss = 0.0
    count = 0
    for i, (x, y) in enumerate(loader):
        if i >= max_iters:
            break
        _, loss = model(x, y)
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


def get_lr(step, config, total_train_tokens=None):
    """Linear warmup then cosine decay over estimated total steps.

    Minimum LR is 10% of base — never reaches zero so training can still
    make small adjustments near the end.
    """
    import math
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    token_count = total_train_tokens if total_train_tokens is not None else 14568
    est_steps_per_epoch = max(1, token_count // config.batch_size)
    total_steps = max(1, est_steps_per_epoch * config.max_epochs)
    progress = min((step - config.warmup_steps) / max(total_steps - config.warmup_steps, 1), 1.0)
    min_lr = config.learning_rate * 0.1
    return min_lr + (config.learning_rate - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def setup_model_from_phase1(config: ModelConfig, actual_vocab: int) -> tuple:
    """Try to load Phase 1 checkpoint and adapt it. Falls back to from-scratch.

    Returns (model, optimizer, start_step, start_epoch, best_val_loss).
    The Phase 1 checkpoint is NEVER overwritten or deleted.
    """
    print("\n  Checking Phase 1 checkpoint compatibility...")
    try:
        phase1_config = load_config_from_checkpoint(EXISTING_CKPT)
    except CheckpointError as e:
        print(f"  WARNING: could not read Phase 1 config: {e}")
        print(f"  Training from scratch instead.")
        model = GPTModel(config)
        optimizer = _create_optimizer(model, config)
        return model, optimizer, 0, 0, float('inf')

    old_vocab = phase1_config.vocab_size
    old_ctx = phase1_config.context_length
    old_layers = phase1_config.n_layers
    old_dim = phase1_config.hidden_dim
    old_heads = phase1_config.n_heads

    # Check architecture compatibility (layers, dim, heads must match)
    arch_compatible = (
        old_layers == config.n_layers
        and old_dim == config.hidden_dim
        and old_heads == config.n_heads
    )

    if not arch_compatible:
        print(f"  Phase 1 architecture differs:")
        print(f"    Checkpoint: layers={old_layers}, heads={old_heads}, dim={old_dim}")
        print(f"    Config:     layers={config.n_layers}, heads={config.n_heads}, dim={config.hidden_dim}")
        print(f"  Cannot transfer weights — training from scratch.")
        print(f"  Phase 1 checkpoint preserved at {EXISTING_CKPT}")
        model = GPTModel(config)
        optimizer = _create_optimizer(model, config)
        return model, optimizer, 0, 0, float('inf')

    # Architecture matches — load weights into a model with the old config
    print(f"  Phase 1 architecture is compatible, loading weights...")
    model = GPTModel(phase1_config)

    try:
        meta = load_checkpoint(EXISTING_CKPT, model)
    except CheckpointError as e:
        print(f"  ERROR: could not load Phase 1 weights: {e}")
        print(f"  Training from scratch instead. Phase 1 checkpoint preserved.")
        model = GPTModel(config)
        optimizer = _create_optimizer(model, config)
        return model, optimizer, 0, 0, float('inf')

    print(f"  Phase 1 loaded: step={meta['step']}, val_loss={meta['val_loss']:.4f}")
    print(f"  Phase 1 config: vocab={old_vocab}, ctx={old_ctx}")

    # Adapt token embeddings if vocab changed
    try:
        model = adapt_model_embeddings(model, old_vocab, actual_vocab, config)
    except AdaptationError as e:
        print(f"  ERROR: embedding adaptation failed: {e}")
        print(f"  Training from scratch instead. Phase 1 checkpoint preserved.")
        model = GPTModel(config)
        optimizer = _create_optimizer(model, config)
        return model, optimizer, 0, 0, float('inf')

    # Adapt position embeddings if context length changed
    try:
        model = adapt_position_embeddings(model, old_ctx, config.context_length, config)
    except AdaptationError as e:
        print(f"  ERROR: position embedding adaptation failed: {e}")
        print(f"  Training from scratch instead. Phase 1 checkpoint preserved.")
        model = GPTModel(config)
        optimizer = _create_optimizer(model, config)
        return model, optimizer, 0, 0, float('inf')

    # Update model's internal config to match adapted dimensions
    model.config = config

    # Create fresh optimizer (adapted params are new)
    optimizer = _create_optimizer(model, config)
    return model, optimizer, 0, 0, float('inf')


def setup_model_from_specialized(
    config: ModelConfig, actual_vocab: int
) -> tuple:
    """Resume from an existing specialized checkpoint.

    Validates compatibility before loading. On mismatch, raises CheckpointError.
    """
    print("\n  Resuming from specialized checkpoint...")

    # Check compatibility first (without loading into model)
    compatible, reason = is_checkpoint_compatible(SPECIALIZED_CKPT, config)
    if not compatible:
        raise CheckpointError(
            f"Specialized checkpoint incompatible with current config: {reason}\n"
            f"Either delete {SPECIALIZED_CKPT} to start fresh, or adjust your config."
        )

    saved_config = load_config_from_checkpoint(SPECIALIZED_CKPT)
    if saved_config is None:
        raise CheckpointError("Specialized checkpoint has no config metadata")

    # Build model with saved config for loading
    model = GPTModel(saved_config)

    optimizer = _create_optimizer(model, config)

    meta = load_checkpoint(SPECIALIZED_CKPT, model, optimizer)

    # Check vocab compatibility
    saved_vocab = saved_config.vocab_size
    if saved_vocab != actual_vocab:
        raise CheckpointError(
            f"Specialized checkpoint vocab ({saved_vocab}) does not match "
            f"current tokenizer vocab ({actual_vocab}). "
            f"Delete {SPECIALIZED_CKPT} and {NEW_TOKENIZER} to retrain from scratch."
        )

    # If context length differs, adapt position embeddings
    if saved_config.context_length != config.context_length:
        model = adapt_position_embeddings(
            model, saved_config.context_length, config.context_length, config
        )

    model.config = config

    start_step = meta["step"]
    start_epoch = meta["epoch"]
    best_val_loss = meta.get("val_loss", float('inf'))

    print(f"  Resumed: step={start_step}, epoch={start_epoch}, "
          f"val_loss={meta['val_loss']:.4f}")

    return model, optimizer, start_step, start_epoch, best_val_loss


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def run_training(config: ModelConfig, resume: bool = False):
    """Full specialized training pipeline.

    On error at any stage, existing checkpoints are preserved. The training
    log is written even on early termination.
    """
    # Prepare data
    try:
        data = prepare_data(config)
    except RuntimeError as e:
        print(f"\nFATAL: Data preparation failed: {e}")
        return None

    tokenizer = data["tokenizer"]
    actual_vocab = data["actual_vocab"]

    # Adapt context length if corpus is small
    total_train_tokens = len(data["train_ids"])
    if total_train_tokens < config.context_length * 20:
        old_ctx = config.context_length
        config.context_length = min(128, max(32, total_train_tokens // 30))
        print(f"\n  Adjusted context_length: {old_ctx} -> {config.context_length}")

    # Create datasets
    train_dataset = TextDataset(data["train_ids"], config)
    val_dataset = TextDataset(data["val_ids"], config)

    print(f"\n[5/5] Setting up model and training...")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Setup model
    start_step = 0
    start_epoch = 0
    best_val_loss = float('inf')

    try:
        if resume and os.path.exists(SPECIALIZED_CKPT):
            model, optimizer, start_step, start_epoch, best_val_loss = \
                setup_model_from_specialized(config, actual_vocab)
        elif os.path.exists(EXISTING_CKPT):
            model, optimizer, start_step, start_epoch, best_val_loss = \
                setup_model_from_phase1(config, actual_vocab)
        else:
            print("\n  No existing checkpoint found, training from scratch...")
            model = GPTModel(config)
            optimizer = _create_optimizer(model, config)
    except CheckpointError as e:
        print(f"\nFATAL: Checkpoint error: {e}")
        print(f"  Existing checkpoints are preserved. Fix the issue and retry.")
        return None
    except AdaptationError as e:
        print(f"\nFATAL: Model adaptation error: {e}")
        print(f"  Existing checkpoints are preserved. Fix the issue and retry.")
        return None

    print(f"  Total model parameters: {model.count_parameters():,}")
    print(f"  Device: CPU, threads: {torch.get_num_threads()}")
    print(f"  Batch size: {config.batch_size}, Context length: {config.context_length}")
    print(f"  Max epochs: {config.max_epochs}, Eval interval: {config.eval_interval}")
    print("-" * 70)

    # Create data loaders
    drop_last = len(train_dataset) > config.batch_size * 2
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config.batch_size, len(train_dataset)),
        shuffle=True,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(config.batch_size, len(val_dataset)),
        shuffle=False,
        drop_last=False,
    )

    # Training loop
    step = start_step
    epoch = start_epoch
    log_entries = []
    start_time = time.time()
    last_train_loss = 0.0
    last_val_loss = 0.0
    train_loss_sum = 0.0
    train_loss_count = 0
    epochs_no_improve = 0
    stopped_early = False

    try:
        for ep in range(epoch, config.max_epochs):
            model.train()

            for x, y in train_loader:
                # Check for shutdown request
                if _shutdown_requested:
                    print("\n  [STOP] Shutdown requested. Saving final checkpoint...")
                    break

                # LR schedule
                lr = get_lr(step, config, total_train_tokens=len(data["train_ids"]))
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                # Forward + backward
                logits, loss = model(x, y)
                optimizer.zero_grad()
                loss.backward()

                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                optimizer.step()
                step += 1
                train_loss_sum += loss.item()
                train_loss_count += 1

                # Periodic eval
                if step % config.eval_interval == 0:
                    val_loss = evaluate_model(
                        model, val_loader, max_iters=config.eval_iters
                    )
                    train_loss = train_loss_sum / max(train_loss_count, 1)
                    train_loss_sum = 0.0
                    train_loss_count = 0
                    last_train_loss = train_loss
                    last_val_loss = val_loss
                    elapsed = time.time() - start_time

                    print(f"  step {step:5d} | epoch {ep} | lr {lr:.2e} | "
                          f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                          f"{elapsed:.1f}s")

                    entry = {
                        "step": step, "epoch": ep,
                        "train_loss": train_loss, "val_loss": val_loss,
                        "lr": lr, "elapsed_s": round(elapsed, 1),
                    }
                    log_entries.append(entry)

                    # Save latest checkpoint (atomic)
                    save_checkpoint(
                        SPECIALIZED_CKPT, model, optimizer, step, ep,
                        train_loss, val_loss, config=config,
                    )

                    # Save best checkpoint (atomic)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                        save_checkpoint(
                            BEST_CKPT, model, optimizer, step, ep,
                            train_loss, val_loss, config=config,
                        )
                        print(f"    -> New best val_loss: {val_loss:.4f} (saved)")
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= config.early_stopping_patience:
                            print(f"\n  [EARLY STOP] No val_loss improvement for "
                                  f"{config.early_stopping_patience} evals. "
                                  f"Stopping at step {step}.")
                            stopped_early = True
                            break

            # Check if we broke out of inner loop
            if _shutdown_requested or stopped_early:
                break

            # End of epoch eval
            epoch_val_loss = evaluate_model(
                model, val_loader, max_iters=config.eval_iters
            )
            last_val_loss = epoch_val_loss
            print(f"=== Epoch {ep} complete | val_loss: {epoch_val_loss:.4f} | "
                  f"best: {best_val_loss:.4f} ===")

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                save_checkpoint(
                    BEST_CKPT, model, optimizer, step, ep,
                    last_train_loss, epoch_val_loss, config=config,
                )

        # Save final checkpoint (unless we were shut down or early-stopped)
        if not _shutdown_requested:
            save_checkpoint(
                SPECIALIZED_CKPT, model, optimizer, step, config.max_epochs,
                last_train_loss, last_val_loss, config=config,
            )

    except RuntimeError as e:
        print(f"\n  ERROR during training: {e}")
        print(f"  Last valid checkpoint is preserved at {SPECIALIZED_CKPT}")
        # Still write the log with what we have
    except KeyboardInterrupt:
        print(f"\n  [KEYBOARD INTERRUPT] Stopping. Last checkpoint preserved.")
        print(f"  Resume with: python llm/train_specialized.py --resume")

    # Write training log (always, even on early stop)
    os.makedirs(LOG_DIR, exist_ok=True)
    log_data = {
        "config": {
            "vocab_size": config.vocab_size,
            "context_length": config.context_length,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "hidden_dim": config.hidden_dim,
            "ff_dim": config.ff_dim,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "max_epochs": config.max_epochs,
        },
        "data_stats": {
            "train_docs": data["stats"]["train_docs"],
            "val_docs": data["stats"]["val_docs"],
            "test_docs": data["stats"]["test_docs"],
            "train_tokens": len(data["train_ids"]),
            "val_tokens": len(data["val_ids"]),
            "test_tokens": len(data["test_ids"]),
            "languages": dict(data["stats"]["languages"]),
            "categories": dict(data["stats"]["categories"]),
        },
        "training": {
            "total_steps": step,
            "best_val_loss": best_val_loss,
            "final_val_loss": last_val_loss,
            "total_time_s": round(time.time() - start_time, 1),
            "interrupted": _shutdown_requested,
            "early_stopped": stopped_early,
        },
        "log_entries": log_entries,
    }
    # Atomic log write
    tmp_log = TRAIN_LOG + ".tmp"
    try:
        with open(tmp_log, "w") as f:
            json.dump(log_data, f, indent=2)
        os.replace(tmp_log, TRAIN_LOG)
    except Exception:
        if os.path.exists(tmp_log):
            os.remove(tmp_log)
        raise

    if _shutdown_requested:
        status = "INTERRUPTED"
    elif stopped_early:
        status = "EARLY STOPPED"
    else:
        status = "complete"
    print(f"\n{'=' * 70}")
    print(f"Training {status}!")
    print(f"  Total steps: {step}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Final val loss: {last_val_loss:.4f}")
    print(f"  Total time: {time.time() - start_time:.1f}s")
    print(f"  Checkpoints: {SPECIALIZED_CKPT}")
    print(f"               {BEST_CKPT}")
    print(f"  Training log: {TRAIN_LOG}")
    print(f"{'=' * 70}")

    return log_data


# ---------------------------------------------------------------------------
# Info
# ---------------------------------------------------------------------------

def show_info():
    """Show corpus and data stats without training."""
    config = get_specialized_config()
    documents = build_corpus()
    print(f"Corpus: {len(documents)} documents")
    print(f"Total chars: {sum(len(d.text) for d in documents):,}")

    pipeline = DataPipeline()
    stats = pipeline.process(documents)
    print()
    print(pipeline.format_report(stats))

    print(f"\nEstimated token count: ~{stats['train_chars'] // 4:,} tokens (4 chars/token approx)")
    print(f"Model target vocab: {config.bpe_vocab_size}")
    print(f"Context length: {config.context_length}")
    print(f"Batch size: {config.batch_size}")
    est_ram = config.vocab_size * config.hidden_dim * 4 / 1024 / 1024
    print(f"Estimated embedding RAM: {est_ram:.1f} MB")
    print(f"CPU threads available: {torch.get_num_threads()}")

    # Check existing checkpoints
    print(f"\nExisting checkpoints:")
    for name, path in [
        ("Phase 1", EXISTING_CKPT),
        ("Specialized", SPECIALIZED_CKPT),
        ("Best", BEST_CKPT),
    ]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            try:
                compat, reason = is_checkpoint_compatible(path, config)
                status = "COMPATIBLE" if compat else f"INCOMPATIBLE ({reason})"
            except CheckpointError as e:
                status = f"ERROR ({e})"
            print(f"  {name}: {path} ({size_mb:.1f} MB) — {status}")
        else:
            print(f"  {name}: not found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specialized training for programming corpus")
    parser.add_argument("--resume", action="store_true", help="Resume from specialized checkpoint")
    parser.add_argument("--info", action="store_true", help="Show corpus stats only")
    args = parser.parse_args()

    if args.info:
        show_info()
    else:
        config = get_specialized_config()
        run_training(config, resume=args.resume)
