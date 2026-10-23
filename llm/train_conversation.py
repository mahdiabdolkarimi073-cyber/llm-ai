"""Conversation fine-tuning script for the local LLM.

Prepares the model for chat by fine-tuning on user/assistant conversation
data alongside the existing programming corpus. This script is READY to run
but does NOT execute automatically — it must be invoked manually:

    python llm/train_conversation.py            # Full conversation training
    python llm/train_conversation.py --resume   # Resume from checkpoint
    python llm/train_conversation.py --info     # Show data stats only

The script:
  - Loads the best specialized checkpoint as starting weights
  - Reuses the existing specialized tokenizer (preserves token-ID mapping)
  - Falls back to training a new tokenizer only if the specialized one is missing
  - Fine-tunes the model on the combined dataset with document-boundary separation
  - Saves checkpoints to checkpoints/conversation_*.pt
  - Preserves all existing checkpoints and tokenizers (never overwrites)

Architecture is kept identical to the specialized model so weights transfer.
"""

import argparse
import json
import math
import os
import sys
import time

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
    CheckpointError,
)
from corpus import build_corpus
from data_pipeline import DataPipeline


CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

BEST_SPECIALIZED = os.path.join(CHECKPOINT_DIR, "specialized_best.pt")
SPECIALIZED_CKPT = os.path.join(CHECKPOINT_DIR, "specialized.pt")
SPECIALIZED_TOK = os.path.join(CHECKPOINT_DIR, "tokenizer_v2.json")

CONV_CKPT = os.path.join(CHECKPOINT_DIR, "conversation.pt")
CONV_BEST_CKPT = os.path.join(CHECKPOINT_DIR, "conversation_best.pt")
CONV_TOKENIZER = os.path.join(CHECKPOINT_DIR, "tokenizer_conv.json")

# Byte value used as document boundary separator (not a valid UTF-8 start byte,
# so it will never appear in real text). This prevents cross-document token
# sequences from being learned as patterns.
_DOC_SEP = ord('\x00')
TRAIN_LOG = os.path.join(LOG_DIR, "conversation_training_log.json")


def get_conversation_config() -> ModelConfig:
    """Config for conversation fine-tuning — matches specialized architecture."""
    return ModelConfig(
        vocab_size=1500,
        bpe_vocab_size=1500,
        context_length=128,
        n_layers=2,
        n_heads=4,
        hidden_dim=128,
        ff_dim=512,
        batch_size=16,
        max_epochs=15,
        eval_interval=30,
        eval_iters=10,
        learning_rate=5e-4,
        weight_decay=0.1,
        warmup_steps=20,
        grad_clip=1.0,
        checkpoint_dir=CHECKPOINT_DIR,
        temperature=0.4,
        top_k=25,
        max_new_tokens=150,
    )


def prepare_conversation_data(config: ModelConfig) -> dict:
    """Build corpus with conversation data, train tokenizer, encode and split."""
    print("\n[1/4] Building corpus with conversation data...")
    documents = build_corpus()
    conv_docs = [d for d in documents if d.category == "conversation"]
    code_docs = [d for d in documents if d.category != "conversation"]
    print(f"  Conversation docs: {len(conv_docs)}")
    print(f"  Code/QA docs: {len(code_docs)}")
    print(f"  Total: {len(documents)}")

    if not conv_docs:
        raise RuntimeError("No conversation documents found in corpus")

    print("\n[2/4] Running data pipeline...")
    pipeline = DataPipeline(
        min_chars=20,
        max_chars=10000,
        train_frac=0.85,
        val_frac=0.10,
        test_frac=0.05,
        seed=42,
    )
    stats = pipeline.process(documents)
    print(pipeline.format_report(stats))

    train_docs, val_docs, test_docs = stats["splits"]
    train_texts = [d.text for d in train_docs]
    val_texts = [d.text for d in val_docs]
    test_texts = [d.text for d in test_docs]

    # Reuse the specialized tokenizer instead of training from scratch.
    # This preserves the existing token-ID mapping so specialized weights
    # transfer correctly — no embedding rebuild needed.
    if os.path.exists(SPECIALIZED_TOK):
        print(f"\n[3/4] Loading specialized tokenizer: {SPECIALIZED_TOK}")
        tokenizer = BPETokenizer(vocab_size=config.bpe_vocab_size)
        tokenizer.load(SPECIALIZED_TOK)
        print(f"  Tokenizer loaded: {tokenizer.vocab_size} tokens")
    else:
        print(f"\n[3/4] Specialized tokenizer not found, training from scratch...")
        tokenizer = BPETokenizer(vocab_size=config.bpe_vocab_size)
        tokenizer.train(train_texts, verbose=True)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    tmp_tok = CONV_TOKENIZER + ".tmp"
    tokenizer.save(tmp_tok)
    os.replace(tmp_tok, CONV_TOKENIZER)
    print(f"  Tokenizer copy saved: {CONV_TOKENIZER}")

    config.vocab_size = tokenizer.vocab_size

    print("\n[4/4] Encoding corpus...")
    def encode_batch(texts, label):
        ids = []
        for t in texts:
            ids.extend(tokenizer.encode(t))
            ids.append(_DOC_SEP)  # document boundary — prevents cross-doc merges
        print(f"  {label}: {len(texts)} docs -> {len(ids):,} tokens")
        return ids

    train_ids = encode_batch(train_texts, "Train")
    val_ids = encode_batch(val_texts, "Val")
    test_ids = encode_batch(test_texts, "Test")

    return {
        "tokenizer": tokenizer,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "actual_vocab": tokenizer.vocab_size,
        "stats": stats,
    }


@torch.no_grad()
def evaluate_model(model, loader, max_iters=20) -> float:
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


def get_lr(step, config):
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    return config.learning_rate * 0.5 * (1 + math.cos(
        math.pi * min(step / (config.max_epochs * 500), 1.0)
    ))


def setup_model(config: ModelConfig, actual_vocab: int) -> tuple:
    """Load best specialized checkpoint as starting point."""
    ckpt_path = BEST_SPECIALIZED if os.path.exists(BEST_SPECIALIZED) else SPECIALIZED_CKPT
    if not os.path.exists(ckpt_path):
        print("  No specialized checkpoint found, training from scratch...")
        model = GPTModel(config)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, betas=(0.9, 0.95),
            weight_decay=config.weight_decay,
        )
        return model, optimizer, 0, float('inf')

    print(f"  Loading specialized checkpoint: {ckpt_path}")
    saved_config = load_config_from_checkpoint(ckpt_path)

    if saved_config is None:
        print("  WARNING: no config in checkpoint, training from scratch...")
        model = GPTModel(config)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, betas=(0.9, 0.95),
            weight_decay=config.weight_decay,
        )
        return model, optimizer, 0, float('inf')

    # Verify architecture compatibility before loading weights
    arch_fields = [("n_layers", saved_config.n_layers, config.n_layers),
                   ("n_heads", saved_config.n_heads, config.n_heads),
                   ("hidden_dim", saved_config.hidden_dim, config.hidden_dim),
                   ("ff_dim", saved_config.ff_dim, config.ff_dim)]
    for name, old_val, new_val in arch_fields:
        if old_val != new_val:
            print(f"  Architecture mismatch: {name} ckpt={old_val} vs config={new_val}")
            print(f"  Cannot transfer weights — training from scratch.")
            model = GPTModel(config)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.learning_rate, betas=(0.9, 0.95),
                weight_decay=config.weight_decay,
            )
            return model, optimizer, 0, float('inf')

    model = GPTModel(saved_config)
    meta = load_checkpoint(ckpt_path, model)
    print(f"  Loaded: step={meta['step']}, val_loss={meta['val_loss']:.4f}")

    if saved_config.vocab_size != actual_vocab:
        print(f"  Vocab mismatch: ckpt={saved_config.vocab_size}, tokenizer={actual_vocab}")
        print(f"  Rebuilding embedding layer for new vocab...")
        old_emb = model.token_embedding.weight.data
        hidden = old_emb.shape[1]
        new_emb = torch.nn.Embedding(actual_vocab, hidden)
        copy_n = min(old_emb.shape[0], actual_vocab)
        new_emb.weight.data[:copy_n] = old_emb[:copy_n]
        if actual_vocab > copy_n:
            mean_emb = old_emb.mean(dim=0, keepdim=True)
            new_emb.weight.data[copy_n:] = mean_emb.expand(
                actual_vocab - copy_n, -1
            ) + torch.randn(actual_vocab - copy_n, hidden) * 0.02
        model.token_embedding = new_emb
        model.lm_head = torch.nn.Linear(hidden, actual_vocab, bias=False)
        model.lm_head.weight = model.token_embedding.weight

    if saved_config.context_length != config.context_length:
        old_pos = model.position_embedding.weight.data
        new_pos = torch.nn.Embedding(config.context_length, hidden)
        copy_len = min(old_pos.shape[0], config.context_length)
        new_pos.weight.data[:copy_len] = old_pos[:copy_len]
        if config.context_length > copy_len:
            mean_pos = old_pos.mean(dim=0, keepdim=True)
            new_pos.weight.data[copy_len:] = mean_pos.expand(
                config.context_length - copy_len, -1
            ) + torch.randn(config.context_length - copy_len, hidden) * 0.02
        model.position_embedding = new_pos

    model.config = config
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )
    return model, optimizer, 0, float('inf')


# Number of eval intervals without improvement before stopping training.
# Prevents overfitting when val_loss keeps rising while train_loss drops.
PATIENCE = 5


def run_training(config: ModelConfig, resume: bool = False):
    try:
        data = prepare_conversation_data(config)
    except RuntimeError as e:
        print(f"\nFATAL: Data preparation failed: {e}")
        return None

    tokenizer = data["tokenizer"]
    actual_vocab = data["actual_vocab"]

    total_train_tokens = len(data["train_ids"])
    if total_train_tokens < config.context_length * 20:
        old_ctx = config.context_length
        config.context_length = min(128, max(32, total_train_tokens // 30))
        print(f"\n  Adjusted context_length: {old_ctx} -> {config.context_length}")

    train_dataset = TextDataset(data["train_ids"], config)
    val_dataset = TextDataset(data["val_ids"], config)

    print(f"\n  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    start_step = 0
    best_val_loss = float('inf')

    if resume and os.path.exists(CONV_CKPT):
        print("\n  Resuming from conversation checkpoint...")
        saved_config = load_config_from_checkpoint(CONV_CKPT)
        if saved_config and saved_config.vocab_size == actual_vocab:
            model = GPTModel(saved_config)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.learning_rate, betas=(0.9, 0.95),
                weight_decay=config.weight_decay,
            )
            meta = load_checkpoint(CONV_CKPT, model, optimizer)
            start_step = meta["step"]
            best_val_loss = meta.get("val_loss", float('inf'))
            model.config = config
        else:
            print("  Checkpoint incompatible, starting fresh...")
            model, optimizer, start_step, best_val_loss = setup_model(config, actual_vocab)
    else:
        model, optimizer, start_step, best_val_loss = setup_model(config, actual_vocab)

    print(f"  Total parameters: {model.count_parameters():,}")
    print(f"  Batch size: {config.batch_size}, Context: {config.context_length}")
    print(f"  Max epochs: {config.max_epochs}")
    print("-" * 60)

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

    step = start_step
    log_entries = []
    start_time = time.time()
    last_train_loss = 0.0
    last_val_loss = 0.0
    no_improve_count = 0

    try:
        for ep in range(config.max_epochs):
            model.train()
            for x, y in train_loader:
                lr = get_lr(step, config)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                logits, loss = model(x, y)
                optimizer.zero_grad()
                loss.backward()
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                step += 1

                if step % config.eval_interval == 0:
                    val_loss = evaluate_model(model, val_loader, config.eval_iters)
                    train_loss = loss.item()
                    last_train_loss = train_loss
                    last_val_loss = val_loss
                    elapsed = time.time() - start_time
                    print(f"  step {step:5d} | epoch {ep} | lr {lr:.2e} | "
                          f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                          f"{elapsed:.1f}s")
                    log_entries.append({
                        "step": step, "epoch": ep,
                        "train_loss": train_loss, "val_loss": val_loss,
                        "lr": lr, "elapsed_s": round(elapsed, 1),
                    })
                    save_checkpoint(
                        CONV_CKPT, model, optimizer, step, ep,
                        train_loss, val_loss, config=config,
                    )
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        no_improve_count = 0
                        save_checkpoint(
                            CONV_BEST_CKPT, model, optimizer, step, ep,
                            train_loss, val_loss, config=config,
                        )
                        print(f"    -> New best val_loss: {val_loss:.4f} (saved)")
                    else:
                        no_improve_count += 1
                        if no_improve_count >= PATIENCE:
                            print(f"\n  [EARLY STOP] No improvement for {PATIENCE} evals.")
                            print(f"  Best val_loss: {best_val_loss:.4f} at step {step - no_improve_count * config.eval_interval}")
                            break

            # Check if early stopping triggered
            if no_improve_count >= PATIENCE:
                break

            epoch_val_loss = evaluate_model(model, val_loader, config.eval_iters)
            last_val_loss = epoch_val_loss
            print(f"=== Epoch {ep} complete | val_loss: {epoch_val_loss:.4f} | "
                  f"best: {best_val_loss:.4f} ===")
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                save_checkpoint(
                    CONV_BEST_CKPT, model, optimizer, step, ep,
                    last_train_loss, epoch_val_loss, config=config,
                )

        save_checkpoint(
            CONV_CKPT, model, optimizer, step, config.max_epochs,
            last_train_loss, last_val_loss, config=config,
        )

    except KeyboardInterrupt:
        print(f"\n  [INTERRUPTED] Checkpoints preserved at {CONV_CKPT}")
    except RuntimeError as e:
        print(f"\n  ERROR: {e}")

    os.makedirs(LOG_DIR, exist_ok=True)
    log_data = {
        "config": {
            "vocab_size": config.vocab_size,
            "context_length": config.context_length,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "hidden_dim": config.hidden_dim,
            "ff_dim": config.ff_dim,
            "learning_rate": config.learning_rate,
            "max_epochs": config.max_epochs,
        },
        "training": {
            "total_steps": step,
            "best_val_loss": best_val_loss,
            "final_val_loss": last_val_loss,
            "total_time_s": round(time.time() - start_time, 1),
        },
        "log_entries": log_entries,
    }
    tmp_log = TRAIN_LOG + ".tmp"
    with open(tmp_log, "w") as f:
        json.dump(log_data, f, indent=2)
    os.replace(tmp_log, TRAIN_LOG)

    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"  Steps: {step}, Best val_loss: {best_val_loss:.4f}")
    print(f"  Checkpoints: {CONV_CKPT}")
    print(f"               {CONV_BEST_CKPT}")
    print(f"  Tokenizer: {CONV_TOKENIZER}")
    print(f"  Log: {TRAIN_LOG}")
    print(f"{'=' * 60}")
    return log_data


def show_info():
    config = get_conversation_config()
    documents = build_corpus()
    conv_docs = [d for d in documents if d.category == "conversation"]
    print(f"Total documents: {len(documents)}")
    print(f"Conversation documents: {len(conv_docs)}")
    print(f"Total chars: {sum(len(d.text) for d in documents):,}")
    print(f"Conv chars: {sum(len(d.text) for d in conv_docs):,}")
    print(f"\nConfig: vocab={config.bpe_vocab_size}, ctx={config.context_length}, "
          f"layers={config.n_layers}, dim={config.hidden_dim}")

    print(f"\nCheckpoints:")
    for name, path in [("Specialized Best", BEST_SPECIALIZED),
                       ("Specialized", SPECIALIZED_CKPT),
                       ("Conversation", CONV_CKPT),
                       ("Conversation Best", CONV_BEST_CKPT)]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"  {name}: {path} ({size_mb:.1f} MB)")
        else:
            print(f"  {name}: not found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conversation fine-tuning")
    parser.add_argument("--resume", action="store_true", help="Resume from conversation checkpoint")
    parser.add_argument("--info", action="store_true", help="Show data stats only")
    args = parser.parse_args()

    if args.info:
        show_info()
    else:
        config = get_conversation_config()
        run_training(config, resume=args.resume)
