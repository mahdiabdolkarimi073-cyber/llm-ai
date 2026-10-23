"""Standalone training script for the Persian conversation + Next.js model.

Trains a fresh model from scratch using only data from corpus_fa.py.
Uses its own independent tokenizer and checkpoint paths — does NOT
touch any existing checkpoint or tokenizer file.

Usage:
    python llm/train_conversation_fa.py            # Full training
    python llm/train_conversation_fa.py --resume   # Resume from checkpoint
    python llm/train_conversation_fa.py --info     # Show data stats only

The script:
  - Builds the Persian corpus from corpus_fa.py
  - Trains an independent BPE tokenizer with special tokens
  - Uses ConversationDataset for loss-masked training
  - Saves checkpoints to checkpoints/conversation_fa*.pt
  - Saves tokenizer to checkpoints/tokenizer_fa.json
  - Never overwrites existing files
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
from dataset import ConversationDataset
from checkpoint import (
    save_checkpoint,
    load_checkpoint,
    load_config_from_checkpoint,
    CheckpointError,
)
from corpus_fa import (
    build_corpus,
    get_special_tokens,
    serialize_conversation,
    USER_TOKEN,
    ASSISTANT_TOKEN,
    END_TOKEN,
)


CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

CONV_FA_CKPT = os.path.join(CHECKPOINT_DIR, "conversation_fa.pt")
CONV_FA_BEST_CKPT = os.path.join(CHECKPOINT_DIR, "conversation_fa_best.pt")
CONV_FA_TOKENIZER = os.path.join(CHECKPOINT_DIR, "tokenizer_fa.json")
TRAIN_LOG = os.path.join(LOG_DIR, "conversation_fa_training_log.json")

PATIENCE = 5


def get_fa_config() -> ModelConfig:
    """Config for the Persian conversation model."""
    return ModelConfig(
        vocab_size=2000,
        bpe_vocab_size=2000,
        context_length=128,
        n_layers=4,
        n_heads=4,
        hidden_dim=256,
        ff_dim=1024,
        batch_size=16,
        max_epochs=20,
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


def prepare_fa_data(config: ModelConfig) -> dict:
    """Build corpus, train independent tokenizer, encode conversations."""
    print("\n[1/3] Building Persian corpus...")
    documents = build_corpus()
    if not documents:
        raise RuntimeError(
            "corpus_fa.py is empty. Add conversation data to CONVERSATIONS "
            "and code to CODE_EXAMPLES before training."
        )
    print(f"  Documents: {len(documents)}")
    print(f"  Total chars: {sum(len(d.text) for d in documents):,}")

    # Serialize conversations into the special-token format
    conv_texts = [d.text for d in documents]
    # Also include raw text for tokenizer training (to cover all byte patterns)
    tok_train_texts = conv_texts[:]

    print("\n[2/3] Training independent BPE tokenizer with special tokens...")
    tokenizer = BPETokenizer(vocab_size=config.bpe_vocab_size)
    tokenizer.train(tok_train_texts, verbose=True)

    # Assign special token IDs after the BPE vocab
    special_tokens = get_special_tokens()
    next_id = tokenizer.vocab_size
    assigned: dict[str, int] = {}
    for token in special_tokens:
        assigned[token] = next_id
        next_id += 1
    tokenizer.add_special_tokens(assigned)
    print(f"  Tokenizer vocab (after special tokens): {tokenizer.vocab_size}")
    print(f"  Special tokens: {assigned}")

    config.vocab_size = tokenizer.vocab_size

    # Save tokenizer
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    tmp_tok = CONV_FA_TOKENIZER + ".tmp"
    tokenizer.save(tmp_tok)
    os.replace(tmp_tok, CONV_FA_TOKENIZER)
    print(f"  Tokenizer saved: {CONV_FA_TOKENIZER}")

    print("\n[3/3] Preparing conversation dataset with loss masking...")
    train_conv, val_conv = _split_conversations(conv_texts, val_frac=0.1)

    train_dataset = ConversationDataset(
        train_conv, tokenizer, config,
        assistant_token=ASSISTANT_TOKEN,
        end_token=END_TOKEN,
        user_token=USER_TOKEN,
    )
    val_dataset = ConversationDataset(
        val_conv, tokenizer, config,
        assistant_token=ASSISTANT_TOKEN,
        end_token=END_TOKEN,
        user_token=USER_TOKEN,
    )

    print(f"  Train conversations: {len(train_conv)}")
    print(f"  Val conversations: {len(val_conv)}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    return {
        "tokenizer": tokenizer,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "actual_vocab": tokenizer.vocab_size,
        "num_conversations": len(conv_texts),
    }


def _split_conversations(conv_texts: list[str], val_frac: float = 0.1) -> tuple[list[str], list[str]]:
    """Split conversations into train and validation."""
    if len(conv_texts) <= 1:
        return conv_texts, conv_texts
    split_idx = int(len(conv_texts) * (1 - val_frac))
    if split_idx == 0:
        split_idx = 1
    return conv_texts[:split_idx], conv_texts[split_idx:]


@torch.no_grad()
def evaluate_model(model, loader, max_iters: int = 20) -> float:
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


def get_lr(step: int, config: ModelConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    return config.learning_rate * 0.5 * (1 + math.cos(
        math.pi * min(step / (config.max_epochs * 500), 1.0)
    ))


def run_training(config: ModelConfig, resume: bool = False):
    try:
        data = prepare_fa_data(config)
    except RuntimeError as e:
        print(f"\nFATAL: {e}")
        return None

    tokenizer = data["tokenizer"]
    actual_vocab = data["actual_vocab"]
    train_dataset = data["train_dataset"]
    val_dataset = data["val_dataset"]

    if len(train_dataset) == 0:
        print("\nFATAL: No training samples produced. Check corpus data size.")
        return None

    start_step = 0
    best_val_loss = float('inf')

    if resume and os.path.exists(CONV_FA_CKPT):
        print("\n  Resuming from Persian conversation checkpoint...")
        saved_config = load_config_from_checkpoint(CONV_FA_CKPT)
        if saved_config and saved_config.vocab_size == actual_vocab:
            model = GPTModel(saved_config)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.learning_rate, betas=(0.9, 0.95),
                weight_decay=config.weight_decay,
            )
            meta = load_checkpoint(CONV_FA_CKPT, model, optimizer)
            start_step = meta["step"]
            best_val_loss = meta.get("val_loss", float('inf'))
            model.config = config
        else:
            print("  Checkpoint incompatible, starting fresh...")
            model = GPTModel(config)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.learning_rate, betas=(0.9, 0.95),
                weight_decay=config.weight_decay,
            )
    else:
        print("\n  Training from scratch...")
        model = GPTModel(config)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, betas=(0.9, 0.95),
            weight_decay=config.weight_decay,
        )

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
        batch_size=min(config.batch_size, max(len(val_dataset), 1)),
        shuffle=False,
        drop_last=False,
    )

    step = start_step
    log_entries: list[dict] = []
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
                        CONV_FA_CKPT, model, optimizer, step, ep,
                        train_loss, val_loss, config=config,
                    )
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        no_improve_count = 0
                        save_checkpoint(
                            CONV_FA_BEST_CKPT, model, optimizer, step, ep,
                            train_loss, val_loss, config=config,
                        )
                        print(f"    -> New best val_loss: {val_loss:.4f} (saved)")
                    else:
                        no_improve_count += 1
                        if no_improve_count >= PATIENCE:
                            print(f"\n  [EARLY STOP] No improvement for {PATIENCE} evals.")
                            print(f"  Best val_loss: {best_val_loss:.4f}")
                            break

            if no_improve_count >= PATIENCE:
                break

            epoch_val_loss = evaluate_model(model, val_loader, config.eval_iters)
            last_val_loss = epoch_val_loss
            print(f"=== Epoch {ep} complete | val_loss: {epoch_val_loss:.4f} | "
                  f"best: {best_val_loss:.4f} ===")
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                save_checkpoint(
                    CONV_FA_BEST_CKPT, model, optimizer, step, ep,
                    last_train_loss, epoch_val_loss, config=config,
                )

        save_checkpoint(
            CONV_FA_CKPT, model, optimizer, step, config.max_epochs,
            last_train_loss, last_val_loss, config=config,
        )

    except KeyboardInterrupt:
        print(f"\n  [INTERRUPTED] Checkpoints preserved at {CONV_FA_CKPT}")
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
            "num_conversations": data["num_conversations"],
        },
        "log_entries": log_entries,
    }
    tmp_log = TRAIN_LOG + ".tmp"
    with open(tmp_log, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_log, TRAIN_LOG)

    print(f"\n{'=' * 60}")
    print(f"Persian conversation training complete!")
    print(f"  Steps: {step}, Best val_loss: {best_val_loss:.4f}")
    print(f"  Checkpoints: {CONV_FA_CKPT}")
    print(f"               {CONV_FA_BEST_CKPT}")
    print(f"  Tokenizer: {CONV_FA_TOKENIZER}")
    print(f"  Log: {TRAIN_LOG}")
    print(f"{'=' * 60}")
    return log_data


def show_info():
    config = get_fa_config()
    documents = build_corpus()
    print(f"Persian Conversation Model — Info")
    print(f"  Documents in corpus_fa: {len(documents)}")
    print(f"  Total chars: {sum(len(d.text) for d in documents):,}")
    print(f"\nConfig: vocab={config.bpe_vocab_size}, ctx={config.context_length}, "
          f"layers={config.n_layers}, dim={config.hidden_dim}")

    print(f"\nCheckpoints:")
    for name, path in [("Persian Conv", CONV_FA_CKPT),
                       ("Persian Conv Best", CONV_FA_BEST_CKPT)]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"  {name}: {path} ({size_mb:.1f} MB)")
        else:
            print(f"  {name}: not found")

    if not documents:
        print(f"\n  WARNING: corpus_fa.py has no data yet. "
              f"Add conversations before training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Persian conversation training")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--info", action="store_true", help="Show data stats only")
    args = parser.parse_args()

    if args.info:
        show_info()
    else:
        config = get_fa_config()
        run_training(config, resume=args.resume)
