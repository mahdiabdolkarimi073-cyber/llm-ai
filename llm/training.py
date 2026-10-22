"""Training loop with real backpropagation, gradient clipping, and loss tracking.

Features:
  - AdamW optimizer with weight decay
  - Gradient clipping
  - Training and validation loss tracking
  - Periodic evaluation
  - Learning rate warmup
  - Atomic checkpoint saving (temp file + rename)
  - Safe error handling: on failure, existing checkpoint is preserved
"""

import math
import os
import time

import torch
from torch.utils.data import DataLoader

from config import ModelConfig
from checkpoint import save_checkpoint, load_checkpoint, CheckpointError
from dataset import TextDataset


def get_lr(step: int, config: ModelConfig) -> float:
    """Linear warmup followed by cosine decay."""
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    decay_steps = config.max_epochs * 1000
    if step > decay_steps:
        return config.learning_rate * 0.1
    return config.learning_rate * 0.5 * (1 + math.cos(
        math.pi * (step - config.warmup_steps) / max(decay_steps - config.warmup_steps, 1)
    ))


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, loader: DataLoader, device: str, max_iters: int = 50) -> float:
    """Compute average loss on a data loader."""
    model.eval()
    total_loss = 0.0
    count = 0
    for i, (x, y) in enumerate(loader):
        if i >= max_iters:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


def train(
    model: torch.nn.Module,
    train_dataset: TextDataset,
    val_dataset: TextDataset,
    config: ModelConfig,
    checkpoint_name: str = "model.pt",
    resume: bool = False,
):
    """Run the full training loop on CPU.

    Uses atomic checkpoint saves. On error or interruption, the last valid
    checkpoint is preserved.

    Returns a history list with train/val losses per eval step.
    """
    device = "cpu"
    model = model.to(device)
    model.train()

    # AdamW optimizer — decay only weight matrices, not biases/layernorm
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate, betas=(0.9, 0.95))

    # drop_last only if we have enough samples; otherwise keep all
    drop_last_train = len(train_dataset) > config.batch_size * 2
    train_loader = DataLoader(
        train_dataset, batch_size=min(config.batch_size, len(train_dataset)),
        shuffle=True, drop_last=drop_last_train
    )
    val_loader = DataLoader(
        val_dataset, batch_size=min(config.batch_size, len(val_dataset)),
        shuffle=False, drop_last=False
    )

    step = 0
    epoch = 0
    history: list[dict] = []

    ckpt_path = os.path.join(config.checkpoint_dir, checkpoint_name)

    # Resume from checkpoint
    if resume and os.path.exists(ckpt_path):
        try:
            meta = load_checkpoint(ckpt_path, model, optimizer)
            step = meta["step"]
            epoch = meta["epoch"]
            print(f"Resumed from checkpoint: step={step}, epoch={epoch}, "
                  f"train_loss={meta['train_loss']:.4f}, val_loss={meta['val_loss']:.4f}")
        except CheckpointError as e:
            print(f"ERROR: Could not resume from checkpoint: {e}")
            print(f"Starting training from scratch. Existing checkpoint preserved at {ckpt_path}")
            step = 0
            epoch = 0

    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Starting training for {config.max_epochs} epochs...")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print("-" * 60)

    start_time = time.time()
    train_loss_val = 0.0
    val_loss = 0.0

    try:
        for ep in range(epoch, config.max_epochs):
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)

                # Learning rate schedule
                lr = get_lr(step, config)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                # Forward pass
                logits, loss = model(x, y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                # Update weights
                optimizer.step()

                step += 1

                # Periodic evaluation
                if step % config.eval_interval == 0:
                    val_loss = evaluate_loss(model, val_loader, device, config.eval_iters)
                    train_loss_val = loss.item()
                    elapsed = time.time() - start_time
                    print(f"  step {step:5d} | epoch {ep} | lr {lr:.2e} | "
                          f"train_loss {train_loss_val:.4f} | val_loss {val_loss:.4f} | "
                          f"{elapsed:.1f}s")

                    history.append({
                        "step": step,
                        "epoch": ep,
                        "train_loss": train_loss_val,
                        "val_loss": val_loss,
                        "lr": lr,
                    })

                    # Save checkpoint (atomic)
                    save_checkpoint(
                        ckpt_path, model, optimizer, step, ep,
                        train_loss_val, val_loss, config=config,
                    )

            epoch = ep + 1
            # End-of-epoch evaluation
            val_loss = evaluate_loss(model, val_loader, device, config.eval_iters)
            print(f"=== Epoch {ep} complete | val_loss: {val_loss:.4f} ===")

        # Final checkpoint (atomic)
        save_checkpoint(ckpt_path, model, optimizer, step, epoch, train_loss_val, val_loss, config=config)

    except KeyboardInterrupt:
        print(f"\n[KEYBOARD INTERRUPT] Stopping. Last checkpoint preserved at {ckpt_path}")
        print(f"Resume with the same command and --resume flag.")
    except RuntimeError as e:
        print(f"\nERROR during training: {e}")
        print(f"Last valid checkpoint is preserved at {ckpt_path}")
        raise

    print(f"\nTraining complete. Final step={step}, val_loss={val_loss:.4f}")
    print(f"Checkpoint saved to {ckpt_path}")

    return history
