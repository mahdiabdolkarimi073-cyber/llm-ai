"""Checkpoint save/load utilities with atomic writes and safe loading.

Saves full model state, optimizer state, training step, epoch, and config.
Uses atomic write (temp file + rename) so a killed process cannot corrupt
an existing valid checkpoint.

All load functions validate checkpoint structure and tensor shapes before
mutating the live model. On any mismatch they raise a clear error and leave
the model untouched.
"""

import os
import tempfile

import torch

from config import ModelConfig


class CheckpointError(Exception):
    """Raised when a checkpoint is missing, corrupt, or incompatible."""


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    train_loss: float,
    val_loss: float,
    config: ModelConfig | None = None,
):
    """Save a complete checkpoint atomically.

    Writes to a temporary file in the same directory, then renames it to
    the target path. This guarantees that *path* is either the previous
    valid checkpoint or the new one — never a half-written file.
    """
    dir_path = os.path.dirname(path) or "."
    os.makedirs(dir_path, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
    }
    if config is not None:
        checkpoint["config"] = {
            "vocab_size": config.vocab_size,
            "context_length": config.context_length,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "hidden_dim": config.hidden_dim,
            "ff_dim": config.ff_dim,
        }

    # Atomic write: temp file in same dir, then os.replace
    fd, tmp_path = tempfile.mkstemp(
        dir=dir_path, suffix=".tmp", prefix=".ckpt_"
    )
    os.close(fd)
    try:
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        # Clean up temp file on failure; leave existing checkpoint intact
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def _validate_checkpoint_structure(checkpoint: dict) -> None:
    """Verify a checkpoint dict has the required keys."""
    required = {"model_state_dict", "step", "epoch", "train_loss", "val_loss"}
    missing = required - set(checkpoint.keys())
    if missing:
        raise CheckpointError(
            f"Checkpoint missing required keys: {missing}"
        )


def _check_shape_compatibility(
    model: torch.nn.Module,
    state_dict: dict,
) -> list[str]:
    """Compare model parameter shapes against a state dict.

    Returns a list of mismatch descriptions (empty if all compatible).
    Does NOT mutate the model.
    """
    model_state = model.state_dict()
    mismatches = []

    all_keys = set(model_state.keys()) | set(state_dict.keys())
    for key in sorted(all_keys):
        if key not in state_dict:
            mismatches.append(f"'{key}': missing from checkpoint")
            continue
        if key not in model_state:
            mismatches.append(f"'{key}': in checkpoint but not in model")
            continue

        ckpt_shape = tuple(state_dict[key].shape)
        model_shape = tuple(model_state[key].shape)
        if ckpt_shape != model_shape:
            mismatches.append(
                f"'{key}': checkpoint shape {ckpt_shape} != model shape {model_shape}"
            )

    return mismatches


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    strict: bool = True,
) -> dict:
    """Load a checkpoint and restore model + optimizer state.

    Validates structure and shape compatibility BEFORE mutating the model.
    On any error, raises CheckpointError and leaves the model unchanged.

    Args:
        path: Path to checkpoint file.
        model: Model to load weights into.
        optimizer: If given, optimizer state is also restored.
        strict: If True, all keys must match exactly. If False, uses
            load_state_dict(strict=False) after shape validation of
            overlapping keys.

    Returns:
        Metadata dict with step, epoch, losses, and config.

    Raises:
        CheckpointError: If file is missing, corrupt, or shapes mismatch.
    """
    if not os.path.exists(path):
        raise CheckpointError(f"Checkpoint file not found: {path}")

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise CheckpointError(f"Failed to load checkpoint file {path}: {e}") from e

    _validate_checkpoint_structure(checkpoint)

    # Shape-check BEFORE mutating the model
    mismatches = _check_shape_compatibility(model, checkpoint["model_state_dict"])
    if mismatches:
        detail = "\n  ".join(mismatches)
        raise CheckpointError(
            f"Checkpoint shape mismatch ({len(mismatches)} issues):\n  {detail}"
        )

    # Safe to load now
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            # Optimizer state mismatch is non-fatal — warn but continue
            print(f"  WARNING: could not restore optimizer state: {e}")

    return {
        "step": checkpoint["step"],
        "epoch": checkpoint["epoch"],
        "train_loss": checkpoint["train_loss"],
        "val_loss": checkpoint["val_loss"],
        "config": checkpoint.get("config"),
    }


def load_config_from_checkpoint(path: str) -> ModelConfig | None:
    """Load just the config from a checkpoint, if present.

    Does not mutate any model. Raises CheckpointError on corrupt files.
    """
    if not os.path.exists(path):
        raise CheckpointError(f"Checkpoint file not found: {path}")

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise CheckpointError(f"Failed to load checkpoint file {path}: {e}") from e

    cfg_dict = checkpoint.get("config")
    if cfg_dict is None:
        return None

    required_keys = {"vocab_size", "context_length", "n_layers", "n_heads",
                     "hidden_dim", "ff_dim"}
    missing = required_keys - set(cfg_dict.keys())
    if missing:
        raise CheckpointError(
            f"Checkpoint config missing keys: {missing}"
        )

    config = ModelConfig()
    config.vocab_size = cfg_dict["vocab_size"]
    config.context_length = cfg_dict["context_length"]
    config.n_layers = cfg_dict["n_layers"]
    config.n_heads = cfg_dict["n_heads"]
    config.hidden_dim = cfg_dict["hidden_dim"]
    config.ff_dim = cfg_dict["ff_dim"]
    return config


def is_checkpoint_compatible(
    path: str,
    config: ModelConfig,
) -> tuple[bool, str]:
    """Check if a checkpoint is compatible with a given config.

    Returns (is_compatible, reason). Does not load the checkpoint into a model.
    """
    if not os.path.exists(path):
        return False, f"File not found: {path}"

    try:
        saved_config = load_config_from_checkpoint(path)
    except CheckpointError as e:
        return False, str(e)

    if saved_config is None:
        return False, "Checkpoint has no config metadata"

    checks = [
        ("n_layers", saved_config.n_layers, config.n_layers),
        ("n_heads", saved_config.n_heads, config.n_heads),
        ("hidden_dim", saved_config.hidden_dim, config.hidden_dim),
        ("ff_dim", saved_config.ff_dim, config.ff_dim),
    ]
    for name, old, new in checks:
        if old != new:
            return False, f"{name}: checkpoint={old}, config={new}"

    return True, "compatible"
