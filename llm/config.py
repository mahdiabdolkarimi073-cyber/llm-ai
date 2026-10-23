"""Model configuration - all hyperparameters in one place.

Edit values here to change model size, vocab, context length, etc.
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # Model architecture
    vocab_size: int = 8000
    context_length: int = 512
    n_layers: int = 4
    n_heads: int = 4
    hidden_dim: int = 256
    ff_dim: int = 1024

    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    dropout: float = 0.1
    batch_size: int = 32
    grad_clip: float = 1.0
    max_epochs: int = 50
    eval_interval: int = 100
    eval_iters: int = 20
    warmup_steps: int = 100
    early_stopping_patience: int = 5

    # Tokenizer
    bpe_vocab_size: int = 8000

    # Checkpoint
    checkpoint_dir: str = "checkpoints"

    # Inference
    temperature: float = 1.0
    top_k: int = 50
    max_new_tokens: int = 200

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.n_heads

    @property
    def total_params_estimate(self) -> int:
        """Rough parameter count estimate."""
        e = self.vocab_size * self.hidden_dim  # token embedding
        p = self.context_length * self.hidden_dim  # position embedding
        per_layer = 4 * self.hidden_dim * self.hidden_dim + 2 * self.hidden_dim * self.ff_dim + 3 * self.hidden_dim
        layers = self.n_layers * per_layer
        lm_head = self.hidden_dim * self.vocab_size
        ln = 4 * self.n_layers * self.hidden_dim
        return e + p + layers + lm_head + ln
