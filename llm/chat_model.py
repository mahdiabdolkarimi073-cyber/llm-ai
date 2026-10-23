"""Chat model engine: loads the real trained model and tokenizer, provides
token-by-token streaming generation with stop control and error handling.

This module is independent of training code. It only reads checkpoints and
tokenizer files — it never modifies them.
"""

import os
import sys
import threading
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig
from model import GPTModel
from tokenizer import BPETokenizer
from checkpoint import load_checkpoint, load_config_from_checkpoint, CheckpointError


CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

AVAILABLE_CHECKPOINTS = {
    "specialized_best": {
        "model": os.path.join(CHECKPOINT_DIR, "specialized_best.pt"),
        "tokenizer": os.path.join(CHECKPOINT_DIR, "tokenizer_v2.json"),
        "label": "مدل تخصصی (بهترین)",
    },
    "specialized": {
        "model": os.path.join(CHECKPOINT_DIR, "specialized.pt"),
        "tokenizer": os.path.join(CHECKPOINT_DIR, "tokenizer_v2.json"),
        "label": "مدل تخصصی",
    },
    "phase1": {
        "model": os.path.join(CHECKPOINT_DIR, "model.pt"),
        "tokenizer": os.path.join(CHECKPOINT_DIR, "tokenizer.json"),
        "label": "مدل پایه (فاز ۱)",
    },
}


class ModelLoadError(Exception):
    pass


class ChatModelEngine:
    """Loads a trained model + tokenizer and generates text token-by-token.

    Generation runs on a background thread so the UI stays responsive.
    A stop flag allows safe interruption mid-generation.
    """

    def __init__(self):
        self.model: GPTModel | None = None
        self.tokenizer: BPETokenizer | None = None
        self.config: ModelConfig | None = None
        self.checkpoint_name: str | None = None
        self._stop_flag = threading.Event()
        self._lock = threading.Lock()
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def list_available(self) -> list[dict]:
        """Return list of checkpoints that exist on disk with their status."""
        results = []
        for key, info in AVAILABLE_CHECKPOINTS.items():
            model_exists = os.path.exists(info["model"])
            tok_exists = os.path.exists(info["tokenizer"])
            results.append({
                "key": key,
                "label": info["label"],
                "model_path": info["model"],
                "tokenizer_path": info["tokenizer"],
                "model_exists": model_exists,
                "tokenizer_exists": tok_exists,
                "available": model_exists and tok_exists,
            })
        return results

    def load(self, checkpoint_key: str = "specialized_best") -> dict:
        """Load model and tokenizer from checkpoint files.

        Returns metadata dict with step, val_loss, params count.
        Raises ModelLoadError on any failure.
        """
        if checkpoint_key not in AVAILABLE_CHECKPOINTS:
            raise ModelLoadError(f"چک‌پوینت ناشناخته: {checkpoint_key}")

        info = AVAILABLE_CHECKPOINTS[checkpoint_key]
        ckpt_path = info["model"]
        tok_path = info["tokenizer"]

        if not os.path.exists(ckpt_path):
            raise ModelLoadError(
                f"فایل چک‌پوینت پیدا نشد: {ckpt_path}\n"
                f"ابتدا مدل را آموزش دهید."
            )
        if not os.path.exists(tok_path):
            raise ModelLoadError(
                f"فایل توکنایزر پیدا نشد: {tok_path}\n"
                f"ابتدا مدل را آموزش دهید."
            )

        try:
            saved_config = load_config_from_checkpoint(ckpt_path)
        except CheckpointError as e:
            raise ModelLoadError(f"خطا در خواندن تنظیمات چک‌پوینت: {e}") from e

        if saved_config is None:
            raise ModelLoadError(
                "چک‌پوینت فاقد تنظیمات است. نمی‌توان مدل را بارگذاری کرد."
            )

        tokenizer = BPETokenizer(vocab_size=saved_config.vocab_size)
        try:
            tokenizer.load(tok_path)
        except Exception as e:
            raise ModelLoadError(f"خطا در بارگذاری توکنایزر: {e}") from e

        if tokenizer.vocab_size != saved_config.vocab_size:
            raise ModelLoadError(
                f"عدم تطابق اندازه واژگان: توکنایزر={tokenizer.vocab_size}, "
                f"چک‌پوینت={saved_config.vocab_size}"
            )

        saved_config.vocab_size = tokenizer.vocab_size

        model = GPTModel(saved_config)
        try:
            meta = load_checkpoint(ckpt_path, model)
        except CheckpointError as e:
            raise ModelLoadError(f"خطا در بارگذاری وزن‌های مدل: {e}") from e

        model.eval()

        with self._lock:
            self.model = model
            self.tokenizer = tokenizer
            self.config = saved_config
            self.checkpoint_name = checkpoint_key
            self._loaded = True

        return {
            "step": meta.get("step", 0),
            "val_loss": meta.get("val_loss", 0.0),
            "params": model.count_parameters(),
            "vocab_size": tokenizer.vocab_size,
            "context_length": saved_config.context_length,
            "n_layers": saved_config.n_layers,
            "hidden_dim": saved_config.hidden_dim,
            "checkpoint_label": AVAILABLE_CHECKPOINTS[checkpoint_key]["label"],
        }

    def unload(self):
        with self._lock:
            self.model = None
            self.tokenizer = None
            self.config = None
            self.checkpoint_name = None
            self._loaded = False

    def stop(self):
        """Request the running generation to stop at the next token."""
        self._stop_flag.set()

    def _is_stopped(self) -> bool:
        return self._stop_flag.is_set()

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_k: int = 40,
    ):
        """Generator that yields text chunks token-by-token.

        Yields dicts:
            {"type": "token", "text": "..."}  — a decoded chunk
            {"type": "done", "text": "..."}   — full text at the end
            {"type": "error", "text": "..."}  — error message
            {"type": "stopped", "text": "..."} — generation was stopped

        The caller is responsible for UI updates on each yield.
        """
        self._stop_flag.clear()

        with self._lock:
            if not self._loaded or self.model is None or self.tokenizer is None:
                yield {"type": "error", "text": "مدل بارگذاری نشده است."}
                return
            model = self.model
            tokenizer = self.tokenizer
            config = self.config

        if not prompt.strip():
            yield {"type": "error", "text": "پیام خالی است."}
            return

        try:
            ids = tokenizer.encode(prompt)
        except Exception as e:
            yield {"type": "error", "text": f"خطا در رمزگذاری متن: {e}"}
            return

        if len(ids) == 0:
            ids = [0]

        max_ctx = config.context_length
        if len(ids) > max_ctx:
            ids = ids[-max_ctx:]

        idx = torch.tensor([ids], dtype=torch.long, device="cpu")

        temp = max(temperature, 1e-8)
        k = top_k if top_k is not None and top_k > 0 else config.top_k
        max_new = max_new_tokens or config.max_new_tokens

        generated_ids = list(ids)
        full_text_parts = []
        import torch.nn.functional as F

        try:
            with torch.no_grad():
                for _ in range(max_new):
                    if self._is_stopped():
                        full = prompt + "".join(full_text_parts)
                        yield {"type": "stopped", "text": full}
                        return

                    idx_cond = idx if idx.size(1) <= max_ctx else idx[:, -max_ctx:]
                    logits, _ = model(idx_cond)
                    logits = logits[:, -1, :] / temp

                    if k is not None and k > 0:
                        v, _ = torch.topk(logits, min(k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = float('-inf')

                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    idx = torch.cat([idx, next_token], dim=1)

                    next_id = next_token[0].item()
                    generated_ids.append(next_id)

                    # Decode incrementally — decode full sequence and diff
                    new_text = tokenizer.decode(generated_ids)
                    if new_text:
                        prev_len = len(prompt) + sum(len(p) for p in full_text_parts)
                        chunk = new_text[prev_len:]
                        if chunk:
                            full_text_parts.append(chunk)
                            yield {"type": "token", "text": chunk}

        except RuntimeError as e:
            yield {"type": "error", "text": f"خطا در تولید متن: {e}"}
            return
        except Exception as e:
            yield {"type": "error", "text": f"خطای غیرمنتظره: {e}"}
            return

        full = prompt + "".join(full_text_parts)
        yield {"type": "done", "text": full}

    def get_status(self) -> dict:
        """Return current model status for UI display."""
        if not self._loaded:
            return {"loaded": False, "available": self.list_available()}
        with self._lock:
            return {
                "loaded": True,
                "checkpoint": self.checkpoint_name,
                "label": AVAILABLE_CHECKPOINTS.get(
                    self.checkpoint_name, {}
                ).get("label", ""),
                "vocab_size": self.tokenizer.vocab_size if self.tokenizer else 0,
                "context_length": self.config.context_length if self.config else 0,
                "params": self.model.count_parameters() if self.model else 0,
            }
