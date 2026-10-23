"""Inference-only engine for the project's local GPT checkpoints."""

import os
import sys
import threading
from collections.abc import Iterator

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checkpoint import CheckpointError, load_checkpoint, load_config_from_checkpoint
from config import ModelConfig
from model import GPTModel
from tokenizer import BPETokenizer


CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

AVAILABLE_CHECKPOINTS = {
    "specialized_best": {
        "model": os.path.join(CHECKPOINT_DIR, "specialized_best.pt"),
        "tokenizer": os.path.join(CHECKPOINT_DIR, "tokenizer_v2.json"),
        "label": "مدل تخصصی، بهترین checkpoint",
    },
    "specialized": {
        "model": os.path.join(CHECKPOINT_DIR, "specialized.pt"),
        "tokenizer": os.path.join(CHECKPOINT_DIR, "tokenizer_v2.json"),
        "label": "مدل تخصصی",
    },
    "phase1": {
        "model": os.path.join(CHECKPOINT_DIR, "model.pt"),
        "tokenizer": os.path.join(CHECKPOINT_DIR, "tokenizer.json"),
        "label": "مدل پایه، فاز اول",
    },
}


class ModelLoadError(Exception):
    """Raised when a checkpoint cannot be used safely."""


class ChatModelEngine:
    """Load local weights and generate only model continuation text."""

    def __init__(self) -> None:
        self.model: GPTModel | None = None
        self.tokenizer: BPETokenizer | None = None
        self.config: ModelConfig | None = None
        self.checkpoint_name: str | None = None
        self._stop_flag = threading.Event()
        self._state_lock = threading.RLock()
        self._generation_lock = threading.Lock()
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        with self._state_lock:
            return self._loaded

    def list_available(self) -> list[dict]:
        results = []
        for key, info in AVAILABLE_CHECKPOINTS.items():
            model_exists = os.path.isfile(info["model"])
            tokenizer_exists = os.path.isfile(info["tokenizer"])
            results.append({
                "key": key,
                "label": info["label"],
                "model_path": info["model"],
                "tokenizer_path": info["tokenizer"],
                "model_exists": model_exists,
                "tokenizer_exists": tokenizer_exists,
                "available": model_exists and tokenizer_exists,
            })
        return results

    def load(self, checkpoint_key: str = "specialized_best") -> dict:
        if checkpoint_key not in AVAILABLE_CHECKPOINTS:
            raise ModelLoadError(f"چک‌پوینت ناشناخته است: {checkpoint_key}")

        info = AVAILABLE_CHECKPOINTS[checkpoint_key]
        if not os.path.isfile(info["model"]):
            raise ModelLoadError(f"فایل وزن مدل پیدا نشد: {info['model']}")
        if not os.path.isfile(info["tokenizer"]):
            raise ModelLoadError(f"فایل توکنایزر پیدا نشد: {info['tokenizer']}")

        try:
            saved_config = load_config_from_checkpoint(info["model"])
        except CheckpointError as error:
            raise ModelLoadError(f"خواندن تنظیمات checkpoint ناموفق بود: {error}") from error

        if saved_config is None:
            raise ModelLoadError("checkpoint تنظیمات معماری ندارد و قابل استفاده نیست.")

        tokenizer = BPETokenizer(vocab_size=saved_config.vocab_size)
        try:
            tokenizer.load(info["tokenizer"])
        except Exception as error:
            raise ModelLoadError(f"بارگذاری توکنایزر ناموفق بود: {error}") from error

        if tokenizer.vocab_size != saved_config.vocab_size:
            raise ModelLoadError(
                "تعداد توکن‌ها سازگار نیست: "
                f"tokenizer={tokenizer.vocab_size}, checkpoint={saved_config.vocab_size}"
            )

        model = GPTModel(saved_config)
        try:
            metadata = load_checkpoint(info["model"], model)
        except CheckpointError as error:
            raise ModelLoadError(f"بارگذاری وزن‌های مدل ناموفق بود: {error}") from error

        model.to("cpu")
        model.eval()
        with self._state_lock:
            self.model = model
            self.tokenizer = tokenizer
            self.config = saved_config
            self.checkpoint_name = checkpoint_key
            self._loaded = True

        return {
            "step": metadata.get("step", 0),
            "val_loss": metadata.get("val_loss", 0.0),
            "params": model.count_parameters(),
            "vocab_size": tokenizer.vocab_size,
            "context_length": saved_config.context_length,
            "n_layers": saved_config.n_layers,
            "hidden_dim": saved_config.hidden_dim,
            "checkpoint_label": info["label"],
        }

    def stop(self) -> None:
        self._stop_flag.set()

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 120,
        temperature: float = 0.35,
        top_k: int = 20,
        repetition_penalty: float = 1.08,
    ) -> Iterator[dict]:
        """Yield continuation chunks; the prompt is never included in output."""
        if not prompt.strip():
            yield {"type": "error", "text": "پیام خالی است."}
            return

        with self._state_lock:
            model = self.model
            tokenizer = self.tokenizer
            config = self.config
            loaded = self._loaded

        if not loaded or model is None or tokenizer is None or config is None:
            yield {"type": "error", "text": "مدل بارگذاری نشده است."}
            return
        if not self._generation_lock.acquire(blocking=False):
            yield {"type": "error", "text": "مدل در حال پاسخ‌گویی به درخواست دیگری است."}
            return

        self._stop_flag.clear()
        try:
            try:
                prompt_ids = tokenizer.encode(prompt)
            except Exception as error:
                yield {"type": "error", "text": f"رمزگذاری پیام ناموفق بود: {error}"}
                return

            if not prompt_ids:
                prompt_ids = [0]
            prompt_ids = prompt_ids[-config.context_length:]
            idx = torch.tensor([prompt_ids], dtype=torch.long, device="cpu")
            generated_ids: list[int] = []
            previous_text = ""
            temperature_value = max(float(temperature), 0.05)
            token_limit = max(1, min(int(max_new_tokens), 256))
            top_k_value = max(1, min(int(top_k), config.vocab_size))

            with torch.inference_mode():
                for _ in range(token_limit):
                    if self._stop_flag.is_set():
                        yield {"type": "stopped", "text": previous_text}
                        return

                    context = idx[:, -config.context_length:]
                    logits, _ = model(context)
                    next_logits = logits[:, -1, :] / temperature_value

                    if repetition_penalty > 1.0 and generated_ids:
                        used_ids = torch.tensor(generated_ids, dtype=torch.long)
                        used_logits = next_logits[0, used_ids]
                        next_logits[0, used_ids] = torch.where(
                            used_logits < 0,
                            used_logits * repetition_penalty,
                            used_logits / repetition_penalty,
                        )

                    values, _ = torch.topk(next_logits, top_k_value)
                    next_logits[next_logits < values[:, [-1]]] = float("-inf")
                    probabilities = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probabilities, num_samples=1)
                    idx = torch.cat([idx, next_token], dim=1)
                    generated_ids.append(int(next_token.item()))

                    current_text = tokenizer.decode(generated_ids)
                    chunk = current_text[len(previous_text):]
                    if chunk:
                        previous_text = current_text
                        yield {"type": "token", "text": chunk}

            yield {"type": "done", "text": previous_text}
        except RuntimeError as error:
            yield {"type": "error", "text": f"خطای تولید متن: {error}"}
        except Exception as error:
            yield {"type": "error", "text": f"خطای غیرمنتظره: {error}"}
        finally:
            self._generation_lock.release()

    def get_status(self) -> dict:
        with self._state_lock:
            if not self._loaded or self.model is None or self.tokenizer is None or self.config is None:
                return {"loaded": False, "available": self.list_available()}
            return {
                "loaded": True,
                "checkpoint": self.checkpoint_name,
                "label": AVAILABLE_CHECKPOINTS[self.checkpoint_name]["label"],
                "vocab_size": self.tokenizer.vocab_size,
                "context_length": self.config.context_length,
                "params": self.model.count_parameters(),
            }
