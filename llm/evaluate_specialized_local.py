"""Lightweight local inference test for the specialized model.

Loads specialized_best.pt with tokenizer_v2.json, runs short generation
on Persian and programming prompts, and prints output clearly.
Supports configurable generation length via CLI args.

Usage:
    python3 llm/evaluate_specialized_local.py
    python3 llm/evaluate_specialized_local.py --max-tokens 30
    python3 llm/evaluate_specialized_local.py --temperature 0.5 --top-k 15

CPU only. No external API calls.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from config import ModelConfig
from tokenizer import BPETokenizer
from model import GPTModel
from checkpoint import load_checkpoint, load_config_from_checkpoint, CheckpointError
from inference import generate_text

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

BEST_CKPT = os.path.join(CHECKPOINT_DIR, "specialized_best.pt")
FALLBACK_CKPT = os.path.join(CHECKPOINT_DIR, "specialized.pt")
TOKENIZER_PATH = os.path.join(CHECKPOINT_DIR, "tokenizer_v2.json")

TEST_PROMPTS: list[dict] = [
    {"category": "Persian conversation", "prompt": "user: سلام\nassistant:"},
    {"category": "Persian conversation", "prompt": "user: چطور می‌توانم یک تابع پایتون بنویسم؟\nassistant:"},
    {"category": "Next.js App Router", "prompt": "// app/page.tsx\nexport default function HomePage() {\n  return"},
    {"category": "Next.js App Router", "prompt": "// app/api/users/route.ts\nimport { NextResponse } from 'next/server'\n\nexport async function GET() {\n  const"},
    {"category": "React components", "prompt": "function TodoApp() {\n  const [todos, setTodos] = "},
    {"category": "TypeScript", "prompt": "interface User {\n  id: number;\n  name: string;\n  "},
    {"category": "Python", "prompt": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return "},
]


def load_model_and_tokenizer() -> tuple[GPTModel, BPETokenizer, ModelConfig]:
    """Load the best specialized checkpoint and matching tokenizer."""
    ckpt_path = BEST_CKPT if os.path.exists(BEST_CKPT) else FALLBACK_CKPT
    tok_path = TOKENIZER_PATH

    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    if not os.path.exists(tok_path):
        print(f"ERROR: Tokenizer not found: {tok_path}")
        sys.exit(1)

    try:
        saved_config = load_config_from_checkpoint(ckpt_path)
    except CheckpointError as e:
        print(f"ERROR: Could not read checkpoint config: {e}")
        sys.exit(1)
    if saved_config is None:
        saved_config = ModelConfig()

    tokenizer = BPETokenizer(vocab_size=saved_config.vocab_size)
    tokenizer.load(tok_path)

    if tokenizer.vocab_size != saved_config.vocab_size:
        print(f"ERROR: Vocab mismatch — tokenizer={tokenizer.vocab_size}, "
              f"checkpoint={saved_config.vocab_size}")
        sys.exit(1)

    model = GPTModel(saved_config)
    try:
        meta = load_checkpoint(ckpt_path, model)
    except CheckpointError as e:
        print(f"ERROR: Could not load checkpoint: {e}")
        sys.exit(1)

    model.eval()
    print(f"Checkpoint: {os.path.basename(ckpt_path)}")
    print(f"  step={meta['step']}, val_loss={meta['val_loss']:.4f}, "
          f"train_loss={meta['train_loss']:.4f}")
    print(f"  architecture: {saved_config.n_layers} layers, "
          f"{saved_config.n_heads} heads, dim={saved_config.hidden_dim}, "
          f"ff={saved_config.ff_dim}, ctx={saved_config.context_length}")
    print(f"  parameters: {model.count_parameters():,}")
    print(f"  vocab: {tokenizer.vocab_size} tokens")
    print()

    return model, tokenizer, saved_config


def run_inference_test(
    model: GPTModel,
    tokenizer: BPETokenizer,
    config: ModelConfig,
    max_tokens: int,
    temperature: float,
    top_k: int,
) -> bool:
    """Run inference on all test prompts. Returns True if all succeed."""
    all_ok = True

    for entry in TEST_PROMPTS:
        category = entry["category"]
        prompt = entry["prompt"]

        try:
            text = generate_text(
                model, tokenizer, prompt, config,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            generated = text[len(prompt):] if text.startswith(prompt) else text
            if not generated.strip():
                generated = "(empty output)"
        except Exception as e:
            generated = f"ERROR: {e}"
            all_ok = False

        print(f"[{category}]")
        print(f"  Prompt: {prompt[:80]}")
        print(f"  Output: {generated[:300]}")
        print()

    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lightweight local evaluation for specialized model"
    )
    parser.add_argument("--max-tokens", type=int, default=40,
                        help="Max tokens to generate per prompt (default: 40)")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature (default: 0.3)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Top-k sampling (default: 20)")
    args = parser.parse_args()

    print("=" * 60)
    print("Specialized Model — Local Inference Test")
    print("=" * 60)
    print()

    model, tokenizer, config = load_model_and_tokenizer()

    print(f"Generation settings: max_tokens={args.max_tokens}, "
          f"temperature={args.temperature}, top_k={args.top_k}")
    print("-" * 60)
    print()

    success = run_inference_test(
        model, tokenizer, config,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print("=" * 60)
    if success:
        print("RESULT: All inference runs completed without errors.")
    else:
        print("RESULT: Some inference runs encountered errors — see above.")
    print()

    print("LIMITATIONS:")
    print(f"  - Model has only {model.count_parameters():,} parameters (~0.6M)")
    print(f"  - Context length is {config.context_length} tokens")
    print(f"  - Vocab size is {tokenizer.vocab_size} — small for Persian text")
    print(f"  - Training was interrupted at step ~1107 due to increasing val_loss")
    print(f"  - Best val_loss was 5.42; final val_loss was 8.29 (overfitting)")
    print(f"  - Output will likely be repetitive or incoherent — this is expected")
    print(f"  - The model was not trained on Persian conversation data")
    print("=" * 60)
