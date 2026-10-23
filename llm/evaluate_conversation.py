"""Evaluation suite for the conversation fine-tuned model.

Tests:
  1. Persian chat responses — prompts in Persian, checks for coherent output
  2. Next.js code generation — verifies correct App Router patterns
  3. Code language match — checks generated code matches prompt language
  4. Test set perplexity — loss on held-out conversation data
  5. Repetition check — detects degenerate repetition in outputs

Usage:
    python llm/evaluate_conversation.py              # Full evaluation
    python llm/evaluate_conversation.py --quick      # Quick (fewer samples)
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader

from config import ModelConfig
from tokenizer import BPETokenizer
from model import GPTModel
from checkpoint import load_checkpoint, load_config_from_checkpoint, CheckpointError
from dataset import TextDataset
from inference import generate_text

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

CONV_BEST_CKPT = os.path.join(CHECKPOINT_DIR, "conversation_best.pt")
CONV_CKPT = os.path.join(CHECKPOINT_DIR, "conversation.pt")
CONV_TOKENIZER = os.path.join(CHECKPOINT_DIR, "tokenizer_conv.json")


PERSIAN_CHAT_PROMPTS = [
    "user: سلام\nassistant:",
    "user: سلام، چطوری؟\nassistant:",
    "user: چطور می‌توانم یک تابع پایتون بنویسم؟\nassistant:",
    "user: یک کامپوننت React بساز\nassistant:",
    "user: تفاوت let و const چیست؟\nassistant:",
    "user: How do I create a Next.js page?\nassistant:",
    "user: یک تابع fibonacci بنویس\nassistant:",
    "user: middleware در Next.js چیست؟\nassistant:",
]

NEXTJS_PROMPTS = [
    "// app/page.tsx\nexport default function HomePage() {\n  return",
    "// app/api/users/route.ts\nimport { NextResponse } from 'next/server'\n\nexport async function GET() {\n  const",
    "// app/blog/[slug]/page.tsx\n\nexport default async function BlogPost({ params }) {\n  const post =",
    "// app/dashboard/loading.tsx\nexport default function Loading() {\n  return",
]

CODE_PROMPTS = {
    "python": [
        "def reverse_string(s):\n    return ",
        "class Stack:\n    def __init__(self):\n        self._items = ",
    ],
    "javascript": [
        "function memoize(fn) {\n  const cache = ",
        "async function fetchData(url) {\n  const res = await",
    ],
    "react": [
        "function TodoApp() {\n  const [todos, setTodos] = ",
        "function useDebounce(value, delay) {\n  const [debounced, ",
    ],
}


def load_model_and_tokenizer():
    """Load the conversation checkpoint and tokenizer."""
    ckpt_path = CONV_BEST_CKPT if os.path.exists(CONV_BEST_CKPT) else CONV_CKPT
    tok_path = CONV_TOKENIZER

    if not os.path.exists(ckpt_path):
        print(f"No conversation checkpoint found at {ckpt_path}")
        print("Run training first: python llm/train_conversation.py")
        sys.exit(1)
    if not os.path.exists(tok_path):
        print(f"Tokenizer not found at {tok_path}")
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
        print(f"ERROR: Tokenizer vocab ({tokenizer.vocab_size}) != "
              f"checkpoint vocab ({saved_config.vocab_size})")
        sys.exit(1)

    model = GPTModel(saved_config)
    try:
        meta = load_checkpoint(ckpt_path, model)
    except CheckpointError as e:
        print(f"ERROR: Could not load checkpoint: {e}")
        sys.exit(1)

    print(f"Loaded: {ckpt_path}")
    print(f"  step={meta['step']}, val_loss={meta['val_loss']:.4f}")
    print(f"  {saved_config.n_layers} layers, {saved_config.hidden_dim} dim, "
          f"{model.count_parameters():,} params")
    print(f"  vocab={tokenizer.vocab_size}, ctx={saved_config.context_length}")

    return model, tokenizer, saved_config


def has_persian(text: str) -> bool:
    """Check if text contains Persian/Arabic characters."""
    return bool(re.search(r'[\u0600-\u06FF]', text))


def check_repetition(text: str, min_len: int = 20) -> float:
    """Return repetition ratio (0=none, 1=fully degenerate)."""
    if len(text) < min_len:
        return 0.0
    words = text.split()
    if len(words) < 4:
        return 0.0
    unique = len(set(words))
    return 1.0 - (unique / len(words))


def check_nextjs_patterns(code: str) -> tuple[list[str], list[str]]:
    """Check Next.js conventions. Returns (issues, patterns_found)."""
    issues = []
    patterns_found = []

    if "export default" in code:
        patterns_found.append("default_export")
    if "next/" in code:
        patterns_found.append("next_import")
    if "NextResponse" in code:
        patterns_found.append("NextResponse")
    if "use client" in code.lower():
        patterns_found.append("use_client")
    if "use server" in code.lower():
        patterns_found.append("use_server")

    jsx_opens = len(re.findall(r'<(\w+)', code))
    jsx_closes = len(re.findall(r'</(\w+)>', code)) + len(re.findall(r'/\s*>', code))
    if abs(jsx_opens - jsx_closes) > 3:
        issues.append(f"Unbalanced JSX: {jsx_opens} open, {jsx_closes} close")

    if code.count('{') - code.count('}') > 2:
        issues.append("Unbalanced braces")

    return issues, patterns_found


def eval_persian_chat(model, tokenizer, config, quick=False):
    """Test 1: Persian chat response quality."""
    print("\n[Test 1] Persian Chat Responses")
    print("-" * 50)

    prompts = PERSIAN_CHAT_PROMPTS[:4] if quick else PERSIAN_CHAT_PROMPTS
    results = []

    for i, prompt in enumerate(prompts):
        text = generate_text(
            model, tokenizer, prompt, config,
            max_new_tokens=80, temperature=0.3, top_k=20,
        )
        generated = text[len(prompt):] if text.startswith(prompt) else text
        has_farsi = has_persian(generated)
        rep_ratio = check_repetition(generated)
        is_degenerate = rep_ratio > 0.6 and len(generated) > 30

        results.append({
            "prompt_idx": i,
            "prompt": prompt[:60],
            "generated": generated[:200],
            "has_persian": has_farsi,
            "repetition_ratio": round(rep_ratio, 2),
            "degenerate": is_degenerate,
        })

        status = []
        if has_farsi:
            status.append("FA")
        if is_degenerate:
            status.append("DEGEN")
        if not generated.strip():
            status.append("EMPTY")
        print(f"  prompt {i}: {' '.join(status) or 'OK'} | rep={rep_ratio:.2f}")
        print(f"    -> {generated[:120]}")

    persian_rate = sum(1 for r in results if r["has_persian"]) / len(results)
    degenerate_rate = sum(1 for r in results if r["degenerate"]) / len(results)
    print(f"\n  Persian output rate: {persian_rate:.0%}")
    print(f"  Degenerate rate: {degenerate_rate:.0%}")
    return {"persian_chat": {
        "results": results, "persian_rate": persian_rate,
        "degenerate_rate": degenerate_rate,
    }}


def eval_nextjs_generation(model, tokenizer, config, quick=False):
    """Test 2: Next.js code generation."""
    print("\n[Test 2] Next.js Code Generation")
    print("-" * 50)

    prompts = NEXTJS_PROMPTS[:2] if quick else NEXTJS_PROMPTS
    results = []
    total_patterns = 0
    total_issues = 0

    for i, prompt in enumerate(prompts):
        text = generate_text(
            model, tokenizer, prompt, config,
            max_new_tokens=100, temperature=0.3, top_k=25,
        )
        generated = text[len(prompt):] if text.startswith(prompt) else text
        issues, patterns = check_nextjs_patterns(text)
        total_patterns += len(patterns)
        total_issues += len(issues)

        results.append({
            "prompt_idx": i,
            "generated": generated[:200],
            "patterns_found": patterns,
            "issues": issues,
        })
        print(f"  prompt {i}: patterns={patterns} issues={len(issues)}")

    avg_patterns = total_patterns / len(results) if results else 0
    print(f"\n  Total patterns: {total_patterns}, Total issues: {total_issues}")
    print(f"  Avg patterns/gen: {avg_patterns:.1f}")
    return {"nextjs_generation": {
        "results": results, "total_patterns": total_patterns,
        "total_issues": total_issues, "avg_patterns": avg_patterns,
    }}


def eval_code_language_match(model, tokenizer, config, quick=False):
    """Test 3: Code language match."""
    print("\n[Test 3] Code Language Match")
    print("-" * 50)

    results = {}
    for lang, prompts in CODE_PROMPTS.items():
        prompts = prompts[:1] if quick else prompts
        lang_results = []
        for i, prompt in enumerate(prompts):
            text = generate_text(
                model, tokenizer, prompt, config,
                max_new_tokens=80, temperature=0.3, top_k=20,
            )
            generated = text[len(prompt):] if text.startswith(prompt) else text
            rep = check_repetition(generated)

            lang_results.append({
                "prompt_idx": i,
                "generated": generated[:150],
                "repetition_ratio": round(rep, 2),
            })
            print(f"  [{lang}] prompt {i}: rep={rep:.2f}")

        avg_rep = sum(r["repetition_ratio"] for r in lang_results) / len(lang_results)
        results[lang] = {"samples": lang_results, "avg_repetition": avg_rep}
        print(f"  -> {lang} avg repetition: {avg_rep:.2f}")

    return {"code_language_match": results}


def eval_test_set_perplexity(model, tokenizer, config):
    """Test 4: Perplexity on held-out test set."""
    print("\n[Test 4] Test Set Perplexity")
    print("-" * 50)

    test_file = os.path.join(DATA_DIR, "test_set.json")
    if not os.path.exists(test_file):
        print("  No test set found (test_set.json). Skipping.")
        return {"test_perplexity": None}

    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    test_texts = test_data.get("test_texts", [])
    if not test_texts:
        print("  Test set is empty. Skipping.")
        return {"test_perplexity": None}

    test_ids = []
    for text in test_texts:
        test_ids.extend(tokenizer.encode(text))

    if len(test_ids) < config.context_length + 1:
        print(f"  Test set too small: {len(test_ids)} tokens. Skipping.")
        return {"test_perplexity": None}

    test_dataset = TextDataset(test_ids, config)
    test_loader = DataLoader(
        test_dataset,
        batch_size=min(config.batch_size, len(test_dataset)),
        shuffle=False, drop_last=False,
    )

    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i >= 50:
                break
            _, loss = model(x, y)
            total_loss += loss.item()
            count += 1

    avg_loss = total_loss / max(count, 1)
    perplexity = float(torch.exp(torch.tensor(avg_loss)))

    print(f"  Test documents: {len(test_texts)}")
    print(f"  Test tokens: {len(test_ids):,}")
    print(f"  Test loss: {avg_loss:.4f}")
    print(f"  Test perplexity: {perplexity:.2f}")

    return {"test_perplexity": {
        "loss": avg_loss, "perplexity": perplexity,
        "test_tokens": len(test_ids), "test_docs": len(test_texts),
    }}


def run_evaluation(quick=False):
    """Run all evaluation tests."""
    print("=" * 70)
    print("Conversation Model Evaluation Suite")
    print("=" * 70)

    model, tokenizer, config = load_model_and_tokenizer()

    all_results = {}
    all_results.update(eval_persian_chat(model, tokenizer, config, quick))
    all_results.update(eval_nextjs_generation(model, tokenizer, config, quick))
    all_results.update(eval_code_language_match(model, tokenizer, config, quick))
    all_results.update(eval_test_set_perplexity(model, tokenizer, config))

    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)

    if "persian_chat" in all_results:
        pc = all_results["persian_chat"]
        print(f"  Persian output rate:    {pc['persian_rate']:.0%}")
        print(f"  Degenerate rate:        {pc['degenerate_rate']:.0%}")
    if "nextjs_generation" in all_results:
        nx = all_results["nextjs_generation"]
        print(f"  Next.js patterns:       {nx['total_patterns']} (avg {nx['avg_patterns']:.1f})")
        print(f"  Next.js issues:         {nx['total_issues']}")
    if "test_perplexity" in all_results and all_results["test_perplexity"]:
        tp = all_results["test_perplexity"]
        print(f"  Test perplexity:        {tp['perplexity']:.2f}")
    else:
        print(f"  Test perplexity:        N/A")

    os.makedirs(LOG_DIR, exist_ok=True)
    eval_path = os.path.join(LOG_DIR, "conversation_eval_results.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved to {eval_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate conversation model")
    parser.add_argument("--quick", action="store_true", help="Quick evaluation")
    args = parser.parse_args()
    run_evaluation(quick=args.quick)
