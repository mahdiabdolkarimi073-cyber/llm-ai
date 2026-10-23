"""Evaluation suite for the Persian conversation + Next.js model.

Tests:
  1. Persian chat responses — prompts in Persian with <|user|>/<|assistant|> format
  2. Next.js code generation — verifies App Router patterns
  3. Stop-at-<|end|> check — verifies generation stops at end token
  4. Code language match — checks generated code matches prompt language
  5. Test set perplexity — loss on held-out conversation data

Usage:
    python llm/evaluate_conversation_fa.py              # Full evaluation
    python llm/evaluate_conversation_fa.py --quick      # Quick (fewer samples)
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
from dataset import ConversationDataset
from inference import generate_text
from corpus_fa import (
    build_corpus,
    USER_TOKEN,
    ASSISTANT_TOKEN,
    END_TOKEN,
    serialize_conversation,
)

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

CONV_FA_BEST_CKPT = os.path.join(CHECKPOINT_DIR, "conversation_fa_best.pt")
CONV_FA_CKPT = os.path.join(CHECKPOINT_DIR, "conversation_fa.pt")
CONV_FA_TOKENIZER = os.path.join(CHECKPOINT_DIR, "tokenizer_fa.json")


PERSIAN_CHAT_PROMPTS = [
    f"{USER_TOKEN}سلام{END_TOKEN}{ASSISTANT_TOKEN}",
    f"{USER_TOKEN}سلام، چطوری؟{END_TOKEN}{ASSISTANT_TOKEN}",
    f"{USER_TOKEN}چطور می‌توانم یک تابع پایتون بنویسم؟{END_TOKEN}{ASSISTANT_TOKEN}",
    f"{USER_TOKEN}یک کامپوننت React بساز{END_TOKEN}{ASSISTANT_TOKEN}",
    f"{USER_TOKEN}تفاوت let و const چیست؟{END_TOKEN}{ASSISTANT_TOKEN}",
    f"{USER_TOKEN}یک تابع fibonacci بنویس{END_TOKEN}{ASSISTANT_TOKEN}",
    f"{USER_TOKEN}middleware در Next.js چیست؟{END_TOKEN}{ASSISTANT_TOKEN}",
    f"{USER_TOKEN}یک صفحه Next.js با TypeScript بساز{END_TOKEN}{ASSISTANT_TOKEN}",
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
    "typescript": [
        "interface User {\n  id: number;\n  name: ",
        "type Result<T> =\n  | { ok: true; value: ",
    ],
    "nextjs": [
        "// app/page.tsx\nexport default function Page() {\n  return <",
        "// app/api/route.ts\nexport async function POST(request: Request) {\n  const body = await",
    ],
}


def load_model_and_tokenizer():
    """Load the Persian conversation checkpoint and tokenizer."""
    ckpt_path = CONV_FA_BEST_CKPT if os.path.exists(CONV_FA_BEST_CKPT) else CONV_FA_CKPT
    tok_path = CONV_FA_TOKENIZER

    if not os.path.exists(ckpt_path):
        print(f"No Persian checkpoint found at {ckpt_path}")
        print("Run training first: python llm/train_conversation_fa.py")
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


def check_end_token_stop(text: str, end_token: str = END_TOKEN) -> bool:
    """Check if the generated text contains the <|end|> token."""
    return end_token in text


def check_nextjs_patterns(code: str) -> tuple[list[str], list[str]]:
    """Check Next.js conventions. Returns (issues, patterns_found)."""
    issues: list[str] = []
    patterns_found: list[str] = []

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


def eval_persian_chat(model, tokenizer, config, quick: bool = False):
    """Test 1: Persian chat response quality."""
    print("\n[Test 1] Persian Chat Responses")
    print("-" * 50)

    prompts = PERSIAN_CHAT_PROMPTS[:4] if quick else PERSIAN_CHAT_PROMPTS
    results: list[dict] = []

    for i, prompt in enumerate(prompts):
        text = generate_text(
            model, tokenizer, prompt, config,
            max_new_tokens=80, temperature=0.3, top_k=20,
        )
        generated = text[len(prompt):] if text.startswith(prompt) else text
        # Strip the end token from the displayed output
        clean = generated.replace(END_TOKEN, "").strip()
        has_farsi = has_persian(clean)
        rep_ratio = check_repetition(clean)
        is_degenerate = rep_ratio > 0.6 and len(clean) > 30
        stopped_at_end = check_end_token_stop(generated)

        results.append({
            "prompt_idx": i,
            "prompt": prompt[:80],
            "generated": clean[:200],
            "has_persian": has_farsi,
            "repetition_ratio": round(rep_ratio, 2),
            "degenerate": is_degenerate,
            "stopped_at_end": stopped_at_end,
        })

        status: list[str] = []
        if has_farsi:
            status.append("FA")
        if is_degenerate:
            status.append("DEGEN")
        if stopped_at_end:
            status.append("END")
        if not clean.strip():
            status.append("EMPTY")
        print(f"  prompt {i}: {' '.join(status) or 'OK'} | rep={rep_ratio:.2f}")
        print(f"    -> {clean[:120]}")

    persian_rate = sum(1 for r in results if r["has_persian"]) / len(results)
    degenerate_rate = sum(1 for r in results if r["degenerate"]) / len(results)
    end_stop_rate = sum(1 for r in results if r["stopped_at_end"]) / len(results)
    print(f"\n  Persian output rate: {persian_rate:.0%}")
    print(f"  Degenerate rate: {degenerate_rate:.0%}")
    print(f"  End-token stop rate: {end_stop_rate:.0%}")
    return {"persian_chat": {
        "results": results, "persian_rate": persian_rate,
        "degenerate_rate": degenerate_rate,
        "end_stop_rate": end_stop_rate,
    }}


def eval_nextjs_generation(model, tokenizer, config, quick: bool = False):
    """Test 2: Next.js code generation."""
    print("\n[Test 2] Next.js Code Generation")
    print("-" * 50)

    prompts = NEXTJS_PROMPTS[:2] if quick else NEXTJS_PROMPTS
    results: list[dict] = []
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


def eval_end_token_stopping(model, tokenizer, config):
    """Test 3: Verify generation stops at <|end|> token."""
    print("\n[Test 3] End-Token Stopping")
    print("-" * 50)

    end_id = tokenizer.special_token_id(END_TOKEN)
    if end_id is None:
        print("  <|end|> token not registered in tokenizer. Skipping.")
        return {"end_token_stopping": None}

    results: list[dict] = []
    test_prompts = PERSIAN_CHAT_PROMPTS[:4]

    for i, prompt in enumerate(test_prompts):
        text = generate_text(
            model, tokenizer, prompt, config,
            max_new_tokens=120, temperature=0.4, top_k=25,
        )
        generated = text[len(prompt):] if text.startswith(prompt) else text
        stopped = END_TOKEN in generated
        # Check if text after end token is empty or just trailing
        after_end = generated.split(END_TOKEN, 1)
        has_excess = stopped and len(after_end) > 1 and after_end[1].strip()

        results.append({
            "prompt_idx": i,
            "stopped_at_end": stopped,
            "has_excess_after_end": bool(has_excess),
        })
        status = "STOPPED" if stopped else "NO_STOP"
        if has_excess:
            status += "+EXCESS"
        print(f"  prompt {i}: {status}")

    stop_rate = sum(1 for r in results if r["stopped_at_end"]) / len(results)
    print(f"\n  End-token stop rate: {stop_rate:.0%}")
    return {"end_token_stopping": {
        "results": results, "stop_rate": stop_rate,
    }}


def eval_code_language_match(model, tokenizer, config, quick: bool = False):
    """Test 4: Code language match."""
    print("\n[Test 4] Code Language Match")
    print("-" * 50)

    results: dict[str, dict] = {}
    for lang, prompts in CODE_PROMPTS.items():
        prompts = prompts[:1] if quick else prompts
        lang_results: list[dict] = []
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
    """Test 5: Perplexity on held-out conversation data."""
    print("\n[Test 5] Test Set Perplexity")
    print("-" * 50)

    documents = build_corpus()
    if not documents:
        print("  Corpus is empty. Skipping.")
        return {"test_perplexity": None}

    # Use the last 10% of documents as a pseudo test set
    split_idx = int(len(documents) * 0.9)
    test_docs = documents[split_idx:]
    if not test_docs:
        test_docs = documents[-1:]

    test_texts = [d.text for d in test_docs]
    test_dataset = ConversationDataset(
        test_texts, tokenizer, config,
        assistant_token=ASSISTANT_TOKEN,
        end_token=END_TOKEN,
        user_token=USER_TOKEN,
    )

    if len(test_dataset) == 0:
        print("  Test set too small. Skipping.")
        return {"test_perplexity": None}

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
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Test loss: {avg_loss:.4f}")
    print(f"  Test perplexity: {perplexity:.2f}")

    return {"test_perplexity": {
        "loss": avg_loss, "perplexity": perplexity,
        "test_samples": len(test_dataset), "test_docs": len(test_texts),
    }}


def run_evaluation(quick: bool = False):
    """Run all evaluation tests."""
    print("=" * 70)
    print("Persian Conversation Model Evaluation Suite")
    print("=" * 70)

    model, tokenizer, config = load_model_and_tokenizer()

    all_results: dict = {}
    all_results.update(eval_persian_chat(model, tokenizer, config, quick))
    all_results.update(eval_nextjs_generation(model, tokenizer, config, quick))
    all_results.update(eval_end_token_stopping(model, tokenizer, config))
    all_results.update(eval_code_language_match(model, tokenizer, config, quick))
    all_results.update(eval_test_set_perplexity(model, tokenizer, config))

    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)

    if "persian_chat" in all_results:
        pc = all_results["persian_chat"]
        print(f"  Persian output rate:    {pc['persian_rate']:.0%}")
        print(f"  Degenerate rate:        {pc['degenerate_rate']:.0%}")
        print(f"  End-token stop rate:    {pc['end_stop_rate']:.0%}")
    if "nextjs_generation" in all_results:
        nx = all_results["nextjs_generation"]
        print(f"  Next.js patterns:       {nx['total_patterns']} (avg {nx['avg_patterns']:.1f})")
        print(f"  Next.js issues:         {nx['total_issues']}")
    if "end_token_stopping" in all_results and all_results["end_token_stopping"]:
        es = all_results["end_token_stopping"]
        print(f"  End-token stop rate:    {es['stop_rate']:.0%}")
    if "test_perplexity" in all_results and all_results["test_perplexity"]:
        tp = all_results["test_perplexity"]
        print(f"  Test perplexity:        {tp['perplexity']:.2f}")
    else:
        print(f"  Test perplexity:        N/A")

    os.makedirs(LOG_DIR, exist_ok=True)
    eval_path = os.path.join(LOG_DIR, "conversation_fa_eval_results.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved to {eval_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Persian conversation model")
    parser.add_argument("--quick", action="store_true", help="Quick evaluation")
    args = parser.parse_args()
    run_evaluation(quick=args.quick)
