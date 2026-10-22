"""Independent evaluation suite for the specialized programming model.

Tests:
  1. Code generation: generate completions from code prompts across languages
  2. Syntax correctness: check generated Python code parses without errors
  3. TypeScript type annotation validity: check basic TS patterns
  4. Next.js App Router examples: verify correct file conventions and patterns
  5. Test set perplexity: compute loss on held-out test data
  6. Language detection: check if generated text matches the prompt's language

Usage:
    python llm/evaluate.py              # Full evaluation
    python llm/evaluate.py --quick      # Quick evaluation (fewer samples)
"""

import argparse
import json
import os
import re
import sys
import ast

# Make package imports work when run directly
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

BEST_CKPT = os.path.join(CHECKPOINT_DIR, "specialized_best.pt")
SPECIALIZED_CKPT = os.path.join(CHECKPOINT_DIR, "specialized.pt")
NEW_TOKENIZER = os.path.join(CHECKPOINT_DIR, "tokenizer_v2.json")


# Evaluation prompts — designed to test different languages and patterns
EVAL_PROMPTS = {
    "python": [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return ",
        "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while ",
        "class Stack:\n    def __init__(self):\n        self._items = ",
        "def debounce(wait):\n    def decorator(func):\n        ",
    ],
    "javascript": [
        "function debounce(func, wait) {\n  let timeout;\n  return function",
        "class EventEmitter {\n  constructor() {\n    this.events = ",
        "async function fetchWithRetry(url, maxRetries = 3) {\n  for",
        "function memoize(fn) {\n  const cache = ",
  ],
    "typescript": [
        "interface User {\n  id: number;\n  name: string;\n  ",
        "type Result<T> =\n  | { success: true; data: T }\n  | ",
        "function safeFetch<T>(url: string): Promise<Result<T>> {\n  try",
        "enum LogLevel {\n  Debug = 'DEBUG',\n  ",
    ],
    "react": [
        "import { useState, useEffect } from 'react';\n\nfunction useFetch(url) {\n  const [data, setData] = ",
        "function TodoApp() {\n  const [todos, setTodos] = ",
        "function useDebounce(value, delay) {\n  const [debouncedValue, ",
        "const ThemeContext = createContext();\n\nfunction ThemeProvider({ children }) {\n  const [theme, ",
    ],
    "nextjs": [
        "// app/page.tsx\nimport { Suspense } from 'react'\n\nexport default function HomePage() {\n  return",
        "// app/api/users/route.ts\nimport { NextRequest, NextResponse } from 'next/server'\n\nexport async function GET(request: NextRequest) {\n  const",
        "// app/blog/[slug]/page.tsx\nimport { notFound } from 'next/navigation'\n\nexport default async function BlogPost({ params }) {\n  const post = ",
        "// app/dashboard/loading.tsx\nexport default function Loading() {\n  return",
    ],
}


def load_model_and_tokenizer():
    """Load the best specialized checkpoint and tokenizer.

    Raises CheckpointError on corrupt or incompatible checkpoints.
    """
    ckpt_path = BEST_CKPT if os.path.exists(BEST_CKPT) else SPECIALIZED_CKPT
    tok_path = NEW_TOKENIZER

    if not os.path.exists(ckpt_path):
        print(f"No specialized checkpoint found at {ckpt_path}")
        print("Run training first: python llm/train_specialized.py")
        sys.exit(1)
    if not os.path.exists(tok_path):
        print(f"Tokenizer not found at {tok_path}")
        sys.exit(1)

    # Load config from checkpoint
    try:
        saved_config = load_config_from_checkpoint(ckpt_path)
    except CheckpointError as e:
        print(f"ERROR: Could not read checkpoint config: {e}")
        sys.exit(1)
    if saved_config is None:
        saved_config = ModelConfig()

    # Load tokenizer
    tokenizer = BPETokenizer(vocab_size=saved_config.bpe_vocab_size)
    tokenizer.load(tok_path)

    # Validate vocab compatibility
    if tokenizer.vocab_size != saved_config.vocab_size:
        print(f"ERROR: Tokenizer vocab ({tokenizer.vocab_size}) does not match "
              f"checkpoint vocab ({saved_config.vocab_size}).")
        print(f"  Tokenizer: {tok_path}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  Retrain with: python llm/train_specialized.py")
        sys.exit(1)

    saved_config.vocab_size = tokenizer.vocab_size

    # Load model
    model = GPTModel(saved_config)
    try:
        meta = load_checkpoint(ckpt_path, model)
    except CheckpointError as e:
        print(f"ERROR: Could not load checkpoint: {e}")
        sys.exit(1)

    print(f"Loaded checkpoint: step={meta['step']}, val_loss={meta['val_loss']:.4f}")
    print(f"Model: {saved_config.n_layers} layers, {saved_config.hidden_dim} dim, "
          f"{model.count_parameters():,} params")

    return model, tokenizer, saved_config


def check_python_syntax(code: str) -> tuple[bool, str | None]:
    """Check if Python code is syntactically valid."""
    # Extract the generated portion (after the prompt)
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)


def check_typescript_patterns(code: str) -> list[str]:
    """Check for common TypeScript validity patterns."""
    issues = []
    # Check for balanced type annotations
    type_annotations = re.findall(r':\s*([A-Z]\w*(?:<[^>]+>)?)', code)
    for ann in type_annotations:
        if '<' in ann and '>' not in ann:
            issues.append(f"Unclosed generic type: {ann}")
    # Check for interface/type balance
    opens = code.count('{')
    closes = code.count('}')
    if abs(opens - closes) > 2:
        issues.append(f"Unbalanced braces: {opens} open, {closes} close")
    # Check for proper string termination
    if code.count("'") % 2 != 0 and code.count('"') % 2 != 0:
        issues.append("Unterminated string literal")
    return issues


def check_nextjs_patterns(code: str) -> list[str]:
    """Check for Next.js App Router conventions in generated code."""
    issues = []
    patterns_found = []

    # Check for server/client directives
    if "'use client'" in code or '"use client"' in code:
        patterns_found.append("use_client_directive")
    if "export default" in code:
        patterns_found.append("default_export")
    if "next/navigation" in code:
        patterns_found.append("next_navigation_import")
    if "next/server" in code:
        patterns_found.append("next_server_import")
    if "NextResponse" in code:
        patterns_found.append("NextResponse_usage")
    if "cookies()" in code:
        patterns_found.append("cookies_usage")
    if "generateStaticParams" in code:
        patterns_found.append("generateStaticParams")
    if "generateMetadata" in code:
        patterns_found.append("generateMetadata")
    if "revalidate" in code:
        patterns_found.append("revalidate")

    # Check for balanced JSX tags (basic)
    jsx_opens = len(re.findall(r'<(\w+)', code))
    jsx_closes = len(re.findall(r'</(\w+)>', code)) + len(re.findall(r'/\s*>', code))
    if abs(jsx_opens - jsx_closes) > 3:
        issues.append(f"Unbalanced JSX tags: {jsx_opens} open, {jsx_closes} close")

    return issues, patterns_found


def detect_language(code: str) -> str:
    """Detect the primary programming language of generated code."""
    if "def " in code and "import " in code and ":" in code:
        return "python"
    if "interface " in code and ": " in code and "type " in code:
        return "typescript"
    if "export default" in code and ("next/" in code or "app/" in code):
        return "nextjs"
    if "useState" in code or "useEffect" in code or "useRef" in code:
        return "react"
    if "function " in code or "const " in code or "=>" in code:
        return "javascript"
    return "unknown"


def eval_code_generation(model, tokenizer, config, quick=False):
    """Test 1: Code generation across languages."""
    print("\n[Test 1] Code Generation")
    print("-" * 50)

    results = {}
    n_samples = 1 if quick else 2
    max_tokens = 80 if quick else 120

    for lang, prompts in EVAL_PROMPTS.items():
        lang_results = []
        for i, prompt in enumerate(prompts[:2] if quick else prompts):
            text = generate_text(
                model, tokenizer, prompt, config,
                max_new_tokens=max_tokens, temperature=0.3, top_k=30,
            )
            generated = text[len(prompt):] if text.startswith(prompt) else text
            detected = detect_language(text)
            correct_lang = detected == lang

            lang_results.append({
                "prompt_idx": i,
                "generated": generated[:200],
                "detected_language": detected,
                "language_match": correct_lang,
            })

            print(f"  [{lang}] prompt {i}: lang_match={correct_lang} ({detected})")

        match_rate = sum(1 for r in lang_results if r["language_match"]) / len(lang_results)
        results[lang] = {"samples": lang_results, "language_match_rate": match_rate}
        print(f"  -> {lang} language match rate: {match_rate:.0%}")

    overall = sum(r["language_match_rate"] for r in results.values()) / len(results)
    print(f"\n  Overall language match rate: {overall:.0%}")
    return {"code_generation": results, "overall_language_match": overall}


def eval_python_syntax(model, tokenizer, config, quick=False):
    """Test 2: Python syntax correctness."""
    print("\n[Test 2] Python Syntax Correctness")
    print("-" * 50)

    prompts = EVAL_PROMPTS["python"]
    if quick:
        prompts = prompts[:2]

    valid_count = 0
    total = 0
    errors = []

    for i, prompt in enumerate(prompts):
        text = generate_text(
            model, tokenizer, prompt, config,
            max_new_tokens=100, temperature=0.2, top_k=20,
        )
        generated = text[len(prompt):] if text.startswith(prompt) else text

        # Try to parse the full text (prompt + generated)
        is_valid, error = check_python_syntax(text)
        total += 1

        if is_valid:
            valid_count += 1
            print(f"  prompt {i}: VALID")
        else:
            errors.append({"prompt_idx": i, "error": error, "generated": generated[:100]})
            print(f"  prompt {i}: INVALID - {error}")

    rate = valid_count / total if total > 0 else 0
    print(f"\n  Python syntax validity: {valid_count}/{total} ({rate:.0%})")
    return {"python_syntax": {"valid": valid_count, "total": total, "rate": rate, "errors": errors}}


def eval_typescript_patterns(model, tokenizer, config, quick=False):
    """Test 3: TypeScript pattern validity."""
    print("\n[Test 3] TypeScript Pattern Validity")
    print("-" * 50)

    prompts = EVAL_PROMPTS["typescript"]
    if quick:
        prompts = prompts[:2]

    clean_count = 0
    total = 0
    all_issues = []

    for i, prompt in enumerate(prompts):
        text = generate_text(
            model, tokenizer, prompt, config,
            max_new_tokens=100, temperature=0.2, top_k=20,
        )
        generated = text[len(prompt):] if text.startswith(prompt) else text

        issues = check_typescript_patterns(text)
        total += 1

        if not issues:
            clean_count += 1
            print(f"  prompt {i}: CLEAN (no pattern issues)")
        else:
            all_issues.append({"prompt_idx": i, "issues": issues})
            print(f"  prompt {i}: ISSUES - {issues}")

    rate = clean_count / total if total > 0 else 0
    print(f"\n  TypeScript pattern validity: {clean_count}/{total} ({rate:.0%})")
    return {"typescript_patterns": {"clean": clean_count, "total": total, "rate": rate, "issues": all_issues}}


def eval_nextjs_patterns(model, tokenizer, config, quick=False):
    """Test 4: Next.js App Router conventions."""
    print("\n[Test 4] Next.js App Router Conventions")
    print("-" * 50)

    prompts = EVAL_PROMPTS["nextjs"]
    if quick:
        prompts = prompts[:2]

    results = []
    total_patterns = 0
    total_issues = 0

    for i, prompt in enumerate(prompts):
        text = generate_text(
            model, tokenizer, prompt, config,
            max_new_tokens=120, temperature=0.3, top_k=25,
        )
        generated = text[len(prompt):] if text.startswith(prompt) else text

        issues, patterns = check_nextjs_patterns(text)
        total_patterns += len(patterns)
        total_issues += len(issues)

        result = {
            "prompt_idx": i,
            "generated": generated[:200],
            "patterns_found": patterns,
            "issues": issues,
        }
        results.append(result)

        print(f"  prompt {i}: patterns={patterns}, issues={len(issues)}")

    print(f"\n  Total patterns found: {total_patterns}")
    print(f"  Total issues: {total_issues}")
    avg_patterns = total_patterns / len(results) if results else 0
    print(f"  Avg patterns per generation: {avg_patterns:.1f}")
    return {"nextjs_patterns": {"results": results, "total_patterns": total_patterns,
                                 "total_issues": total_issues, "avg_patterns": avg_patterns}}


def eval_test_set_perplexity(model, tokenizer, config):
    """Test 5: Test set perplexity (loss on held-out data)."""
    print("\n[Test 5] Test Set Perplexity")
    print("-" * 50)

    test_file = os.path.join(DATA_DIR, "test_set.json")
    if not os.path.exists(test_file):
        print("  No test set found. Run training first.")
        return {"test_perplexity": None}

    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    test_texts = test_data["test_texts"]
    test_ids = []
    for text in test_texts:
        test_ids.extend(tokenizer.encode(text))

    if len(test_ids) < config.context_length + 1:
        print(f"  Test set too small: {len(test_ids)} tokens")
        return {"test_perplexity": None}

    test_dataset = TextDataset(test_ids, config)
    test_loader = DataLoader(test_dataset, batch_size=min(config.batch_size, len(test_dataset)),
                              shuffle=False, drop_last=False)

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

    return {"test_perplexity": {"loss": avg_loss, "perplexity": perplexity,
                                 "test_tokens": len(test_ids), "test_docs": len(test_texts)}}


def run_evaluation(quick=False):
    """Run all evaluation tests."""
    print("=" * 70)
    print("Specialized Model Evaluation Suite")
    print("=" * 70)

    model, tokenizer, config = load_model_and_tokenizer()

    all_results = {}

    # Run all tests
    all_results.update(eval_code_generation(model, tokenizer, config, quick))
    all_results.update(eval_python_syntax(model, tokenizer, config, quick))
    all_results.update(eval_typescript_patterns(model, tokenizer, config, quick))
    all_results.update(eval_nextjs_patterns(model, tokenizer, config, quick))
    all_results.update(eval_test_set_perplexity(model, tokenizer, config))

    # Summary
    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)

    if "overall_language_match" in all_results:
        print(f"  Language match rate:     {all_results['overall_language_match']:.0%}")
    if "python_syntax" in all_results and all_results["python_syntax"]:
        ps = all_results["python_syntax"]
        print(f"  Python syntax validity:  {ps['valid']}/{ps['total']} ({ps['rate']:.0%})")
    if "typescript_patterns" in all_results and all_results["typescript_patterns"]:
        ts = all_results["typescript_patterns"]
        print(f"  TS pattern cleanliness:  {ts['clean']}/{ts['total']} ({ts['rate']:.0%})")
    if "nextjs_patterns" in all_results and all_results["nextjs_patterns"]:
        nx = all_results["nextjs_patterns"]
        print(f"  Next.js patterns found:  {nx['total_patterns']} (avg {nx['avg_patterns']:.1f}/gen)")
        print(f"  Next.js issues:          {nx['total_issues']}")
    if "test_perplexity" in all_results and all_results["test_perplexity"]:
        tp = all_results["test_perplexity"]
        print(f"  Test perplexity:         {tp['perplexity']:.2f}")
    else:
        print(f"  Test perplexity:         N/A")

    # Save results
    os.makedirs(LOG_DIR, exist_ok=True)
    eval_path = os.path.join(LOG_DIR, "evaluation_results.json")
    with open(eval_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved to {eval_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate specialized model")
    parser.add_argument("--quick", action="store_true", help="Quick evaluation (fewer samples)")
    args = parser.parse_args()
    run_evaluation(quick=args.quick)
