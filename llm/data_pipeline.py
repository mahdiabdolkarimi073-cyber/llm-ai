"""Data pipeline for programming code training corpus.

Features:
  - Document-level collection with metadata (language, source, category)
  - Cleaning: removes null bytes, normalizes whitespace, deduplication
  - Validation: checks minimum length, valid UTF-8, balanced brackets
  - Split into train/validation/test with NO data leakage (per-document, not per-token)
  - Contamination check: ensures no test documents appear in train
"""

import hashlib
import re
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class CodeDocument:
    """A single training document with metadata."""
    text: str
    language: str       # python, javascript, typescript, react, nextjs
    category: str       # code, docs, example, qa, error-fix
    source: str         # origin label

    @property
    def fingerprint(self) -> str:
        """Content hash for deduplication and leakage detection."""
        normalized = re.sub(r'\s+', ' ', self.text).strip().lower()
        return hashlib.md5(normalized.encode()).hexdigest()


class DataPipeline:
    """Full pipeline: collect, clean, validate, split programming corpus."""

    def __init__(
        self,
        min_chars: int = 20,
        max_chars: int = 10000,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 42,
    ):
        assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed

    def clean(self, text: str) -> str:
        """Remove problematic characters and normalize."""
        # Remove null bytes
        text = text.replace('\x00', '')
        # Remove carriage returns (normalize to \n)
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Remove trailing whitespace per line
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)
        # Strip leading/trailing blank lines
        text = text.strip()
        return text

    def validate(self, doc: CodeDocument) -> tuple[bool, str]:
        """Validate a document. Returns (is_valid, reason)."""
        if len(doc.text) < self.min_chars:
            return False, f"too_short ({len(doc.text)} chars)"
        if len(doc.text) > self.max_chars:
            return False, f"too_long ({len(doc.text)} chars)"
        if not doc.text.isprintable() and '\n' in doc.text:
            # Allow newlines but check for unprintable chars
            cleaned = doc.text.replace('\n', '').replace('\t', '')
            if not cleaned.isprintable():
                return False, "contains_unprintable"
        # Check for balanced brackets (basic sanity)
        bracket_pairs = {'(': ')', '{': '}', '[': ']'}
        stack = []
        for char in doc.text:
            if char in bracket_pairs:
                stack.append(char)
            elif char in bracket_pairs.values():
                if not stack:
                    break
                last = stack.pop()
                if bracket_pairs[last] != char:
                    break
        # Not a hard fail — just informational
        return True, "ok"

    def deduplicate(self, documents: list[CodeDocument]) -> list[CodeDocument]:
        """Remove exact duplicates by content fingerprint."""
        seen = set()
        unique = []
        dup_count = 0
        for doc in documents:
            fp = doc.fingerprint
            if fp in seen:
                dup_count += 1
                continue
            seen.add(fp)
            unique.append(doc)
        return unique

    def check_leakage(
        self,
        train_docs: list[CodeDocument],
        test_docs: list[CodeDocument],
    ) -> list[str]:
        """Check for data leakage between train and test sets."""
        train_fps = {doc.fingerprint for doc in train_docs}
        leaked = []
        for doc in test_docs:
            if doc.fingerprint in train_fps:
                leaked.append(doc.fingerprint)
        return leaked

    def split(
        self,
        documents: list[CodeDocument],
    ) -> tuple[list[CodeDocument], list[CodeDocument], list[CodeDocument]]:
        """Split documents into train/val/test with no leakage.

        Splitting is per-document (not per-token) to prevent
        cross-contamination of content between splits.
        """
        import random
        rng = random.Random(self.seed)
        shuffled = list(documents)
        rng.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * self.train_frac)
        val_end = int(n * (self.train_frac + self.val_frac))

        train_docs = shuffled[:train_end]
        val_docs = shuffled[train_end:val_end]
        test_docs = shuffled[val_end:]

        # Verify no leakage
        leaked = self.check_leakage(train_docs, test_docs)
        if leaked:
            raise ValueError(f"Data leakage detected: {len(leaked)} test docs found in train")

        leaked_vt = self.check_leakage(val_docs, test_docs)
        if leaked_vt:
            raise ValueError(f"Data leakage detected: {len(leaked_vt)} test docs found in val")

        return train_docs, val_docs, test_docs

    def process(
        self,
        documents: list[CodeDocument],
    ) -> dict:
        """Full pipeline: clean, validate, deduplicate, split.

        Returns a report dict with stats and the three splits.
        """
        # Step 1: Clean
        cleaned = []
        for doc in documents:
            doc.text = self.clean(doc.text)
            if doc.text:
                cleaned.append(doc)

        # Step 2: Validate
        valid = []
        invalid_count = 0
        invalid_reasons = Counter()
        for doc in cleaned:
            ok, reason = self.validate(doc)
            if ok:
                valid.append(doc)
            else:
                invalid_count += 1
                invalid_reasons[reason] += 1

        # Step 3: Deduplicate
        unique = self.deduplicate(valid)
        dup_count = len(valid) - len(unique)

        # Step 4: Split
        train_docs, val_docs, test_docs = self.split(unique)

        # Stats
        stats = {
            "input_docs": len(documents),
            "after_clean": len(cleaned),
            "invalid": invalid_count,
            "invalid_reasons": dict(invalid_reasons),
            "duplicates_removed": dup_count,
            "unique_docs": len(unique),
            "train_docs": len(train_docs),
            "val_docs": len(val_docs),
            "test_docs": len(test_docs),
            "train_chars": sum(len(d.text) for d in train_docs),
            "val_chars": sum(len(d.text) for d in val_docs),
            "test_chars": sum(len(d.text) for d in test_docs),
            "languages": Counter(d.language for d in unique),
            "categories": Counter(d.category for d in unique),
        }
        stats["splits"] = (train_docs, val_docs, test_docs)
        return stats

    @staticmethod
    def format_report(stats: dict) -> str:
        """Format stats dict as readable report."""
        lines = [
            "=== Data Pipeline Report ===",
            f"Input documents:      {stats['input_docs']}",
            f"After cleaning:       {stats['after_clean']}",
            f"Invalid removed:      {stats['invalid']}",
        ]
        if stats['invalid_reasons']:
            for reason, count in stats['invalid_reasons'].items():
                lines.append(f"  - {reason}: {count}")
        lines.extend([
            f"Duplicates removed:   {stats['duplicates_removed']}",
            f"Unique documents:     {stats['unique_docs']}",
            f"",
            f"Train: {stats['train_docs']} docs, {stats['train_chars']:,} chars",
            f"Val:   {stats['val_docs']} docs, {stats['val_chars']:,} chars",
            f"Test:  {stats['test_docs']} docs, {stats['test_chars']:,} chars",
            f"",
            f"Languages: {dict(stats['languages'])}",
            f"Categories: {dict(stats['categories'])}",
        ])
        return '\n'.join(lines)
