"""Character vocabulary helpers for CharGPT."""

from __future__ import annotations

from pathlib import Path

import torch


class CharTokenizer:
    def __init__(self, chars: list[str]) -> None:
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        # Unknown chars (rare) map to space if present, else first token
        fallback = self.stoi.get(" ", 0)
        return [self.stoi.get(ch, fallback) for ch in text]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.itos[i] for i in ids)

    def save(self, path: Path) -> None:
        path.write_text("".join(self.chars), encoding="utf-8")

    def extend_with_text(self, text: str) -> int:
        """Append any new characters; keep existing ids stable. Returns count added."""
        existing = set(self.chars)
        added = sorted(ch for ch in set(text) if ch not in existing)
        if not added:
            return 0
        self.chars = list(self.chars) + added
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        return len(added)

    def extend_with_files(self, paths: list[Path]) -> int:
        """Append chars from many files; keep existing ids. Returns total added."""
        added = 0
        for path in paths:
            added += self.extend_with_text(path.read_text(encoding="utf-8"))
        return added

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        chars = sorted(set(text))
        return cls(chars)

    @classmethod
    def from_files(cls, paths: list[Path]) -> CharTokenizer:
        """Union vocabulary over all given text files (sorted unique chars)."""
        chars: set[str] = set()
        for path in paths:
            chars.update(path.read_text(encoding="utf-8"))
        return cls(sorted(chars))

    @classmethod
    def load(cls, path: Path) -> CharTokenizer:
        chars = list(path.read_text(encoding="utf-8"))
        return cls(chars)
