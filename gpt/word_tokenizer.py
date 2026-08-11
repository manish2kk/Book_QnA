"""Word / punctuation tokenizer for word-level GPT pretraining."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import torch

# Words (incl. Unicode letters/digits) and each non-space punctuation mark.
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

UNK = "<unk>"
NL = "<nl>"
SPECIALS = (UNK, NL)


class WordTokenizer:
    def __init__(self, tokens: list[str]) -> None:
        if UNK not in tokens:
            tokens = [UNK, *[t for t in tokens if t != UNK]]
        self.tokens = tokens
        self.stoi = {t: i for i, t in enumerate(tokens)}
        self.itos = {i: t for i, t in enumerate(tokens)}

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Split text into word/punct tokens; newlines become <nl>."""
        out: list[str] = []
        parts = text.split("\n")
        for i, part in enumerate(parts):
            out.extend(TOKEN_RE.findall(part))
            if i < len(parts) - 1:
                out.append(NL)
        return out

    def encode(self, text: str) -> list[int]:
        unk = self.stoi[UNK]
        return [self.stoi.get(tok, unk) for tok in self.tokenize(text)]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        pieces: list[str] = []
        for i in ids:
            tok = self.itos.get(i, UNK)
            if tok == NL:
                pieces.append("\n")
            elif tok == UNK:
                if pieces and not pieces[-1].endswith("\n"):
                    pieces.append(" ")
                pieces.append(UNK)
            elif re.fullmatch(r"[^\w\s]", tok):
                pieces.append(tok)
            else:
                if pieces and not pieces[-1].endswith(("\n", " ")):
                    # space before words; no space after opening-ish already handled
                    if not (pieces and re.fullmatch(r"[\"'(\[]", pieces[-1])):
                        pieces.append(" ")
                pieces.append(tok)
        return "".join(pieces).strip()

    def save(self, path: Path) -> None:
        # One token per line (tokens never contain raw newlines).
        path.write_text("\n".join(self.tokens) + "\n", encoding="utf-8")

    def extend_with_tokens(self, new_tokens: list[str]) -> int:
        """Append unknown tokens; keep existing ids stable. Returns count added."""
        existing = set(self.tokens)
        added = [t for t in new_tokens if t not in existing and t not in SPECIALS]
        # Keep deterministic order for newly added tokens.
        added = sorted(set(added))
        if not added:
            return 0
        self.tokens = list(self.tokens) + added
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
        self.itos = {i: t for i, t in enumerate(self.tokens)}
        return len(added)

    def extend_with_files(self, paths: list[Path], min_freq: int = 1) -> int:
        counts: Counter[str] = Counter()
        for path in paths:
            counts.update(self.tokenize(path.read_text(encoding="utf-8")))
        candidates = [t for t, n in counts.items() if n >= min_freq and t not in SPECIALS]
        return self.extend_with_tokens(candidates)

    @classmethod
    def from_files(cls, paths: list[Path], min_freq: int = 1) -> WordTokenizer:
        counts: Counter[str] = Counter()
        for path in paths:
            counts.update(cls.tokenize(path.read_text(encoding="utf-8")))
        words = sorted(t for t, n in counts.items() if n >= min_freq and t not in SPECIALS)
        return cls([*SPECIALS, *words])

    @classmethod
    def load(cls, path: Path) -> WordTokenizer:
        lines = path.read_text(encoding="utf-8").splitlines()
        tokens = [ln for ln in lines if ln != ""]
        return cls(tokens)
