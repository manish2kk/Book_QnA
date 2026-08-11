#!/usr/bin/env python3
"""Check extracted book .txt files for likely garbage / wrong words.

Default: heuristic extraction/OCR issues (no extra packages).
Optional: --spell uses pyspellchecker on ASCII English-looking words
(Sanskrit/IAST words are skipped to limit false positives).

True grammar (subject-verb, etc.) is not covered here — use LanguageTool /
a word processor on English-only excerpts; domain terms will still false-flag.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DIR = ROOT / "book1_txt"

# Leftover PDF mojibake / broken encodings often seen before cleaning.
MOJIBAKE_RE = re.compile(r"[àáâãäåæçèéêëìíîïðñòóôõøùúûüýþÿ÷¤þ]")
# Font-encoded Devanagari leftovers (colon + letter clusters, ö markers).
FONT_ENC_RE = re.compile(r"[A-Za-z]:[A-Za-zö]|[öÎÂØðòùÜ¶¤½¾¹]")
# Very long "words" with mixed scripts or digit noise (not long Sanskrit compounds).
WEIRD_WORD_RE = re.compile(
    r"\b(?=\w*[A-Za-z])(?=\w*\d)\w{6,}\b"  # letters+digits blob
)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|[^\W\d_][\w'\-]*", re.UNICODE)
IAST_RE = re.compile(r"[āīūṛṝḷḹṅñṇṭḍṣśṃḥĀĪŪṚṜḶḸṄÑṆṬḌṢŚṂḤ]")
# English-ish: pure ASCII letters, optional internal apostrophe/hyphen.
ASCII_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'\-]{1,}$")


def iter_txt_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.txt"))


def line_issues(line: str, line_no: int) -> list[str]:
    hits: list[str] = []
    if MOJIBAKE_RE.search(line):
        hits.append(f"L{line_no}: possible mojibake / broken IAST")
    if FONT_ENC_RE.search(line) and not IAST_RE.search(line):
        # Many true IAST lines are fine; flag font-encoding style leftovers.
        if re.search(r"[A-Za-z]:[öA-Za-z]|ö", line):
            hits.append(f"L{line_no}: possible font-encoded Devanagari leftover")
    for m in WEIRD_WORD_RE.finditer(line):
        hits.append(f"L{line_no}: weird token {m.group(0)!r}")
    if "\x00" in line or any(ord(ch) < 9 for ch in line if ch not in "\t\n\r"):
        hits.append(f"L{line_no}: control characters")
    return hits


def collect_ascii_words(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for w in WORD_RE.findall(text):
        if IAST_RE.search(w):
            continue
        if not ASCII_WORD_RE.match(w):
            continue
        if len(w) < 3:
            continue
        # Skip ALLCAPS headings-ish short tokens somewhat
        counts[w.lower()] += 1
    return counts


def spellcheck_words(
    counts: Counter[str], max_report: int
) -> list[tuple[str, int]]:
    try:
        from spellchecker import SpellChecker
    except ImportError as exc:
        raise SystemExit(
            "Spell check needs: pip install pyspellchecker\n"
            f"Import error: {exc}"
        ) from exc

    spell = SpellChecker(language="en")
    # Domain / yoga allowlist (lowercase)
    allow = {
        "prana",
        "pranayama",
        "kundalini",
        "chakra",
        "chakras",
        "asana",
        "asanas",
        "mudra",
        "mudras",
        "bandha",
        "bandhas",
        "nadi",
        "nadis",
        "mantra",
        "mantras",
        "yoga",
        "yogi",
        "yogis",
        "yogic",
        "hatha",
        "raja",
        "bhakti",
        "jnana",
        "karma",
        "samadhi",
        "dhyana",
        "pranava",
        "om",
        "aum",
        "shakti",
        "shiva",
        "sushumna",
        "ida",
        "pingala",
        "muladhara",
        "swadhisthana",
        "manipura",
        "anahata",
        "vishuddha",
        "ajna",
        "sahasrara",
        "kumbhaka",
        "puraka",
        "rechaka",
        "japa",
        "tapas",
        "guru",
        "gurus",
        "sanskrit",
        "vedas",
        "upanishads",
        "gita",
        "brahman",
        "atman",
        "maya",
        "samsara",
        "moksha",
        "nirvana",
        "tamas",
        "rajas",
        "sattva",
        "gunas",
        "prakriti",
        "purusha",
    }
    unknown = []
    for word, n in counts.items():
        if word in allow:
            continue
        if word in spell:
            continue
        # pyspellchecker: word not in frequency list
        unknown.append((word, n))
    unknown.sort(key=lambda x: (-x[1], x[0]))
    return unknown[:max_report]


def check_file(path: Path, spell: bool, max_spell: int, max_line_hits: int) -> int:
    text = path.read_text(encoding="utf-8")
    print(f"\n=== {path.name} ({len(text):,} chars) ===")

    line_hits: list[str] = []
    for i, line in enumerate(text.splitlines(), start=1):
        line_hits.extend(line_issues(line, i))

    if line_hits:
        print(f"Extraction / garbage flags: {len(line_hits)}")
        for msg in line_hits[:max_line_hits]:
            print(f"  {msg}")
        if len(line_hits) > max_line_hits:
            print(f"  ... {len(line_hits) - max_line_hits} more")
    else:
        print("Extraction / garbage flags: 0")

    # Corpus-hapax ASCII words (often typos if English)
    counts = collect_ascii_words(text)
    hapax = sorted(w for w, n in counts.items() if n == 1 and len(w) >= 6)
    print(f"Rare ASCII words (appear once, len>=6): {len(hapax)}")
    if hapax:
        preview = ", ".join(hapax[:40])
        print(f"  e.g. {preview}")
        if len(hapax) > 40:
            print(f"  ... {len(hapax) - 40} more")

    if spell:
        bad = spellcheck_words(counts, max_spell)
        print(f"English spellcheck suspects (top {len(bad)}):")
        for w, n in bad:
            print(f"  {w!r}  (count={n})")

    return len(line_hits)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=DEFAULT_DIR,
        help="book1_txt/ folder or a single .txt file",
    )
    parser.add_argument(
        "--spell",
        action="store_true",
        help="Also run English spellcheck (pip install pyspellchecker)",
    )
    parser.add_argument("--max-spell", type=int, default=80)
    parser.add_argument("--max-line-hits", type=int, default=40)
    args = parser.parse_args()

    files = iter_txt_files(args.path)
    if not files:
        print(f"No .txt files at {args.path}", file=sys.stderr)
        return 1

    total = 0
    for f in files:
        total += check_file(f, args.spell, args.max_spell, args.max_line_hits)

    print(f"\nDone. Files={len(files)} extraction_flags={total}")
    print(
        "\nGrammar tip: for English sentence grammar, paste a chapter into\n"
        "LanguageTool (https://languagetool.org) or a word processor.\n"
        "Expect many false alarms on Sanskrit names and IAST spellings."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
