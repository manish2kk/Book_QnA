#!/usr/bin/env python3
"""Convert PDFs in book1/ to cleaned continuous .txt files in book1_txt/."""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf

INPUT_DIR = Path(__file__).resolve().parent / "book1"
OUTPUT_DIR = Path(__file__).resolve().parent / "book1_txt"

# Standalone page numbers: Arabic or Roman numerals
PAGE_NUMBER_RE = re.compile(
    r"^(?:\d{1,4}|[ivxlcdm]{1,12})$",
    re.IGNORECASE,
)

# Collapse 3+ blank lines into 2; normalize whitespace within lines lightly
MULTI_BLANK_RE = re.compile(r"\n{3,}")
# Hard hyphen, soft hyphen (U+00AD), en/em dash at PDF line wrap inside a word
HYPHEN_BREAK_RE = re.compile(r"(\w)[\-\u00ad–—]\n(\w)")

# Font-encoded Devanagari bija syllables -> IAST English sounds.
# Longer tokens first so e.g. A:òö wins over A:ö / Aö.
DEVANAGARI_FONT_TO_IAST: dict[str, str] = {
    "A:òö": "auṃ",
    "A:ðö": "oṃ",
    "Oðö": "aiṃ",
    "A:ö": "āṃ",
    "K:ö": "khaṃ",
    "G:ö": "ghaṃ",
    "N:ö": "ṇaṃ",
    "T:ö": "thaṃ",
    "D:ö": "dhaṃ",
    "S:ö": "śaṃ",
    "\\:ö": "ṣaṃ",
    "v:ö": "vaṃ",
    "s:ö": "saṃ",
    "b:ö": "baṃ",
    "B:ö": "bhaṃ",
    "m:ö": "maṃ",
    "y:ö": "yaṃ",
    "l:ö": "laṃ",
    "t:ö": "taṃ",
    "n:ö": "naṃ",
    "p:ö": "paṃ",
    "c:ö": "caṃ",
    "W:ö": "ñaṃ",
    "x:ö": "kṣaṃ",
    "g:ö": "gaṃ",
    "j:ö": "jaṃ",
    "rö": "raṃ",
    "Rö": "ṭaṃ",
    "Zö": "ṭhaṃ",
    "dö": "daṃ",
    "Pö": "phaṃ",
    "kö": "kaṃ",
    "{ö": "ṅaṃ",
    "Cö": "chaṃ",
    "J:ö": "jhaṃ",
    "Xö": "ṭaṃ",
    "Yö": "ṭhaṃ",
    "Aö": "aṃ",
    "Eö": "iṃ",
    "Iö": "īṃ",
    "uö": "uṃ",
    "Uö": "ūṃ",
    "?ö": "ṛṃ",
    "@ö": "ṝṃ",
    ";ö": "ḷṃ",
    "=ö": "ḹṃ",
    "Oö": "eṃ",
    "hö": "haṃ",
    "AH": "aḥ",
}

_DEVANAGARI_FONT_RE = re.compile(
    "|".join(
        re.escape(token)
        for token in sorted(DEVANAGARI_FONT_TO_IAST, key=len, reverse=True)
    )
)


def transliterate_devanagari_font(text: str) -> str:
    """Replace font-encoded Sanskrit syllables with IAST romanization."""
    return _DEVANAGARI_FONT_RE.sub(
        lambda m: DEVANAGARI_FONT_TO_IAST[m.group(0)], text
    )


# PDF romanization uses a broken character set; map to proper IAST.
_IAST_CHAR_FIXES: list[tuple[str, str]] = [
    ("à", "ā"),
    ("á", "ā"),
    ("ü", "ṃ"),
    ("÷", "ś"),
    ("ù", "ṣ"),
    ("ã", "ī"),
    ("å", "ū"),
    ("ç", "ṛ"),
    ("é", "ṝ"),
    ("ë", "ḷ"),
    ("í", "ḹ"),
    ("ï", "ṅ"),
    ("¤", "ñ"),
    ("õ", "ṇ"),
    ("þ", "ḥ"),
    (".N", "ṃ"),
]

# ó is ṭ at token start (bija glosses) and ḍ elsewhere (maṇḍala, rūḍha, …).
_IAST_T_FIXES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?<![A-Za-zāīūṛṝḷḹṅñṇṭḍṣśṃḥà-ÿ÷])óhaü"), "ṭhaṃ"),
    (re.compile(r"(?<![A-Za-zāīūṛṝḷḹṅñṇṭḍṣśṃḥà-ÿ÷])óaü"), "ṭaṃ"),
    (re.compile(r"(?<![A-Za-zāīūṛṝḷḹṅñṇṭḍṣśṃḥà-ÿ÷])óhaṃ"), "ṭhaṃ"),
    (re.compile(r"(?<![A-Za-zāīūṛṝḷḹṅñṇṭḍṣśṃḥà-ÿ÷])óaṃ"), "ṭaṃ"),
]


def fix_garbled_iast(text: str) -> str:
    """Repair mojibake IAST from the PDF into readable English sounds."""
    for src, dst in _IAST_CHAR_FIXES:
        text = text.replace(src, dst)
    for pattern, dst in _IAST_T_FIXES:
        text = pattern.sub(dst, text)
    text = text.replace("ó", "ḍ")
    return text


# ---------------------------------------------------------------------------
# OCR digit-in-word fixes (scanned PDFs often turn m → n1/1n, l → 1, nj → 11)
# ---------------------------------------------------------------------------

# Exact token replacements (as they appear in extracted text).
OCR_DIGIT_WORD_FIXES: dict[str, str] = {
    "Pata11jali": "Patanjali",
    "con1fortable": "comfortable",
    "co1nfortable": "comfortable",
    "con1fortably": "comfortably",
    "ani1nals": "animals",
    "moven1ent": "movement",
    "moven1ents": "movements",
    "move1nent": "movement",
    "n1oven1ent": "movement",
    "n1ovement": "movement",
    "extren1ely": "extremely",
    "n1attress": "mattress",
    "ston1ach": "stomach",
    "n1oderation": "moderation",
    "recon1mended": "recommended",
    "Pawann1uktasana": "Pawanmuktasana",
    "n1ind": "mind",
    "1nind": "mind",
    "n1eans": "means",
    "becon1e": "become",
    "becon1es": "becomes",
    "beco1nes": "becomes",
    "n1ethod": "method",
    "aln1ost": "almost",
    "pe1formance": "performance",
    "con1mon": "common",
    "san1e": "same",
    "1nay": "may",
    "n1ay": "may",
    "1norning": "morning",
    "n1uscles": "muscles",
    "TI1is": "This",
    "glauco1na": "glaucoma",
    "paln1s": "palms",
    "arn1": "arm",
    "arn1s": "arms",
    "thun1b": "thumb",
    "tin1es": "times",
    "tin1e": "time",
    "n1inute": "minute",
    "simhaga1janasana": "simhagarjanasana",
    "1nuch": "much",
    "Durati9n": "Duration",
    "n1akes": "makes",
    "forn1": "form",
    "forn1s": "forms",
    "fro1n": "from",
    "fron1": "from",
    "en1pty": "empty",
    "han1strings": "hamstrings",
    "inflan1mation": "inflammation",
    "Intern1ediate": "Intermediate",
    "n1anifest": "manifest",
    "tl1e": "the",
    "tl1en": "then",
    "maxin1um": "maximum",
    "1naximum": "maximum",
    "tn1nk": "trunk",
    "t1unk": "trunk",
    "ren1ain": "remain",
    "fo1ward": "forward",
    "n1aking": "making",
    "n1aintaining": "maintaining",
    "lin1bs": "limbs",
    "1nanipura": "manipura",
    "horn1onal": "hormonal",
    "c1n": "cm",
    "abdo1nen": "abdomen",
    "abdon1en": "abdomen",
    "forearn1s": "forearms",
    "1nayurasana": "mayurasana",
    "atte1npted": "attempted",
    "n1ere": "mere",
    "pranayan1a": "pranayama",
    "n1odified": "modified",
    "1s": "is",
    "n1udra": "mudra",
    "son1e": "some",
    "sha1nbhavi": "shambhavi",
    "n1eals": "meals",
    "n1ooladhara": "mooladhara",
    "n1oola": "moola",
    "n1aha": "maha",
    "sn1ooth": "smooth",
    "s1noothly": "smoothly",
    "anna11Ulya": "annamaya",
    "minin1ize": "minimize",
    "minimun1": "minimum",
    "b1owing": "blowing",
    "sum1ner": "summer",
    "desc1ibe": "describe",
    "transfor11Ultion": "transformation",
    "who1e": "whole",
    "vo1nit": "vomit",
    "swal1ow": "swallow",
    "wil1": "will",
    "sheetkra1na": "sheetkrama",
    "ren1ove": "remove",
    "perfo1med": "performed",
    "perforn1": "perform",
    "syn1ptoms": "symptoms",
    "stin1ulates": "stimulates",
    "n1ust": "must",
    "a1Ulhata": "anahata",
    "a1Ulhad": "anahad",
    "In1agine": "Imagine",
    "i1nportant": "important",
    "11Ulnipura": "manipura",
    "11Ulni": "mani",
    "dynamisn1": "dynamism",
    "diaphragn1": "diaphragm",
    "sternun1": "sternum",
    "n1aha": "maha",
}

# Safe inside-token substitutions (m broken as n1 / 1n; nj as 11).
_OCR_SAFE_SUBS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?<=[A-Za-z])n1(?=[A-Za-z])"), "m"),
    (re.compile(r"^n1(?=[A-Za-z])"), "m"),
    (re.compile(r"(?<=[A-Za-z])n1\b"), "m"),
    (re.compile(r"(?<=[A-Za-z])1n(?=[A-Za-z])"), "m"),
    (re.compile(r"^1n(?=[A-Za-z])"), "m"),
    (re.compile(r"(?<=[A-Za-z])11(?=[A-Za-z])"), "nj"),
]

_OCR_DIGIT_TOKEN_RE = re.compile(r"\b(?=\w*[A-Za-z])(?=\w*\d)\w+\b")


def _skip_ocr_digit_token(tok: str) -> bool:
    """True for ordinals, footnotes, chem formulas — not spelling errors."""
    if re.match(r"^\d+(st|nd|rd|th)$", tok, re.I):
        return True
    if re.match(r"^(CO2|C02|C0|T\d+)$", tok, re.I):
        return True
    if re.match(r"^\d+year$", tok, re.I):
        return True
    # Footnote: heart17, disease2 — but NOT OCR m→n1 (minimun1, sternun1).
    m = re.match(r"^([A-Za-z][A-Za-z']*)(\d+)$", tok)
    if m:
        word, digits = m.group(1), m.group(2)
        if len(digits) == 1 and word[-1].lower() in "nmrl":
            return False
        return True
    # Page junk 6Take / 19Ross — but keep OCR forms like 11Ulnipura / 1nay.
    if re.match(r"^\d+[A-Z]", tok):
        if re.match(r"^(1n|11U|11)", tok):
            return False
        return True
    return False


def _heuristic_ocr_token(tok: str) -> str:
    """Apply safe digit→letter repairs when not in the explicit map."""
    t = tok
    for pat, repl in _OCR_SAFE_SUBS:
        t = pat.sub(repl, t)
    if t != tok and not re.search(r"\d", t):
        return t
    # Remaining interior 1 often stands for l (forward, will, the, …).
    if re.search(r"[A-Za-z]1[A-Za-z]", t):
        t2 = t.replace("1", "l")
        if not re.search(r"\d", t2):
            return t2
    if t.startswith("1") and len(t) > 1 and t[1].isalpha():
        t2 = "m" + t[1:]
        if not re.search(r"\d", t2):
            return t2
    return tok


def fix_ocr_digit_words(text: str) -> tuple[str, int]:
    """Fix OCR letter/digit confusions. Returns (new_text, replacement_count)."""
    count = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal count
        tok = m.group(0)
        # Explicit map wins (even if skip heuristics would ignore the token).
        if tok in OCR_DIGIT_WORD_FIXES:
            count += 1
            return OCR_DIGIT_WORD_FIXES[tok]
        if _skip_ocr_digit_token(tok):
            return tok
        fixed = _heuristic_ocr_token(tok)
        if fixed != tok:
            count += 1
            return fixed
        return tok

    return _OCR_DIGIT_TOKEN_RE.sub(repl, text), count


# Common non-digit OCR letter / punctuation confusions.
OCR_TYPO_FIXES: dict[str, str] = {
    "peiformed": "performed",
    "maxirnum": "maximum",
    "gland:t.dar": "glandular",
    "afi:er": "after",
    "ftaturing": "featuring",
    "brahrnarishi": "brahmarishi",
    "nmli": "nadi",
    "rishh": "rishi",
    r"\Vhen": "When",
    "Jaalandharoddyaananamoolabandhaa": "Jalandharoddyanamoolabandhaa",
    "Bandhatrayesminparicheeyamaane": "Bandhatrayesmin paricheeyamaane",
    "njalpanti": "ye jalpanti",
    "J alandhara": "Jalandhara",
    "Atlu.Jasane": "Atha asane",
    "drif!he": "dridhe",
}

_OCR_TYPO_RE = re.compile(
    r"|".join(
        re.escape(k) for k in sorted(OCR_TYPO_FIXES, key=len, reverse=True)
    )
)

# "Note:Word" / "bandhas:jalandhara" → insert space after colon when missing.
_COLON_SPACE_RE = re.compile(r"([:;,])([A-Za-z])")


def fix_common_ocr_typos(text: str) -> tuple[str, int]:
    """Fix letter-level OCR typos and missing space after punctuation."""
    count = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return OCR_TYPO_FIXES[m.group(0)]

    text = _OCR_TYPO_RE.sub(repl, text)
    # Only add space after : when it looks like "Note:Word" (letter after colon).
    def colon_space(m: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return f"{m.group(1)} {m.group(2)}"

    text = _COLON_SPACE_RE.sub(colon_space, text)
    return text, count


def is_garbage_extraction_line(line: str) -> bool:
    """True for leftover font/encoding junk lines (not readable English/IAST)."""
    s = line.strip()
    if not s:
        return False
    # Keep tiny punctuation-only separators? Drop pure junk fragments.
    if s in {"Â.Ã-", "\\:", "· '", "'", "-", "--", "- - -"}:
        return True
    if len(s) < 3 and re.fullmatch(r"[\W_·À-ÿÂÃ\\]+", s or ""):
        return True

    # High density of non-text / private-use / odd symbols.
    weird = len(re.findall(r"[^\w\s.,;:'\"!?()\[\]\-–—/]", s, flags=re.UNICODE))
    letters = len(re.findall(r"[A-Za-zāīūṛṝḷḹṅñṇṭḍṣśṃḥĀĪŪ]", s))
    if letters == 0 and weird >= 2:
        return True
    if len(s) >= 8 and weird >= max(4, len(s) // 3) and letters < weird:
        return True
    # Classic font-encoded Devanagari leftovers (strong signal).
    if is_font_encoded_line(s) and letters < 12:
        return True
    # Mixed junk: braces, % encodings, private-use / exotic symbols.
    if re.search(r"[{}%]\s*[A-Za-z0-9].*[{}%]|YIU\d|Cfi0ol|2tsf|CI14q", s):
        return True
    if re.search(r"[{}\x80-\x9fƏƈƍƌƉµ¶]", s) and (
        letters < max(8, len(s) // 2) or weird >= 3
    ):
        return True
    # Leftover encoded "shloka" crumbs before a real IAST line.
    if re.search(r"[µ¶]", s) and letters < 15:
        return True
    # Mostly quotes/accents with almost no words.
    if letters <= 3 and weird >= 3 and len(s) >= 6:
        return True
    return False


def strip_garbage_extraction_lines(text: str) -> tuple[str, int]:
    """Drop lines that are pure PDF font/encoding garbage."""
    kept: list[str] = []
    removed = 0
    for line in text.splitlines():
        if is_garbage_extraction_line(line):
            removed += 1
            continue
        kept.append(line)
    out = "\n".join(kept)
    out = MULTI_BLANK_RE.sub("\n\n", out)
    return out.strip() + "\n", removed


def _edge_alpha(s: str, *, from_end: bool) -> str | None:
    """First alphabetic character from the start or end, skipping punctuation/digits/hyphens."""
    text = s.rstrip() if from_end else s.lstrip()
    chars = reversed(text) if from_end else text
    for ch in chars:
        if ch.isalpha():
            return ch
    return None


def _wrap_join_gap(left: str, right: str) -> str | None:
    """
    How to join two adjacent wrapped lines.
    Returns '' (hyphenated word, no space), ' ' (space), or None.
    """
    left_r = left.rstrip()
    right_l = right.lstrip()
    if not left_r or not right_l:
        return None

    # Word-breaking hyphen at EOL (letter- then continuation): merge with no space
    # only when both edge letters are lowercase.
    if left_r[-1] in "-­–—" and len(left_r) >= 2 and left_r[-2].isalpha():
        a = left_r[-2]
        b = _edge_alpha(right_l, from_end=False)
        if a.islower() and b is not None and b.islower():
            return ""

    # Strip edge punctuation/digits/quotes/brackets/hyphens until a letter remains.
    # Join if the next line's edge letter is lowercase (current line may end with
    # lowercase or uppercase, e.g. "A" / "I" wrapping onto "well chosen…").
    a = _edge_alpha(left_r, from_end=True)
    b = _edge_alpha(right_l, from_end=False)
    if a is not None and b is not None and b.islower():
        return " "
    return None


def _merge_wrapped(left: str, right: str, gap: str) -> str:
    left_r = left.rstrip()
    right_l = right.lstrip()
    if gap == "":
        if left_r[-1] in "-­–—":
            left_r = left_r[:-1]
        return left_r + right_l
    return left_r + gap + right_l


def join_lowercase_wrapped_lines(text: str) -> tuple[str, int]:
    """
    Join mid-sentence PDF line wraps.

    Ignores trailing/leading non-letters (punctuation, digits, quotes, brackets,
    hyphens) when finding edge letters. Joins when the next line's edge letter is
    lowercase (current line may end in lower or upper case, e.g. "A" + "well…").
    Soft/hard hyphen at EOL still merges into one word when both letters are
    lowercase. Does not join across blank lines.
    """
    lines = text.splitlines()
    if not lines:
        return text, 0

    out: list[str] = []
    joins = 0
    i = 0
    while i < len(lines):
        cur = lines[i]
        if i + 1 < len(lines) and cur.strip() and lines[i + 1].strip():
            gap = _wrap_join_gap(cur, lines[i + 1])
            if gap is not None:
                joined = _merge_wrapped(cur, lines[i + 1], gap)
                joins += 1
                i += 2
                while i < len(lines) and lines[i].strip():
                    gap2 = _wrap_join_gap(joined, lines[i])
                    if gap2 is None:
                        break
                    joined = _merge_wrapped(joined, lines[i], gap2)
                    joins += 1
                    i += 1
                out.append(joined)
                continue
        out.append(cur)
        i += 1

    result = "\n".join(out)
    if text.endswith("\n"):
        result += "\n"
    return result, joins


def clean_extracted_text(text: str) -> tuple[str, dict[str, int]]:
    """Run all post-extraction text repairs. Returns text + fix counts."""
    stats: dict[str, int] = {}
    text, stats["ocr_digit"] = fix_ocr_digit_words(text)
    text, stats["ocr_typo"] = fix_common_ocr_typos(text)
    text, stats["garbage_lines"] = strip_garbage_extraction_lines(text)
    text, stats["line_joins"] = join_lowercase_wrapped_lines(text)
    text = MULTI_BLANK_RE.sub("\n\n", text)
    if not text.endswith("\n"):
        text += "\n"
    return text, stats


def is_font_encoded_line(line: str) -> bool:
    """True for legacy Devanagari-font lines like sT:av:rö j:¤m:ö …"""
    s = line.strip()
    if not s:
        return False
    colon_letters = len(re.findall(r"[A-Za-z]:", s))
    specials = len(re.findall(r"[öÎÂØðòùÜ¶¤½¾¹\*\\]", s))
    english = len(
        re.findall(
            r"\b(?:the|and|are|that|this|with|from|which|represented|"
            r"letters|Sanskrit|vibrations|Nadis|Chakra|each)\b",
            s,
            re.I,
        )
    )
    if english >= 3:
        return False
    if colon_letters >= 3 and specials >= 1:
        return True
    if colon_letters >= 2 and specials >= 2:
        return True
    # Short verse lines that are almost pure encoding
    if specials >= 2 and colon_letters >= 1 and len(s) <= 80 and english == 0:
        return True
    return False


def is_garbled_iast_line(line: str) -> bool:
    """True for PDF romanization lines that follow encoded shlokas."""
    s = line.strip()
    if not s or is_font_encoded_line(s):
        return False
    if not re.search(r"[àáâãäåæçèéêëìíîïñòóôõöùúûüýþÿ÷¤]", s):
        return False
    # Mostly Latin + diacritics / danda marks
    return bool(
        re.fullmatch(r"[A-Za-zà-ÿ÷øþ¤\.\,\|\;\:\-\s\"\'\(\)\/]+", s)
    )


def replace_font_encoded_shlokas(text: str) -> str:
    """
    Drop font-encoded Devanagari lines when a romanization follows,
    and repair that romanization into proper IAST English sounds.
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if is_font_encoded_line(lines[i]):
            j = i
            while j < len(lines) and (
                is_font_encoded_line(lines[j]) or not lines[j].strip()
            ):
                # Keep going through encoded lines; allow blank lines inside a verse
                if not lines[j].strip():
                    # peek: if next nonempty is still encoded, consume blank
                    k = j + 1
                    while k < len(lines) and not lines[k].strip():
                        k += 1
                    if k < len(lines) and is_font_encoded_line(lines[k]):
                        j = k
                        continue
                    break
                j += 1

            # Skip blanks / short markers between encoded block and romanization
            k = j
            interstitial: list[str] = []
            while k < len(lines):
                t = lines[k].strip()
                if not t:
                    k += 1
                    continue
                if t in {"OM", "!", "||"} or re.fullmatch(r"[|!\.\s]+", t):
                    interstitial.append(t)
                    k += 1
                    continue
                break

            if k < len(lines) and is_garbled_iast_line(lines[k]):
                out.extend(interstitial)
                while k < len(lines) and (
                    is_garbled_iast_line(lines[k]) or not lines[k].strip()
                ):
                    if lines[k].strip():
                        out.append(fix_garbled_iast(lines[k]))
                    else:
                        if out and out[-1] != "":
                            out.append("")
                    k += 1
                while out and out[-1] == "":
                    out.pop()
                i = k
                continue

            # No romanization pair: leave encoded lines for bija/other handling
            out.extend(lines[i:j])
            i = j
            continue

        out.append(lines[i])
        i += 1

    return "\n".join(out)


def is_page_number(line: str) -> bool:
    return bool(PAGE_NUMBER_RE.match(line.strip()))


def is_header_like(line: str) -> bool:
    """Heuristic for running headers / footers (short title-like lines)."""
    s = line.strip()
    if not s or is_page_number(s):
        return False
    if len(s) > 60 or len(s.split()) > 8:
        return False
    # Body sentences usually end with sentence punctuation.
    if s.endswith((".", "!", "?", ",", ";", ":", '"', "'")):
        return False
    return True


def extract_page_lines(page: pymupdf.Page) -> list[str]:
    text = page.get_text("text") or ""
    return [line.rstrip() for line in text.splitlines()]


def detect_running_headers(pages_lines: list[list[str]]) -> set[str]:
    """Find short lines that repeatedly appear at the top or bottom of pages."""
    edge_counts: Counter[str] = Counter()
    for lines in pages_lines:
        nonempty = [ln.strip() for ln in lines if ln.strip()]
        if not nonempty:
            continue
        for candidate in {nonempty[0], nonempty[-1]}:
            if not is_header_like(candidate):
                continue
            edge_counts[candidate] += 1

    # Chapter running headers may only span a short section.
    min_hits = 3
    return {line for line, count in edge_counts.items() if count >= min_hits}


def nonempty_edge(lines: list[str], from_end: bool) -> tuple[int | None, str | None]:
    indices = range(len(lines) - 1, -1, -1) if from_end else range(len(lines))
    for idx in indices:
        stripped = lines[idx].strip()
        if stripped:
            return idx, stripped
    return None, None


def strip_edge_noise(lines: list[str], headers: set[str]) -> list[str]:
    """Remove page numbers and running headers only at page edges."""
    result = list(lines)

    def strip_from_start() -> None:
        while True:
            idx, stripped = nonempty_edge(result, from_end=False)
            if idx is None or stripped is None:
                break
            if is_page_number(stripped) or stripped in headers:
                del result[: idx + 1]
                continue
            break

    def strip_from_end() -> None:
        while True:
            idx, stripped = nonempty_edge(result, from_end=True)
            if idx is None or stripped is None:
                break

            if is_page_number(stripped) or stripped in headers:
                del result[idx:]
                continue

            # Pattern: content, page number, header  OR  content, header, page number
            prev_idx, prev = nonempty_edge(result[:idx], from_end=True)
            if prev is not None and prev_idx is not None:
                if is_page_number(prev) and (
                    stripped in headers or is_header_like(stripped)
                ):
                    del result[prev_idx:]
                    continue
                if is_page_number(stripped) and (
                    prev in headers or is_header_like(prev)
                ):
                    del result[prev_idx:]
                    continue
            break

    strip_from_start()
    strip_from_end()
    return result


def clean_page_lines(lines: list[str], headers: set[str]) -> list[str]:
    edged = strip_edge_noise(lines, headers)
    cleaned: list[str] = []
    for line in edged:
        stripped = line.strip()
        if not stripped:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        cleaned.append(stripped)
    while cleaned and cleaned[0] == "":
        cleaned.pop(0)
    while cleaned and cleaned[-1] == "":
        cleaned.pop()
    return cleaned


def pages_to_continuous_text(pages_lines: list[list[str]], headers: set[str]) -> str:
    chunks: list[str] = []
    for page_idx, lines in enumerate(pages_lines):
        # Keep title-page text intact (book name may match a running header).
        page_headers: set[str] = set() if page_idx == 0 else headers
        cleaned = clean_page_lines(lines, page_headers)
        if cleaned:
            chunks.append("\n".join(cleaned))

    text = "\n\n".join(chunks)
    # Rejoin words split across line breaks with a hyphen
    text = HYPHEN_BREAK_RE.sub(r"\1\2", text)
    # Prefer the PDF's own romanization under each shloka; drop font glyphs
    text = replace_font_encoded_shlokas(text)
    text = fix_garbled_iast(text)
    text = transliterate_devanagari_font(text)
    text, _stats = clean_extracted_text(text)
    text = MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip() + "\n"


def convert_pdf(pdf_path: Path, output_dir: Path) -> Path:
    out_path = output_dir / (pdf_path.stem + ".txt")
    doc = pymupdf.open(pdf_path)
    try:
        pages_lines = [extract_page_lines(doc[i]) for i in range(doc.page_count)]
        headers = detect_running_headers(pages_lines)
        text = pages_to_continuous_text(pages_lines, headers)
    finally:
        doc.close()

    out_path.write_text(text, encoding="utf-8")
    return out_path


def repair_existing_txt(output_dir: Path) -> int:
    """Apply OCR / garbage cleanups to already-converted .txt files."""
    files = sorted(output_dir.glob("*.txt"))
    if not files:
        print(f"No .txt files in {output_dir}")
        return 0
    for path in files:
        original = path.read_text(encoding="utf-8")
        fixed, stats = clean_extracted_text(original)
        total = sum(stats.values())
        if total == 0 and fixed == original:
            print(f"OK {path.name} (no fixes)")
            continue
        path.write_text(fixed, encoding="utf-8")
        print(
            f"Repaired {path.name} "
            f"(digit={stats['ocr_digit']}, typo={stats['ocr_typo']}, "
            f"garbage_lines={stats['garbage_lines']}, "
            f"line_joins={stats['line_joins']})"
        )
    return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repair-txt",
        action="store_true",
        help="Re-scan book1_txt/*.txt and apply OCR digit-in-word fixes "
        "(does not re-convert PDFs).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-convert PDFs even if .txt already exists.",
    )
    args = parser.parse_args()

    if args.repair_txt:
        if not OUTPUT_DIR.is_dir():
            print(f"Output folder not found: {OUTPUT_DIR}", file=sys.stderr)
            return 1
        return repair_existing_txt(OUTPUT_DIR)

    if not INPUT_DIR.is_dir():
        print(f"Input folder not found: {INPUT_DIR}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files found in {INPUT_DIR}")
        return 0

    for pdf_path in pdfs:
        out_path = OUTPUT_DIR / (pdf_path.stem + ".txt")
        if out_path.exists() and not args.force:
            print(f"Skipped {pdf_path.name} (already converted: {out_path.name})")
            continue
        out_path = convert_pdf(pdf_path, OUTPUT_DIR)
        size = out_path.stat().st_size
        print(f"Wrote {out_path.name} ({size:,} bytes) from {pdf_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
