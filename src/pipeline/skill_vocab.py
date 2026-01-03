# pipeline/skill_vocab.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from rapidfuzz import fuzz, process


def _norm(s: str) -> str:
    s = s.strip().lower()
    s = "".join(ch for ch in s if ch.isalpha() or ch in " -'")
    s = " ".join(s.split())
    return s


@dataclass(frozen=True)
class Vocab:
    canon: List[str]              # canonical names
    norm: List[str]               # normalized names (same order)
    norm_to_canon: Dict[str, str] # mapping normalized -> canonical


def load_vocab(path: Path) -> Vocab:
    lines = path.read_text(encoding="utf-8").splitlines()
    canon = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]

    norm = []
    norm_to_canon: Dict[str, str] = {}
    for c in canon:
        n = _norm(c)
        if not n:
            continue
        # keep first occurrence if duplicates normalize the same
        norm_to_canon.setdefault(n, c)
        norm.append(n)

    # keep norm list unique while preserving order
    seen = set()
    norm_unique = []
    for n in norm:
        if n not in seen:
            seen.add(n)
            norm_unique.append(n)

    return Vocab(canon=canon, norm=norm_unique, norm_to_canon=norm_to_canon)


def snap_name(raw_ocr: str, vocab: Vocab) -> tuple[str, int]:
    q = _norm(raw_ocr)
    if len(q) < 2 or not vocab.norm:
        return ("", 0)

    # get top 2 for a margin check
    matches = process.extract(
        q,
        vocab.norm,
        scorer=fuzz.ratio,
        limit=2,
    )
    if not matches:
        return ("", 0)

    best_norm, best_score, _ = matches[0]
    second_score = matches[1][1] if len(matches) > 1 else 0
    if best_score >= 92:
        return (vocab.norm_to_canon[best_norm], int(best_score))
    if best_score >= 85 and (best_score - second_score) >= 6:
        return (vocab.norm_to_canon[best_norm], int(best_score))

    return ("", int(best_score))


VOCAB_PATH = Path(__file__).with_name("skills_vocab.txt")
VOCAB = load_vocab(VOCAB_PATH)