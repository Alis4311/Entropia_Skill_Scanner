# pipeline/skill_vocab.py
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple
from entropia_skillscanner.resources import resource_path
from rapidfuzz import fuzz, process


def _norm(s: str) -> str:
    s = s.strip().lower()
    s = "".join(ch for ch in s if ch.isalpha() or ch in " -'")
    s = " ".join(s.split())
    return s


@dataclass(frozen=True)
class Vocab:
    canon: List[str]              # canonical names
    norm: List[str]               # normalized names (unique, same order)
    norm_to_canon: Dict[str, str] # mapping normalized -> canonical

def load_vocab(path: Path) -> Vocab:
    lines = path.read_text(encoding="utf-8").splitlines()
    canon = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]

    norm: List[str] = []
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
    norm_unique: List[str] = []
    for n in norm:
        if n not in seen:
            seen.add(n)
            norm_unique.append(n)

    return Vocab(canon=canon, norm=norm_unique, norm_to_canon=norm_to_canon)


# ---- OCR-variant helper ----
# Common OCR confusions observed in Entropia skill UI (tesseract):
# i<->l and l<->t show up a lot ("Aim"->"Alm", "Biology"->"Blotogy").
_OCR_SUBS: Dict[str, Tuple[str, ...]] = {
    "l": ("l", "i", "t"),
    "i": ("i", "l", "t"),
    "t": ("i", "l" , "t"),
    "c": ("c", "g"),
    "g": ("g", "c"),
}


def _ocr_variants(q: str, *, max_variants: int = 64) -> List[str]:
    """
    Generate a bounded set of variants for common OCR confusions.
    Keeps runtime bounded.
    """
    choices = [_OCR_SUBS.get(ch, (ch,)) for ch in q]

    out: List[str] = []
    for tup in product(*choices):
        out.append("".join(tup))
        if len(out) >= max_variants:
            break

    # de-dupe preserving order
    seen = set()
    uniq: List[str] = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def snap_name(raw_ocr: str, vocab: Vocab) -> tuple[str, int]:
    """
    Snap an OCR string to the nearest canonical skill name.

    - Normalize query
    - Restrict candidate pool by length (reduces false positives, helps margins)
    - Try original query + bounded OCR-variants (i<->l, l<->t) and keep best score
    - Apply strict thresholds so we don't over-snap
    """
    q = _norm(raw_ocr)
    if len(q) < 2 or not vocab.norm:
        return ("", 0)

    # Restrict candidates by length band (helps both accuracy and speed).
    # +/-1 covers common missing/extra glyphs without opening floodgates.
    pool = [n for n in vocab.norm if abs(len(n) - len(q)) <= 1] or vocab.norm

    # For very short queries, restrict to exact-length when possible
    if len(q) <= 4:
        exact = [n for n in pool if len(n) == len(q)]
        if exact:
            pool = exact

    best_norm = ""
    best_score = -1

    queries = [q]
    subs_count = sum(1 for ch in q if ch in _OCR_SUBS)
    if subs_count <= 4:
        queries += _ocr_variants(q)

    # Find best across variants
    for qq in queries:
        matches = process.extract(qq, pool, scorer=fuzz.ratio, limit=1)
        if not matches:
            continue
        cand_norm, score, _ = matches[0]
        score_i = int(score)
        if score_i > best_score:
            best_score = score_i
            best_norm = cand_norm
            if best_score == 100:
                break

    if not best_norm or best_score < 0:
        return ("", 0)

    matches2 = process.extract(q, pool, scorer=fuzz.ratio, limit=2)
    second_score = int(matches2[1][1]) if matches2 and len(matches2) > 1 else 0

    if best_score >= 92:
        return (vocab.norm_to_canon[best_norm], best_score)
    if best_score >= 85 and (best_score - second_score) >= 6:
        return (vocab.norm_to_canon[best_norm], best_score)

    return ("", best_score)


VOCAB_PATH = resource_path("src/pipeline/skills_vocab.txt")
VOCAB = load_vocab(VOCAB_PATH)
