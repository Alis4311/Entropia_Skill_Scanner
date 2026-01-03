# extract_skill_name.py
from __future__ import annotations

import hashlib
from typing import Dict, List

import cv2 as cv
import numpy as np
import pytesseract


# -------- internal helpers --------

def _hash_img(img: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(img.tobytes())
    h.update(str(img.shape).encode("utf-8"))
    return h.hexdigest()


def _preprocess(name_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocess skill-name ROI for OCR.
    Optimized for bright text on dark Entropia UI background.
    """
    gray = cv.cvtColor(name_bgr, cv.COLOR_BGR2GRAY)

    # Boost contrast
    gray = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX)

    # Invert: black text on white bg for Tesseract
    gray = 255 - gray

    # Otsu threshold
    _, bw = cv.threshold(
        gray, 0, 255,
        cv.THRESH_BINARY | cv.THRESH_OTSU
    )

    # Light morphology to clean edges
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    bw = cv.morphologyEx(bw, cv.MORPH_OPEN, kernel)

    return bw


def _clean_text(s: str) -> str:
    s = s.strip()

    # Common OCR quirks
    s = s.replace("|", "I")
    s = s.replace("’", "'")

    # Keep only characters skill names actually use
    s = "".join(ch for ch in s if ch.isalpha() or ch in " -'")

    # Normalize whitespace
    s = " ".join(s.split())
    return s


def _tesseract_config() -> str:
    return (
        '--oem 3 --psm 6 '
        '-c "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-\' "'
    )


# -------- public API --------

# Cache still useful if the *same* ROI is processed repeatedly (rare),
# but keep it for compatibility.
_ocr_cache: Dict[str, str] = {}


def extract_skill_names_batched(name_bgr_list: List[np.ndarray]) -> List[str]:
    """
    Batched OCR: one pytesseract call for N rows.

    Returns list of cleaned skill names aligned to input order.
    """
    if not name_bgr_list:
        return []

    # Preprocess each crop (cheap compared to OCR)
    bw_list = [_preprocess(bgr) for bgr in name_bgr_list]

    # Normalize widths so we can stack cleanly
    max_w = max(im.shape[1] for im in bw_list)

    # Pad + separator to discourage cross-row line merges
    pad_x = 10
    pad_y = 6
    sep_h = 18

    sep = 255 * np.ones((sep_h, max_w + 2 * pad_x), dtype=np.uint8)

    blocks = []
    for im in bw_list:
        if im.shape[1] < max_w:
            im = cv.copyMakeBorder(im, 0, 0, 0, max_w - im.shape[1], cv.BORDER_CONSTANT, value=255)
        im = cv.copyMakeBorder(im, pad_y, pad_y, pad_x, pad_x, cv.BORDER_CONSTANT, value=255)
        blocks.append(im)

    big = blocks[0]
    for b in blocks[1:]:
        big = np.vstack([big, sep, b])

    raw = pytesseract.image_to_string(big, lang="eng", config=_tesseract_config())

    # Split into non-empty lines, then clean
    lines = [ln for ln in raw.splitlines() if ln.strip()]

    cleaned: List[str] = []
    for ln in lines:
        name = _clean_text(ln)
        if len(name) < 2:
            name = ""
        cleaned.append(name)

    # Force length match (caller expects same count)
    if len(cleaned) < len(name_bgr_list):
        cleaned += [""] * (len(name_bgr_list) - len(cleaned))
    else:
        cleaned = cleaned[: len(name_bgr_list)]

    return cleaned


def extract_skill_name(name_bgr: np.ndarray) -> str:
    """
    Backwards-compatible single-ROI API.
    Uses cache + batched OCR (N=1).
    """
    key = _hash_img(name_bgr)
    if key in _ocr_cache:
        return _ocr_cache[key]

    name = extract_skill_names_batched([name_bgr])[0]
    _ocr_cache[key] = name
    return name
