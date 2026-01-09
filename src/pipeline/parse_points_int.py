# src/pipeline/parse_points_int.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List, Sequence, Tuple

import cv2
import numpy as np
import pytesseract


@dataclass(frozen=True)
class PointsIntResult:
    value: Optional[int]
    text: str
    confidence: float
    reason: str


@dataclass(frozen=True)
class PointsIntBatchResult:
    values: List[Optional[int]]
    results: List[PointsIntResult]
    ok_count: int
    reason: str


def _prep_points_int_band(
    points_bgr: np.ndarray,
    *,
    upscale: int,
    blur: bool,
    top_band_frac: float,
) -> np.ndarray:
    """
    Returns bw_num (top-band binarized digits region) for one ROI.
    Mirrors parse_points_int preprocessing, but returns the processed image.
    """
    gray = cv2.cvtColor(points_bgr, cv2.COLOR_BGR2GRAY)

    if upscale and upscale > 1:
        gray = cv2.resize(
            gray,
            (gray.shape[1] * upscale, gray.shape[0] * upscale),
            interpolation=cv2.INTER_CUBIC,
        )

    if blur:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if float(np.mean(bw)) < 127.0:
        bw = 255 - bw

    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    H, W = bw.shape[:2]
    cut = int(round(np.clip(top_band_frac, 0.20, 1.00) * H))
    bw_num = bw[:cut, :]
    return bw_num


def parse_points_int_batch(
    points_rois: Sequence[np.ndarray],
    *,
    upscale: int = 3,
    blur: bool = True,
    top_band_frac: float = 0.55,
    debug_dir: Union[str, Path, None] = None,
    min_confidence: float = 0.50,
) -> PointsIntBatchResult:
    """
    TRUE batched parse: one pytesseract call total.

    Strategy:
      - preprocess each ROI into bw_num (top band digits)
      - pad to same width and stack vertically with separators
      - run pytesseract.image_to_data ONCE
      - assign detected words to rows by their y-position (top)
      - per row: join words -> digits -> int, confidence=mean(word conf)/100
    """
    n = len(points_rois)
    if n == 0:
        return PointsIntBatchResult(values=[], results=[], ok_count=0, reason="empty input")

    dbg = Path(debug_dir) if debug_dir else None
    if dbg:
        dbg.mkdir(parents=True, exist_ok=True)

    # Preprocess all
    bw_list: List[np.ndarray] = []
    for i, roi in enumerate(points_rois):
        if roi is None or roi.size == 0:
            bw_list.append(np.zeros((10, 10), dtype=np.uint8))
            continue
        bw_num = _prep_points_int_band(roi, upscale=upscale, blur=blur, top_band_frac=top_band_frac)
        bw_list.append(bw_num)
        if dbg:
            cv2.imwrite(str(dbg / f"row_{i:02d}_bw_num.png"), bw_num)

    # Stack with known row boundaries
    max_w = max(im.shape[1] for im in bw_list)
    pad_x = 10
    pad_y = 6
    sep_h = 18

    row_spans: List[Tuple[int, int]] = []  # (y0,y1) in big image for each row block (excluding sep)
    blocks: List[np.ndarray] = []

    y_cursor = 0
    for im in bw_list:
        if im.shape[1] < max_w:
            im = cv2.copyMakeBorder(im, 0, 0, 0, max_w - im.shape[1], cv2.BORDER_CONSTANT, value=255)
        im = cv2.copyMakeBorder(im, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=255)

        y0 = y_cursor
        y1 = y_cursor + im.shape[0]
        row_spans.append((y0, y1))

        blocks.append(im)
        y_cursor = y1 + sep_h  # include separator gap

    sep = 255 * np.ones((sep_h, max_w + 2 * pad_x), dtype=np.uint8)
    big = blocks[0]
    for b in blocks[1:]:
        big = np.vstack([big, sep, b])

    if dbg:
        cv2.imwrite(str(dbg / "batch_big.png"), big)

    # One OCR call total
    config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
    data = pytesseract.image_to_data(big, config=config, output_type=pytesseract.Output.DICT)

    # Collect per-row words + confidences
    row_words: List[List[Tuple[int, str]]] = [[] for _ in range(n)]  # (left, text)
    row_confs: List[List[float]] = [[] for _ in range(n)]

    texts = data.get("text", []) or []
    confs = data.get("conf", []) or []
    tops  = data.get("top", []) or []
    lefts = data.get("left", []) or []

    for t, c, y, x in zip(texts, confs, tops, lefts):
        if not t or not str(t).strip():
            continue

        # confidence may be "-1"
        try:
            cf = float(c)
        except Exception:
            cf = -1.0
        if cf < 0:
            continue

        # assign to row by y
        yi = int(y)
        row_idx = None
        for i, (y0, y1) in enumerate(row_spans):
            if y0 <= yi < y1:
                row_idx = i
                break
        if row_idx is None:
            continue

        row_words[row_idx].append((int(x), str(t)))
        row_confs[row_idx].append(cf)

    results: List[PointsIntResult] = []
    values: List[Optional[int]] = []
    ok = 0

    for i in range(n):
        words = [w for _, w in sorted(row_words[i], key=lambda p: p[0])]
        raw_text = " ".join(words).strip()
        digits = "".join(ch for ch in raw_text if ch.isdigit())

        conf = 0.0
        if row_confs[i]:
            conf = float(np.clip(np.mean(row_confs[i]) / 100.0, 0.0, 1.0))

        if digits == "":
            r = PointsIntResult(None, raw_text, conf, "no digits recognized")
            results.append(r)
            values.append(None)
            continue

        try:
            val = int(digits)
        except Exception:
            r = PointsIntResult(None, raw_text, conf, "int conversion failed")
            results.append(r)
            values.append(None)
            continue

        r = PointsIntResult(val, digits, conf, "ok")
        results.append(r)
        values.append(val)
        if conf >= min_confidence:
            ok += 1

    reason = "ok" if ok == n else f"ok {ok}/{n} (min_conf={min_confidence:.2f})"
    return PointsIntBatchResult(values=values, results=results, ok_count=ok, reason=reason)

@dataclass(frozen=True)
class PointsIntResult:
    value: Optional[int]
    text: str
    confidence: float
    reason: str


def parse_points_int(
    points_bgr: np.ndarray,
    *,
    upscale: int = 3,
    blur: bool = True,
    psm: int = 7,
    top_band_frac: float = 0.55,  # NEW: keep only top part (removes the bar)
    debug_dir: Union[str, Path, None] = None,
) -> PointsIntResult:
    """
    Parse INTEGER part from points_roi (no decimals yet).
    """
    if points_bgr is None or points_bgr.size == 0:
        return PointsIntResult(None, "", 0.0, "empty input")

    dbg = Path(debug_dir) if debug_dir else None
    if dbg:
        dbg.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dbg / "00_points_roi.png"), points_bgr)

    gray = cv2.cvtColor(points_bgr, cv2.COLOR_BGR2GRAY)

    # Upscale helps OCR a lot on UI fonts
    if upscale and upscale > 1:
        gray = cv2.resize(
            gray,
            (gray.shape[1] * upscale, gray.shape[0] * upscale),
            interpolation=cv2.INTER_CUBIC,
        )

    if blur:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if float(np.mean(bw)) < 127.0:
        bw = 255 - bw

    # Light cleanup
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    # Further crop to remove bar
    H, W = bw.shape[:2]
    cut = int(round(np.clip(top_band_frac, 0.20, 1.00) * H))
    bw_num = bw[:cut, :]

    if dbg:
        cv2.imwrite(str(dbg / "01_gray.png"), gray)
        cv2.imwrite(str(dbg / "02_bw.png"), bw)
        cv2.imwrite(str(dbg / "03_bw_num.png"), bw_num)

    config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789"

    txt = pytesseract.image_to_string(bw_num, config=config) or ""
    txt = txt.strip()
    digits = "".join(ch for ch in txt if ch.isdigit())

    conf = _tesseract_confidence(bw_num, config=config)

    if digits == "":
        return PointsIntResult(None, txt, conf, "no digits recognized")

    try:
        val = int(digits)
    except Exception:
        return PointsIntResult(None, txt, conf, "int conversion failed")

    return PointsIntResult(val, digits, conf, "ok")


def _tesseract_confidence(bw: np.ndarray, config: str) -> float:
    """
    Returns a rough confidence in [0..1] using pytesseract image_to_data.
    """
    try:
        data = pytesseract.image_to_data(bw, config=config, output_type=pytesseract.Output.DICT)
        confs = []
        for c in data.get("conf", []):
            try:
                ci = float(c)
                if ci >= 0:  # -1 means "not a word"
                    confs.append(ci)
            except Exception:
                pass
        if not confs:
            return 0.0
        # conf is 0..100
        return float(np.clip(np.mean(confs) / 100.0, 0.0, 1.0))
    except Exception:
        return 0.0
