from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .parse_points_int import PointsIntResult, parse_points_int
from .parse_points_decimal import PointsDecimalResult, parse_points_decimal_from_bar


@dataclass(frozen=True)
class PointsValueResult:
    value: Optional[float]          # int + decimal (e.g. 7480.89)
    int_value: Optional[int]
    decimal_value: Optional[float]  # 0.00..0.99
    confidence: float
    reason: str

    int_res: PointsIntResult
    dec_res: PointsDecimalResult


def parse_points_value(
    points_bgr: np.ndarray,
    *,
    # int OCR options
    upscale: int = 3,
    blur: bool = True,
    psm: int = 7,
    top_band_frac: float = 0.55,
    # decimal options
    debug_dir: Union[str, Path, None] = None,
) -> PointsValueResult:
    """
    points_roi -> float value (integer OCR + decimal bar).

    Debug layout (if debug_dir is provided):
      debug_dir/
        int/...
        dec/...
    """
    dbg = Path(debug_dir) if debug_dir else None

    int_dbg = (dbg / "int") if dbg else None
    dec_dbg = (dbg / "dec") if dbg else None

    int_res = parse_points_int(
        points_bgr,
        upscale=upscale,
        blur=blur,
        psm=psm,
        top_band_frac=top_band_frac,
        debug_dir=int_dbg,
    )

    dec_res = parse_points_decimal_from_bar(
        points_bgr,
        debug_dir=dec_dbg,
    )

    # Combine
    if int_res.value is None:
        # If integer failed, we can't produce a numeric value.
        conf = float(np.clip(0.7 * int_res.confidence + 0.3 * dec_res.confidence, 0.0, 1.0))
        return PointsValueResult(
            value=None,
            int_value=None,
            decimal_value=dec_res.decimal,
            confidence=conf,
            reason=f"int failed: {int_res.reason}",
            int_res=int_res,
            dec_res=dec_res,
        )

    # Decimal may be missing; default to 0.00 if we have an int
    dec_val = dec_res.decimal if dec_res.decimal is not None else 0.0
    value = float(int_res.value + dec_val)

    conf = float(np.clip(0.7 * int_res.confidence + 0.3 * dec_res.confidence, 0.0, 1.0))
    reason = "ok"
    if dec_res.decimal is None:
        reason = f"ok (decimal missing: {dec_res.reason})"
    elif not dec_res.reason.startswith("ok"):
        reason = f"ok (decimal warn: {dec_res.reason})"

    return PointsValueResult(
        value=value,
        int_value=int_res.value,
        decimal_value=dec_val,
        confidence=conf,
        reason=reason,
        int_res=int_res,
        dec_res=dec_res,
    )
