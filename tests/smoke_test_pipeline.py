from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2 as cv
import numpy as np

# Adjust these imports to your actual package layout
from pipeline.run_pipeline import run_pipeline, PipelineConfig


RowOut = Tuple[str, str]  # (skill_name, "1234.56")


VALUE_RE = re.compile(r"^\d+(\.\d{2})?$")  # allow ints or 2 decimals (depending on your formatting)
NAME_BAD = {"", "Placeholder", "Unknown", "N/A"}


@dataclass(frozen=True)
class SmokeConfig:
    min_rows: int = 5                 # low threshold: single screenshot should have some rows
    min_named_ratio: float = 0.80     # at least 80% names extracted
    min_value_ratio: float = 0.95     # at least 95% values parse as numeric
    require_decimals: bool = False    # flip to True once decimals are guaranteed everywhere


def _load_bgr(path: Path) -> np.ndarray:
    img = cv.imread(str(path), cv.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to load image: {path}")
    return img


def _is_value_ok(s: str, require_decimals: bool) -> bool:
    s = s.strip()
    if not VALUE_RE.match(s):
        return False
    if require_decimals and "." not in s:
        return False
    return True


def _run_one(cfg: PipelineConfig, img_path: Path, scfg: SmokeConfig, debug: bool) -> None:
    bgr = _load_bgr(img_path)
    rows, status = run_pipeline(cfg, bgr, debug=debug, debug_dir=img_path.parent / "_debug")

    assert isinstance(status, str) and status, f"{img_path.name}: empty status"
    assert isinstance(rows, list), f"{img_path.name}: rows not a list"
    assert len(rows) >= scfg.min_rows, f"{img_path.name}: too few rows ({len(rows)})"

    names = [n for (n, _) in rows]
    values = [v for (_, v) in rows]

    named_ok = sum(1 for n in names if n.strip() not in NAME_BAD)
    values_ok = sum(1 for v in values if _is_value_ok(v, scfg.require_decimals))

    named_ratio = named_ok / max(1, len(rows))
    value_ratio = values_ok / max(1, len(rows))

    assert named_ratio >= scfg.min_named_ratio, (
        f"{img_path.name}: name ratio too low "
        f"({named_ratio:.2%}, {named_ok}/{len(rows)})"
    )
    assert value_ratio >= scfg.min_value_ratio, (
        f"{img_path.name}: value ratio too low "
        f"({value_ratio:.2%}, {values_ok}/{len(rows)})"
    )

    # Ensure no duplicate exact rows (often indicates row-splitting bug)
    assert len(set(rows)) >= int(0.7 * len(rows)), (
        f"{img_path.name}: too many duplicate rows, possible row extraction issue"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images_dir", type=str, help="Folder containing screenshots (png/jpg)")
    ap.add_argument("--debug", action="store_true", help="Enable pipeline debug output")
    ap.add_argument("--min-rows", type=int, default=5)
    ap.add_argument("--min-named-ratio", type=float, default=0.80)
    ap.add_argument("--min-value-ratio", type=float, default=0.95)
    ap.add_argument("--require-decimals", action="store_true")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    assert images_dir.exists(), f"Missing folder: {images_dir}"

    paths: List[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
        paths.extend(images_dir.glob(ext))
    paths = sorted(paths)

    assert paths, f"No images found in {images_dir}"

    scfg = SmokeConfig(
        min_rows=args.min_rows,
        min_named_ratio=args.min_named_ratio,
        min_value_ratio=args.min_value_ratio,
        require_decimals=args.require_decimals,
    )

    cfg = PipelineConfig()

    failed = 0
    for p in paths:
        try:
            _run_one(cfg, p, scfg, debug=args.debug)
            print(f"[OK]   {p.name}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {p.name}: {e}")

    if failed:
        raise SystemExit(f"{failed}/{len(paths)} images failed smoke test")
    print(f"All {len(paths)} images passed.")


if __name__ == "__main__":
    main()
