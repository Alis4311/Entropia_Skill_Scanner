from __future__ import annotations

import argparse
import cProfile
import io
import pstats
from pathlib import Path

import cv2 as cv

from pipeline.run_pipeline import run_pipeline, PipelineConfig


def profile_once(image_path: Path, *, debug: bool = False, top: int = 40) -> int:
    if not image_path.exists():
        print(f"ERROR: file not found: {image_path}")
        return 2

    bgr = cv.imread(str(image_path), cv.IMREAD_COLOR)
    if bgr is None:
        print(f"ERROR: could not read image: {image_path}")
        return 2

    cfg = PipelineConfig()

    pr = cProfile.Profile()
    pr.enable()
    rows, status = run_pipeline(cfg, bgr, debug=debug, debug_dir=None, logger=None)
    pr.disable()

    print(f"status: {status}")
    print(f"rows:   {len(rows)}")

    # Print a readable summary
    s = io.StringIO()
    stats = pstats.Stats(pr, stream=s)

    # Commonly useful: time spent in the function itself (tottime)
    # Another good view is cumulative time (cumtime)
    stats.sort_stats("tottime")
    stats.print_stats(top)
    stats.sort_stats("cumtime")
    stats.print_stats(top)
    print("\n=== cProfile (top by tottime) ===")
    print(s.getvalue())

    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path, help="Path to a screenshot (png/jpg)")
    ap.add_argument("--top", type=int, default=40, help="How many rows of stats to print")
    ap.add_argument("--debug", action="store_true", help="Run pipeline with debug=True (usually slower)")
    args = ap.parse_args()
    return profile_once(args.image, debug=args.debug, top=args.top)


if __name__ == "__main__":
    raise SystemExit(main())
