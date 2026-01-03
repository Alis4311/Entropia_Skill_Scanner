from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2 as cv
import numpy as np

from pipeline.run_pipeline import run_pipeline, PipelineConfig


RowOut = Tuple[str, str]  # ("Placeholder", "1234.56")


def _iter_inputs(paths: List[Path], exts={".png", ".jpg", ".jpeg", ".bmp", ".webp"}) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted([x for x in p.rglob("*") if x.suffix.lower() in exts and x.is_file()]))
        else:
            out.append(p)
    return out


def _load_bgr(path: Path) -> np.ndarray:
    bgr = cv.imread(str(path), cv.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return bgr


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="entropia-skillscanner", description="Headless regression runner for skill scanner.")
    ap.add_argument("inputs", nargs="+", help="Image files or directories (recursively scanned).")
    ap.add_argument("--debug-dir", default=None, help="If set, pipeline debug artifacts go here.")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of CSV.")
    ap.add_argument("--fail-on-empty", action="store_true", help="Fail if no rows are produced.")
    ap.add_argument("--norm-width", type=int, default=1400)
    ap.add_argument("--min-table-density", type=float, default=0.010)
    args = ap.parse_args(argv)

    input_paths = _iter_inputs([Path(x) for x in args.inputs])
    if not input_paths:
        print("No input images found.", file=sys.stderr)
        return 2

    cfg = PipelineConfig(norm_width=args.norm_width, min_table_density=args.min_table_density)

    all_results = []
    overall_ok = True

    for img_path in input_paths:
        debug_dir = Path(args.debug_dir) / img_path.stem if args.debug_dir else None

        logs: List[str] = []
        def logger(msg: str):
            logs.append(msg)

        try:
            bgr = _load_bgr(img_path)
            rows, status = run_pipeline(
                cfg,
                bgr,
                debug_dir=debug_dir,
                logger=logger,
            )
        except Exception as e:
            rows, status = [], f"error:exception:{e}"

        ok = (status == "ok")
        if args.fail_on_empty and not rows:
            ok = False
            if status == "ok":
                status = "error:empty"

        overall_ok = overall_ok and ok

        all_results.append(
            {
                "input": str(img_path),
                "status": status,
                "rows": rows,      # list of [name,value] pairs
                "logs": logs,
            }
        )

    if args.json:
        print(json.dumps(all_results, indent=2))
    else:
        # CSV to stdout: input, status, skill_name, skill_value
        print("input,status,skill_name,skill_value")
        for r in all_results:
            if not r["rows"]:
                print(f"{r['input']},{r['status']},,")
                continue
            for name, val in r["rows"]:
                print(f"{r['input']},{r['status']},{name},{val}")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
