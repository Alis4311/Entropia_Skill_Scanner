from __future__ import annotations

from pathlib import Path
import sys
import cv2

from pipeline import (
    extract_skill_window,
    extract_table,
    extract_rows,
    extract_columns,
)

from pipeline.parse_points_decimal import parse_points_decimal_from_bar


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: python tests/debug_points_decimal_from_screenshot.py <screenshot.png> <out_dir>")
        return 2

    input_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    debug_dir = out_dir / "_debug"

    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"ERROR: input not found: {input_path.resolve()}")
        return 2

    bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"ERROR: could not read image: {input_path}")
        return 2

    # 1) Full screenshot -> normalized skills window crop
    sw = extract_skill_window(
        bgr,
        norm_width=1400,
        debug_dir=debug_dir / "skills_window",
    )
    if sw.norm_bgr is None:
        print(f"ERROR: skills window detection failed: {sw.reason} (conf={sw.confidence:.3f})")
        return 3

    cv2.imwrite(str(out_dir / "01_norm_skills_window.png"), sw.norm_bgr)

    # 2) normalized -> fixed table crop
    ft = extract_table(
        sw.norm_bgr,
        debug_dir=debug_dir / "fixed_table",
    )
    x, y, w, h = ft.table_bbox
    table_bgr = sw.norm_bgr[y : y + h, x : x + w]
    cv2.imwrite(str(out_dir / "02_table_roi.png"), table_bgr)

    if not ft.valid:
        print(f"WARNING: fixed table validation failed: {ft.reason} (density={ft.density:.4f})")

    # 3) table ROI -> rows
    rows = extract_rows(
        table_bgr,
        debug_dir=out_dir / "03_rows",
    )

    print(f"OK: rows={len(rows.row_images)} reason={rows.reason}")
    if len(rows.row_images) == 0:
        print("ERROR: no rows extracted")
        return 4

    # 4) rows -> columns
    cols = extract_columns(rows.row_images, debug_dir=out_dir / "04_cols")
    if len(cols) != len(rows.row_images):
        print(f"WARNING: columns count mismatch: cols={len(cols)} rows={len(rows.row_images)}")

    # 5) per row: points_bgr -> decimal (with debug)
    dec_root = out_dir / "05_decimals"
    dec_root.mkdir(parents=True, exist_ok=True)

    for i, c in enumerate(cols, start=1):
        points_bgr = c.points_bgr
        if points_bgr is None or points_bgr.size == 0:
            print(f"Row {i:02d}: ERROR points_bgr empty")
            continue

        row_dbg = dec_root / f"row_{i:02d}"
        res = parse_points_decimal_from_bar(
            points_bgr,
            debug_dir=row_dbg,
        )

        print(f"Row {i:02d}: frac={res.fraction} dec={res.decimal} conf={res.confidence:.2f} reason={res.reason}")

    print(f"Done. Output: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
