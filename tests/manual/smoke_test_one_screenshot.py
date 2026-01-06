from pathlib import Path
import cv2

from pipeline import extract_skill_window, extract_table, extract_rows, extract_columns, parse_points_int

INPUT = Path("data\screenshots\Entropia 2026-01-02 23.45.03.png")

OUT_DIR = Path("out_test_one")
DEBUG_DIR = OUT_DIR / "_debug"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    if not INPUT.exists():
        print(f"ERROR: input not found: {INPUT.resolve()}")
        return 2

    bgr = cv2.imread(str(INPUT), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"ERROR: could not read image: {INPUT}")
        return 2

    # 1) Full screenshot -> normalized skills window crop (in-memory)
    sw = extract_skill_window(
        bgr,
        norm_width=1400,
        debug_dir=DEBUG_DIR / "skills_window",
    )
    if sw.norm_bgr is None:
        print(f"ERROR: skills window detection failed: {sw.reason} (conf={sw.confidence:.3f})")
        return 3

    cv2.imwrite(str(OUT_DIR / "01_norm_skills_window.png"), sw.norm_bgr)

    # 2) normalized -> fixed table crop
    ft = extract_table(
        sw.norm_bgr,
        debug_dir=DEBUG_DIR / "fixed_table",
    )
    x, y, w, h = ft.table_bbox
    table_bgr = sw.norm_bgr[y : y + h, x : x + w]
    cv2.imwrite(str(OUT_DIR / "02_table_roi.png"), table_bgr)

    if not ft.valid:
        print(f"WARNING: fixed table validation failed: {ft.reason} (density={ft.density:.4f})")

    # 3) table ROI -> rows
    rows = extract_rows(
        table_bgr,
        debug_dir=OUT_DIR / "03_rows",
    )

    print(f"OK: rows={len(rows.row_images)} reason={rows.reason}")
    if len(rows.row_images) == 0:
        print("ERROR: no rows extracted")
        return 4

    EXPECTED = 12
    if len(rows.row_images) != EXPECTED:
        print(f"ERROR: expected {EXPECTED} rows, got {len(rows.row_images)}")
        return 5
    cols = extract_columns(rows.row_images, debug_dir=OUT_DIR / "04_cols")

    res = parse_points_int(cols[0].points_bgr, debug_dir=OUT_DIR / "05_points")
    print(res.value, res.text, res.confidence, res.reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
