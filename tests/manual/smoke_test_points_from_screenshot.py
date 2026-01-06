from pathlib import Path
import cv2

from pipeline import extract_skill_window, extract_table, extract_rows
from pipeline.extract_columns import extract_columns_from_rows
from pipeline.parse_points_int import parse_points_int_batch
from pipeline.parse_points_value import parse_points_value
from pipeline.extract_skill_name import extract_skill_name

# ---- Hardcoded input ----
INPUT = Path("data/screenshots/1024x768-125.png")

# ---- Output folders ----
OUT_DIR = Path("out_points_test")
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

    # 1) screenshot -> normalized skills window
    sw = extract_skill_window(
        bgr,
        norm_width=1400,
        debug_dir=DEBUG_DIR / "01_skill_window",
    )
    if sw.norm_bgr is None:
        print(f"ERROR: skills window failed: {sw.reason} (conf={sw.confidence:.3f})")
        return 3
    cv2.imwrite(str(OUT_DIR / "01_norm_skills_window.png"), sw.norm_bgr)

    # 2) normalized -> table ROI
    tbl = extract_table(
        sw.norm_bgr,
        debug_dir=DEBUG_DIR / "02_table",
    )
    if tbl.table_bgr is None:
        print(f"ERROR: table crop failed: {tbl.reason}")
        return 4
    cv2.imwrite(str(OUT_DIR / "02_table_roi.png"), tbl.table_bgr)
    if not tbl.valid:
        print(f"WARNING: table validation failed: {tbl.reason} (density={tbl.density:.4f})")

    # 3) table ROI -> rows
    rows = extract_rows(
        tbl.table_bgr,
        debug_dir=OUT_DIR / "03_rows",
    )
    if not rows.row_images:
        print(f"ERROR: row extraction failed: {rows.reason}")
        return 5

    # 4) rows -> columns (we only need points ROI)
    cols = extract_columns_from_rows(rows.row_images, debug_dir=OUT_DIR / "04_columns")
    points_rois = [c.points_bgr for c in cols]
    name_rois = [c.name_bgr for c in cols]

    for i, roi in enumerate(points_rois):
        r = parse_points_value(roi, debug_dir=OUT_DIR / "05_points_value" / f"row_{i:02d}")
        print(i, r.value, r.confidence, r.reason)
    
    for i, roi in enumerate(name_rois):
        r = extract_skill_name(roi)
        print(i,r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
