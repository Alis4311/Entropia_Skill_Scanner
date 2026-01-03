from pathlib import Path
from pipeline.extract_skill_name import extract_skill_name
import cv2 as cv
# ---- Hardcoded input ----
INPUT = Path("out_test_one/04_cols/row_01/01_name_roi.png")
def main() -> int:
    if not INPUT.exists():
        print(f"ERROR: input not found: {INPUT.resolve()}")
        return 2

    bgr = cv.imread(str(INPUT), cv.IMREAD_COLOR)
    if bgr is None:
        print(f"ERROR: could not read image: {INPUT}")
        return 2
    
    print(extract_skill_name(bgr))

if __name__ == "__main__":
    raise SystemExit(main())