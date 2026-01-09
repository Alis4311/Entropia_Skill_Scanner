from pathlib import Path
import os

def configure_ocr_temp_dir() -> None:
    d = Path.home() / ".entropia_skillscanner" / "tmp"
    d.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = str(d)
    os.environ["TEMP"] = str(d)
    
configure_ocr_temp_dir()