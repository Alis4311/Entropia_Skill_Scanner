from pathlib import Path
import time

_LOG_PATH = Path.home() / "entropia_skillscanner_timing.log"

def timing_log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with _LOG_PATH.open("a", encoding="utf-8", errors="replace") as f:
        f.write(f"{ts} {msg}\n")