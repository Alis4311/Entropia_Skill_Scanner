# src/entropia_skillscanner/audio_feedback.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import sys
import time
import threading

try:
    import winsound  # Windows only (built-in)
except Exception:  # pragma: no cover
    winsound = None  # type: ignore


def _resource_path(relative: str) -> Path:
    """
    Resolve a resource path that works:
    - in development (normal package files)
    - in PyInstaller onefile (sys._MEIPASS)
    - in PyInstaller onedir
    """
    # PyInstaller sets sys._MEIPASS to the temp extraction dir (onefile)
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return (base / relative).resolve()


def _dev_asset_path(relative: str) -> Path:
    """
    Dev fallback: when running from source tree, the assets live alongside the package.
    """
    return (Path(__file__).resolve().parent / relative).resolve()


@dataclass(frozen=True)
class AudioFeedbackConfig:
    enabled: bool = True

    # Rate limit only for the "start click", to avoid spam if user triggers frequently.
    start_min_interval_s: float = 0.6


class AudioFeedback:
    """
    Small sound-effects service with:
    - enable toggle
    - start-click rate limiting
    - non-blocking playback
    - PyInstaller-friendly resource resolution
    """

    def __init__(self, cfg: AudioFeedbackConfig | None = None):
        self.cfg = cfg or AudioFeedbackConfig()
        self._lock = threading.Lock()
        self._last_start_ts = 0.0

        
        self._click_rel = "entropia_skillscanner/assets/sfx/click.wav"
        self._success_rel = "entropia_skillscanner/assets/sfx/success.wav"
        self._warn_rel = "entropia_skillscanner/assets/sfx/warn.wav"

    def play_start(self) -> None:
        if not self._should_play_start():
            return
        self._play(self._click_rel)

    def play_success(self) -> None:
        self._play(self._success_rel)

    def play_warn(self) -> None:
        self._play(self._warn_rel)

    def play_error(self) -> None:
        # If you don’t want a separate error sound, keep it same as warn.
        self._play(self._warn_rel)

    def _should_play_start(self) -> bool:
        print("_should_play_start")
        if not self.cfg.enabled:
            return False
        now = time.monotonic()
        with self._lock:
            if (now - self._last_start_ts) < self.cfg.start_min_interval_s:
                return False
            self._last_start_ts = now
            return True

    def _resolve_wav(self, relative: str) -> Path:
        # First try PyInstaller-style resource root
        p = _resource_path(relative)
        if p.exists():
            return p

        # Then try dev tree (package-relative)
        p2 = _dev_asset_path(relative.replace("entropia_skillscanner/", ""))
        if p2.exists():
            return p2

        # Final fallback: return the PyInstaller path even if missing (caller will handle)
        return p

    def _play(self, relative: str) -> None:
        if not self.cfg.enabled:
            return
        if winsound is None:
            return

        wav_path = self._resolve_wav(relative)
        if not wav_path.exists():
            # Fail silently: audio should never break the app.
            return

        try:
            # Non-blocking playback
            winsound.PlaySound(str(wav_path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception:
            # Fail silently: audio should never break the app.
            return
