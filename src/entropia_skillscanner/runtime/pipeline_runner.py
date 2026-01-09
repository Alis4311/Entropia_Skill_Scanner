from __future__ import annotations

import hashlib
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import cv2 as cv
import numpy as np
from PIL import Image

from entropia_skillscanner.core import PipelineResult
from pipeline.run_pipeline import PipelineConfig, run_pipeline


PipelineStart = Callable[[], None]
PipelineProgress = Callable[[str], None]
PipelineCompletion = Callable[[Any], None]


def run_pipeline_sync(
    cfg: PipelineConfig,
    bgr: np.ndarray,
    *,
    debug: bool = False,
    debug_dir=None,
    logger: Optional[Callable[[str], None]] = None,
) -> PipelineResult:
    return run_pipeline(cfg, bgr, debug=debug, debug_dir=debug_dir, logger=logger)


@dataclass
class _PendingFrame:
    bgr: np.ndarray
    clip_hash: str


class PipelineRunner:
    """
    Encapsulates clipboard polling and background pipeline work.
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        *,
        debug: bool = False,
        poll_ms: int = 400,
        ui_dispatch: Callable[[Callable[[], None], Optional[int]], None],
        on_started: Optional[PipelineStart] = None,
        on_progress: Optional[PipelineProgress] = None,
        on_completed: Optional[PipelineCompletion] = None,
    ) -> None:
        self.cfg = cfg
        self.debug = debug
        self.poll_ms = poll_ms
        self.ui_dispatch = ui_dispatch
        self.on_started = on_started
        self.on_progress = on_progress
        self.on_completed = on_completed

        self._auto = True
        self._last_clip_hash: Optional[str] = None
        self._last_clip_ts: float = 0

        self._worker_busy = False
        self._pending: Optional[_PendingFrame] = None
        self._results_q: "queue.Queue[tuple[str, object]]" = queue.Queue()

    # -------- public controls --------

    @property
    def auto_poll_enabled(self) -> bool:
        return self._auto

    def set_auto_poll(self, enabled: bool) -> None:
        self._auto = enabled

    def start_polling(self) -> None:
        self.ui_dispatch(self._poll_clipboard, self.poll_ms)
        self.ui_dispatch(self._drain_results_queue, 50)

    def enqueue_bgr(self, bgr: np.ndarray, *, clip_hash: Optional[str] = None) -> None:
        if clip_hash is None:
            clip_hash = self._hash_bgr(bgr)
        self._enqueue(_PendingFrame(bgr=bgr, clip_hash=clip_hash))

    # -------- clipboard --------

    def _poll_clipboard(self) -> None:
        try:
            if self._auto:
                img = ImageGrab.grabclipboard()
                pil_img = self._normalize_clipboard_image(img)
                if pil_img is not None:
                    clip_hash = self._hash_pil(pil_img)
                    now = time.time()
                    if clip_hash != self._last_clip_hash and (now - self._last_clip_ts) > 0.2:
                        self._last_clip_hash = clip_hash
                        self._last_clip_ts = now
                        bgr = self._pil_to_bgr(pil_img)
                        self._enqueue(_PendingFrame(bgr=bgr, clip_hash=clip_hash))
            else:
                self._emit_progress("paused (auto-poll off)")
        except Exception as e:
            self._emit_progress(f"error (clipboard): {e}")

        self.ui_dispatch(self._poll_clipboard, self.poll_ms)

    @staticmethod
    def _normalize_clipboard_image(obj):
        if obj is None:
            return None
        if isinstance(obj, Image.Image):
            return obj.convert("RGB")
        return None

    @staticmethod
    def _hash_pil(pil_img: "Image.Image") -> str:
        small = pil_img.convert("L").resize((320, 180))
        return hashlib.sha1(small.tobytes()).hexdigest()

    @staticmethod
    def _hash_bgr(bgr: np.ndarray) -> str:
        small = cv.cvtColor(cv.resize(bgr, (320, 180)), cv.COLOR_BGR2GRAY)
        return hashlib.sha1(small.tobytes()).hexdigest()

    @staticmethod
    def _pil_to_bgr(pil_img: "Image.Image") -> np.ndarray:
        rgb = np.array(pil_img)
        return cv.cvtColor(rgb, cv.COLOR_RGB2BGR)

    # -------- worker --------

    def _enqueue(self, pending: _PendingFrame) -> None:
        if self._worker_busy:
            self._pending = pending
            self._emit_progress("queued (new screenshot)")
            return

        self._worker_busy = True
        self._pending = None
        self._emit_started()

        threading.Thread(target=self._worker_run_pipeline, args=(pending,), daemon=True).start()

    def _worker_run_pipeline(self, pending: _PendingFrame) -> None:
        def worker_logger(msg: str):
            self._results_q.put(("log", msg))

        try:
            result = run_pipeline_sync(
                self.cfg,
                pending.bgr,
                debug=self.debug,
                debug_dir=None,
                logger=worker_logger,
            )
            self._results_q.put(("ok", result))
        except Exception as e:
            self._results_q.put(("err", e))

    def _drain_results_queue(self) -> None:
        try:
            while True:
                kind, payload = self._results_q.get_nowait()

                if kind == "log":
                    self._emit_progress(str(payload))
                elif kind == "err":
                    self._worker_busy = False
                    self._emit_completed(payload if isinstance(payload, Exception) else Exception(str(payload)))
                elif kind == "ok":
                    self._worker_busy = False
                    self._emit_completed(payload) 

                if (not self._worker_busy) and self._pending is not None:
                    next_frame = self._pending
                    self._pending = None
                    self._enqueue(next_frame)
        except queue.Empty:
            pass

        self.ui_dispatch(self._drain_results_queue, 50)

    # -------- callbacks --------

    def _emit_started(self) -> None:
        if self.on_started:
            self.on_started()

    def _emit_progress(self, msg: str) -> None:
        if self.on_progress:
            self.on_progress(msg)

    def _emit_completed(self, result: Any) -> None:
        if self.on_completed:
            self.on_completed(result)


# Pillow import only for typing clarity; grabbed lazily.
try:
    from PIL import ImageGrab 
except ImportError:
    raise SystemExit("Missing dependency: pillow (pip install pillow)")
