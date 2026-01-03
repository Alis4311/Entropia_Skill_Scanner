import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import time
from pathlib import Path
import hashlib
import threading
import queue

import numpy as np
import cv2 as cv

try:
    from PIL import ImageGrab, Image
except ImportError:
    raise SystemExit("Missing dependency: pillow (pip install pillow)")

from pipeline.run_pipeline import run_pipeline, PipelineConfig
from entropia_skillscanner.exporter import ExportError, build_export, write_csv
from entropia_skillscanner.models import SkillRow


POLL_MS = 400
HIGHLIGHT_LAST_N = 12


class SkillScannerApp(tk.Tk):
    def __init__(self, cfg=None, debug=False):
        super().__init__()
        self.title("Entropia Skill Scanner")
        self.geometry("840x620")

        self.cfg = cfg or PipelineConfig()
        self.debug = debug

        # Data model: list[SkillRow]
        self.rows = []
        self._last_added_indices = []

        # Clipboard dedupe
        self._last_clip_hash = None
        self._last_clip_ts = 0

        # Worker / async pipeline
        self._worker_busy = False
        self._pending_bgr = None          # latest-wins slot
        self._results_q = queue.Queue()   # ("log"|"ok"|"err", payload)

        self._build_ui()
        self._set_status("waiting for screenshot")

        # Start UI loops
        self.after(POLL_MS, self._poll_clipboard)
        self.after(50, self._drain_results_queue)

    # ---------------- UI ----------------

    def _build_ui(self):
        # Top bar
        top = ttk.Frame(self, padding=(10, 10, 10, 6))
        top.pack(fill="x")

        self.status_var = tk.StringVar(value="")
        self.status_lbl = ttk.Label(top, textvariable=self.status_var, font=("Segoe UI", 11))
        self.status_lbl.pack(side="left")

        ttk.Separator(self).pack(fill="x")

        # Table
        mid = ttk.Frame(self, padding=(10, 8, 10, 8))
        mid.pack(fill="both", expand=True)

        cols = ("skill", "value", "added")
        self.tree = ttk.Treeview(mid, columns=cols, show="headings", height=20)
        self.tree.heading("skill", text="Skill name")
        self.tree.heading("value", text="Skill value")
        self.tree.heading("added", text="Added")

        self.tree.column("skill", width=420, anchor="w")
        self.tree.column("value", width=140, anchor="e")
        self.tree.column("added", width=180, anchor="w")

        vsb = ttk.Scrollbar(mid, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        mid.grid_rowconfigure(0, weight=1)
        mid.grid_columnconfigure(0, weight=1)

        # Tags for highlighting new rows
        self.tree.tag_configure("new", background="#e9f5ff")  # light highlight
        self.tree.tag_configure("err", background="#ffe9e9")

        # Bottom controls
        bot = ttk.Frame(self, padding=(10, 6, 10, 10))
        bot.pack(fill="x")

        ttk.Button(bot, text="Export CSV…", command=self._export_csv).pack(side="left")
        ttk.Button(bot, text="Clear", command=self._clear).pack(side="left", padx=(8, 0))

        self.auto_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bot, text="Auto-poll clipboard", variable=self.auto_var).pack(side="right")

    def _set_status(self, s: str):
        self.status_var.set(f"Status: {s}")

    # ---------------- Clipboard polling ----------------

    def _poll_clipboard(self):
        try:
            if self.auto_var.get():
                img = ImageGrab.grabclipboard()
                pil_img = self._normalize_clipboard_image(img)
                if pil_img is not None:
                    clip_hash = self._hash_pil(pil_img)

                    # debounce + dedupe
                    now = time.time()
                    if clip_hash != self._last_clip_hash and (now - self._last_clip_ts) > 0.2:
                        self._last_clip_hash = clip_hash
                        self._last_clip_ts = now
                        self._handle_screenshot(pil_img)
            else:
                self._set_status("paused (auto-poll off)")
        except Exception as e:
            self._set_status(f"error (clipboard): {e}")

        self.after(POLL_MS, self._poll_clipboard)

    @staticmethod
    def _normalize_clipboard_image(obj):
        # ImageGrab.grabclipboard() may return:
        # - PIL.Image.Image
        # - list of filenames
        # - None
        if obj is None:
            return None
        if isinstance(obj, Image.Image):
            return obj.convert("RGB")
        # If it's a list of file paths, ignore for now (you can extend later)
        return None

    @staticmethod
    def _hash_pil(pil_img: "Image.Image") -> str:
        """
        Faster than hashing full-res: downscale + grayscale.
        Still good enough for dedupe.
        """
        small = pil_img.convert("L").resize((320, 180))
        return hashlib.sha1(small.tobytes()).hexdigest()

    # ---------------- Pipeline integration (async) ----------------

    def _handle_screenshot(self, pil_img: "Image.Image"):
        rgb = np.array(pil_img)
        bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
        self._enqueue_screenshot_for_processing(bgr)

    def _enqueue_screenshot_for_processing(self, bgr: np.ndarray):
        """
        Latest-wins: if worker is busy, overwrite pending with newest.
        If idle, start immediately.
        """
        if self._worker_busy:
            self._pending_bgr = bgr
            self._set_status("queued (new screenshot)")
            return

        self._worker_busy = True
        self._pending_bgr = None
        self._set_status("extracting...")

        threading.Thread(target=self._worker_run_pipeline, args=(bgr,), daemon=True).start()

    def _worker_run_pipeline(self, bgr: np.ndarray):
        """
        Runs on background thread. Never touches Tk directly.
        """
        def worker_logger(msg: str):
            # send status updates to UI thread
            self._results_q.put(("log", msg))

        try:
            rows, status = run_pipeline(
                self.cfg,
                bgr,
                debug=self.debug,
                debug_dir=None,
                logger=worker_logger,
            )
            self._results_q.put(("ok", (rows, status)))
        except Exception as e:
            self._results_q.put(("err", str(e)))

    def _drain_results_queue(self):
        """
        Runs on Tk thread. Applies logs/results and kicks next pending job if any.
        """
        try:
            while True:
                kind, payload = self._results_q.get_nowait()

                if kind == "log":
                    self._set_status(str(payload))

                elif kind == "err":
                    self._worker_busy = False
                    self._set_status(f"error (pipeline): {payload}")

                elif kind == "ok":
                    rows, status = payload
                    self._worker_busy = False

                    if rows:
                        self._append_rows(rows)
                        self._set_status(status or f"done (+{len(rows)} rows)")
                    else:
                        self._set_status(status or "done (no rows)")

                # After any terminal event, if we have a pending screenshot, run it next.
                if (not self._worker_busy) and (self._pending_bgr is not None):
                    next_bgr = self._pending_bgr
                    self._pending_bgr = None
                    self._enqueue_screenshot_for_processing(next_bgr)

        except queue.Empty:
            pass

        self.after(50, self._drain_results_queue)

    # ---------------- Data / Table rendering ----------------

    def _append_rows(self, extracted_rows):
        # extracted_rows expected: iterable of (skill_name, skill_value_str_or_float)
        start_idx = len(self.rows)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        for name, val in extracted_rows:
            try:
                fval = float(val)
            except Exception:
                continue
            self.rows.append(SkillRow(name=str(name), value=fval, added=ts))

        end_idx = len(self.rows) - 1
        self._last_added_indices = list(range(start_idx, end_idx + 1))

        self._refresh_table()

    def _refresh_table(self):
        self.tree.delete(*self.tree.get_children())

        last_set = set(self._last_added_indices[-HIGHLIGHT_LAST_N:]) if self._last_added_indices else set()

        for i, r in enumerate(self.rows):
            tags = ("new",) if i in last_set else ()
            self.tree.insert("", "end", values=(r.name, f"{r.value:.2f}", r.added), tags=tags)

        # Scroll to bottom
        if self.rows:
            last = self.tree.get_children()[-1]
            self.tree.see(last)

    # ---------------- Actions ----------------

    def _export_csv(self):
        if not self.rows:
            messagebox.showinfo("Export CSV", "No rows to export yet.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Export extracted skills",
        )
        if not path:
            return

        try:
            result = build_export(self.rows)
            write_csv(result, Path(path))
        except ExportError as e:
            messagebox.showerror("Export CSV", f"Export failed:\n{e}")
            return
        except Exception as e:
            messagebox.showerror("Export CSV", f"Failed to export:\n{e}")
            return

        messagebox.showinfo("Export CSV", f"Exported {len(self.rows)} rows with categories.")

    def _clear(self):
        self.rows.clear()
        self._last_added_indices = []
        self._refresh_table()
        self._set_status("waiting for screenshot")


def main():
    cfg = PipelineConfig()
    app = SkillScannerApp(cfg=cfg, debug=False)
    app.mainloop()


if __name__ == "__main__":
    main()
