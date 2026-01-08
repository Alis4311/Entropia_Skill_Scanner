import time
from pathlib import Path
from decimal import Decimal
from typing import Callable, Optional, Sequence, Union

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from entropia_skillscanner.config import AppConfig, load_app_config
from entropia_skillscanner.core import PipelineRow, PipelineResult, SkillRow
from entropia_skillscanner.exporter import ExportError, build_export, write_csv
from entropia_skillscanner.runtime import PipelineRunner
from entropia_skillscanner.view_model import SkillScannerViewModel

from pipeline.professions import compute_professions
from pipeline.profession_store import get_profession_weights

from entropia_skillscanner.import_skills_csv import load_skill_rows_from_export_csv, ImportError

##for bundling:
import pytesseract
from entropia_skillscanner.resources import resource_path, is_frozen
import os

HIGHLIGHT_LAST_N = 12


class SkillScannerApp(tk.Tk):
    def __init__(
        self,
        *,
        app_cfg: Optional[AppConfig] = None,
        runner_factory: Optional[Callable[..., PipelineRunner]] = None,
    ):
        super().__init__()
        self.title("Entropia Skill Scanner")
        self.geometry("840x620")

        self.app_cfg = app_cfg or load_app_config()

        try:
            self.app_cfg.validate()
        except Exception as e: 
            messagebox.showerror("Configuration error: ", str(e))
        self.pipeline_cfg = self.app_cfg.pipeline_config
        if is_frozen():
            pytesseract.pytesseract.tesseract_cmd = str(
                resource_path("tesseract/tesseract.exe")
            )
            os.environ["TESSDATA_PREFIX"] = str(
                resource_path("tesseract/tessdata")
            )
        self.view_model = SkillScannerViewModel()
        self._last_added_indices = []

        self._build_ui()
        self._subscriptions = [
            self.view_model.subscribe("rows", self._on_rows_changed),
            self.view_model.subscribe("status", self._on_status_changed),
            self.view_model.subscribe("warnings", self._on_warnings_changed),
        ]

        self.view_model.set_status("waiting for screenshot")

        # Runner
        self.runner = (runner_factory or self._default_runner_factory)(
            cfg=self.pipeline_cfg,
            debug=self.app_cfg.debug_pipeline,
            ui_dispatch=self._dispatch,
            on_started=self._on_pipeline_started,
            on_progress=self._on_pipeline_progress,
            on_completed=self._on_pipeline_completed,
        )
        self.runner.set_auto_poll(self.auto_var.get())
        self.runner.start_polling()

        # Start UI loops
        self.after(0, lambda: None)  # no-op to ensure Tk loop initialized

    # ---------------- UI ----------------

    def _build_ui(self):
        # Top bar
        top = ttk.Frame(self, padding=(10, 10, 10, 6))
        top.pack(fill="x")

        self.status_var = tk.StringVar(value="")
        self.status_lbl = ttk.Label(top, textvariable=self.status_var, font=("Segoe UI", 11))
        self.status_lbl.pack(side="left")

        self.warnings_var = tk.StringVar(value="")
        self.warnings_lbl = ttk.Label(
            top,
            textvariable=self.warnings_var,
            font=("Segoe UI", 10),
            foreground="#b26b00",
        )
        self.warnings_lbl.pack(side="right")

        ttk.Separator(self).pack(fill="x")

        # Notebook (Skills / Professions)
        mid = ttk.Frame(self, padding=(10, 8, 10, 8))
        mid.pack(fill="both", expand=True)

        nb = ttk.Notebook(mid)
        nb.pack(fill="both", expand=True)

        # ---- Skills tab ----
        skills_tab = ttk.Frame(nb)
        nb.add(skills_tab, text="Skills")

        cols = ("skill", "value", "added")
        self.tree = ttk.Treeview(skills_tab, columns=cols, show="headings", height=20)
        self.tree.heading("skill", text="Skill name")
        self.tree.heading("value", text="Skill value")
        self.tree.heading("added", text="Added")

        self.tree.column("skill", width=420, anchor="w")
        self.tree.column("value", width=140, anchor="e")
        self.tree.column("added", width=180, anchor="w")

        vsb = ttk.Scrollbar(skills_tab, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        skills_tab.grid_rowconfigure(0, weight=1)
        skills_tab.grid_columnconfigure(0, weight=1)

        self.tree.tag_configure("new", background="#e9f5ff")
        self.tree.tag_configure("err", background="#ffe9e9")

        # ---- Professions tab ----
        prof_tab = ttk.Frame(nb)
        nb.add(prof_tab, text="Professions")

        pcols = ("profession", "value", "flags")
        self.prof_tree = ttk.Treeview(prof_tab, columns=pcols, show="headings", height=20)
        self.prof_tree.heading("profession", text="Profession")
        self.prof_tree.heading("value", text="Value")
        self.prof_tree.heading("flags", text="Flags")

        self.prof_tree.column("profession", width=420, anchor="w")
        self.prof_tree.column("value", width=140, anchor="e")
        self.prof_tree.column("flags", width=180, anchor="w")

        pvsb = ttk.Scrollbar(prof_tab, orient="vertical", command=self.prof_tree.yview)
        self.prof_tree.configure(yscrollcommand=pvsb.set)

        self.prof_tree.grid(row=0, column=0, sticky="nsew")
        pvsb.grid(row=0, column=1, sticky="ns")

        prof_tab.grid_rowconfigure(0, weight=1)
        prof_tab.grid_columnconfigure(0, weight=1)

        self.prof_tree.tag_configure("warn", background="#fff6e5")  # light warning highlight

        # Bottom controls
        bot = ttk.Frame(self, padding=(10, 6, 10, 10))
        bot.pack(fill="x")

        ttk.Button(bot, text="Load Skills CSV…", command=self.on_load_skills_csv).pack(side="left")
        ttk.Button(bot, text="Export CSV…", command=self._export_csv).pack(side="left", padx=(8, 0))
        ttk.Button(bot, text="Clear", command=self._clear).pack(side="left", padx=(8, 0))

        self.auto_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            bot,
            text="Auto-poll clipboard",
            variable=self.auto_var,
            command=self._on_toggle_auto_poll,
        ).pack(side="right")

    def _set_status(self, s: str):
        self.view_model.set_status(s)

    # ---------------- Runner integration ----------------

    def _default_runner_factory(self, **kwargs) -> PipelineRunner:
        return PipelineRunner(**kwargs)

    def _dispatch(self, fn: Callable[[], None], delay_ms: Optional[int] = None) -> None:
        if delay_ms is None:
            self.after_idle(fn)
        else:
            self.after(delay_ms, fn)

    def _on_toggle_auto_poll(self) -> None:
        self.runner.set_auto_poll(self.auto_var.get())
        if not self.auto_var.get():
            self._set_status("paused (auto-poll off)")

    def _on_pipeline_started(self) -> None:
        self.view_model.set_warnings(())
        self._set_status("extracting...")

    def _on_pipeline_progress(self, msg: str) -> None:
        self._set_status(msg)

    def _on_pipeline_completed(self, result: Union[PipelineResult, Exception]) -> None:
        if isinstance(result, Exception):
            self._set_status(f"error (pipeline): {result}")
            return

        if result.rows:
            self._append_rows(result.rows)
            self._set_status(result.status or f"done (+{len(result.rows)} rows)")
        else:
            self._set_status(result.status or "done (no rows)")

    # ---------------- Data / Table rendering ----------------

    def _append_rows(self, extracted_rows: Sequence[PipelineRow]):
        # extracted_rows expected: iterable of PipelineRow
        start_idx = len(self.view_model.rows)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        new_rows = []
        for row in extracted_rows:
            try:
                fval = float(row.value)
            except Exception:
                continue
            new_rows.append(SkillRow(name=str(row.name), value=fval, added=ts))

        if not new_rows:
            return

        end_idx = start_idx + len(new_rows) - 1
        self._last_added_indices = list(range(start_idx, end_idx + 1))
        self.view_model.set_warnings(())
        self.view_model.append_rows(new_rows)

    def _refresh_table(self):
        self.tree.delete(*self.tree.get_children())

        last_set = set(self._last_added_indices[-HIGHLIGHT_LAST_N:]) if self._last_added_indices else set()

        for i, r in enumerate(self.view_model.rows):
            tags = ("new",) if i in last_set else ()
            self.tree.insert("", "end", values=(r.name, f"{r.value:.2f}", r.added), tags=tags)

        # Scroll to bottom
        if self.view_model.rows:
            last = self.tree.get_children()[-1]
            self.tree.see(last)

        self._refresh_professions()

    # ---------------- Actions ----------------

    def on_load_skills_csv(self) -> None:
        fp = filedialog.askopenfilename(
            title="Load export CSV (reads [Skills] only)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not fp:
            return

        try:
            res = load_skill_rows_from_export_csv(Path(fp), added_label="imported", strict=True)
        except ImportError as e:
            messagebox.showerror("Import failed", str(e))
            return
        except Exception as e:
            messagebox.showerror("Import failed", f"Unexpected error:\n{e}")
            return

        self._last_added_indices = []  # don’t highlight imported rows as “new”

        self.view_model.set_rows(res.rows)
        self.view_model.set_warnings(res.warnings)

        self._set_status(f"loaded {len(self.view_model.rows)} skills from CSV")

    def _export_csv(self):
        if not self.view_model.rows:
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
            result = build_export(
                self.view_model.rows,
                app_config=self.app_cfg,
            )
            write_csv(result, Path(path))
        except ExportError as e:
            messagebox.showerror("Export CSV", f"Export failed:\n{e}")
            return
        except Exception as e:
            messagebox.showerror("Export CSV", f"Failed to export:\n{e}")
            return

        messagebox.showinfo("Export CSV", f"Exported {len(self.view_model.rows)} rows with categories.")

    def _clear(self):
        self.view_model.set_rows([])
        self._last_added_indices = []
        self.view_model.set_warnings(())
        self._set_status("waiting for screenshot")

    def _refresh_professions(self):
        """
        Recompute professions from current skills and render.
        Non-strict so it never hard-fails mid-scan.
        """
        self.prof_tree.delete(*self.prof_tree.get_children())
        warnings = list(self.view_model.warnings)

        if not self.view_model.rows:
            return

        # Build skill->Decimal map from scanned rows
        skills = {r.name: Decimal(str(r.value)) for r in self.view_model.rows}

        try:
            weights = get_profession_weights(self.app_cfg.professions_weights_path)
            prof_vals = compute_professions(
                skills=skills,
                profession_weights=weights,
                strict=False,
            )
        except Exception as e:
            # Show a single error row
            self.prof_tree.insert("", "end", values=("ERROR", "", str(e)), tags=("warn",))
            warnings.append(f"profession computation failed: {e}")
            self.view_model.set_warnings(warnings)
            return

        # Sort by value desc (best for debugging)
        items = sorted(prof_vals.items(), key=lambda kv: kv[1].value, reverse=True)

        for prof, pv in items:
            flags = []
            if pv.missing_skills:
                flags.append("MISSING_SKILLS")
                warnings.append(f"{prof}: missing skills ({', '.join(pv.missing_skills)})")
            if pv.pct_sum != Decimal("100"):
                flags.append(f"PCT_SUM={pv.pct_sum}")
                warnings.append(f"{prof}: pct sum is {pv.pct_sum}")

            tag = ("warn",) if flags else ()
            self.prof_tree.insert(
                "",
                "end",
                values=(prof, format(pv.value, ".2f"), ";".join(flags)),
                tags=tag,
            )

        self.view_model.set_warnings(warnings)

    # ---------------- View-model bindings ----------------

    def _on_rows_changed(self, rows: Sequence[SkillRow]) -> None:
        # Keep local cache for convenience
        self._refresh_table()

    def _on_status_changed(self, status: str) -> None:
        self.status_var.set(f"Status: {status}")

    def _on_warnings_changed(self, warnings: Sequence[str]) -> None:
        self.warnings_var.set("; ".join(warnings))


def main():
    app_cfg = load_app_config()
    app = SkillScannerApp(app_cfg=app_cfg)
    app.mainloop()


if __name__ == "__main__":
    main()
