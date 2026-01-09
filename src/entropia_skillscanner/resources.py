from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


APP_NAME_DEFAULT = "Entropia Skill Scanner"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"))


def bundled_base_dir() -> Path:
    """
    Base directory for bundled resources.

    Frozen: PyInstaller extraction dir (sys._MEIPASS)
    Dev: repo root
    """
    if is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return _repo_root()


def resource_path(rel: str) -> Path:
    """
    Resolve a resource path relative to the bundled base directory.
    Use this for read-only shipped resources (icons, default JSON files, etc.).
    """
    return (bundled_base_dir() / rel).resolve()


def exe_dir() -> Path:
    """
    Directory where the executable lives (useful for external files shipped alongside the exe).
    """
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path.cwd().resolve()


def user_data_dir(app_name: str = APP_NAME_DEFAULT) -> Path:
    """
    Writable per-user directory for app data.
    """
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(Path.home())
    d = (Path(base) / app_name).resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_user_data_file(rel: str, *, app_name: str = APP_NAME_DEFAULT) -> Path:
    """
    Ensure a writable copy of a shipped default exists in user data dir.

    Typical use: data/professions.json, data/professions_list.json, caches, etc.

    - Frozen: copies from bundled resource into %LOCALAPPDATA%/<app>/...
    - Dev: still copies from repo-root resource into the same user data dir
           (so dev behavior matches production behavior).
    """
    rel_path = Path(rel)
    dst = (user_data_dir(app_name) / rel_path).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        return dst

    src = resource_path(rel_path.as_posix())

    if not src.exists():
        raise FileNotFoundError(f"Bundled resource not found: {src}")

    shutil.copyfile(src, dst)
    return dst


def user_data_path(rel: str, *, app_name: str = APP_NAME_DEFAULT, ensure_parent: bool = True) -> Path:
    """
    Get a path inside the user data dir without copying any defaults.
    """
    p = (user_data_dir(app_name) / Path(rel)).resolve()
    if ensure_parent:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p
