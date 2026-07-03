from __future__ import annotations

import atexit
import shutil
from pathlib import Path


def cleanup_temp_uploads(upload_dir: Path) -> None:
    """Delete temporary upload contents while keeping the folder and .gitkeep."""

    upload_dir.mkdir(parents=True, exist_ok=True)
    for item in upload_dir.iterdir():
        if item.name == ".gitkeep":
            continue
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            try:
                item.unlink()
            except FileNotFoundError:
                pass


def register_cleanup(upload_dir: Path) -> None:
    atexit.register(cleanup_temp_uploads, upload_dir)
