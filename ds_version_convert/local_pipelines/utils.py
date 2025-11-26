from __future__ import annotations

import shutil
from pathlib import Path


def ensure_destination(dest_root: Path, overwrite: bool = False) -> None:
    if dest_root.exists():
        if not any(dest_root.iterdir()):
            return
        if not overwrite:
            raise FileExistsError(f"Destination path {dest_root} already exists")
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)


def clone_tree(source_root: Path, dest_root: Path, *, overwrite: bool = False) -> None:
    ensure_destination(dest_root, overwrite=overwrite)
    shutil.copytree(source_root, dest_root)
