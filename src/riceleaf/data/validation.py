from __future__ import annotations

from pathlib import Path
from riceleaf.preprocessing.image_io import is_valid_image


def list_images(raw_dir: str | Path, extensions: list[str]) -> list[Path]:
    raw = Path(raw_dir)
    exts = {e.lower() for e in extensions}
    return [p for p in raw.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def filter_valid_images(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    valid, invalid = [], []
    for p in paths:
        (valid if is_valid_image(p) else invalid).append(p)
    return valid, invalid
