from __future__ import annotations

from pathlib import Path
from typing import Any
from omegaconf import DictConfig, OmegaConf


def _resolve_defaults(config_path: Path) -> list[Path]:
    raw = OmegaConf.load(config_path)
    defaults = raw.get("defaults", []) if isinstance(raw, DictConfig) else []
    paths: list[Path] = []
    for item in defaults:
        if isinstance(item, str):
            paths.append((config_path.parent / item).resolve())
    return paths


def load_config(config_path: str | Path, default_path: str | Path = "configs/default.yml") -> DictConfig:
    """Load a YAML config and merge it with default.yml.

    Experiment configs may include a `defaults` list with relative paths.
    Values in the specific experiment config override defaults.
    """
    config_path = Path(config_path).resolve()
    default_path = Path(default_path).resolve()

    configs: list[Any] = []
    if default_path.exists():
        configs.append(OmegaConf.load(default_path))
    for p in _resolve_defaults(config_path):
        if p.exists() and p != default_path:
            configs.append(OmegaConf.load(p))
    configs.append(OmegaConf.load(config_path))
    cfg = OmegaConf.merge(*configs)
    if "defaults" in cfg:
        del cfg["defaults"]
    return cfg


def save_config(cfg: DictConfig, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=out_path)
