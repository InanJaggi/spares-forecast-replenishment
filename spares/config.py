"""Load the customer config (config/corvane.toml) with optional overrides."""
from __future__ import annotations

import copy
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "corvane.toml"


def load_config(path: str | Path = DEFAULT_CONFIG, overrides: dict | None = None) -> dict:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    if overrides:
        cfg = merge(cfg, overrides)
    return cfg


def merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out
