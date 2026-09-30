"""Configuration: packaged defaults, optional YAML override file, env overrides."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "holdfast.yaml"


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _apply_set(cfg: dict, spec: str) -> dict:
    for item in filter(None, (s.strip() for s in spec.split(","))):
        path, _, raw = item.partition("=")
        keys = [k.strip() for k in path.split(".") if k.strip()]
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = yaml.safe_load(raw.strip()) if raw.strip() else None
    return cfg


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    override = path or os.getenv("HOLDFAST_CONFIG")
    if override and Path(override).resolve() != DEFAULT_CONFIG:
        with open(override, encoding="utf-8") as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    if os.getenv("HOLDFAST_SET"):
        cfg = _apply_set(cfg, os.environ["HOLDFAST_SET"])
    return cfg
