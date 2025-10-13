#!/usr/bin/env python3
"""Shared environment + path helpers for the SCDL pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PipelinePaths:
    """Canonical project directories and frequently used file locations."""

    project_dir: Path
    out_dir: Path
    preview: Path
    final: Path
    mask_npy: Path
    mask_preview: Path
    roi_bbox: Path
    log_file: Path


def _resolve_project_dir(default_root: Optional[Path]) -> Path:
    """Resolve the project directory using SCDL_PROJECT_DIR when available."""

    env_dir = os.environ.get("SCDL_PROJECT_DIR", "").strip()
    if env_dir:
        candidate = Path(env_dir).expanduser()
        if not candidate.is_absolute() and default_root is not None:
            candidate = default_root / candidate
        resolved = candidate.resolve(strict=False)
        if resolved.exists():
            return resolved
    if default_root is not None:
        return default_root.resolve(strict=False)
    return Path(__file__).resolve().parent


def get_pipeline_paths(default_root: Optional[Path] = None) -> PipelinePaths:
    """Return shared directories/log paths and ensure they exist."""

    project_dir = _resolve_project_dir(default_root)
    out_dir = project_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    preview = out_dir / "preview.png"
    final = out_dir / "final.png"
    mask_npy = out_dir / "user_importance.npy"
    mask_preview = out_dir / "user_importance_preview.png"
    roi_bbox = out_dir / "roi_bbox.txt"
    log_file = out_dir / "scdl_pipeline.log"
    return PipelinePaths(
        project_dir=project_dir,
        out_dir=out_dir,
        preview=preview,
        final=final,
        mask_npy=mask_npy,
        mask_preview=mask_preview,
        roi_bbox=roi_bbox,
        log_file=log_file,
    )


def env_path(key: str, default: Path | str, *, base: Optional[Path] = None) -> Path:
    """Resolve a filesystem path from ENV with sensible fallbacks."""

    raw = os.environ.get(key, "").strip()
    if raw:
        candidate = Path(raw).expanduser()
    else:
        candidate = Path(default).expanduser()
    if not candidate.is_absolute() and base is not None:
        candidate = base / candidate
    return candidate.resolve(strict=False)
