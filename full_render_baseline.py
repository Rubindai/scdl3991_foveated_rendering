#!/usr/bin/env python3
"""Standalone full-frame render baseline (no foveation) for Blender 4.5.2 + RTX 3060 (OPTIX only)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import bpy

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scdl import StageTimer, TimerRecord, env_path, get_logger
from scdl.logging import log_devices, log_environment
from scdl.runtime import (
    cycles_devices,
    ensure_optix_device,
    require_blender_version,
    set_cycles_scene_device,
)


@dataclass(frozen=True)
class BaselinePaths:
    """Output and logging destinations for the baseline render."""

    project_dir: Path
    out_dir: Path
    final: Path
    log_file: Path


@dataclass(frozen=True)
class FullRenderConfig:
    """Render configuration derived from shared pipeline environment variables."""

    base_samples: int
    min_samples: int
    max_samples: int
    adaptive_threshold: float
    adaptive_min_samples: int
    filter_glossy: float
    clamp_direct: Optional[float]
    clamp_indirect: Optional[float]
    auto_samples: bool
    caustics: bool

    @classmethod
    def from_env(cls) -> "FullRenderConfig":
        """Parse configuration using the same knobs as the foveated pipeline."""

        base_samples = _env_positive_int("SCDL_FOVEATED_BASE_SPP", default=192)
        min_samples = _env_positive_int("SCDL_FOVEATED_MIN_SPP", default=32)
        max_samples = _env_positive_int("SCDL_FOVEATED_MAX_SPP", default=768)
        adaptive_threshold = float(os.getenv("SCDL_ADAPTIVE_THRESHOLD", "0.02"))
        adaptive_min_samples = _env_positive_int("SCDL_ADAPTIVE_MIN_SAMPLES", default=min_samples)
        filter_glossy = float(os.getenv("SCDL_FOVEATED_FILTER_GLOSSY", "0.8"))
        clamp_direct = _env_optional_float("SCDL_FOVEATED_CLAMP_DIRECT")
        clamp_indirect = _env_optional_float("SCDL_FOVEATED_CLAMP_INDIRECT")
        auto_samples = _env_bool("SCDL_FOVEATED_AUTOSPP", default=True)
        caustics = _env_bool("SCDL_FOVEATED_CAUSTICS", default=False)

        if min_samples > base_samples:
            raise RuntimeError("[Baseline] SCDL_FOVEATED_MIN_SPP cannot exceed SCDL_FOVEATED_BASE_SPP.")
        if max_samples < base_samples:
            raise RuntimeError("[Baseline] SCDL_FOVEATED_MAX_SPP cannot be lower than SCDL_FOVEATED_BASE_SPP.")
        if adaptive_min_samples < min_samples:
            raise RuntimeError("[Baseline] SCDL_ADAPTIVE_MIN_SAMPLES must be ≥ SCDL_FOVEATED_MIN_SPP.")

        return cls(
            base_samples=base_samples,
            min_samples=min_samples,
            max_samples=max_samples,
            adaptive_threshold=adaptive_threshold,
            adaptive_min_samples=adaptive_min_samples,
            filter_glossy=filter_glossy,
            clamp_direct=clamp_direct,
            clamp_indirect=clamp_indirect,
            auto_samples=auto_samples,
            caustics=caustics,
        )


def _env_positive_int(name: str, *, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"[Baseline] {name} must be an integer (received '{raw}').") from exc
    if value <= 0:
        raise RuntimeError(f"[Baseline] {name} must be greater than zero (received {value}).")
    return value


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"[Baseline] {name} must be truthy/falsy (received '{raw}').")


def _env_optional_float(name: str) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"[Baseline] {name} must be a float (received '{raw}').") from exc


def get_baseline_paths(default_root: Optional[Path] = None) -> BaselinePaths:
    """Resolve project-relative paths, respecting SCDL_PROJECT_DIR overrides."""

    base_root = (default_root or ROOT).resolve()
    project_dir = env_path("SCDL_PROJECT_DIR", base_root, base=base_root)
    out_dir = project_dir / "out_full_render"
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "final.png"
    log_file = out_dir / "full_render.log"
    return BaselinePaths(project_dir=project_dir, out_dir=out_dir, final=final, log_file=log_file)


def configure_cycles(scene: bpy.types.Scene, cfg: FullRenderConfig) -> int:
    """Apply Cycles render settings and return the effective sample count."""

    scene.render.engine = "CYCLES"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "16"

    cycles = scene.cycles
    cycles.device = "GPU"
    cycles.use_fast_gi = False
    cycles.max_bounces = 8
    cycles.transparent_max_bounces = 8
    cycles.caustics_reflective = cfg.caustics
    cycles.caustics_refractive = cfg.caustics

    cycles.use_adaptive_sampling = True
    cycles.adaptive_threshold = cfg.adaptive_threshold
    cycles.adaptive_min_samples = cfg.adaptive_min_samples

    effective = max(cfg.min_samples, min(cfg.max_samples, cfg.base_samples))
    cycles.samples = effective

    cycles.use_denoising = True
    cycles.denoiser = "OPTIX"
    for view_layer in scene.view_layers:
        view_layer.cycles.use_denoising = True

    cycles.blur_glossy = cfg.filter_glossy
    if cfg.clamp_direct is not None:
        cycles.sample_clamp_direct = cfg.clamp_direct
    if cfg.clamp_indirect is not None:
        cycles.sample_clamp_indirect = cfg.clamp_indirect

    return effective


def log_configuration(logger, cfg: FullRenderConfig, paths: BaselinePaths) -> None:
    """Emit environment and render configuration details."""

    log_environment(
        logger,
        {
            "project_dir": paths.project_dir.as_posix(),
            "output_dir": paths.out_dir.as_posix(),
            "base_samples": cfg.base_samples,
            "min_samples": cfg.min_samples,
            "max_samples": cfg.max_samples,
            "adaptive_threshold": cfg.adaptive_threshold,
            "adaptive_min_samples": cfg.adaptive_min_samples,
            "filter_glossy": cfg.filter_glossy,
            "clamp_direct": cfg.clamp_direct,
            "clamp_indirect": cfg.clamp_indirect,
            "auto_samples": cfg.auto_samples,
            "caustics": cfg.caustics,
            "assumed_coverage": 1.0,
        },
    )


def summarize_timings(logger, timings: Iterable[TimerRecord]) -> None:
    """Log a consolidated summary of recorded durations."""

    entries = list(timings)
    if not entries:
        return
    logger.info("[step] Timing summary")
    for record in entries:
        logger.info("[timer] %s %.3fs", record.name, record.seconds)


def main() -> None:
    """Execute the baseline render inside Blender."""

    paths = get_baseline_paths(ROOT)
    logger = get_logger("scdl.full_render_baseline", default_path=paths.log_file)
    logger.info("[step] Baseline full-frame render starting")

    info = require_blender_version((4, 5, 2))
    logger.info("[system] Blender %s (commit %s)", info.version_string, info.build_commit)
    logger.info("[system] Background mode: %s", info.is_background)

    device_override = os.getenv("SCDL_CYCLES_DEVICE", "OPTIX").strip().upper()
    if device_override != "OPTIX":
        raise RuntimeError(f"[Baseline] SCDL_CYCLES_DEVICE must be OPTIX (received '{device_override}').")

    matched_devices = ensure_optix_device("RTX 3060")
    set_cycles_scene_device()
    devices = cycles_devices()

    cfg = FullRenderConfig.from_env()
    log_configuration(logger, cfg, paths)
    log_devices(logger, devices)
    if matched_devices:
        log_devices(logger, [f"OPTIX REQUIRED {name}" for name in matched_devices])

    scene = bpy.context.scene
    scene.render.filepath = str(paths.final)

    timings: List[TimerRecord] = []
    with StageTimer("cycles.configure", logger=logger, records=timings):
        effective = configure_cycles(scene, cfg)
        logger.info("[env] effective_samples=%d", effective)

    with StageTimer("render.execute", logger=logger, records=timings):
        bpy.ops.render.render(write_still=True)

    logger.info("[OK] Baseline render saved to %s", paths.final.as_posix())
    summarize_timings(logger, timings)


if __name__ == "__main__":
    main()
