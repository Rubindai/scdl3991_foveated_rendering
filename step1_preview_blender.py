#!/usr/bin/env python3
"""Step 1 — Preview render configured for Blender 4.5.2 + RTX 3060 (OPTIX only)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import bpy

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scdl import StageTimer, get_logger, get_pipeline_paths
from scdl.logging import log_devices, log_environment
from scdl.runtime import (
    cycles_devices,
    ensure_optix_device,
    require_blender_version,
    set_cycles_scene_device,
)


@dataclass(frozen=True)
class PreviewConfig:
    """Strongly typed configuration extracted from environment variables."""

    target_short: int
    samples: int
    adaptive_threshold: float
    adaptive_min_samples: int
    denoise: bool
    blur_glossy: float
    clamp_direct: Optional[float]
    clamp_indirect: Optional[float]
    caustics: bool
    seed: int


def _require_positive_int(value: str, *, var: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:  # pragma: no cover - defensive validation
        raise RuntimeError(f"[Step1] {var} must be an integer, received '{value}'.") from exc
    if parsed <= 0:
        raise RuntimeError(f"[Step1] {var} must be greater than zero (received {parsed}).")
    return parsed


def _parse_optional_float(value: Optional[str], *, var: str) -> Optional[float]:
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError as exc:  # pragma: no cover - defensive validation
        raise RuntimeError(f"[Step1] {var} must be a float, received '{value}'.") from exc


def _parse_bool(value: Optional[str], *, default: bool, var: str) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"[Step1] {var} must be truthy/falsy (received '{value}').")


def read_preview_config() -> PreviewConfig:
    """Read configuration from environment variables and validate values."""

    target_short = _require_positive_int(os.getenv("SCDL_PREVIEW_SHORT", "448"), var="SCDL_PREVIEW_SHORT")
    samples = _require_positive_int(os.getenv("SCDL_PREVIEW_SPP", "16"), var="SCDL_PREVIEW_SPP")
    adaptive_threshold = float(os.getenv("SCDL_PREVIEW_ADAPTIVE_THRESHOLD", "0.04"))
    adaptive_min_samples = _require_positive_int(os.getenv("SCDL_PREVIEW_MIN_SPP", "4"), var="SCDL_PREVIEW_MIN_SPP")
    denoise = _parse_bool(os.getenv("SCDL_PREVIEW_DENOISE"), default=True, var="SCDL_PREVIEW_DENOISE")
    blur_glossy = float(os.getenv("SCDL_PREVIEW_BLUR_GLOSSY", "0.2"))
    clamp_direct = _parse_optional_float(os.getenv("SCDL_PREVIEW_CLAMP_DIRECT"), var="SCDL_PREVIEW_CLAMP_DIRECT")
    clamp_indirect = _parse_optional_float(os.getenv("SCDL_PREVIEW_CLAMP_INDIRECT"), var="SCDL_PREVIEW_CLAMP_INDIRECT")
    caustics = _parse_bool(os.getenv("SCDL_PREVIEW_CAUSTICS"), default=False, var="SCDL_PREVIEW_CAUSTICS")
    seed = _require_positive_int(os.getenv("SCDL_PREVIEW_SEED", "42"), var="SCDL_PREVIEW_SEED")

    return PreviewConfig(
        target_short=target_short,
        samples=samples,
        adaptive_threshold=adaptive_threshold,
        adaptive_min_samples=adaptive_min_samples,
        denoise=denoise,
        blur_glossy=blur_glossy,
        clamp_direct=clamp_direct,
        clamp_indirect=clamp_indirect,
        caustics=caustics,
        seed=seed,
    )


def configure_resolution(scene: bpy.types.Scene, target_short: int) -> tuple[int, int]:
    """Snap the render resolution to multiples of 16 while preserving aspect."""

    render = scene.render
    rx = max(1, int(render.resolution_x))
    ry = max(1, int(render.resolution_y))

    if rx >= ry:
        new_y = target_short
        new_x = int(round(target_short * (rx / ry)))
    else:
        new_x = target_short
        new_y = int(round(target_short * (ry / rx)))

    def snap16(value: int) -> int:
        snapped = max(16, (value // 16) * 16)
        if snapped == 0:
            raise RuntimeError("[Step1] Resolution snapped to zero, refusing to continue.")  # pragma: no cover
        return snapped

    render.resolution_x = snap16(new_x)
    render.resolution_y = snap16(new_y)
    render.resolution_percentage = 100
    render.film_transparent = False

    return render.resolution_x, render.resolution_y


def configure_cycles(scene: bpy.types.Scene, cfg: PreviewConfig) -> None:
    """Apply Cycles settings optimised for deterministic previews."""

    cycles = scene.cycles
    cycles.device = "GPU"
    cycles.samples = cfg.samples
    cycles.use_adaptive_sampling = True
    cycles.adaptive_threshold = cfg.adaptive_threshold
    cycles.adaptive_min_samples = cfg.adaptive_min_samples

    cycles.use_denoising = cfg.denoise
    cycles.denoiser = "OPTIX"
    for view_layer in scene.view_layers:
        view_layer.cycles.use_denoising = cfg.denoise
        view_layer.cycles.denoising_store_passes = cfg.denoise

    cycles.caustics_reflective = cfg.caustics
    cycles.caustics_refractive = cfg.caustics
    cycles.blur_glossy = cfg.blur_glossy

    if cfg.clamp_direct is not None:
        cycles.sample_clamp_direct = cfg.clamp_direct
    if cfg.clamp_indirect is not None:
        cycles.sample_clamp_indirect = cfg.clamp_indirect

    cycles.seed = cfg.seed


def configure_output(render: bpy.types.RenderSettings, path: Path) -> None:
    """Set render output path and format."""

    render.engine = "CYCLES"
    render.image_settings.file_format = "PNG"
    render.image_settings.color_depth = "8"
    render.filepath = str(path)


def log_configuration(logger, cfg: PreviewConfig, resolution: tuple[int, int], devices: Iterable[str]) -> None:
    """Emit structured logs describing the configuration."""

    rx, ry = resolution
    log_environment(
        logger,
        {
            "preview_short": cfg.target_short,
            "resolution": f"{rx}x{ry}",
            "samples": cfg.samples,
            "adaptive_threshold": cfg.adaptive_threshold,
            "adaptive_min_samples": cfg.adaptive_min_samples,
            "denoise": cfg.denoise,
            "blur_glossy": cfg.blur_glossy,
            "clamp_direct": cfg.clamp_direct,
            "clamp_indirect": cfg.clamp_indirect,
            "caustics": cfg.caustics,
            "seed": cfg.seed,
        },
    )
    log_devices(logger, devices)


def main() -> None:
    """Entry point executed inside Blender."""

    paths = get_pipeline_paths(Path(__file__).resolve().parent)
    logger = get_logger("scdl.step1.preview", default_path=paths.log_file)

    logger.info("[step] Step1 preview render starting")

    info = require_blender_version((4, 5, 2))
    logger.info("[system] Blender %s (commit %s)", info.version_string, info.build_commit)
    logger.info("[system] Background mode: %s", info.is_background)

    device_override = os.getenv("SCDL_CYCLES_DEVICE", "OPTIX").strip().upper()
    if device_override != "OPTIX":
        raise RuntimeError(f"[Step1] SCDL_CYCLES_DEVICE must be OPTIX (received '{device_override}').")

    matched_devices = ensure_optix_device("RTX 3060")
    set_cycles_scene_device()
    devices = cycles_devices()

    cfg = read_preview_config()
    scene = bpy.context.scene

    configure_output(scene.render, paths.preview)

    resolution: tuple[int, int]
    with StageTimer("preview.configure", logger=logger):
        resolution = configure_resolution(scene, cfg.target_short)
        configure_cycles(scene, cfg)

    log_configuration(logger, cfg, resolution, devices)
    if matched_devices:
        log_devices(logger, [f"OPTIX REQUIRED {name}" for name in matched_devices])

    with StageTimer("preview.render", logger=logger):
        bpy.ops.render.render(write_still=True)

    logger.info("[OK] Preview saved to %s", paths.preview.as_posix())


if __name__ == "__main__":
    main()
