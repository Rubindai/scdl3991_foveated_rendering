#!/usr/bin/env python3
"""
Render a single frame at a chosen SPP in either Baseline or Foveated mode.

Prompts interactively, launches Blender headless, and writes outputs ONLY to:
single_render_out/ (logs) and single_render_out/renders/{mode}_{spp}.png.

No files are written to the benchmark sweep directory.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SINGLE_OUT = PROJECT_ROOT / "single_render_out"
RENDERS_DIR = SINGLE_OUT / "renders"

# Ensure local package imports work inside Blender
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # Detect Blender context
    import bpy  # type: ignore

    INSIDE_BLENDER = True
except ImportError:
    INSIDE_BLENDER = False


def load_scdl_env() -> Dict[str, str]:
    env_path = PROJECT_ROOT / ".scdl.env"
    entries: Dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.split("#", 1)[0].strip().strip("'").strip('"')
            entries[k.strip()] = v
    return entries


def prompt_mode() -> str:
    while True:
        raw = input("Mode (baseline/foveated) [baseline]: ").strip().lower()
        if not raw:
            return "BASELINE"
        if raw in {"baseline", "b"}:
            return "BASELINE"
        if raw in {"foveated", "f"}:
            return "FOVEATED"
        print("Please enter 'baseline' or 'foveated'.")


def prompt_spp() -> int:
    while True:
        raw = input("Target SPP (positive integer) [32]: ").strip()
        if not raw:
            return 32
        try:
            val = int(raw)
            if val <= 0:
                print("SPP must be positive.")
                continue
            return val
        except ValueError:
            print("Invalid integer. Try again.")


def resolve_blend(env_file: Dict[str, str]) -> Path:
    blend_str = env_file.get("BLEND_FILE") or os.getenv("BLEND_FILE")
    if not blend_str:
        raise RuntimeError("Set BLEND_FILE in .scdl.env or environment.")
    blend = PROJECT_ROOT / blend_str
    if not blend.exists():
        raise RuntimeError(f"Blend file not found at {blend}")
    return blend


if INSIDE_BLENDER:
    import time
    from scdl import get_logger
    from scdl.logging import log_devices, log_environment
    from scdl.runtime import (
        cycles_devices,
        ensure_optix_device,
        require_blender_version,
        set_cycles_scene_device,
    )
    import full_render_baseline
    import step3_singlepass_foveated

    def run_blender_single() -> None:
        mode = os.getenv("SINGLE_MODE", "BASELINE").upper()
        spp_raw = os.getenv("SINGLE_SPP", "32")
        try:
            target_spp = int(spp_raw)
            if target_spp <= 0:
                raise ValueError
        except ValueError:
            print(f"[Single] Invalid SINGLE_SPP '{spp_raw}'.")
            sys.exit(1)

        SINGLE_OUT.mkdir(parents=True, exist_ok=True)
        RENDERS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = SINGLE_OUT / "single_render.log"
        logger = get_logger(f"scdl.single.{mode.lower()}", default_path=log_path)

        require_blender_version((4, 5, 4))
        expected_gpu = os.getenv("SCDL_EXPECTED_GPU", "RTX 3060")
        ensure_optix_device(expected_gpu)
        set_cycles_scene_device()
        devices = cycles_devices()
        log_devices(logger, devices)

        scene = bpy.context.scene
        view_layer = bpy.context.window.view_layer
        if hasattr(view_layer, "cycles"):
            view_layer.cycles.pass_debug_sample_count = False

        if mode == "BASELINE":
            cfg = full_render_baseline.FullRenderConfig.from_env()
            full_render_baseline.configure_cycles(scene, cfg)
            log_environment(
                logger,
                {
                    "mode": "baseline",
                    "samples": cfg.base_samples,
                    "min_samples": cfg.min_samples,
                    "max_samples": cfg.max_samples,
                    "adaptive_threshold": cfg.adaptive_threshold,
                    "adaptive_min_samples": cfg.adaptive_min_samples,
                    "light_sampling_threshold": cfg.light_sampling_threshold,
                    "expected_gpu": expected_gpu,
                },
            )
        elif mode == "FOVEATED":
            cfg = step3_singlepass_foveated.FoveatedConfig.from_env()
            out_dir = Path(os.getenv("SCDL_OUT_DIR") or (PROJECT_ROOT / "out"))
            mask_npy = out_dir / "user_importance.npy"
            mask_exr = out_dir / "user_importance_mask.exr"
            if not mask_npy.exists() or not mask_exr.exists():
                logger.error("[Single] Missing mask files; run the foveated pipeline first.")
                sys.exit(1)
            mask_stats = step3_singlepass_foveated.mask_statistics(mask_npy, cfg)
            mask_image = step3_singlepass_foveated.load_mask_image(mask_exr)
            lo, hi, gamma = mask_stats.lo, mask_stats.hi, mask_stats.gamma
            group = step3_singlepass_foveated.ensure_foveation_group()
            injected = 0
            for material in bpy.data.materials:
                if step3_singlepass_foveated.inject_group_into_material(
                    material,
                    mask_image,
                    group,
                    lo,
                    hi,
                    gamma,
                    cfg.reuse_textures,
                    cfg.lq_diffuse,
                    logger,
                ):
                    injected += 1
            logger.info("[Single] Injected materials: %d", injected)
            step3_singlepass_foveated.configure_cycles(scene, cfg, mask_stats)
            log_environment(
                logger,
                {
                    "mode": "foveated",
                    "samples": cfg.base_samples,
                    "min_samples": cfg.min_samples,
                    "max_samples": cfg.max_samples,
                    "adaptive_threshold": cfg.adaptive_threshold,
                    "adaptive_min_samples": cfg.adaptive_min_samples,
                    "mask_coverage": mask_stats.coverage,
                    "mask_hq": mask_stats.coverage_hq,
                    "mask_mid": mask_stats.coverage_mid,
                    "mask_lq": mask_stats.coverage_lq,
                    "expected_gpu": expected_gpu,
                },
            )
        else:
            logger.error("[Single] Unknown mode %s", mode)
            sys.exit(1)

        scene.cycles.samples = target_spp
        scene.cycles.adaptive_min_samples = min(scene.cycles.adaptive_min_samples, target_spp)
        scene.render.filepath = str(RENDERS_DIR / f"{mode.lower()}_{target_spp}.png")

        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        dt = time.time() - t0
        logger.info("[Single] Finished %s_%d.png in %.2fs", mode.lower(), target_spp, dt)
        print(f"[Single] Render complete: {scene.render.filepath} ({dt:.2f}s)")

    if __name__ == "__main__":
        run_blender_single()
        sys.exit(0)


def main() -> None:
    env_file = load_scdl_env()
    mode = prompt_mode()
    target_spp = prompt_spp()
    blend = resolve_blend(env_file)

    blender_exe = (
        os.getenv("SCDL_LINUX_BLENDER_EXE")
        or os.getenv("BLENDER_EXE")
        or env_file.get("SCDL_LINUX_BLENDER_EXE")
        or env_file.get("BLENDER_EXE")
    )
    if not blender_exe:
        raise RuntimeError("Set SCDL_LINUX_BLENDER_EXE or BLENDER_EXE (env or .scdl.env).")

    SINGLE_OUT.mkdir(parents=True, exist_ok=True)
    RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SINGLE_OUT / "single_render.log"

    cmd = [
        blender_exe,
        str(blend),
        "-b",
        "-E",
        "CYCLES",
        "--factory-startup",
        "--addons",
        "cycles",
        "-P",
        str(SCRIPT_DIR / "render_single.py"),
    ]
    env = os.environ.copy()
    env["SINGLE_MODE"] = mode
    env["SINGLE_SPP"] = str(target_spp)
    env["SINGLE_OUT"] = str(SINGLE_OUT)
    env["SCDL_LOG_MODE"] = "stdout"

    print(f"[Single] Launching Blender: {' '.join(cmd)}")
    print(f"[Single] Mode={mode}, SPP={target_spp}")
    print(f"[Single] Log: {log_path}")

    with open(log_path, "a", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        if process.stdout:
            for line in process.stdout:
                f.write(line)
                f.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

    dest_path = RENDERS_DIR / f"{mode.lower()}_{target_spp}.png"
    print(f"[Single] Done. Image: {dest_path}")


if __name__ == "__main__" and not INSIDE_BLENDER:
    main()
