#!/usr/bin/env python3
"""
Benchmark sweep for foveated vs. baseline renders.

- Outside Blender: orchestrates optional render sweeps and computes LPIPS/SSIM/PSNR
  against a reference image (out_full_render/final.png by default), then plots three
  graphs (one per metric) with both curves on the same axes.
- Inside Blender: renders a batch for either BASELINE or FOVEATED mode using the
  provided SPP list, writing images to benchmark_sweep_out/renders/.

Reference point: the final result of the full render pipeline (baseline) at your
chosen high SPP (default 512). All metrics compare against that image.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUT_ROOT = PROJECT_ROOT / "benchmark_sweep_out"
RENDERS_DIR = OUT_ROOT / "renders"
REFERENCE_PATH = PROJECT_ROOT / "out_full_render" / "final.png"

try:
    import bpy  # type: ignore

    INSIDE_BLENDER = True
except ImportError:
    INSIDE_BLENDER = False

# ==========================
# BLENDER RENDER PATH
# ==========================
if INSIDE_BLENDER:
    import time
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scdl import StageTimer, get_logger
    from scdl.logging import log_devices, log_environment
    from scdl.runtime import (
        cycles_devices,
        ensure_optix_device,
        require_blender_version,
        set_cycles_scene_device,
    )
    import full_render_baseline
    import step3_singlepass_foveated

    def _parse_spp_list(raw: str) -> List[int]:
        levels: List[int] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                value = int(part)
            except ValueError:
                continue
            if value > 0:
                levels.append(value)
        return sorted(set(levels))

    def run_blender_batch() -> None:
        mode = os.getenv("SCDL_BENCHMARK_MODE", "BASELINE").upper()
        spp_list_raw = os.getenv("SCDL_BENCHMARK_SPP_LIST", "")
        if not spp_list_raw:
            print("[Batch] No SPP list provided; nothing to do.")
            return
        spp_levels = _parse_spp_list(spp_list_raw)

        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        RENDERS_DIR.mkdir(parents=True, exist_ok=True)
        log_file = OUT_ROOT / f"benchmark_{mode.lower()}.log"

        logger = get_logger(f"scdl.benchmark.{mode.lower()}", default_path=log_file)
        logger.info("[Batch] Starting %s benchmark batch", mode)
        logger.info("[Batch] SPP Levels: %s", spp_levels)

        require_blender_version((4, 5, 4))
        expected_gpu = os.getenv("SCDL_EXPECTED_GPU", "RTX 3060")
        ensure_optix_device(expected_gpu)
        set_cycles_scene_device()
        devices = cycles_devices()
        log_devices(logger, devices)

        scene = bpy.context.scene
        skip_mask = False
        
        batch_times: Dict[int, float] = {}

        if mode == "BASELINE":
            cfg = full_render_baseline.FullRenderConfig.from_env()
            natural_cap = full_render_baseline.configure_cycles(scene, cfg)
            log_environment(
                logger,
                {
                    "mode": "baseline",
                    "base_samples": cfg.base_samples,
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
            mask_npy = Path(os.getenv("SCDL_OUT_DIR", PROJECT_ROOT / "out")) / "user_importance.npy"
            mask_exr = Path(os.getenv("SCDL_OUT_DIR", PROJECT_ROOT / "out")) / "user_importance_mask.exr"
            if not mask_npy.exists() or not mask_exr.exists():
                logger.error("[Batch] Missing mask files; run the pipeline first.")
                return
            mask_stats = step3_singlepass_foveated.mask_statistics(mask_npy, cfg)
            lo, hi, gamma = mask_stats.lo, mask_stats.hi, mask_stats.gamma
            mask_image = step3_singlepass_foveated.load_mask_image(mask_exr)
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
            logger.info("[Batch] Injected materials: %d", injected)
            sampling = step3_singlepass_foveated.configure_cycles(scene, cfg, mask_stats)
            natural_cap = sampling.effective_samples
            log_environment(
                logger,
                {
                    "mode": "foveated",
                    "base_samples": cfg.base_samples,
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
            logger.error("[Batch] Unknown mode: %s", mode)
            return

        for spp in spp_levels:
            if spp <= 0:
                logger.info("[Batch] Skipping non-positive SPP %s", spp)
                continue
            
            if spp > natural_cap:
                logger.info("[Batch] Overriding cap %s with requested SPP %s", natural_cap, spp)

            actual_spp = spp
            scene.cycles.samples = actual_spp
            scene.cycles.adaptive_min_samples = min(scene.cycles.adaptive_min_samples, actual_spp)
            scene.render.filepath = str(RENDERS_DIR / f"{mode.lower()}_{spp}.png")

            t0 = time.time()
            bpy.ops.render.render(write_still=True)
            dt = time.time() - t0
            logger.info("[Batch] Finished %s_%d.png in %.2fs", mode.lower(), spp, dt)
            batch_times[spp] = dt

        # Write render times to sidecar JSON
        (OUT_ROOT / f"times_{mode.lower()}.json").write_text(json.dumps(batch_times, indent=2))

    if __name__ == "__main__":
        run_blender_batch()
        sys.exit(0)

# ==========================
# ORCHESTRATOR + METRICS
# ==========================
import math
import numpy as np
from PIL import Image


def load_image(path: Path) -> np.ndarray:
    """Load an image as float32 [0,1]."""

    with Image.open(path) as im:
        arr = np.array(im)
    
    is_16bit = (arr.dtype == np.uint16)
    arr = arr.astype(np.float32)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    
    return arr / 65535.0 if is_16bit else arr / 255.0


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0.0:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("scikit-image is required for SSIM (pip install scikit-image).") from exc
    if a.ndim == 3 and a.shape[-1] == 3:
        score = 0.0
        for c in range(3):
            score += structural_similarity(a[..., c], b[..., c], data_range=1.0)
        return score / 3.0
    return float(structural_similarity(a, b, data_range=1.0))


def lpips_distance(a: np.ndarray, b: np.ndarray, model=None) -> float:
    import torch
    import lpips  # type: ignore

    to_tensor = lambda x: torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)
    a_t = to_tensor(a).float()
    b_t = to_tensor(b).float()
    a_t = a_t * 2.0 - 1.0
    b_t = b_t * 2.0 - 1.0
    
    if model is not None:
        device = next(model.parameters()).device
        loss_fn = model
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        import lpips
        loss_fn = lpips.LPIPS(net="vgg").to(device).eval()

    a_t = a_t.to(device)
    b_t = b_t.to(device)
    with torch.no_grad():
        dist = loss_fn(a_t, b_t)
    dist_val = float(dist.item())
    if not math.isfinite(dist_val):
        raise RuntimeError("LPIPS returned a non-finite value.")
    return dist_val


def collect_metrics(reference: Path, mode: str, spp_values: List[int], lpips_model=None, time_data: Dict[str, float] = None) -> Dict[int, Dict[str, float]]:
    results: Dict[int, Dict[str, float]] = {}
    ref = load_image(reference)

    # Initialize LPIPS model if not provided
    if lpips_model is None:
        try:
            import torch
            import lpips
            # Force CPU to avoid OOM
            device = torch.device("cpu")
            lpips_model = lpips.LPIPS(net="vgg").to(device).eval()
        except Exception as exc:
            print(f"[Metrics] Warning: Could not initialize LPIPS model: {exc}")

    import gc
    import torch

    for spp in spp_values:
        # We want to plot metrics even if we don't have time data, and vice versa.
        # The only hard requirement is the image existing for visual metrics.
        img_path = RENDERS_DIR / f"{mode}_{spp}.png"
        
        lp, ssim_val, psnr_val = float("nan"), float("nan"), float("nan")
        
        if img_path.exists():
            img = load_image(img_path)
            if img.shape == ref.shape:
                try:
                    lp = lpips_distance(img, ref, model=lpips_model)
                except Exception as exc:
                    print(f"[Metrics] LPIPS failed for {img_path}: {exc}")
                try:
                    ssim_val = ssim_score(img, ref)
                except Exception as exc:
                    print(f"[Metrics] SSIM failed for {img_path}: {exc}")
                psnr_val = psnr(img, ref)
            else:
                print(f"[Metrics] Skipping {img_path} (shape {img.shape} != reference {ref.shape})")
        
        render_time = float("nan")
        if time_data:
            render_time = time_data.get(str(spp), float("nan"))
            
        # Only add to results if we have at least one valid metric (visual or time)
        if not (math.isnan(lp) and math.isnan(ssim_val) and math.isnan(psnr_val) and math.isnan(render_time)):
            results[spp] = {"lpips": lp, "ssim": ssim_val, "psnr": psnr_val, "time": render_time}
        
        # Aggressive cleanup to avoid OOM on limited VRAM
        torch.cuda.empty_cache()
        gc.collect()
        
    return results


def plot_curves(metrics: Dict[str, Dict[int, Dict[str, float]]], metric: str, ylabel: str, title: str, outfile: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, AutoMinorLocator

    palette = {"baseline": "#1f77b4", "foveated": "#e45756"}
    markers = {"baseline": "o", "foveated": "D"}  # Circle vs Diamond to distinguish overlapping
    
    plt.figure(figsize=(12, 7))  # Larger plot
    plt.title(title, fontsize=16, fontweight="bold", pad=15)
    
    max_x = 0
    y_values: List[float] = []
    
    for label, data in metrics.items():
        xs = []
        ys = []
        for x in sorted(data.keys()):
            val = data[x].get(metric)
            if val is None or not math.isfinite(val):
                continue
            xs.append(x)
            ys.append(val)
        
        if not xs:
            continue
            
        y_values.extend(ys)
        max_x = max(max_x, max(xs))
        
        plt.plot(
            xs,
            ys,
            marker=markers.get(label, "o"),
            linestyle="-",
            linewidth=2.5,
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=2,
            alpha=0.85,
            color=palette.get(label, None),
            label=label.title(),
        )

    # Add a small margin to the X-axis limits so end points aren't clipped
    x_margin = max_x * 0.02 if max_x > 0 else 1
    plt.xlim(left=0, right=max_x + x_margin)
    
    if y_values:
        y_min = min(y_values)
        y_max = max(y_values)
        margin = 0.05 * (y_max - y_min if y_max != y_min else 1.0)
        plt.ylim(bottom=y_min - margin, top=y_max + margin)

    # Intelligent ticking
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins='auto', steps=[1, 2, 5, 10]))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    plt.legend(fontsize=11, framealpha=0.9)
    plt.grid(True, which='major', linestyle='-', alpha=0.6)
    plt.grid(True, which='minor', linestyle=':', alpha=0.3)
    
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def build_spp_range(start: int, stop: int, step: int) -> List[int]:
    if start < 0:
        start = 0
    values = list(range(start, stop + 1, step))
    if values and values[0] != 0:
        values.insert(0, 0)  # ensure x-axis starts at 0 even if not rendered
    return sorted(set(values))


def load_scdl_env() -> Dict[str, str]:
    env_path = PROJECT_ROOT / ".scdl.env"
    entries: Dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.split("#", 1)[0].strip().strip("'").strip('"')
            entries[k.strip()] = v
    return entries


def _get_env_bool(name: str, env_file: Dict[str, str], default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        raw = env_file.get(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be truthy/falsy (received '{raw}').")


def _mask_stats_for_cap(env_file: Dict[str, str]) -> Tuple[float, float, float, float]:
    """Return coverage, hq, mid, lq fractions using the same auto-threshold logic as Step 3."""

    import numpy as np

    out_dir = Path(env_file.get("SCDL_OUT_DIR") or os.getenv("SCDL_OUT_DIR") or (PROJECT_ROOT / "out"))
    mask_path = out_dir / "user_importance.npy"
    if not mask_path.exists():
        raise RuntimeError(f"Mask array not found at {mask_path}; run the pipeline first.")

    arr = np.load(mask_path).astype(np.float32)
    flat = arr[np.isfinite(arr)].ravel()
    if flat.size == 0:
        return 0.0, 0.0, 0.0, 1.0

    mask_bias = float(env_file.get("SCDL_FOVEATED_MASK_BIAS", os.getenv("SCDL_FOVEATED_MASK_BIAS", "0.08")))
    auto_thresh = _get_env_bool("SCDL_FOVEATED_AUTO_THRESH", env_file, True)
    auto_lo_push = float(env_file.get("SCDL_FOVEATED_AUTO_LO_PUSH", os.getenv("SCDL_FOVEATED_AUTO_LO_PUSH", "0.25")))
    auto_hi_pull = float(env_file.get("SCDL_FOVEATED_AUTO_HI_PULL", os.getenv("SCDL_FOVEATED_AUTO_HI_PULL", "0.15")))
    auto_mask_bias_gain = float(
        env_file.get("SCDL_FOVEATED_AUTO_MASK_BIAS_GAIN", os.getenv("SCDL_FOVEATED_AUTO_MASK_BIAS_GAIN", "0.12"))
    )

    base_p_lo, base_p_hi = 0.2, 0.8
    coverage = float(flat.mean())

    if auto_thresh:
        p_lo = min(0.9, base_p_lo + (1.0 - coverage) * auto_lo_push)
        p_hi_candidate = base_p_hi + (coverage - 0.5) * auto_hi_pull
        p_hi = max(p_lo + 0.05, min(0.99, p_hi_candidate))
        bias_dynamic = mask_bias + (1.0 - coverage) * auto_mask_bias_gain
    else:
        p_lo = base_p_lo
        p_hi = base_p_hi
        bias_dynamic = mask_bias

    q_lo_base, q_hi_base = np.quantile(flat, [p_lo, p_hi])
    lo = max(0.0, min(1.0, q_lo_base + bias_dynamic))
    hi = max(lo + 1e-6, min(1.0, q_hi_base + bias_dynamic))

    coverage_hq = float((flat >= hi).mean())
    coverage_lq = float((flat <= lo).mean())
    coverage_mid = max(0.0, min(1.0, 1.0 - coverage_hq - coverage_lq))
    return coverage, coverage_hq, coverage_mid, coverage_lq


def _predict_foveated_effective(env_file: Dict[str, str]) -> float:
    """Predict effective samples using the same math as Step 3 (without bpy)."""

    coverage, hq, mid, lq = _mask_stats_for_cap(env_file)

    base_samples = _get_env_number("SCDL_FOVEATED_BASE_SPP", env_file, typ="int")
    min_samples = _get_env_number("SCDL_FOVEATED_MIN_SPP", env_file, typ="int")
    max_samples = _get_env_number("SCDL_FOVEATED_MAX_SPP", env_file, typ="int")
    auto_samples = _get_env_bool("SCDL_FOVEATED_AUTOSPP", env_file, True)
    auto_bias = float(env_file.get("SCDL_FOVEATED_AUTOSPP_FLOOR", os.getenv("SCDL_FOVEATED_AUTOSPP_FLOOR", "0.35")))
    auto_gamma = float(env_file.get("SCDL_FOVEATED_AUTOSPP_GAMMA", os.getenv("SCDL_FOVEATED_AUTOSPP_GAMMA", "1.35")))
    auto_taper = float(env_file.get("SCDL_FOVEATED_AUTOSPP_TAPER", os.getenv("SCDL_FOVEATED_AUTOSPP_TAPER", "0.65")))
    min_floor_fraction = float(
        env_file.get("SCDL_FOVEATED_MIN_FLOOR_FRACTION", os.getenv("SCDL_FOVEATED_MIN_FLOOR_FRACTION", "0.5"))
    )
    importance_mid_weight = float(
        env_file.get("SCDL_FOVEATED_IMPORTANCE_MID_WEIGHT", os.getenv("SCDL_FOVEATED_IMPORTANCE_MID_WEIGHT", "0.35"))
    )
    lq_sample_bias = float(env_file.get("SCDL_FOVEATED_LQ_SAMPLE_BIAS", os.getenv("SCDL_FOVEATED_LQ_SAMPLE_BIAS", "0.85")))
    lq_min_scale = float(env_file.get("SCDL_FOVEATED_LQ_MIN_SCALE", os.getenv("SCDL_FOVEATED_LQ_MIN_SCALE", "0.35")))

    importance = max(0.0, min(1.0, hq + importance_mid_weight * mid))
    lq_scale = max(lq_min_scale, 1.0 - lq_sample_bias * lq)
    taper = auto_taper + (1.0 - auto_taper) * importance

    if auto_samples:
        coverage_term = importance ** auto_gamma
        sample_scale = taper * (auto_bias + (1.0 - auto_bias) * coverage_term) * lq_scale
        min_floor = int(
            round(min_samples * (min_floor_fraction + (1.0 - min_floor_fraction) * importance) * lq_scale)
        )
        min_floor = max(1, min(min_samples, min_floor))
        scaled = int(round(base_samples * sample_scale))
        effective = max(min_floor, min(max_samples, scaled))
    else:
        effective = max(min_samples, min(max_samples, base_samples))

    return float(effective)


def verify_metric_dependencies() -> None:
    missing: List[str] = []
    for mod in ("torch", "lpips", "skimage.metrics", "matplotlib"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise RuntimeError(
            "Missing metric dependencies: "
            + ", ".join(missing)
            + ". Activate the scdl-foveated environment or install them (conda env update --file environment.yml --prune)."
        )


def _get_env_number(name: str, env_file: Dict[str, str], *, typ: str) -> float:
    raw = os.getenv(name)
    if raw is None:
        raw = env_file.get(name)
    if raw is None:
        raise RuntimeError(f"Missing required setting {name} (set in .scdl.env or environment).")
    raw = raw.strip()
    try:
        return int(raw) if typ == "int" else float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a {typ} (received '{raw}').") from exc


def _get_env_number_with_fallback(primary: str, fallback: str, env_file: Dict[str, str], *, typ: str) -> float:
    raw = os.getenv(primary)
    if raw is None:
        raw = env_file.get(primary)
    if raw is None:
        raw = os.getenv(fallback)
    if raw is None:
        raw = env_file.get(fallback)
    if raw is None:
        raise RuntimeError(f"Missing required setting {primary} (fallback {fallback}) in .scdl.env or environment.")
    raw = raw.strip()
    try:
        return int(raw) if typ == "int" else float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{primary} (fallback {fallback}) must be a {typ} (received '{raw}').") from exc


def compute_baseline_cap(env_file: Dict[str, str]) -> int:
    base = _get_env_number("SCDL_BASELINE_SPP", env_file, typ="int")
    scale_raw = os.getenv("SCDL_BASELINE_SPP_SCALE") or env_file.get("SCDL_BASELINE_SPP_SCALE") or "1.0"
    try:
        scale = float(scale_raw.strip())
    except ValueError as exc:
        raise RuntimeError(f"SCDL_BASELINE_SPP_SCALE must be a float (received '{scale_raw}').") from exc
    base = int(round(base * scale))
    min_spp = _get_env_number_with_fallback("SCDL_BASELINE_MIN_SPP", "SCDL_FOVEATED_MIN_SPP", env_file, typ="int")
    max_spp = _get_env_number_with_fallback("SCDL_BASELINE_MAX_SPP", "SCDL_FOVEATED_MAX_SPP", env_file, typ="int")
    return max(min_spp, min(max_spp, base))


def compute_foveated_cap(env_file: Dict[str, str]) -> int:
    return int(round(_predict_foveated_effective(env_file)))


def prompt_val(prompt: str, default: int) -> int:
    """Ask the user for an integer input, falling back to default on empty."""
    while True:
        raw = input(f"{prompt} [Default: {default}]: ").strip()
        if not raw:
            return default
        try:
            val = int(raw)
            if val < 0:
                print("Please enter a positive integer.")
                continue
            return val
        except ValueError:
            print("Invalid integer. Try again.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark sweep for foveated vs baseline renders.")
    parser.add_argument("--blend", type=Path, default=None, help="Blend file to render (defaults to BLEND_FILE in .scdl.env).")
    parser.add_argument("--spp-start", type=int, default=0, help="Starting SPP (x-axis starts at 0).")
    parser.add_argument("--spp-step", type=int, default=32, help="Step size for SPP sweep.")
    parser.add_argument("--baseline-stop", type=int, default=None, help="End SPP for baseline curve (default: pipeline cap).")
    parser.add_argument("--foveated-stop", type=int, default=None, help="End SPP for foveated curve (default: pipeline cap).")
    parser.add_argument("--no-render", action="store_true", help="Skip rendering; only compute metrics/plots from existing images.")
    parser.add_argument("--batch", action="store_true", help="Run in non-interactive mode (accept defaults/args).")
    return parser.parse_args()


def launch_blender(mode: str, spp_values: List[int], blend: Path, env_file: Dict[str, str]) -> None:
    blender_exe = (
        os.getenv("SCDL_LINUX_BLENDER_EXE")
        or os.getenv("BLENDER_EXE")
        or env_file.get("SCDL_LINUX_BLENDER_EXE")
        or env_file.get("BLENDER_EXE")
    )
    if not blender_exe:
        raise RuntimeError("Set SCDL_LINUX_BLENDER_EXE or BLENDER_EXE (env or .scdl.env) to point to Blender 4.5.4.")
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
        str(SCRIPT_DIR / "benchmark_sweep.py"),
    ]
    env = os.environ.copy()
    env["SCDL_BENCHMARK_MODE"] = mode.upper()
    env["SCDL_BENCHMARK_SPP_LIST"] = ",".join(str(x) for x in spp_values if x > 0)
    # For sweep consistency, force foveated runs to honour exact SPP inputs (no autospp scaling).
    if mode.upper() == "FOVEATED":
        env["SCDL_FOVEATED_AUTOSPP"] = "0"
    env["SCDL_CLEAR_OUT_DIR"] = env.get("SCDL_CLEAR_OUT_DIR", "0")
    subprocess.run(cmd, check=True, env=env)


def detect_max_spp(mode: str) -> int:
    """Scan the renders directory to find the highest available SPP level."""
    max_val = 0
    if not RENDERS_DIR.exists():
        return 0
    prefix = f"{mode.lower()}_"
    suffix = ".png"
    for p in RENDERS_DIR.iterdir():
        if p.name.startswith(prefix) and p.name.endswith(suffix):
            try:
                val = int(p.name[len(prefix):-len(suffix)])
                max_val = max(max_val, val)
            except ValueError:
                continue
    return max_val


def main() -> None:
    args = parse_args()

    env_file = load_scdl_env()

    blend = args.blend
    if blend is None:
        blend_str = None
        if "BLEND_FILE" in env_file:
            blend_str = env_file["BLEND_FILE"]
        elif os.getenv("BLEND_FILE"):
            blend_str = os.getenv("BLEND_FILE")
        if blend_str:
            blend = PROJECT_ROOT / blend_str
    if blend is None or not blend.exists():
        raise RuntimeError("Blend file not found. Provide --blend or set BLEND_FILE in .scdl.env.")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RENDERS_DIR.mkdir(parents=True, exist_ok=True)

    baseline_cap = compute_baseline_cap(env_file)
    foveated_cap = compute_foveated_cap(env_file)

    # Determine defaults based on mode
    default_baseline_stop = baseline_cap
    default_foveated_stop = foveated_cap
    
    if args.no_render:
        # If not rendering, try to use existing files as the default range
        detected_base = detect_max_spp("baseline")
        if detected_base > 0:
            default_baseline_stop = detected_base
            
        detected_fov = detect_max_spp("foveated")
        if detected_fov > 0:
            default_foveated_stop = detected_fov

    if not args.batch:
        print(f"\n[Benchmark] Interactive Configuration (Press Enter to accept default)")
        baseline_stop = prompt_val("Baseline End SPP", default=default_baseline_stop)
        foveated_stop = prompt_val("Foveated End SPP", default=default_foveated_stop)
        step_size = prompt_val("SPP Step Size", default=args.spp_step)
    else:
        baseline_stop = args.baseline_stop if args.baseline_stop is not None else default_baseline_stop
        foveated_stop = args.foveated_stop if args.foveated_stop is not None else default_foveated_stop
        step_size = args.spp_step

    baseline_spp = build_spp_range(args.spp_start, baseline_stop, step_size)
    foveated_spp = build_spp_range(args.spp_start, foveated_stop, step_size)

    if not args.no_render:
        if not REFERENCE_PATH.exists():
            raise RuntimeError(f"Reference image missing at {REFERENCE_PATH}. Run the full baseline render first.")
        print(f"[Sweep] Baseline SPP levels: {baseline_spp}")
        print(f"[Sweep] Foveated SPP levels: {foveated_spp}")
        launch_blender("BASELINE", baseline_spp, blend, env_file)
        launch_blender("FOVEATED", foveated_spp, blend, env_file)

    verify_metric_dependencies()

    shared_model = None
    try:
        import torch
        import lpips
        # Force CPU for LPIPS to avoid OOM on 2k+ images with 6GB GPU (VGG features are huge)
        device = torch.device("cpu")
        shared_model = lpips.LPIPS(net="vgg").to(device).eval()
    except Exception as e:
        print(f"[Main] Warning: Could not load shared LPIPS model: {e}")

    def parse_times_from_log(log_path: Path) -> Dict[str, float]:
        """Fallback: parse render times from existing logs if JSON is missing."""
        extracted = {}
        if not log_path.exists():
            return extracted
        import re
        # Example log line: ... [Batch] Finished baseline_2.png in 3.29s
        pattern = re.compile(r"Finished (\w+)_(\d+)\.png in ([\d\.]+)s")
        for line in log_path.read_text().splitlines():
            match = pattern.search(line)
            if match:
                spp = match.group(2)
                duration = float(match.group(3))
                extracted[spp] = duration
        return extracted

    times_baseline = {}
    times_foveated = {}
    
    # Try loading JSON first
    try:
        times_baseline = json.loads((OUT_ROOT / "times_baseline.json").read_text())
    except Exception:
        pass
    
    # Fallback to log parsing if empty (legacy runs)
    if not times_baseline:
        print("[Main] times_baseline.json missing, parsing log...")
        times_baseline = parse_times_from_log(OUT_ROOT / "benchmark_baseline.log")

    try:
        times_foveated = json.loads((OUT_ROOT / "times_foveated.json").read_text())
    except Exception:
        pass
        
    if not times_foveated:
        print("[Main] times_foveated.json missing, parsing log...")
        times_foveated = parse_times_from_log(OUT_ROOT / "benchmark_foveated.log")

    metrics = {
        "baseline": collect_metrics(REFERENCE_PATH, "baseline", baseline_spp, lpips_model=shared_model, time_data=times_baseline),
        "foveated": collect_metrics(REFERENCE_PATH, "foveated", foveated_spp, lpips_model=shared_model, time_data=times_foveated),
    }

    (OUT_ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plot_curves(
        metrics,
        "lpips",
        "LPIPS Distance (lower is better)",
        "Perceptual Quality: LPIPS",
        OUT_ROOT / "lpips.png",
    )
    plot_curves(
        metrics,
        "ssim",
        "SSIM (higher is better)",
        "Structural Similarity: SSIM",
        OUT_ROOT / "ssim.png",
    )
    plot_curves(
        metrics,
        "psnr",
        "PSNR (dB, higher is better)",
        "Peak Signal-to-Noise Ratio: PSNR",
        OUT_ROOT / "psnr.png",
    )
    plot_curves(
        metrics,
        "time",
        "Render Time (s)",
        "Render Performance: Time vs. SPP",
        OUT_ROOT / "time.png",
    )

    # Write markdown table
    all_spp = sorted(set(baseline_spp + foveated_spp))
    table_lines = [
        "| SPP | Baseline Time | Baseline LPIPS | Baseline SSIM | Baseline PSNR | Foveated Time | Foveated LPIPS | Foveated SSIM | Foveated PSNR |",
        "|-----|---------------|----------------|---------------|---------------|---------------|----------------|---------------|---------------|",
    ]
    for spp in all_spp:
        b = metrics["baseline"].get(spp)
        f = metrics["foveated"].get(spp)
        def fmt(entry, key):
            if entry is None:
                return "N/A"
            val = entry.get(key)
            if val is None or (isinstance(val, float) and (val != val)):
                return "N/A"
            return f"{val:.4f}"
        row = [
            str(spp),
            fmt(b, "time"),
            fmt(b, "lpips"),
            fmt(b, "ssim"),
            fmt(b, "psnr"),
            fmt(f, "time"),
            fmt(f, "lpips"),
            fmt(f, "ssim"),
            fmt(f, "psnr"),
        ]
        table_lines.append("| " + " | ".join(row) + " |")
    (OUT_ROOT / "benchmark_results.md").write_text("\n".join(table_lines))

    write_unified_log(baseline_spp, foveated_spp)
    print("[Done] Metrics written to", OUT_ROOT)


def write_unified_log(baseline_spp: List[int], foveated_spp: List[int]) -> None:
    """Emit a single consolidated log combining render logs and metrics locations."""

    run_log = OUT_ROOT / "benchmark_run.log"
    baseline_log = OUT_ROOT / "benchmark_baseline.log"
    foveated_log = OUT_ROOT / "benchmark_foveated.log"

    lines: List[str] = []
    lines.append("[Sweep] Baseline SPP levels: " + ",".join(map(str, baseline_spp)))
    lines.append("[Sweep] Foveated SPP levels: " + ",".join(map(str, foveated_spp)))
    lines.append(f"[Sweep] Metrics: {OUT_ROOT / 'metrics.json'}")
    lines.append(f"[Sweep] Table: {OUT_ROOT / 'benchmark_results.md'}")
    lines.append(f"[Sweep] Plots: {OUT_ROOT / 'lpips.png'}, {OUT_ROOT / 'ssim.png'}, {OUT_ROOT / 'psnr.png'}, {OUT_ROOT / 'time.png'}")

    def append_file(tag: str, path: Path) -> None:
        if not path.exists():
            lines.append(f"[Sweep] {tag} log missing: {path}")
            return
        lines.append(f"[Sweep] ===== Begin {tag} log ({path}) =====")
        lines.extend(path.read_text().splitlines())
        lines.append(f"[Sweep] ===== End {tag} log =====")

    append_file("baseline", baseline_log)
    append_file("foveated", foveated_log)

    run_log.write_text("\n".join(lines))


if __name__ == "__main__" and not INSIDE_BLENDER:
    main()
