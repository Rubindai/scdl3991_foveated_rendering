#!/usr/bin/env python3
# step4_optional_luxcore_render.py
# Optional LuxCore render guided by the DINO USER_IMPORTANCE mask (invoked from WSL).
# The top-level pipeline only calls this script when SCDL_USE_WSL_FINAL=1.

import os
import time
import array
from pathlib import Path
import argparse

import numpy as np
import imageio.v3 as iio
from PIL import Image

import pyluxcore  # pip install pyluxcore (WSL conda env)
from typing import Optional, Dict

from logging_utils import get_logger
from scdl_config import env_path, get_pipeline_paths

# =========== CONFIG (edit) ===========
PATHS = get_pipeline_paths(Path(__file__).resolve().parent)
PROJECT_DIR = PATHS.project_dir
EXPORT_DIR = PATHS.export_dir     # from Blender FileSaver
OUT_DIR = PATHS.out_dir
LOGGER = get_logger("scdl_step4_render", PATHS.log_file)


def log_info(msg: str):
    LOGGER.info(msg)


def log_warn(msg: str):
    LOGGER.warning(msg)


def log_error(msg: str):
    LOGGER.error(msg)

# Allow overriding the config path (prefer the exact path produced by FileSaver)
CFG_PATH = env_path("SCDL_CFG_PATH", EXPORT_DIR / "render.cfg", base=PROJECT_DIR)

# Precomputed DINO mask path (produced by step2_dino_mask.py)
MASK_NPY = PATHS.mask_npy
PREVIEW_PATH = PATHS.preview

# Final render (defaults; can be overridden by CLI/env)
FINAL_W = int(os.environ.get("SCDL_FINAL_W", "1024"))
FINAL_H = int(os.environ.get("SCDL_FINAL_H", "1024"))
FINAL_HALTSPS = int(os.environ.get("SCDL_HALTSPS", "1200"))
FINAL_PATH = PATHS.final

# Adaptive noise estimation cadence
NOISE_WARMUP_SPP = int(os.environ.get("SCDL_NOISE_WARMUP", "32"))
NOISE_STEP_SPP = int(os.environ.get("SCDL_NOISE_STEP", "32"))
HALT_TIME = int(os.environ.get("SCDL_HALTTIME", "500"))
HALT_THRESHOLD = float(os.environ.get("SCDL_HALTTHRESH", str(8/256)))
USE_GPU_DEFAULT = int(os.environ.get("SCDL_USE_GPU", "0")) != 0
IMPORTANCE_FROM_FILE = int(os.environ.get("SCDL_IMPORTANCE_FROM_FILE", "0")) != 0
IMPORTANCE_FILE_PATH = os.environ.get("SCDL_IMPORTANCE_FILE", "")
IMPORTANCE_MIN = float(os.environ.get("SCDL_IMPORTANCE_MIN", "0.2"))  # clamp the minimum per-pixel importance
# =====================================


# ---------- LuxCore helpers ----------
def make_props_from_cfg(cfg_path: Path, overrides: Optional[Dict] = None):
    # Add cfg directory so relative references in cfg resolve
    pyluxcore.AddFileNameResolverPath(str(cfg_path.parent))
    props = pyluxcore.Properties(str(cfg_path))
    # Also add the directory of the referenced scene.scn so meshes/textures resolve
    try:
        scene_file = props.Get("scene.file").GetString()
        scene_path = Path(scene_file)
        if not scene_path.is_absolute():
            scene_path = (cfg_path.parent / scene_path).resolve()
        pyluxcore.AddFileNameResolverPath(str(scene_path.parent))
    except Exception:
        pass
    if overrides:
        for k, v in overrides.items():
            props.Set(pyluxcore.Property(k, v))
    return props


def _write_importance_map_file(mask: np.ndarray, out_dir: Path) -> Optional[str]:
    """Write the importance mask to a file LuxCore can read (prefer EXR)."""
    m = np.clip(mask.astype(np.float32), 0.0, 1.0)
    path = Path(IMPORTANCE_FILE_PATH) if IMPORTANCE_FILE_PATH else (out_dir / "user_importance_input.exr")
    try:
        iio.imwrite(path.as_posix(), m)
        return path.resolve().as_posix()
    except Exception:
        try:
            alt = path.with_suffix(".png")
            iio.imwrite(alt.as_posix(), (m * 255).astype(np.uint8))
            return alt.resolve().as_posix()
        except Exception:
            return None


def _try_upload_runtime(film, mask_buf) -> bool:
    """Attempt runtime upload of USER_IMPORTANCE via available Film APIs."""
    try:
        if hasattr(film, "UpdateOutput"):
            film.UpdateOutput(pyluxcore.FilmOutputType.USER_IMPORTANCE, mask_buf, 0)
            return True
    except Exception:
        pass
    try:
        if hasattr(film, "SetOutput"):
            film.SetOutput(pyluxcore.FilmOutputType.USER_IMPORTANCE, mask_buf)
            return True
    except Exception:
        pass
    try:
        ch_enum = getattr(pyluxcore, "FilmChannelType", None)
        if ch_enum is not None and hasattr(film, "SetChannel"):
            film.SetChannel(ch_enum.USER_IMPORTANCE, mask_buf)
            return True
    except Exception:
        pass
    return False


def single_pass_with_user_importance(props, mask: np.ndarray, out_path: Path, haltspp: int,
                                     halttime: Optional[int] = HALT_TIME,
                                     haltthreshold: Optional[float] = HALT_THRESHOLD,
                                     use_gpu: bool = False,
                                     importance_from_file: bool = IMPORTANCE_FROM_FILE):
    tmp_exr = (out_path.parent / "user_importance.exr").as_posix()
    noise_exr = (out_path.parent / "noise.exr").as_posix()
    scount_exr = (out_path.parent / "samplecount.exr").as_posix()
    if use_gpu:
        props.SetFromString("""
            renderengine.type = PATHOCL
            opencl.gpu.use = 1
            opencl.cpu.use = 0
            film.opencl.enable = 1
            sampler.type = SOBOL
            opencl.devices.select = ""
        """)
    else:
        props.SetFromString("""
            renderengine.type = PATHCPU
            opencl.gpu.use = 0
            film.opencl.enable = 0
            sampler.type = SOBOL
        """)

    props.SetFromString(f"""
        film.noiseestimation.warmup = {NOISE_WARMUP_SPP}
        film.noiseestimation.step = {NOISE_STEP_SPP}
        film.outputs.0.type = RGB_IMAGEPIPELINE
        film.outputs.0.index = 0
        film.outputs.0.filename = "{out_path.as_posix()}"
        film.outputs.1.type = USER_IMPORTANCE
        film.outputs.1.filename = "{tmp_exr}"
        film.outputs.2.type = NOISE
        film.outputs.2.filename = "{noise_exr}"
        film.outputs.3.type = SAMPLECOUNT
        film.outputs.3.filename = "{scount_exr}"
        batch.haltspp = {haltspp} 0
        batch.halttime = {int(halttime) if halttime else 0}
        batch.haltthreshold = {float(haltthreshold) if haltthreshold else 0}
        batch.haltthreshold.warmup = {NOISE_WARMUP_SPP}
        batch.haltthreshold.step = {NOISE_STEP_SPP}
        batch.haltthreshold.filter.enable = False
        batch.haltthreshold.stoprendering.enable = True
    """)

    if importance_from_file:
        path = _write_importance_map_file(mask, out_path.parent)
        if path:
            try:
                props.SetFromString(f"""
                    sampler.userimportance.enable = 1
                    sampler.userimportance.mapfile = "{path}"
                """)
            except Exception:
                pass
            try:
                props.SetFromString(f"film.userimportance.filename = \"{path}\"")
            except Exception:
                pass

    try:
        config = pyluxcore.RenderConfig(props)
    except Exception as e:
        if use_gpu:
            log_warn(f"[GPU] Falling back to CPU due to: {e}")
            props.SetFromString("""
                renderengine.type = PATHCPU
                opencl.gpu.use = 0
                film.opencl.enable = 0
                sampler.type = SOBOL
                opencl.devices.select = ""
            """)
            config = pyluxcore.RenderConfig(props)
        else:
            raise
    session = pyluxcore.RenderSession(config)

    mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
    if IMPORTANCE_MIN > 0.0:
        # Ensure every pixel gets a baseline sampling probability to avoid zeroed tiles
        a = float(max(0.0, min(0.95, IMPORTANCE_MIN)))
        mask = mask * (1.0 - a) + a

    H, W = mask.shape
    try:
        fw = int(props.Get("film.width").GetInt())
        fh = int(props.Get("film.height").GetInt())
    except Exception:
        fw, fh = W, H
    assert (W, H) == (fw, fh), f"Mask shape {mask.shape} must match film {fh}x{fw}"
    buf = array.array("f", np.ascontiguousarray(mask, dtype=np.float32).ravel())
    film = session.GetFilm()
    uploaded = _try_upload_runtime(film, buf)
    if not uploaded:
        if importance_from_file:
            log_info("[INFO] USER_IMPORTANCE runtime upload unavailable; using file-based map if the config accepted it.")
        else:
            log_warn("[WARN] USER_IMPORTANCE runtime upload unavailable and file-based map disabled; rendering without it.")

    session.Start()
    t0 = time.time()
    while not session.HasDone():
        time.sleep(0.25)
        if halttime and halttime > 0 and (time.time() - t0) > (halttime + 5):
            log_warn("[WARN] Manual time cutoff reached; stopping session.")
            break
    film.SaveOutputs()
    session.Stop()


def main():
    parser = argparse.ArgumentParser(description="DINO-guided LuxCore render")
    parser.add_argument("--use-gpu", dest="use_gpu", action="store_true", default=None,
                        help="Prefer GPU (PATHOCL); falls back to CPU on failure")
    parser.add_argument("--no-gpu", dest="use_gpu", action="store_false", help="Force CPU PATH integrator")
    parser.set_defaults(use_gpu=USE_GPU_DEFAULT)
    parser.add_argument("--haltspp", type=int, default=FINAL_HALTSPS, help="Eye SPP cap")
    parser.add_argument("--halttime", type=int, default=HALT_TIME, help="Time cap (seconds)")
    parser.add_argument("--haltthr", type=float, default=HALT_THRESHOLD, help="Noise threshold")
    args = parser.parse_args()

    assert CFG_PATH.exists(), f"Missing {CFG_PATH}"

    pyluxcore.Init()

    assert PREVIEW_PATH.exists(), f"Missing preview image at {PREVIEW_PATH}. Run the Blender preview step first."
    assert MASK_NPY.exists(), f"Missing mask: {MASK_NPY}. Run step2_dino_mask.py before final render."

    final_props = make_props_from_cfg(CFG_PATH, overrides={"film.width": FINAL_W, "film.height": FINAL_H})
    try:
        fw = int(final_props.Get("film.width").GetInt())
        fh = int(final_props.Get("film.height").GetInt())
    except Exception:
        fw, fh = FINAL_W, FINAL_H

    mask = np.load(MASK_NPY).astype(np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mh, mw = mask.shape
    if (mw, mh) != (fw, fh):
        # LuxCore expects the mask to match the render dimensions; resize when necessary
        m_img = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8), mode="L")
        m_img = m_img.resize((fw, fh), resample=Image.BILINEAR)
        mask = (np.asarray(m_img).astype(np.float32) / 255.0)
        iio.imwrite((OUT_DIR / "user_importance_resized.png").as_posix(), (mask * 255).astype(np.uint8))

    single_pass_with_user_importance(final_props, mask, FINAL_PATH,
                                     haltspp=args.haltspp, halttime=args.halttime,
                                     haltthreshold=args.haltthr, use_gpu=args.use_gpu)

    log_info(f"[DONE] Final: {FINAL_PATH}")
    log_info(f"[INFO] Mask saved at: {OUT_DIR/'user_importance.npy'}")


if __name__ == "__main__":
    main()
