#!/usr/bin/env python3
"""Step 2 — DINOv3 saliency mask generation (CUDA-only, Flash Attention required)."""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import decode_image, read_file
import kornia as K
import kornia.morphology as KM

import transformers
from transformers import AutoImageProcessor, AutoModel
from torch.nn.attention import SDPBackend, sdpa_kernel

try:
    import OpenEXR
    import Imath
except Exception as exc:  # pragma: no cover - module import failure surfaces early
    raise RuntimeError(
        "[Step2/DINO] OpenEXR Python bindings (OpenEXR, Imath) are required."
    ) from exc

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scdl import StageTimer, TimerRecord, get_logger, get_pipeline_paths
from scdl.logging import log_devices, log_environment
from scdl.runtime import ensure_flash_attention, platform_summary, torch_device_summary


class Step2Error(RuntimeError):
    """Explicit error type for Step 2 failures."""


@dataclass(frozen=True)
class SaliencyConfig:
    """Validated environment configuration for DINOv3 saliency."""

    local_dir: str
    device: str
    preview_path: Path
    out_dir: Path
    mask_npy: Path
    mask_exr: Path
    preview_png: Path
    make_preview: bool
    preview_mode: str
    preview_alpha: float
    resize: int
    perc_lo: float
    perc_hi: float
    gamma: float
    morph_kernel: int

    @classmethod
    def from_env(cls, paths) -> "SaliencyConfig":
        """Construct configuration from environment variables."""

        make_preview = _env_bool("SCDL_USER_IMPORTANCE_PREVIEW", default=True)
        preview_mode = os.getenv("SCDL_USER_IMPORTANCE_PREVIEW_MODE", "gray").strip().lower()
        if preview_mode not in {"gray", "color"}:
            raise Step2Error("SCDL_USER_IMPORTANCE_PREVIEW_MODE must be 'color' or 'gray'.")

        preview_alpha = float(os.getenv("SCDL_USER_IMPORTANCE_PREVIEW_ALPHA", "0.6"))
        resize = _env_positive_int("SCDL_DINO_SIZE", default=336)
        perc_lo = float(os.getenv("SCDL_PERC_LO", "0.60"))
        perc_hi = float(os.getenv("SCDL_PERC_HI", "0.995"))
        gamma = float(os.getenv("SCDL_MASK_GAMMA", "0.70"))
        morph_kernel = _env_positive_int("SCDL_MORPH_K", default=3)

        local_dir = os.getenv("SCDL_DINO_LOCAL_DIR", "models/dinov3-vitl16").strip()
        if not local_dir:
            raise Step2Error("SCDL_DINO_LOCAL_DIR must point to a local DINOv3 repository.")
        preview_override = os.getenv("SCDL_PREVIEW_PATH", "")
        preview_path = Path(preview_override).expanduser() if preview_override else paths.preview

        device_override = os.getenv("SCDL_MASK_DEVICE", "").strip().lower()
        if device_override and not device_override.startswith("cuda"):
            raise Step2Error("SCDL_MASK_DEVICE must reference a CUDA device (e.g. cuda or cuda:0).")
        device_raw = device_override if device_override else "cuda"
        device = device_raw if ":" in device_raw else f"{device_raw}:0"

        out_dir_override = os.getenv("SCDL_OUT_DIR", "").strip()
        out_dir = Path(out_dir_override).expanduser() if out_dir_override else paths.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        mask_npy = out_dir / "user_importance.npy"
        mask_exr = out_dir / "user_importance_mask.exr"
        preview_png = out_dir / "user_importance_preview.png"

        return cls(
            local_dir=str(Path(local_dir).resolve()),
            device=device,
            preview_path=preview_path,
            out_dir=out_dir,
            mask_npy=mask_npy,
            mask_exr=mask_exr,
            preview_png=preview_png,
            make_preview=make_preview,
            preview_mode=preview_mode,
            preview_alpha=preview_alpha,
            resize=resize,
            perc_lo=perc_lo,
            perc_hi=perc_hi,
            gamma=gamma,
            morph_kernel=morph_kernel,
        )


def _env_positive_int(name: str, *, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise Step2Error(f"{name} must be an integer (received '{raw}').") from exc
    if value <= 0:
        raise Step2Error(f"{name} must be greater than zero (received {value}).")
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
    raise Step2Error(f"{name} must be truthy/falsy (received '{raw}').")


def load_image_tensor(path: Path) -> torch.Tensor:
    """Load a PNG preview file into a `[1,C,H,W]` tensor."""

    if not path.exists():
        raise Step2Error(f"Preview image not found at {path}.")
    data = read_file(str(path))
    image = decode_image(data)
    if image.ndim != 3 or image.shape[0] not in (1, 3, 4):
        raise Step2Error(f"Unsupported preview tensor shape {tuple(image.shape)}.")
    if image.shape[0] == 4:
        image = image[:3]
    return image.unsqueeze(0)


def to_numpy_hwc_uint8(img_bchw: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor `[1,C,H,W]` into `H×W×C` uint8 array."""

    return img_bchw.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()


def save_exr_single_channel(path: Path, array_f32_hw: np.ndarray, channel_name: str = "Y") -> None:
    """Persist a single-channel float32 EXR file."""

    if array_f32_hw.ndim != 2:
        raise Step2Error(f"EXR writer expects H×W float32 array (received shape {array_f32_hw.shape}).")
    height, width = array_f32_hw.shape
    header = OpenEXR.Header(width, height)
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    header["channels"] = {channel_name: Imath.Channel(pixel_type)}
    out = OpenEXR.OutputFile(str(path), header)
    try:
        out.writePixels({channel_name: array_f32_hw.astype(np.float32).tobytes()})
    finally:
        out.close()


def contrast_stretch(x: torch.Tensor, lo: float, hi: float, eps: float = 1e-6) -> torch.Tensor:
    """GPU-friendly percentile stretching to normalise saliency response."""

    xf = x.float()
    height, width = int(xf.shape[-2]), int(xf.shape[-1])
    pooled = F.avg_pool2d(xf, kernel_size=4, stride=4) if min(height, width) >= 128 else xf
    q_lo = torch.quantile(pooled, lo)
    q_hi = torch.quantile(pooled, hi)
    stretched = (xf - q_lo) / (q_hi - q_lo + eps)
    return stretched.clamp(0, 1).to(x.dtype)


def cls_patch_cosine_saliency(last_hidden: torch.Tensor, num_register_tokens: int) -> torch.Tensor:
    """Compute cosine similarity between CLS token and patch tokens."""

    cls = last_hidden[:, 0:1, :]
    patches = last_hidden[:, 1 + num_register_tokens :, :]
    cls_n = F.normalize(cls, dim=-1)
    patches_n = F.normalize(patches, dim=-1)
    return torch.einsum("bid,bjd->bij", cls_n, patches_n).squeeze(1)


def run_saliency(
    cfg: SaliencyConfig,
    processor: AutoImageProcessor,
    model: AutoModel,
    logger,
    timings: List[TimerRecord],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Execute the full saliency pipeline and return mask, preview RGB, and metadata."""

    device_obj = torch.device(cfg.device)

    with StageTimer("preview.load", logger=logger, records=timings):
        img_bchw = load_image_tensor(cfg.preview_path)
        height, width = int(img_bchw.shape[-2]), int(img_bchw.shape[-1])
        image_np = to_numpy_hwc_uint8(img_bchw)

    with StageTimer("dino.preprocess", logger=logger, records=timings):
        inputs = processor(images=image_np, return_tensors="pt", do_resize=True, size=cfg.resize)
        pixel_values = inputs["pixel_values"].to(cfg.device, dtype=torch.bfloat16, non_blocking=True)
        pixel_values = pixel_values.contiguous(memory_format=torch.channels_last)

    gpu_sync = lambda: torch.cuda.synchronize(device=device_obj)

    with StageTimer("dino.forward", logger=logger, records=timings, sync_callbacks=[gpu_sync]):
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16), torch.inference_mode():
                outputs = model(pixel_values=pixel_values)
        last_hidden = outputs.last_hidden_state

    with StageTimer("dino.post", logger=logger, records=timings):
        with torch.inference_mode():
            num_register_tokens = int(getattr(model.config, "num_register_tokens", 0))
            sal = cls_patch_cosine_saliency(last_hidden, num_register_tokens)
            batch, patches = sal.shape
            if batch != 1:
                raise Step2Error(f"Unexpected batch size from DINOv3 output: {batch}")

            patch = int(getattr(model.config, "patch_size", 16))
            in_h, in_w = int(pixel_values.shape[-2]), int(pixel_values.shape[-1])
            grid_h, grid_w = in_h // patch, in_w // patch
            if grid_h * grid_w != patches:
                g = int(round(math.sqrt(patches)))
                if g * g != patches:
                    raise Step2Error(f"Cannot infer patch grid for {patches} tokens.")
                grid_h = grid_w = g

            sal_grid = sal.reshape(1, 1, grid_h, grid_w)
            sal_full = F.interpolate(sal_grid, size=(height, width), mode="bilinear", align_corners=False)
            sal_full = (sal_full.clamp(-1, 1) + 1.0) * 0.5

            sal_full = contrast_stretch(sal_full, cfg.perc_lo, cfg.perc_hi)
            sal_full = torch.pow(sal_full, cfg.gamma)

            kernel = torch.ones(cfg.morph_kernel, cfg.morph_kernel, device=device_obj, dtype=sal_full.dtype)
            sal_proc = KM.closing(sal_full, kernel)
            sal_proc = K.filters.gaussian_blur2d(sal_proc, (cfg.morph_kernel, cfg.morph_kernel), (0.5 * cfg.morph_kernel, 0.5 * cfg.morph_kernel))
            sal_proc = sal_proc.clamp(0, 1).squeeze(0).squeeze(0)
            mask_np = sal_proc.detach().float().cpu().numpy()

    metrics = {"height": height, "width": width}
    return mask_np, image_np, metrics


def write_outputs(
    cfg: SaliencyConfig,
    mask_np: np.ndarray,
    image_np: np.ndarray,
    logger,
    timings: List[TimerRecord],
) -> None:
    """Persist NPY/EXR outputs and optional preview image."""

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    with StageTimer("io.save", logger=logger, records=timings):
        np.save(str(cfg.mask_npy), mask_np.astype(np.float32))
        save_exr_single_channel(cfg.mask_exr, mask_np, channel_name="Y")

    if cfg.make_preview:
        with StageTimer("preview.write", logger=logger, records=timings):
            _write_preview(cfg, mask_np, image_np, logger)


def _write_preview(cfg: SaliencyConfig, mask_np: np.ndarray, image_np: np.ndarray, logger) -> None:
    """Write the optional preview PNG, either grayscale or color overlay."""

    try:
        import cv2
    except Exception as exc:  # pragma: no cover - dependency failure surfaces here
        raise Step2Error("OpenCV (cv2) is required to produce the preview PNG.") from exc

    mask_u8 = np.clip(mask_np * 255.0, 0, 255).astype(np.uint8)

    if cfg.preview_mode == "color":
        bgr_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        heat_bgr = cv2.applyColorMap(mask_u8, cv2.COLORMAP_VIRIDIS)
        blended = cv2.addWeighted(bgr_image, 1.0, heat_bgr, float(cfg.preview_alpha), 0.0)
        if not cv2.imwrite(str(cfg.preview_png), blended):
            raise Step2Error(f"Failed to write preview overlay at {cfg.preview_png}.")
    else:
        if not cv2.imwrite(str(cfg.preview_png), mask_u8):
            raise Step2Error(f"Failed to write preview grayscale mask at {cfg.preview_png}.")

    logger.info("[OK] Preview mask saved to %s", cfg.preview_png.as_posix())


def summarize_timings(logger, timings: Iterable[TimerRecord]) -> None:
    """Log a consolidated timing summary."""

    entries = list(timings)
    if not entries:
        return
    logger.info("[step] Timing summary")
    for record in entries:
        logger.info("[timer] %s %.3fs", record.name, record.seconds)


def main() -> None:
    """Run Step 2 inside the configured Conda environment."""

    paths = get_pipeline_paths(Path(__file__).resolve().parent)
    logger = get_logger("scdl.step2.dino", default_path=paths.log_file)
    logger.info("[step] Step2 DINO saliency starting")

    cfg = SaliencyConfig.from_env(paths)
    expected_gpu = os.getenv("SCDL_EXPECTED_GPU", "RTX 3060")
    log_environment(
        logger,
        {
            "platform": platform_summary(),
            "local_dir": cfg.local_dir,
            "preview_path": cfg.preview_path,
            "resize": cfg.resize,
            "perc_lo": cfg.perc_lo,
            "perc_hi": cfg.perc_hi,
            "gamma": cfg.gamma,
            "morph_kernel": cfg.morph_kernel,
            "make_preview": cfg.make_preview,
            "preview_mode": cfg.preview_mode,
            "device": cfg.device,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "kornia_version": getattr(K, "__version__", "unknown"),
            "expected_gpu": expected_gpu,
        },
    )

    timings: List[TimerRecord] = []

    try:
        device_obj = torch.device(cfg.device)
        torch.cuda.set_device(device_obj)
        with StageTimer("env.validate", logger=logger, records=timings):
            device_name = torch_device_summary(expected_gpu)
            torch.set_float32_matmul_precision("high")
            ensure_flash_attention()
        log_devices(logger, [f"CUDA {device_name}"])

        with StageTimer(
            "model.load_processor",
            logger=logger,
            records=timings,
            sync_callbacks=[lambda: torch.cuda.synchronize(device=device_obj)],
        ):
            processor = AutoImageProcessor.from_pretrained(cfg.local_dir, local_files_only=True)
            model = AutoModel.from_pretrained(cfg.local_dir, dtype=torch.bfloat16, local_files_only=True)

        if hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation("sdpa")
        model = model.to(cfg.device).to(memory_format=torch.channels_last).eval()

        mask_np, image_np, metadata = run_saliency(cfg, processor, model, logger, timings)
        write_outputs(cfg, mask_np, image_np, logger, timings)
        logger.info(
            "[OK] Saliency mask written (%dx%d)", metadata["width"], metadata["height"]
        )
        logger.info("[OK] Outputs: %s, %s", cfg.mask_npy.as_posix(), cfg.mask_exr.as_posix())
        summarize_timings(logger, timings)
    except Exception as exc:
        logger.exception("[ERROR] Step2 failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
