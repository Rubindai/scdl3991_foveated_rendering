#!/usr/bin/env python3
# step2_dino_mask.py  — USER script integrated
# Purpose: Run under WSL to compute a USER_IMPORTANCE mask from out/preview.png
# using DINOv3. Saves out/user_importance.npy and visualization PNGs.
#
# Key characteristics:
# - CLS–patch cosine saliency built on DINOv3 forward features for crisp maps.
# - Mixed-precision inference (if CUDA) with inference-mode to reduce overhead.
# - Adaptive mask refinement with morphological pad + feather to avoid seams.
# - Optional extra debug outputs:
#     SCDL_SAVE_ATTENTION=1 → out/attention_mask.png
#     SCDL_SAVE_FOVEATED=1  → out/foveated_preview.png
# - Choose the backbone with SCDL_DINO_ARCH (vitl16|vits16); defaults to ViT-L/16.
# - Automatic mixed precision on CUDA when SCDL_MASK_AMP=1 (default).

from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence
import time

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np
import imageio.v3 as iio
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms as T

try:
    import cv2  # type: ignore
except ImportError as exc:  # pragma: no cover - hard failure
    raise RuntimeError("OpenCV (cv2) is required for step2_dino_mask.py") from exc

from logging_utils import get_logger
from scdl_config import env_path, get_pipeline_paths

# =========== PATHS / CONFIG ===========
PATHS = get_pipeline_paths(Path(__file__).resolve().parent)
PROJECT_DIR = PATHS.project_dir
OUT_DIR = PATHS.out_dir
PREVIEW_PATH = PATHS.preview
EXR_OUTPUT_PATH = OUT_DIR / "fovea_mask.exr"

# DINOv3 architecture selection + assets
_ARCH_ALIASES = {
    "vitl16": "vitl16",
    "large": "vitl16",
    "l": "vitl16",
    "vits16": "vits16",
    "small": "vits16",
    "s": "vits16",
}
DINO_ARCH_RAW = os.environ.get("SCDL_DINO_ARCH", "vitl16").strip().lower()
DINO_ARCH = _ARCH_ALIASES.get(DINO_ARCH_RAW, "vitl16")
DINO_HUB_ENTRY = {
    "vitl16": "dinov3_vitl16",
    "vits16": "dinov3_vits16",
}[DINO_ARCH]
_DEFAULT_WEIGHTS = {
    "vitl16": PROJECT_DIR / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "vits16": PROJECT_DIR / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
}

# DINOv3 local clone + weights
REPO_DIR = env_path("SCDL_DINO_REPO", PROJECT_DIR / "dinov3", base=PROJECT_DIR)
WEIGHT_PATH = env_path(
    "SCDL_DINO_WEIGHTS",
    _DEFAULT_WEIGHTS[DINO_ARCH],
    base=PROJECT_DIR,
)

# Outputs
MASK_NPY = PATHS.mask_npy
MASK_PREVIEW = PATHS.mask_preview
LOGGER = get_logger("scdl_step2", PATHS.log_file)


def log_info(msg: str): LOGGER.info(msg)
def log_warn(msg: str): LOGGER.warning(msg)
def log_error(msg: str): LOGGER.error(msg)


def fail(msg: str, code: int = 1):
    LOGGER.error(msg)
    sys.exit(code)


USE_AMP = torch.cuda.is_available() and int(os.environ.get("SCDL_MASK_AMP", "1")) != 0
ALLOW_TF32 = torch.cuda.is_available() and int(os.environ.get("SCDL_MASK_ALLOW_TF32", "1")) != 0
DEVICE_OVERRIDE_RAW = os.environ.get("SCDL_MASK_DEVICE", "").strip()
DEVICE_OVERRIDE = DEVICE_OVERRIDE_RAW if DEVICE_OVERRIDE_RAW else None
MASK_SMOOTH_K = max(1, int(float(os.environ.get("SCDL_MASK_SMOOTH_K", "1"))))
MASK_GAMMA = max(0.01, float(os.environ.get("SCDL_MASK_GAMMA", os.environ.get("GAMMA", "1.0"))))
MASK_PAD_FRAC = float(os.environ.get("SCDL_MASK_PAD_FRAC", "0.0125"))
MASK_FEATHER_FRAC = float(os.environ.get("SCDL_MASK_FEATHER_FRAC", "0.006"))
MASK_BLUR_SIGMA = float(os.environ.get("SCDL_MASK_BLUR_SIGMA", "1.5"))
EARLY_EXIT_ENABLED = int(os.environ.get("SCDL_MASK_EARLY_EXIT", "1")) != 0
EARLY_EPS = max(1e-4, float(os.environ.get("SCDL_MASK_EARLY_EPS", "0.002")))
EARLY_MIN_SCALES = max(1, int(os.environ.get("SCDL_MASK_EARLY_MIN", "1")))
GC_ROI_PAD_FRAC = float(os.environ.get("SCDL_GC_ROI_PAD_FRAC", "0.05"))


def _amp_dtype() -> torch.dtype:
    override = os.environ.get("SCDL_MASK_AMP_DTYPE", "").strip().lower()
    if override in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if override in {"fp16", "float16"}:
        return torch.float16
    try:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


# ---- Preprocess (no resize here; we letterbox manually) ----
def _make_preprocess():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


# ---- Load DINOv3 model via the repo's torch.hub entrypoint ----
def load_dinov3_model(repo_dir: Path, hub_entry: str, weight_path: Path, device=None, *, device_override: str | None = None):
    if not repo_dir.exists():
        fail(
            f"[DINO] Missing repo at {repo_dir}. Set SCDL_DINO_REPO to your local clone of facebookresearch/dinov3.")
    if not weight_path.exists():
        fail(f"[DINO] Missing weights: {weight_path}")

    threads_raw = os.environ.get("SCDL_MASK_THREADS", "").strip()
    if threads_raw.isdigit():
        try:
            torch.set_num_threads(max(1, int(threads_raw)))
            log_info(f"[DINO] torch.set_num_threads({threads_raw})")
        except Exception as exc:
            log_warn(f"[DINO] set_num_threads failed: {exc}")

    if device_override:
        try:
            device = torch.device(device_override)
        except Exception as exc:
            fail(f"[DINO] Invalid SCDL_MASK_DEVICE='{device_override}': {exc}")
    else:
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if device.type == "cuda" and not torch.cuda.is_available():
        fail("[DINO] CUDA device requested but torch.cuda.is_available() is False.")

    preprocess = _make_preprocess()

    try:
        model = torch.hub.load(str(repo_dir), hub_entry,
                               source='local', weights=str(weight_path))
    except Exception as e:
        fail(f"[DINO] torch.hub.load failed (repo={repo_dir}): {e}")

    # Quick structure sanity check for DINOv3 ViT models
    model.eval().to(device)

    if int(os.environ.get("SCDL_MASK_COMPILE", "0")) != 0:
        mode = os.environ.get("SCDL_MASK_COMPILE_MODE", "reduce-overhead")
        try:
            model = torch.compile(model, mode=mode)
            log_info(f"[DINO] torch.compile enabled (mode={mode}).")
        except Exception as exc:
            log_warn(f"[DINO] torch.compile failed ({exc}); continuing without compile.")

    if ALLOW_TF32 and device.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            log_info("[DINO] TF32 math enabled for faster inference.")
        except Exception as exc:
            log_warn(f"[DINO] Could not enable TF32 ({exc}).")

    if device.type == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
        except Exception as exc:
            log_warn(f"[DINO] Could not enable cuDNN benchmark ({exc}).")

    if device_override:
        log_info(f"[DINO] Using explicit device override: {device}")
    else:
        log_info(f"[DINO] Using device: {device}")

    return model, preprocess, device


# ---- Utilities ----
def _resize_letterbox(img_hw3: np.ndarray, size: int = 224):
    """Return letterboxed RGB image (size x size) and metadata to unpad."""
    H, W = img_hw3.shape[:2]
    scale = min(size / max(1, H), size / max(1, W))
    newH = max(1, int(round(H * scale)))
    newW = max(1, int(round(W * scale)))

    interpolation = cv2.INTER_LINEAR
    if newH < H or newW < W:
        interpolation = cv2.INTER_AREA

    resized = cv2.resize(img_hw3, (newW, newH), interpolation=interpolation)
    if resized.ndim == 2:
        resized = np.repeat(resized[:, :, None], 3, axis=2)

    canvas = np.zeros((size, size, resized.shape[2]), dtype=resized.dtype)
    top = (size - newH) // 2
    left = (size - newW) // 2
    canvas[top:top + newH, left:left + newW, :resized.shape[2]] = resized

    return canvas, (top, left, newH, newW, H, W)


def _normalize01(t: torch.Tensor) -> torch.Tensor:
    lo = float(t.min())
    hi = float(t.max())
    return (t - lo) / (hi - lo + 1e-6) if hi > lo else torch.zeros_like(t)


def _parse_scales(scales_str: str, *, lo: int = 160, hi: int = 896) -> list[int]:
    values: list[int] = []
    for token in scales_str.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            val = int(token)
        except ValueError:
            log_warn(f"[DINO] Ignoring non-integer scale '{token}'.")
            continue
        if val < lo or val > hi:
            log_warn(f"[DINO] Ignoring out-of-range scale {val} (expected {lo}-{hi}).")
            continue
        values.append(val)
    return values


def _parse_scale_weights(weights_str: str, count: int) -> torch.Tensor | None:
    tokens = [tok.strip() for tok in weights_str.split(",") if tok.strip()]
    if not tokens:
        return None
    if len(tokens) != count:
        fail(f"[DINO] SCDL_MASK_SCALE_WEIGHTS must provide {count} values (got {len(tokens)}).")
    try:
        weights = torch.tensor([float(tok) for tok in tokens], dtype=torch.float32)
    except ValueError:
        fail("[DINO] Could not parse SCDL_MASK_SCALE_WEIGHTS as floats.")
    if torch.all(weights == 0):
        fail("[DINO] SCDL_MASK_SCALE_WEIGHTS must not sum to zero.")
    weights = torch.clamp(weights, min=0.0)
    if torch.all(weights == 0):
        fail("[DINO] SCDL_MASK_SCALE_WEIGHTS must contain positive values.")
    weights /= weights.sum()
    return weights


def _auto_scale_list(height: int, width: int) -> list[int]:
    short = max(1, min(height, width))
    if short <= 256:
        ratios = [1.0]
    elif short <= 384:
        ratios = [0.75, 1.0]
    elif short <= 512:
        ratios = [0.75, 0.875, 1.0]
    elif short <= 768:
        ratios = [0.6, 0.8, 1.0]
    else:
        ratios = [0.5, 0.7, 1.0]
    scales = {max(32, int(round(short * r))) for r in ratios}
    return sorted(scales)


def _combine_scale_maps(maps: Sequence[torch.Tensor], mode: str, weights: torch.Tensor | None) -> torch.Tensor:
    if len(maps) == 1:
        return maps[0]
    stack = torch.stack(list(maps), dim=0)
    mode = mode.lower()
    if mode == "max":
        return stack.max(dim=0).values
    if mode == "median":
        return stack.median(dim=0).values
    if mode == "weighted" and weights is not None:
        w = weights.to(stack.device).view(-1, 1, 1)
        return torch.sum(stack * w, dim=0)
    if mode not in {"mean", "weighted"}:
        fail(f"[DINO] Unknown SCDL_MASK_REDUCE='{mode}'.")
    return stack.mean(dim=0)


def pad_and_feather_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("pad_and_feather_mask expects a 2D array.")

    h, w = mask.shape
    short_side = max(1, min(h, w))
    pad_px = int(round(short_side * MASK_PAD_FRAC))
    pad_px = max(1, pad_px)
    kernel_size = 2 * pad_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    expanded = cv2.dilate(mask.astype(np.float32), kernel, iterations=1)

    feather_sigma = max(0.0, float(short_side) * MASK_FEATHER_FRAC)
    if feather_sigma > 0:
        expanded = cv2.GaussianBlur(expanded, (0, 0), feather_sigma)

    expanded = np.clip(expanded, 0.0, 1.0)
    return expanded.astype(np.float32)


def _mask_refinement_metrics(mask: np.ndarray) -> dict[str, float]:
    coverage = float(np.mean(mask))
    mid_ratio = float(np.mean((mask > 0.15) & (mask < 0.85)))
    edge_strip = np.concatenate([
        mask[0:2, :].flatten(),
        mask[-2:, :].flatten(),
        mask[:, 0:2].flatten(),
        mask[:, -2:].flatten(),
    ])
    edge_mean = float(np.mean(edge_strip)) if edge_strip.size else 0.0
    try:
        binary = (mask >= 0.6).astype(np.uint8)
        components, _ = cv2.connectedComponents(binary)
        components = max(int(components) - 1, 0)
    except Exception:
        components = 1
    return {
        "coverage": coverage,
        "mid_ratio": mid_ratio,
        "edge_mean": edge_mean,
        "components": float(components),
    }


def _should_apply_grabcut(metrics: dict[str, float]) -> bool:
    if metrics["coverage"] < 0.005 or metrics["coverage"] > 0.995:
        return False
    if metrics["mid_ratio"] < 0.08 and metrics["edge_mean"] < 0.015 and metrics["components"] <= 1:
        return False
    return True


def _should_apply_watershed(metrics: dict[str, float]) -> bool:
    if metrics["coverage"] < 0.01 or metrics["coverage"] > 0.985:
        return False
    if metrics["components"] <= 1 and metrics["mid_ratio"] < 0.18:
        return False
    return True


def _auto_grabcut_iters(height: int, width: int) -> int:
    short = max(1, min(height, width))
    if short <= 512:
        return 4
    return 5


def _resolve_grabcut_iters(height: int, width: int) -> int:
    iters_env = os.environ.get("SCDL_MASK_GRABCUT_ITERS", "").strip().lower()
    if iters_env == "auto":
        return _auto_grabcut_iters(height, width)
    if iters_env.isdigit():
        return max(1, int(iters_env))
    return _auto_grabcut_iters(height, width)


# ===== Your core token helpers (ported from script.py) =====
PATCH_SIZE = 16  # ViT-*/16 (all DINOv3 ViTs here use 16px patches)


@torch.no_grad()
def get_patch_tokens_and_cls(model, x: torch.Tensor, patch_size: int = PATCH_SIZE):
    """
    Return:
      patch_tokens: [B, P, D]
      cls_token:    [B, D] or None
      H_p, W_p:     patch grid size
      R:            inferred #register tokens (may be 0)
    """
    B, C, H, W = x.shape
    H_p = H // patch_size
    W_p = W // patch_size
    P_expected = H_p * W_p

    tokens_seq = None
    cls = None

    # Preferred path: forward_features returns dict with normalized tokens.
    if hasattr(model, "forward_features"):
        out = model.forward_features(x)
        if isinstance(out, dict):
            # Dinov2/3-style dict keys (common)
            if "x_norm_patchtokens" in out:
                patch_tokens = out["x_norm_patchtokens"]  # [B, P, D]
                cls = out.get("x_norm_clstoken", None)    # [B, D] or None
                return patch_tokens, cls, H_p, W_p, 0
            # Some builds might return raw tokens
            if "tokens" in out and isinstance(out["tokens"], torch.Tensor) and out["tokens"].ndim == 3:
                tokens_seq = out["tokens"]  # [B, N, D]

    # Fallback: get_intermediate_layers
    if tokens_seq is None:
        if hasattr(model, "get_intermediate_layers"):
            tokens_seq = model.get_intermediate_layers(
                x, n=1, return_class_token=True, norm=True)[0]  # [B, N, D]
        else:
            raise RuntimeError(
                "Cannot obtain patch tokens. Need model.forward_features (dict) or model.get_intermediate_layers.")

    # Split CLS / registers / patches from the sequence
    B, N, D = tokens_seq.shape
    R = max(0, N - 1 - P_expected)  # inferred #register tokens
    cls = tokens_seq[:, 0, :]
    patch_tokens = tokens_seq[:, 1 + R:, :]  # [B, P_expected, D]
    if patch_tokens.shape[1] != P_expected:
        raise RuntimeError(
            f"Patch count mismatch: got {patch_tokens.shape[1]}, expected {P_expected}. "
            "Check patch size or how tokens are returned.")
    return patch_tokens, cls, H_p, W_p, R


@torch.no_grad()
def make_attention_map_from_cls_similarity(patch_tokens: torch.Tensor,
                                           cls_token: torch.Tensor | None,
                                           H_p: int, W_p: int) -> torch.Tensor:
    """Cosine attention to CLS (0..1) with pseudo-CLS fallback."""
    patches_n = F.normalize(patch_tokens, dim=-1)  # [B, P, D]
    if cls_token is None:
        cls_n = F.normalize(patches_n.mean(dim=1), dim=-1)  # [B, D]
    else:
        cls_n = F.normalize(cls_token, dim=-1)
    scores = torch.einsum("bpd,bd->bp", patches_n, cls_n)  # [B, P]
    attn = scores.reshape(-1, H_p, W_p)[0]
    a_min, a_max = attn.min(), attn.max()
    attn = (attn - a_min) / (a_max - a_min + 1e-8)
    return attn


# ---- USER core: single-scale attention on letterboxed image ----
def _use_amp_for_device(device: torch.device | str, requested: bool) -> bool:
    if not requested:
        return False
    dev_str = str(device)
    return torch.cuda.is_available() and dev_str.startswith("cuda")


@torch.no_grad()
def _user_saliency_single_scale(img_np_hw3: np.ndarray, model, preprocess, device,
                                out_wh, size: int = 336, *, use_amp: bool = False):
    img_let, meta = _resize_letterbox(img_np_hw3, size=size)
    x_cpu = preprocess(img_let).unsqueeze(0)
    if str(device).startswith("cuda"):
        try:
            x_cpu = x_cpu.pin_memory()
        except Exception:
            pass
        x = x_cpu.to(device, non_blocking=True)
    else:
        x = x_cpu.to(device)
    amp_ctx = nullcontext()
    if _use_amp_for_device(device, use_amp):
        device_type = str(device).split(":", 1)[0]
        amp_ctx = torch.amp.autocast(device_type=device_type, dtype=_amp_dtype())
    with torch.inference_mode():
        with amp_ctx:
            patch_tokens, cls_token, H_p, W_p, _ = get_patch_tokens_and_cls(
                model, x, patch_size=PATCH_SIZE)
            attn = make_attention_map_from_cls_similarity(
                patch_tokens, cls_token, H_p, W_p)
            attn = _normalize01(attn)
    attn = attn.float()

    # Optional smoothing + gamma (keep existing knobs)
    k = MASK_SMOOTH_K
    if k > 1:
        attn = F.avg_pool2d(
            attn[None, None, ...], kernel_size=k, stride=1, padding=k // 2).squeeze()
        attn = _normalize01(attn)
    attn = torch.pow(attn.clamp(0.0, 1.0), MASK_GAMMA)

    # Upsample to preview size, removing the letterbox
    size_sq = F.interpolate(attn[None, None, ...], size=(
        size, size), mode="bicubic", align_corners=False).squeeze()
    top, left, newH, newW, H0, W0 = meta
    size_sq = size_sq[top:top + newH, left:left + newW]
    out = F.interpolate(size_sq[None, None, ...], size=(
        H0, W0), mode="bicubic", align_corners=False).squeeze()
    return out


def _resolve_scale_settings(height: int, width: int) -> tuple[list[int], str, torch.Tensor | None, list[float] | None]:
    reduce_mode = os.environ.get("SCDL_MASK_REDUCE", "mean").strip().lower()
    scales_str = os.environ.get("SCDL_MASK_SCALES", "auto").strip()
    if scales_str.lower() == "auto":
        sizes = _auto_scale_list(height, width)
        sizes = [min(768, max(160, s)) for s in sizes]
    else:
        sizes = _parse_scales(scales_str, hi=768)
    if not sizes:
        fail("[DINO] No valid scales configured for saliency core.")

    weights_tensor: torch.Tensor | None = None
    weights_list: list[float] | None = None
    if reduce_mode == "weighted":
        raw_weights = os.environ.get("SCDL_MASK_SCALE_WEIGHTS", "")
        weights_tensor = _parse_scale_weights(raw_weights, len(sizes))
        if weights_tensor is None:
            reduce_mode = "mean"
        else:
            weights_list = [float(v) for v in weights_tensor.tolist()]
    order = list(range(len(sizes)))
    order.sort(key=lambda idx: sizes[idx], reverse=True)
    if order != list(range(len(sizes))):
        sizes = [sizes[i] for i in order]
        if weights_tensor is not None:
            idx_tensor = torch.tensor(order, dtype=torch.long)
            weights_tensor = weights_tensor[idx_tensor]
            weights_list = [float(v) for v in weights_tensor.tolist()]
    return sizes, reduce_mode, weights_tensor, weights_list


@torch.no_grad()
def dense_feature_importance_mask(img_np_hw3: np.ndarray, model, preprocess, device, out_wh,
                                  *, scale_cfg: tuple[list[int], str, torch.Tensor | None, list[float] | None]):
    """Compute saliency/importance via CLS–patch cosine similarity."""
    amp_enabled = _use_amp_for_device(device, USE_AMP)
    sizes, reduce_mode, weight_tensor, weights_list = scale_cfg
    reduce_mode = reduce_mode.lower()
    needs_stack = reduce_mode == "median"
    maps_for_stack: list[torch.Tensor] = [] if needs_stack else []
    fused_accum: torch.Tensor | None = None
    fused_prev: torch.Tensor | None = None
    current_map: torch.Tensor | None = None
    total_weight = 0.0
    used_count = 0

    for idx, sz in enumerate(sizes):
        new_map = _user_saliency_single_scale(
            img_np_hw3, model, preprocess, device, out_wh, size=sz, use_amp=amp_enabled)
        used_count += 1

        if needs_stack:
            maps_for_stack.append(new_map)

        if reduce_mode == "weighted" and weight_tensor is not None and weights_list is not None:
            weight = weights_list[idx]
            total_weight += weight
            if fused_accum is None:
                fused_accum = new_map * weight
            else:
                fused_accum = fused_accum + new_map * weight
            current_map = fused_accum / max(total_weight, 1e-8)
        elif reduce_mode == "max":
            fused_accum = new_map if fused_accum is None else torch.maximum(fused_accum, new_map)
            current_map = fused_accum
        elif reduce_mode == "median":
            current_weights = None
            if weight_tensor is not None:
                current_weights = weight_tensor[:used_count]
            current_map = _combine_scale_maps(maps_for_stack, reduce_mode, current_weights)
        else:  # mean or fallback
            fused_accum = new_map if fused_accum is None else fused_accum + new_map
            current_map = fused_accum / float(used_count)

        if current_map is None:
            raise RuntimeError("Saliency aggregation failed.")

        if EARLY_EXIT_ENABLED and fused_prev is not None and used_count >= EARLY_MIN_SCALES:
            delta = torch.mean(torch.abs(current_map - fused_prev)).item()
            if delta <= EARLY_EPS:
                log_info(f"[DINO] Early exit after {used_count} of {len(sizes)} scales (Δ={delta:.4f}).")
                break
        fused_prev = current_map.detach()

    if current_map is None:
        raise RuntimeError("No saliency scales were evaluated.")

    fused_norm = _normalize01(current_map)
    used_sizes = sizes[:used_count]
    log_info(f"[DINO] Saliency core scales={used_sizes}")
    return fused_norm.detach().cpu().numpy().astype(np.float32)


# ---- Optional edge-aware refinement (GrabCut). Skipped if OpenCV unavailable. ----
def refine_with_grabcut(img_rgb: np.ndarray, prob: np.ndarray, iters: int | None = None):
    use_gc = int(os.environ.get("SCDL_MASK_USE_GRABCUT", "1")) != 0
    if not use_gc:
        return prob

    H, W = prob.shape
    lo_q = float(os.environ.get("SCDL_GC_LO_Q", "0.35"))
    hi_q = float(os.environ.get("SCDL_GC_HI_Q", "0.65"))
    lo_q = max(0.0, min(1.0, lo_q))
    hi_q = max(0.0, min(1.0, hi_q))
    if hi_q <= lo_q:
        hi_q = min(0.99, max(lo_q + 0.05, hi_q))
    lo, hi = np.quantile(prob, (lo_q, hi_q))

    mask = np.full((H, W), cv2.GC_PR_BGD, np.uint8)
    mask[prob <= lo] = cv2.GC_BGD
    mask[prob >= hi] = cv2.GC_FGD
    # Optional split: near-fg uncertain band as PR_FGD to tighten seeds
    mid_t = (lo + hi) * 0.5
    band = (prob > lo) & (prob < hi)
    prfg = band & (prob >= mid_t)
    mask[prfg] = cv2.GC_PR_FGD

    border = max(0, int(os.environ.get("SCDL_GC_BORDER", "8")))
    if border > 0:
        mask[:border, :] = cv2.GC_BGD
        mask[-border:, :] = cv2.GC_BGD
        mask[:, :border] = cv2.GC_BGD
        mask[:, -border:] = cv2.GC_BGD

    # Restrict GrabCut to the uncertain region for speed (crop to ROI + pad)
    uncertain = (mask == cv2.GC_PR_BGD) | (mask == cv2.GC_PR_FGD)
    if np.any(uncertain):
        ys, xs = np.where(uncertain)
        pad = int(round(min(H, W) * GC_ROI_PAD_FRAC))
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(H, int(ys.max()) + pad + 1)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(W, int(xs.max()) + pad + 1)
    else:
        # No uncertain pixels → nothing to refine
        out = prob.copy()
        out[mask == cv2.GC_BGD] = 0.0
        m = out.max()
        if m > 0:
            out /= m
        return out

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    iters_env = os.environ.get("SCDL_MASK_GRABCUT_ITERS", "").strip().lower()
    if iters_env == "auto":
        iters = _auto_grabcut_iters(H, W)
    elif iters_env.isdigit():
        iters = int(iters_env)
    elif iters is None:
        iters = _auto_grabcut_iters(H, W)

    iters = max(1, int(iters))
    log_info(f"[DINO] GrabCut iterations → {iters}")

    # Run GrabCut only on ROI and paste labels back
    mask_roi = mask[y0:y1, x0:x1].copy()
    img_roi = img_rgb[y0:y1, x0:x1, :]
    roi_h, roi_w = mask_roi.shape[:2]
    max_mp = float(os.environ.get("SCDL_GC_ROI_MAX_MP", "1.5"))
    max_area = int(round(max_mp * 1_000_000))
    area = roi_h * roi_w
    scaled = False
    if area > max_area and roi_h > 1 and roi_w > 1:
        scale = (max_area / float(area)) ** 0.5
        new_w = max(1, int(round(roi_w * scale)))
        new_h = max(1, int(round(roi_h * scale)))
        if new_w < roi_w and new_h < roi_h:
            mask_roi = cv2.resize(mask_roi, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            img_roi = cv2.resize(img_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
            scaled = True
    # Choose iterations based on ROI size
    if iters is None:
        iters = _auto_grabcut_iters(img_roi.shape[0], img_roi.shape[1])
    try:
        cv2.grabCut(img_roi, mask_roi, None, bgdModel,
                    fgdModel, iters, cv2.GC_INIT_WITH_MASK)
    except Exception as exc:
        fail(f"[DINO] GrabCut refinement failed: {exc}")
    if scaled:
        mask_roi = cv2.resize(mask_roi, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    mask[y0:y1, x0:x1] = mask_roi
    fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)

    out = prob.copy()
    out[~fg] = 0.0
    m = out.max()
    if m > 0:
        out /= m
    return out


# ---- Optional watershed refinement for tighter object contours ----
def refine_with_watershed(img_rgb: np.ndarray, prob: np.ndarray):
    use_ws = int(os.environ.get("SCDL_MASK_WATERSHED", "1")) != 0
    if not use_ws:
        return prob

    if prob.ndim != 2 or prob.size == 0:
        return prob

    H, W = prob.shape
    ws_lo_q = float(os.environ.get("SCDL_WS_LO_Q", "0.30"))
    ws_hi_q = float(os.environ.get("SCDL_WS_HI_Q", "0.92"))
    ws_lo_q = max(0.0, min(1.0, ws_lo_q))
    ws_hi_q = max(0.0, min(1.0, ws_hi_q))
    if ws_hi_q <= ws_lo_q:
        ws_hi_q = min(0.99, max(ws_lo_q + 0.05, ws_hi_q))

    prob_norm = np.clip(prob.astype(np.float32), 0.0, 1.0)

    blur_sigma = float(os.environ.get("SCDL_WS_BLUR", "0.8"))
    if blur_sigma > 0:
        prob_norm = cv2.GaussianBlur(
            prob_norm, ksize=(0, 0), sigmaX=blur_sigma)

    lo, hi = np.quantile(prob_norm, (ws_lo_q, ws_hi_q))
    sure_fg = (prob_norm >= hi)
    sure_bg = (prob_norm <= lo)

    kernel_size = int(os.environ.get("SCDL_WS_KERNEL", "5"))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if int(os.environ.get("SCDL_WS_FG_CLOSE", "1")):
        sure_fg = cv2.morphologyEx(sure_fg.astype(
            np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    if int(os.environ.get("SCDL_WS_BG_OPEN", "1")):
        sure_bg = cv2.morphologyEx(sure_bg.astype(
            np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)

    markers = np.zeros((H, W), dtype=np.int32)
    markers[sure_bg] = 1
    markers[sure_fg] = 2

    img_u8 = (img_rgb.astype(np.uint8))
    if img_u8.ndim == 2:
        img_u8 = np.stack([img_u8, img_u8, img_u8], axis=-1)
    img_bgr = cv2.cvtColor(img_u8[:, :, :3], cv2.COLOR_RGB2BGR)

    try:
        markers = cv2.watershed(img_bgr, markers)
    except Exception as exc:
        fail(f"[DINO] Watershed refinement failed: {exc}")

    seg = (markers == 2).astype(np.uint8)
    seg_close = max(0, int(os.environ.get("SCDL_WS_SEG_CLOSE", "0")))
    if seg_close > 0:
        seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE,
                               kernel, iterations=seg_close)
    seg = cv2.GaussianBlur(seg.astype(np.float32), ksize=(0, 0), sigmaX=1.0)
    seg = np.clip(seg, 0.0, 1.0)

    blend = float(os.environ.get("SCDL_WS_BLEND", "0.45"))
    blend = max(0.0, min(1.0, blend))
    refined = (1.0 - blend) * prob_norm + blend * seg
    refined = np.clip(refined, 0.0, 1.0)
    return refined.astype(np.float32)


def main():
    if not PREVIEW_PATH.exists():
        fail(f"[DINO] Missing preview: {PREVIEW_PATH}. Run the Blender preview step first.")

    img = iio.imread(PREVIEW_PATH)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]
    H, W = img.shape[:2]

    sizes, reduce_mode, weights_tensor, weights_list = _resolve_scale_settings(H, W)
    scale_cfg = (sizes, reduce_mode, weights_tensor, weights_list)

    model, preprocess, device = load_dinov3_model(
        REPO_DIR, DINO_HUB_ENTRY, WEIGHT_PATH, device_override=DEVICE_OVERRIDE)
    log_info(f"[DINO] Using architecture '{DINO_ARCH}' (hub entry '{DINO_HUB_ENTRY}').")
    if _use_amp_for_device(device, USE_AMP):
        log_info("[DINO] Mixed precision enabled (torch.amp.autocast).")
    else:
        reason = "device is not CUDA" if not str(device).startswith("cuda") else "SCDL_MASK_AMP=0"
        log_info(f"[DINO] Mixed precision disabled ({reason}).")

    t_sal = time.perf_counter()
    mask_raw = dense_feature_importance_mask(
        img, model, preprocess, device, (W, H), scale_cfg=scale_cfg)
    log_info(f"[DINO] Saliency core time: {time.perf_counter() - t_sal:.2f}s")

    metrics = _mask_refinement_metrics(mask_raw)
    log_info(
        f"[DINO] Mask metrics → coverage={metrics['coverage']:.3f} mid={metrics['mid_ratio']:.3f} edge={metrics['edge_mean']:.3f} components={metrics['components']:.0f}")

    if int(os.environ.get("SCDL_MASK_USE_GRABCUT", "1")) != 0 and _should_apply_grabcut(metrics):
        t_ref_gc = time.perf_counter()
        mask_raw = refine_with_grabcut(
            img[:, :, :3].astype(np.uint8), mask_raw, iters=None)
        log_info(f"[DINO] GrabCut refinement time: {time.perf_counter() - t_ref_gc:.2f}s")
    else:
        log_info("[DINO] GrabCut skipped (mask already stable).")

    if int(os.environ.get("SCDL_MASK_WATERSHED", "1")) != 0 and _should_apply_watershed(metrics):
        t_ref_ws = time.perf_counter()
        mask_raw = refine_with_watershed(img[:, :, :3].astype(np.uint8), mask_raw)
        log_info(f"[DINO] Watershed refinement time: {time.perf_counter() - t_ref_ws:.2f}s")
    else:
        log_info("[DINO] Watershed skipped (mask contours sufficient).")

    np.save(MASK_NPY, mask_raw)

    lo, hi = np.percentile(mask_raw, 2), np.percentile(mask_raw, 98)
    spread = max(hi - lo, 1e-6)
    viz = np.clip((mask_raw - lo) / spread, 0, 1)
    if spread < 0.03:
        grid = (np.indices(mask_raw.shape).sum(0) % 16 == 0).astype(np.float32) * 0.08
        viz = np.clip(viz * (1.0 - 0.08) + grid, 0, 1)
    iio.imwrite(MASK_PREVIEW, (viz * 255).astype(np.uint8))
    log_info(
        f"[OK] Saved raw mask: {MASK_NPY}  shape={mask_raw.shape}  min={mask_raw.min():.3f} max={mask_raw.max():.3f} mean={mask_raw.mean():.3f}")
    log_info(f"[OK] Saved raw viz : {MASK_PREVIEW}")

    final_mask = pad_and_feather_mask(mask_raw)

    try:
        if MASK_BLUR_SIGMA > 0:
            final_mask = cv2.GaussianBlur(final_mask, (0, 0), MASK_BLUR_SIGMA)
            m_min, m_max = final_mask.min(), final_mask.max()
            if m_max > m_min:
                final_mask = (final_mask - m_min) / (m_max - m_min)
    except Exception as exc:
        fail(f"Gaussian blur failed in OpenCV: {exc}")

    mask_float32 = final_mask.astype(np.float32)
    try:
        import OpenEXR  # type: ignore
        import Imath    # type: ignore
    except Exception as exc:
        fail(f"OpenEXR bindings are required to write fovea_mask.exr ({exc}).")

    try:
        header = OpenEXR.Header(W, H)
        header['channels'] = {
            'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        }
        exr_file = OpenEXR.OutputFile(str(EXR_OUTPUT_PATH), header)
        exr_file.writePixels({'R': mask_float32.tobytes()})
        exr_file.close()
        log_info(f"[OK] Saved float EXR mask for Blender -> {EXR_OUTPUT_PATH}")
    except Exception as exc:
        fail(f"Failed to save EXR via OpenEXR: {exc}")

    if int(os.environ.get("SCDL_SAVE_ATTENTION", "0")):
        attn_path = OUT_DIR / "attention_mask.png"
        iio.imwrite(attn_path, (viz * 255).astype(np.uint8))
        log_info(f"[OK] Saved attention map: {attn_path}")
    if int(os.environ.get("SCDL_SAVE_FOVEATED", "0")):
        pil = Image.fromarray(img[:, :, :3].astype(np.uint8))
        Hh, Ww = pil.height, pil.width
        mask_t = torch.from_numpy(final_mask).float()
        mfull = F.interpolate(mask_t[None, None, ...], size=(Hh, Ww), mode="bilinear", align_corners=False)[0, 0]
        lo_w = max(1, Ww // max(1, int(os.environ.get("SCDL_FOV_LORES", "8"))))
        lo_h = max(1, Hh // max(1, int(os.environ.get("SCDL_FOV_LORES", "8"))))
        lo_img = pil.resize((lo_w, lo_h), Image.BILINEAR).resize((Ww, Hh), Image.BILINEAR)
        hi = T.ToTensor()(pil)
        lo = T.ToTensor()(lo_img)
        out = (mfull.unsqueeze(0) * hi + (1 - mfull.unsqueeze(0)) * lo).clamp(0, 1)
        fpath = OUT_DIR / "foveated_preview.png"
        iio.imwrite(fpath, (out.mul(255).byte().permute(1, 2, 0).numpy()))
        log_info(f"[OK] Saved foveated preview: {fpath}")

if __name__ == "__main__":
    main()
