#!/usr/bin/env python3
# step2_dino_mask.py  — USER script integrated
# Purpose: Run under WSL to compute a USER_IMPORTANCE mask from out/preview.png
# using DINOv3. Saves out/user_importance.npy and visualization PNGs.
#
# What changed vs previous version:
# - The core saliency is now computed from your CLS–patch cosine method that
#   extracts patch tokens via forward_features/get_intermediate_layers
#   (ported from script.py). This generally gives crisper, single-object maps.
# - You can toggle back to the previous implementation via
#     SCDL_MASK_CORE=builtin
#   (defaults to "user").
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
from torchvision.transforms import functional as TF
import torchvision.transforms as T

from logging_utils import get_logger
from scdl_config import env_path, get_pipeline_paths

# =========== PATHS / CONFIG ===========
PATHS = get_pipeline_paths(Path(__file__).resolve().parent)
PROJECT_DIR = PATHS.project_dir
OUT_DIR = PATHS.out_dir
PREVIEW_PATH = PATHS.preview

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

    if ALLOW_TF32 and device.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            log_info("[DINO] TF32 math enabled for faster inference.")
        except Exception as exc:
            log_warn(f"[DINO] Could not enable TF32 ({exc}).")

    if device_override:
        log_info(f"[DINO] Using explicit device override: {device}")
    else:
        log_info(f"[DINO] Using device: {device}")

    return model, preprocess, device


# ---- Utilities ----
def _resize_letterbox(img_hw3: np.ndarray, size: int = 224):
    """Return letterboxed RGB uint8 image (size x size) and metadata to unpad."""
    H, W = img_hw3.shape[:2]
    scale = min(size / H, size / W)
    newH = int(round(H * scale))
    newW = int(round(W * scale))

    pil = Image.fromarray(img_hw3)
    pil_resized = TF.resize(pil, (newH, newW), antialias=True)

    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    top = (size - newH) // 2
    left = (size - newW) // 2
    canvas.paste(pil_resized, (left, top))

    # WRITABLE NumPy copy to avoid torchvision warnings
    canvas_np = np.array(canvas, dtype=np.uint8, copy=True)
    return canvas_np, (top, left, newH, newW, H, W)


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
        log_warn(f"[DINO] Ignoring SCDL_MASK_SCALE_WEIGHTS (expected {count} values, got {len(tokens)}).")
        return None
    try:
        weights = torch.tensor([float(tok) for tok in tokens], dtype=torch.float32)
    except ValueError:
        log_warn("[DINO] Could not parse SCDL_MASK_SCALE_WEIGHTS as floats; ignoring.")
        return None
    if torch.all(weights == 0):
        log_warn("[DINO] Scale weights sum to zero; ignoring.")
        return None
    weights = torch.clamp(weights, min=0.0)
    if torch.all(weights == 0):
        log_warn("[DINO] Scale weights are non-positive; ignoring.")
        return None
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
        log_warn(f"[DINO] Unknown SCDL_MASK_REDUCE='{mode}', defaulting to mean.")
    return stack.mean(dim=0)


def _auto_grabcut_iters(height: int, width: int) -> int:
    short = max(1, min(height, width))
    if short <= 512:
        return 6
    if short <= 960:
        return 8
    return 10


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
    x = preprocess(img_let).unsqueeze(0).to(device)  # [1,3,size,size]
    amp_ctx = nullcontext()
    if _use_amp_for_device(device, use_amp):
        device_type = str(device).split(":", 1)[0]
        amp_ctx = torch.autocast(device_type=device_type, dtype=torch.float16)
    with amp_ctx:
        patch_tokens, cls_token, H_p, W_p, _ = get_patch_tokens_and_cls(
            model, x, patch_size=PATCH_SIZE)
        attn = make_attention_map_from_cls_similarity(
            patch_tokens, cls_token, H_p, W_p)
        attn = _normalize01(attn)
    attn = attn.float()

    # Optional smoothing + gamma (keep existing knobs)
    k = max(1, int(os.environ.get("SCDL_MASK_SMOOTH_K", "1")))
    if k > 1:
        attn = F.avg_pool2d(
            attn[None, None, ...], kernel_size=k, stride=1, padding=k // 2).squeeze()
        attn = _normalize01(attn)
    gamma = max(0.01, float(os.environ.get(
        "SCDL_MASK_GAMMA", os.environ.get("GAMMA", "1.0"))))
    attn = torch.pow(attn.clamp(0.0, 1.0), gamma)

    # Upsample to preview size, removing the letterbox
    size_sq = F.interpolate(attn[None, None, ...], size=(
        size, size), mode="bicubic", align_corners=False).squeeze()
    top, left, newH, newW, H0, W0 = meta
    size_sq = size_sq[top:top + newH, left:left + newW]
    out = F.interpolate(size_sq[None, None, ...], size=(
        H0, W0), mode="bicubic", align_corners=False).squeeze()
    return out


@torch.no_grad()
def dense_feature_importance_mask(img_np_hw3: np.ndarray, model, preprocess, device, out_wh):
    """Compute saliency/importance. Default: use USER method; set SCDL_MASK_CORE=builtin to revert."""
    core = os.environ.get("SCDL_MASK_CORE", "user").strip().lower()
    amp_enabled = _use_amp_for_device(device, USE_AMP)

    reduce_mode = os.environ.get("SCDL_MASK_REDUCE", "mean").strip().lower()
    height, width = img_np_hw3.shape[:2]

    if core == "builtin":
        scales_str = os.environ.get("SCDL_MASK_SCALES", "336").strip()
        if scales_str.lower() == "auto":
            sizes = _auto_scale_list(height, width)
            sizes = [min(448, max(160, s)) for s in sizes]
        else:
            sizes = _parse_scales(scales_str, hi=448) or [336]
        weight_tensor: torch.Tensor | None = None
        if reduce_mode == "weighted":
            raw_weights = os.environ.get("SCDL_MASK_SCALE_WEIGHTS", "")
            weight_tensor = _parse_scale_weights(raw_weights, len(sizes))
            if weight_tensor is None:
                reduce_mode = "mean"
        maps = []
        for sz in sizes:
            maps.append(_user_saliency_single_scale(
                img_np_hw3, model, preprocess, device, out_wh, size=sz, use_amp=amp_enabled))
        sal = _combine_scale_maps(maps, reduce_mode, weight_tensor)
        sal = _normalize01(sal)
        log_info(f"[DINO] Saliency core (builtin) scales={sizes}")
        return sal.detach().cpu().numpy().astype(np.float32)

    # USER path (default)
    # Default to a single high-resolution scale for efficiency. For higher quality,
    # you can use multiple scales by setting the environment variable, e.g.:
    # SCDL_MASK_SCALES="336,392,448"
    scales_str = os.environ.get("SCDL_MASK_SCALES", "448").strip()
    if scales_str.lower() == "auto":
        sizes = _auto_scale_list(height, width)
        sizes = [min(768, max(160, s)) for s in sizes]
    else:
        sizes = _parse_scales(scales_str, hi=768) or [448]
    weight_tensor: torch.Tensor | None = None
    if reduce_mode == "weighted":
        raw_weights = os.environ.get("SCDL_MASK_SCALE_WEIGHTS", "")
        weight_tensor = _parse_scale_weights(raw_weights, len(sizes))
        if weight_tensor is None:
            reduce_mode = "mean"
    maps = []
    for sz in sizes:
        maps.append(_user_saliency_single_scale(
            img_np_hw3, model, preprocess, device, out_wh, size=sz, use_amp=amp_enabled))
    sal = _combine_scale_maps(maps, reduce_mode, weight_tensor)
    sal = _normalize01(sal)
    log_info(f"[DINO] Saliency core scales={sizes}")
    return sal.detach().cpu().numpy().astype(np.float32)


# ---- Optional edge-aware refinement (GrabCut). Skipped if OpenCV unavailable. ----
def refine_with_grabcut(img_rgb: np.ndarray, prob: np.ndarray, iters: int | None = None):
    use_gc = int(os.environ.get("SCDL_MASK_USE_GRABCUT", "1")) != 0
    if not use_gc:
        return prob
    try:
        import cv2
    except Exception:
        log_warn("[WARN] OpenCV not found; skipping GrabCut refinement.")
        return prob

    H, W = prob.shape
    lo_q = float(os.environ.get("SCDL_GC_LO_Q", "0.20"))
    hi_q = float(os.environ.get("SCDL_GC_HI_Q", "0.80"))
    lo_q = max(0.0, min(1.0, lo_q))
    hi_q = max(0.0, min(1.0, hi_q))
    if hi_q <= lo_q:
        hi_q = min(0.99, max(lo_q + 0.05, hi_q))
    lo, hi = np.quantile(prob, lo_q), np.quantile(prob, hi_q)

    mask = np.full((H, W), cv2.GC_PR_BGD, np.uint8)
    mask[prob <= lo] = cv2.GC_BGD
    mask[prob >= hi] = cv2.GC_FGD

    border = max(0, int(os.environ.get("SCDL_GC_BORDER", "8")))
    if border > 0:
        mask[:border, :] = cv2.GC_BGD
        mask[-border:, :] = cv2.GC_BGD
        mask[:, :border] = cv2.GC_BGD
        mask[:, -border:] = cv2.GC_BGD

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

    cv2.grabCut(img_rgb, mask, None, bgdModel,
                fgdModel, iters, cv2.GC_INIT_WITH_MASK)
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
    try:
        import cv2
    except Exception:
        log_warn("[WARN] OpenCV not found; skipping watershed refinement.")
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
        import cv2
        prob_norm = cv2.GaussianBlur(
            prob_norm, ksize=(0, 0), sigmaX=blur_sigma)

    lo = np.quantile(prob_norm, ws_lo_q)
    hi = np.quantile(prob_norm, ws_hi_q)
    sure_fg = (prob_norm >= hi)
    sure_bg = (prob_norm <= lo)
    unknown = ~(sure_fg | sure_bg)

    import cv2
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
        log_warn(f"[WARN] Watershed failed ({exc}); skipping refinement.")
        return prob

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
        fail(
            f"[DINO] Missing preview: {PREVIEW_PATH}. Run the Blender preview step first.")

    img = iio.imread(PREVIEW_PATH)
    # Ensure 3-channel RGB
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]
    H, W = img.shape[:2]

    model, preprocess, device = load_dinov3_model(
        REPO_DIR, DINO_HUB_ENTRY, WEIGHT_PATH, device_override=DEVICE_OVERRIDE)
    log_info(f"[DINO] Using architecture '{DINO_ARCH}' (hub entry '{DINO_HUB_ENTRY}').")
    if _use_amp_for_device(device, USE_AMP):
        log_info("[DINO] Mixed precision enabled (torch.autocast).")
    else:
        reason = "device is not CUDA" if not str(device).startswith("cuda") else "SCDL_MASK_AMP=0"
        log_info(f"[DINO] Mixed precision disabled ({reason}).")

    # ------- Build saliency mask (now via USER attention by default) -------
    t_sal = time.perf_counter()
    mask = dense_feature_importance_mask(
        img, model, preprocess, device, (W, H))
    log_info(f"[DINO] Saliency core time: {time.perf_counter() - t_sal:.2f}s")

    # Refinements (edge-aware + watershed)
    t_ref_gc = time.perf_counter()
    mask = refine_with_grabcut(
        img[:, :, :3].astype(np.uint8), mask, iters=None)
    log_info(f"[DINO] GrabCut refinement time: {time.perf_counter() - t_ref_gc:.2f}s")
    t_ref_ws = time.perf_counter()
    mask = refine_with_watershed(img[:, :, :3].astype(np.uint8), mask)
    log_info(f"[DINO] Watershed refinement time: {time.perf_counter() - t_ref_ws:.2f}s")

    # --- Save the raw mask and its visualization for sanity checks FIRST ---
    np.save(MASK_NPY, mask)

    # Create and save visualization from the raw mask
    lo, hi = np.percentile(mask, 2), np.percentile(mask, 98)
    spread = max(hi - lo, 1e-6)
    viz = np.clip((mask - lo) / spread, 0, 1)
    if spread < 0.03:
        grid = (np.indices(mask.shape).sum(0) %
                16 == 0).astype(np.float32) * 0.08
        viz = np.clip(viz * (1.0 - 0.08) + grid, 0, 1)
    iio.imwrite(MASK_PREVIEW, (viz * 255).astype(np.uint8))
    log_info(
        f"[OK] Saved raw mask: {MASK_NPY}  shape={mask.shape}  min={mask.min():.3f} max={mask.max():.3f} mean={mask.mean():.3f}")
    log_info(f"[OK] Saved raw viz : {MASK_PREVIEW}")


    # --- NOW, soften and save the EXR mask for Blender ---
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - hard failure
        fail(f"OpenCV is required for mask post-processing but was not found ({exc}).")

    try:
        # Apply a small Gaussian blur to soften the mask and prevent hard edges
        blur_sigma = float(os.environ.get("SCDL_MASK_BLUR_SIGMA", "1.5"))
        if blur_sigma > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), blur_sigma)
            # Renormalize after blur
            m_min, m_max = mask.min(), mask.max()
            if m_max > m_min:
                mask = (mask - m_min) / (m_max - m_min)
    except Exception as exc:  # pragma: no cover - hard failure
        fail(f"Gaussian blur failed in OpenCV: {exc}")

    # Save the blurred mask as a float EXR file for Blender (fail hard otherwise)
    exr_path = OUT_DIR / "fovea_mask.exr"
    mask_float32 = mask.astype(np.float32)
    def _save_exr_via_openexr() -> bool:
        try:
            import OpenEXR  # type: ignore
            import Imath    # type: ignore
        except Exception:
            return False
        try:
            H, W = mask_float32.shape
            header = OpenEXR.Header(W, H)
            header['channels'] = {
                'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
            }
            exr_file = OpenEXR.OutputFile(str(exr_path), header)
            exr_file.writePixels({'R': mask_float32.tobytes()})
            exr_file.close()
            log_info(f"[OK] Saved float EXR mask for Blender -> {exr_path}")
            return True
        except Exception as e:
            log_error(f"Failed to save EXR via OpenEXR: {e}")
            return False

    def _save_exr_via_opencv() -> bool:
        try:
            import cv2  # type: ignore
        except Exception:
            return False
        try:
            ok = cv2.imwrite(str(exr_path), mask_float32)
            if ok:
                log_info(f"[OK] Saved EXR mask via OpenCV -> {exr_path}")
            else:
                log_warn("[WARN] OpenCV reported failure writing EXR mask.")
            return bool(ok)
        except Exception as e:
            log_warn(f"[WARN] OpenCV could not write EXR ({e})")
            return False

    def _save_exr_via_imageio() -> bool:
        try:
            iio.imwrite(exr_path, mask_float32)
            log_info(f"[OK] Saved EXR mask via imageio -> {exr_path}")
            return True
        except Exception as e:
            log_warn(f"[WARN] imageio could not write EXR ({e})")
            return False

    wrote_exr = (
        _save_exr_via_openexr()
        or _save_exr_via_opencv()
        or _save_exr_via_imageio()
    )
    if not wrote_exr:
        fail(
            "Failed to save fovea mask as EXR. "
            "Ensure OpenEXR/python bindings are installed or enable OpenCV EXR support "
            "(OPENCV_IO_ENABLE_OPENEXR=1)."
        )

    # Optional extra debug outputs
    if int(os.environ.get("SCDL_SAVE_ATTENTION", "0")):
        attn_path = OUT_DIR / "attention_mask.png"
        iio.imwrite(attn_path, (viz * 255).astype(np.uint8))
        log_info(f"[OK] Saved attention map: {attn_path}")
    if int(os.environ.get("SCDL_SAVE_FOVEATED", "0")):
        # Quick foveated preview: hi * mask + lo * (1-mask)
        pil = Image.fromarray(img[:, :, :3].astype(np.uint8))
        Hh, Ww = pil.height, pil.width
        mask_t = torch.from_numpy(mask).float()
        mfull = F.interpolate(mask_t[None, None, ...], size=(
            Hh, Ww), mode="bilinear", align_corners=False)[0, 0]
        lo_w = max(1, Ww // max(1, int(os.environ.get("SCDL_FOV_LORES", "8"))))
        lo_h = max(1, Hh // max(1, int(os.environ.get("SCDL_FOV_LORES", "8"))))
        lo_img = pil.resize((lo_w, lo_h), Image.BILINEAR).resize(
            (Ww, Hh), Image.BILINEAR)
        hi = T.ToTensor()(pil)
        lo = T.ToTensor()(lo_img)
        out = (mfull.unsqueeze(0) * hi +
               (1 - mfull.unsqueeze(0)) * lo).clamp(0, 1)
        fpath = OUT_DIR / "foveated_preview.png"
        iio.imwrite(fpath, (out.mul(255).byte().permute(1, 2, 0).numpy()))
        log_info(f"[OK] Saved foveated preview: {fpath}")

if __name__ == "__main__":
    main()
