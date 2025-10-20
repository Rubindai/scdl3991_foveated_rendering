#!/usr/bin/env python3
# step2_dino_mask.py — First-run optimized DINOv3 saliency (offline, no fallbacks, OpenEXR writer)
#
# Outputs (matches Step 3 expectations):
#   out/user_importance.npy
#   out/user_importance_mask.exr
# Optional:
#   out/user_importance_preview.png
#
# Preview toggle (env):
#   SCDL_USER_IMPORTANCE_PREVIEW=0|1        # default 0 (off)
#   SCDL_USER_IMPORTANCE_PREVIEW_MODE=color|gray   # default color
#   SCDL_USER_IMPORTANCE_PREVIEW_ALPHA=0.6  # only for color overlay
#
# Other env:
#   SCDL_DINO_LOCAL_DIR    # default models/dinov3-vitl16  (local HF folder; offline)
#   SCDL_PREVIEW_PATH      # default out/preview.png
#   SCDL_OUT_DIR           # default out
#   SCDL_DINO_SIZE         # default 336 (try 336/384 for crisper, still one pass)
#   SCDL_PERC_LO/H I       # default 0.60 / 0.995 (contrast stretch)
#   SCDL_MASK_GAMMA        # default 0.70  (gamma <1 broadens highlights; >1 sharpens)
#   SCDL_MORPH_K           # default 3     (small GPU clean-up)
from __future__ import annotations

import os, math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend  # force Flash SDPA :contentReference[oaicite:4]{index=4}

import numpy as np
from torchvision.io import read_file, decode_image
import kornia as K
import kornia.morphology as KM

try:
    from transformers import AutoModel, AutoImageProcessor
except Exception as e:
    raise RuntimeError("[Step2/DINOv3] transformers is required and must be recent enough for dtype=...") from e

# EXR writer (no OpenCV)
try:
    import OpenEXR, Imath
except Exception as e:
    raise RuntimeError("[Step2/DINOv3] OpenEXR Python module is required. Install 'OpenEXR' and 'Imath'.") from e


def die(msg: str) -> None:
    raise RuntimeError(f"[Step2/DINOv3] {msg}")


def require_cuda_ampere() -> None:
    if not torch.cuda.is_available():
        die("CUDA is not available.")
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        die(f"Requires NVIDIA Ampere or newer (found compute capability {major}.{minor}).")
    if not torch.cuda.is_bf16_supported():
        die("BF16 not supported by this PyTorch/CUDA build or GPU.")
    # Enable TF32 GEMMs (Ampere): official control :contentReference[oaicite:5]{index=5}
    torch.set_float32_matmul_precision("high")


def assert_flash_sdpa_available() -> None:
    # Minimal runtime probe to ensure Flash SDPA backend is usable (single tiny attention call).
    B, H, T, D = 1, 4, 256, 64
    q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            o = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        if not torch.isfinite(o).all():
            die("Flash SDPA probe produced non-finite output.")
    except Exception as e:
        die(f"Flash SDPA backend is not available/usable: {e}")


def load_image_tensor(path: Path) -> torch.Tensor:
    if not path.exists():
        die(f"Preview image not found: {path}")
    data = read_file(str(path))
    img = decode_image(data)  # [C,H,W], uint8
    if img.ndim != 3 or img.shape[0] not in (1, 3, 4):
        die(f"Unsupported image shape from decode: {list(img.shape)}")
    if img.shape[0] == 4:  # drop alpha
        img = img[:3]
    return img.unsqueeze(0)  # [1,C,H,W]


def to_numpy_hwc_uint8(img_bchw: torch.Tensor) -> np.ndarray:
    return img_bchw.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()


def save_exr_single_channel(path: Path, array_f32_hw: np.ndarray, channel_name: str = "Y") -> None:
    if array_f32_hw.ndim != 2:
        die(f"EXR writer expects HxW float array; got shape {array_f32_hw.shape}")
    H, W = array_f32_hw.shape
    header = OpenEXR.Header(W, H)
    pix_type = Imath.PixelType(Imath.PixelType.FLOAT)
    header["channels"] = {channel_name: Imath.Channel(pix_type)}
    # fastest: uncompressed (keep default). If you want smaller files, you may set DWAA/DWAB.
    out = OpenEXR.OutputFile(str(path), header)
    try:
        out.writePixels({channel_name: array_f32_hw.astype(np.float32).tobytes()})
    finally:
        out.close()


def cls_patch_cosine_saliency(last_hidden: torch.Tensor, num_register_tokens: int) -> torch.Tensor:
    # last_hidden: [B, 1 + R + P, D] → returns [B, P] cosine similarity between CLS and patch tokens.
    cls = last_hidden[:, 0:1, :]                            # [B,1,D]
    patches = last_hidden[:, 1 + num_register_tokens :, :]  # [B,P,D]  (skip DINOv3 register tokens)
    cls_n = F.normalize(cls, dim=-1)
    patches_n = F.normalize(patches, dim=-1)
    return torch.einsum("bid,bjd->bij", cls_n, patches_n).squeeze(1)  # [B,P]


def main():
    device = "cuda"
    dtype = torch.bfloat16  # Ampere supports BF16

    # Paths / env
    LOCAL_DIR    = os.getenv("SCDL_DINO_LOCAL_DIR", "models/dinov3-vitl16").strip()
    PREVIEW_PATH = Path(os.getenv("SCDL_PREVIEW_PATH", "out/preview.png"))
    OUT_DIR      = Path(os.getenv("SCDL_OUT_DIR", "out"))
    MASK_NPY     = OUT_DIR / "user_importance.npy"
    MASK_EXR     = OUT_DIR / "user_importance_mask.exr"
    PREVIEW_PNG  = OUT_DIR / "user_importance_preview.png"

    MAKE_PREVIEW = int(os.getenv("SCDL_USER_IMPORTANCE_PREVIEW", "1"))
    PREV_MODE    = os.getenv("SCDL_USER_IMPORTANCE_PREVIEW_MODE", "gray").lower()  # color | gray
    PREV_ALPHA   = float(os.getenv("SCDL_USER_IMPORTANCE_PREVIEW_ALPHA", "0.6"))

    if PREV_MODE not in ("color", "gray"):
        die("SCDL_USER_IMPORTANCE_PREVIEW_MODE must be 'color' or 'gray'.")

    if not LOCAL_DIR:
        die("SCDL_DINO_LOCAL_DIR must point to a local HF-format DINOv3 directory (offline).")
    LOCAL_DIR = str(Path(LOCAL_DIR).resolve())

    # Checks
    require_cuda_ampere()
    assert_flash_sdpa_available()

    # Load model / processor strictly offline; use dtype= (modern API)
    try:
        processor = AutoImageProcessor.from_pretrained(LOCAL_DIR, local_files_only=True)
        model     = AutoModel.from_pretrained(LOCAL_DIR, dtype=dtype, local_files_only=True)
    except TypeError as te:
        die(f"Transformers here doesn't support dtype=... in from_pretrained(): {te}")
    except Exception as e:
        die(f"Failed to load DINOv3 locally at {LOCAL_DIR}: {e}")

    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("sdpa")
    model = model.to(device).to(memory_format=torch.channels_last).eval()

    # Read preview and preprocess
    img_bchw = load_image_tensor(PREVIEW_PATH)      # [1,C,H,W], uint8
    H, W = int(img_bchw.shape[-2]), int(img_bchw.shape[-1])
    img_np = to_numpy_hwc_uint8(img_bchw)           # HWC uint8

    # (still single forward) optionally bump input size for crisper grid
    resize = int(os.getenv("SCDL_DINO_SIZE", "336"))
    inputs = processor(images=img_np, return_tensors="pt", do_resize=True, size=resize)
    pixel_values = inputs["pixel_values"].to(device, dtype=dtype, non_blocking=True)
    pixel_values = pixel_values.contiguous(memory_format=torch.channels_last)

    # Forward with forced Flash SDPA + BF16 autocast
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        with torch.amp.autocast(device_type="cuda", dtype=dtype), torch.inference_mode():
            outputs = model(pixel_values=pixel_values)
    last_hidden = outputs.last_hidden_state  # [B, 1+R+P, D]

    # CLS–patch cosine saliency
    num_register_tokens = int(getattr(model.config, "num_register_tokens", 0))
    sal = cls_patch_cosine_saliency(last_hidden, num_register_tokens=num_register_tokens)  # [B,P]
    B, P = sal.shape
    if B != 1:
        die(f"Unexpected batch size in output: {B}")

    # Patch grid from config patch size
    patch = int(getattr(model.config, "patch_size", 16))
    in_h, in_w = int(pixel_values.shape[-2]), int(pixel_values.shape[-1])
    gh, gw = in_h // patch, in_w // patch
    if gh * gw != P:
        g = int(round(math.sqrt(P)))
        if g * g != P:
            die(f"Cannot infer patch grid: P={P}, in_h={in_h}, in_w={in_w}, patch={patch}")
        gh = gw = g

    sal_grid = sal.reshape(1, 1, gh, gw)
    sal_full = F.interpolate(sal_grid, size=(H, W), mode="bilinear", align_corners=False)  # [1,1,H,W]
    sal_full = (sal_full.clamp(-1, 1) + 1.0) * 0.5  # [-1,1] -> [0,1]

    # Contrast stretch + gamma to expand dynamic range (GPU)
    def contrast_stretch(x, lo=0.60, hi=0.995, eps=1e-6):
        # Compute quantiles in FP32 on-GPU on a small downsample for speed,
       # then stretch the full-res map.
        xf  = x.float()
        Hq, Wq = int(xf.shape[-2]), int(xf.shape[-1])
        if min(Hq, Wq) >= 128:
            xs = F.avg_pool2d(xf, kernel_size=4, stride=4)
        else:
            xs = xf
        qlo = torch.quantile(xs, lo)
        qhi = torch.quantile(xs, hi)
        y   = (xf - qlo) / (qhi - qlo + eps)
        return y.clamp(0, 1).to(x.dtype)

    lo = float(os.getenv("SCDL_PERC_LO", "0.60"))
    hi = float(os.getenv("SCDL_PERC_HI", "0.995"))
    gamma = float(os.getenv("SCDL_MASK_GAMMA", "0.70"))

    sal_full = contrast_stretch(sal_full, lo, hi)
    sal_full = torch.pow(sal_full, gamma)

    # Light cleanup (GPU)
    k = int(os.getenv("SCDL_MORPH_K", "3"))
    k = max(1, int(k))
    kernel = torch.ones(k, k, device=device, dtype=sal_full.dtype)
    sal_proc = KM.closing(sal_full, kernel)
    sal_proc = K.filters.gaussian_blur2d(sal_proc, (k, k), (0.5 * k, 0.5 * k))
    sal_proc = sal_proc.clamp(0, 1).squeeze(0).squeeze(0)  # [H,W]

    # Save data outputs (consumed by Step 3)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mask_np = sal_proc.detach().float().cpu().numpy()
    np.save(str(MASK_NPY), mask_np)
    save_exr_single_channel(MASK_EXR, mask_np, channel_name="Y")

    # Optional Preview (based on the saved NPY)
    if MAKE_PREVIEW == 1:
        try:
            import cv2
        except Exception as e:
            die("OpenCV (cv2) is required for user_importance_preview; install opencv-python.")
        mask_from_npy = np.load(str(MASK_NPY))
        m8 = np.clip(mask_from_npy * 255.0, 0, 255).astype(np.uint8)

        if PREV_MODE == "color":
            # Colorized heatmap OVERLAYED on the preview (viridis colormap) :contentReference[oaicite:6]{index=6}
            prev_rgb = img_bchw.squeeze(0).permute(1,2,0).contiguous().cpu().numpy()
            if prev_rgb.dtype != np.uint8:
                prev_rgb = prev_rgb.astype(np.uint8)
            heat_bgr = cv2.applyColorMap(m8, cv2.COLORMAP_VIRIDIS)
            prev_bgr = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2BGR)
            over = cv2.addWeighted(prev_bgr, 1.0, heat_bgr, float(PREV_ALPHA), 0.0)
            if not cv2.imwrite(str(PREVIEW_PNG), over):
                die(f"Failed to write preview PNG at {PREVIEW_PNG}")
        else:
            # Plain grayscale preview of the mask (no overlay)
            if not cv2.imwrite(str(PREVIEW_PNG), m8):
                die(f"Failed to write preview PNG at {PREVIEW_PNG}")

    print(f"[Step2/DINOv3] Wrote: {MASK_NPY.name}, {MASK_EXR.name}" + (", user_importance_preview.png" if MAKE_PREVIEW==1 else "") + f" in {OUT_DIR}")


if __name__ == "__main__":
    main()
