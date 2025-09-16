
#!/usr/bin/env python3
# step2_dino_mask.py
# Purpose: Run under WSL to compute a USER_IMPORTANCE mask from out/preview.png
# using DINOv3. Saves out/user_importance.npy and visualization PNGs.

import os
import sys
from pathlib import Path
import numpy as np
import imageio.v3 as iio
import torch
import torch.nn.functional as F
from torchvision import transforms

from logging_utils import get_logger
from scdl_config import env_path, get_pipeline_paths


# =========== CONFIG (edit) ===========
PATHS = get_pipeline_paths(Path(__file__).resolve().parent)
PROJECT_DIR = PATHS.project_dir
OUT_DIR = PATHS.out_dir
PREVIEW_PATH = PATHS.preview

# DINOv3 (local clone + weights)
REPO_DIR = env_path("SCDL_DINO_REPO", PROJECT_DIR / "dinov3", base=PROJECT_DIR)
WEIGHT_PATH = env_path("SCDL_DINO_WEIGHTS",
                      PROJECT_DIR / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
                      base=PROJECT_DIR)

# Outputs
MASK_NPY = PATHS.mask_npy
MASK_PREVIEW = PATHS.mask_preview
ROI_BBOX_TXT = PATHS.roi_bbox
LOGGER = get_logger("scdl_step2", PATHS.log_file)


def log_info(msg: str):
    LOGGER.info(msg)


def log_warn(msg: str):
    LOGGER.warning(msg)


def fail(msg: str, code: int = 1):
    LOGGER.error(msg)
    sys.exit(code)
# =====================================


def _make_preprocess():
    # torchvision Resize may not support antialias kwarg in older versions
    try:
        resize = transforms.Resize((224, 224), antialias=True)
    except TypeError:
        resize = transforms.Resize((224, 224))
    return transforms.Compose([
        transforms.ToTensor(),
        resize,
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


def load_dinov3_vits16(repo_dir: Path, weight_path: Path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = _make_preprocess()
    try:
        sys.path.insert(0, str(repo_dir))
        from dinov3.models.vision_transformer import vit_small  # type: ignore
    except Exception as e:
        fail(f"[DINO] Failed to import official dinov3 from {repo_dir}: {e}")
    try:
        model = vit_small(patch_size=16, init_values=1.0, block_chunks=0)
        sd = torch.load(str(weight_path), map_location="cpu")
        state = sd.get("state_dict", sd.get("model", sd))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            log_warn(f"[DINO] Warning: unexpected keys ignored: {list(unexpected)[:8]}")
        if missing:
            log_warn(f"[DINO] Warning: missing keys: {list(missing)[:8]}")
        model.eval().to(device)
    except Exception as e:
        fail(f"[DINO] Failed to load model/weights: {weight_path}: {e}")
    return model, preprocess, device


@torch.no_grad()
def dense_feature_energy_mask(img_np_hw3: np.ndarray, model, preprocess, device, out_wh):
    x = preprocess(img_np_hw3).unsqueeze(0).to(device)  # [1,3,224,224]
    feats = model.forward_features(x)
    # Expect keys from dinov3 VisionTransformer
    if not isinstance(feats, dict) or ('x_norm_patchtokens' not in feats and 'x' not in feats):
        raise RuntimeError(
            "Unexpected DINOv3 forward_features output; missing patch tokens")
    if 'x_norm_patchtokens' in feats:
        patch = feats['x_norm_patchtokens']  # [B, P, D]
        cls = feats.get('x_norm_clstoken', None)       # [B, D]
    else:
        patch = feats['x']
        cls = None
    B, P, D = patch.shape
    side = int(P ** 0.5)
    assert side * side == P, f"Patch tokens not square: P={P}"

    # Normalize tokens
    patch_n = F.normalize(patch, dim=-1)
    grid = patch_n.reshape(B, side, side, D)
    ex = torch.linalg.vector_norm(
        grid[:, :, 1:, :] - grid[:, :, :-1, :], dim=-1)  # [B,H,W-1]
    ey = torch.linalg.vector_norm(
        grid[:, 1:, :, :] - grid[:, :-1, :, :], dim=-1)  # [B,H-1,W]
    e = torch.zeros(B, side, side, device=grid.device, dtype=grid.dtype)
    e[:, :, 1:] += ex
    e[:, 1:, :] += ey

    if cls is not None:
        cls_n = F.normalize(cls, dim=-1).unsqueeze(2)  # [B,D,1]
        sal = torch.matmul(patch_n, cls_n).squeeze(2)  # [B,P]
        sal = sal.reshape(B, side, side)
        sal = (sal - sal.amin(dim=(1, 2), keepdim=True)) / (sal.amax(dim=(1,
                                                                          2), keepdim=True) - sal.amin(dim=(1, 2), keepdim=True) + 1e-6)
    else:
        sal = torch.zeros_like(e)

    w_edge = float(os.environ.get("SCDL_MASK_EDGE_WEIGHT", "0.5"))
    w_edge = max(0.0, min(1.0, w_edge))
    m = (w_edge * e + (1.0 - w_edge) * sal)[:, None, :, :]  # [B,1,H,W]
    out_w, out_h = out_wh
    m = F.interpolate(m, size=(out_h, out_w), mode="bicubic",
                      align_corners=False).squeeze()
    m = F.avg_pool2d(m[None, None, ...], kernel_size=5,
                     stride=1, padding=2).squeeze()
    lo, hi = float(m.min()), float(m.max())
    if hi > lo:
        m = (m - lo) / (hi - lo)
    m = torch.pow(m, 0.7)  # gentle emphasis
    return m.detach().cpu().numpy().astype(np.float32)  # HxW float32 in [0,1]


def main():
    if not PREVIEW_PATH.exists():
        fail(f"[DINO] Missing preview: {PREVIEW_PATH}. Run the Blender preview step first.")
    if not REPO_DIR.exists():
        fail(f"[DINO] Missing DINO repo: {REPO_DIR}")
    if not WEIGHT_PATH.exists():
        fail(f"[DINO] Missing DINO weights: {WEIGHT_PATH}")

    img = iio.imread(PREVIEW_PATH)
    # Ensure 3-channel RGB
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3:
        c = img.shape[2]
        if c == 4:
            img = img[:, :, :3]
        elif c > 4:
            img = img[:, :, :3]
        elif c == 1:
            img = np.concatenate([img, img, img], axis=-1)
    H, W = img.shape[0], img.shape[1]

    model, preprocess, device = load_dinov3_vits16(REPO_DIR, WEIGHT_PATH)
    mask = dense_feature_energy_mask(img, model, preprocess, device, (W, H))
    np.save(MASK_NPY, mask)

    # Robust contrast for preview visualization only
    import numpy as _np
    lo, hi = _np.percentile(mask, 2), _np.percentile(mask, 98)
    viz = (mask - lo) / max(1e-6, (hi - lo))
    viz = _np.clip(viz, 0, 1)
    iio.imwrite(MASK_PREVIEW, (viz * 255).astype(_np.uint8))
    log_info(f"[OK] Saved mask: {MASK_NPY}  shape={mask.shape}  min={mask.min():.3f} max={mask.max():.3f} mean={mask.mean():.3f}")
    log_info(f"[OK] Saved viz : {MASK_PREVIEW}")

    # Overlay preview for sanity: red-tinted mask overlay
    try:
        p8 = img.astype(_np.uint8)
        if p8.ndim == 2:
            p8 = _np.stack([p8, p8, p8], axis=-1)
        if p8.shape[2] > 3:
            p8 = p8[:, :, :3]
        a = _np.clip(mask, 0, 1).astype(_np.float32)
        a = (a - a.min()) / (a.max() - a.min() + 1e-6)
        a = a * 0.6  # 60% max tint
        overlay = p8.astype(_np.float32)
        overlay[:, :, 0] = overlay[:, :, 0] * (1 - a) + 255 * a
        overlay = _np.clip(overlay, 0, 255).astype(_np.uint8)
        overlay_path = OUT_DIR / 'mask_overlay.png'
        iio.imwrite(overlay_path, overlay)
        log_info(f"[OK] Saved overlay: {overlay_path}")
    except Exception as _e:
        log_warn(f"[WARN] Could not save mask overlay: {_e}")

    # Save a normalized ROI bbox based on area budget (quantile threshold)
    area = float(os.environ.get("SCDL_ROI_AREA", "0.20"))
    area = max(0.01, min(0.95, area))
    thr = float(_np.quantile(mask.astype(_np.float32), 1.0 - area))
    ys, xs = _np.where(mask >= thr)
    if xs.size > 0:
        min_x, max_x = int(xs.min()), int(xs.max())
        min_y, max_y = int(ys.min()), int(ys.max())
        nx0 = max(0.0, min_x / W)
        nx1 = min(1.0, (max_x + 1) / W)
        ny0 = max(0.0, min_y / H)
        ny1 = min(1.0, (max_y + 1) / H)
        ROI_BBOX_TXT.write_text(f"{nx0} {nx1} {ny0} {ny1}\n")
        log_info(
            f"[OK] ROI bbox saved → {ROI_BBOX_TXT}  area≈{(nx1-nx0)*(ny1-ny0):.1%} thr={thr:.3f}")
    else:
        log_warn("[WARN] Mask produced empty ROI; skipping bbox save.")


if __name__ == "__main__":
    main()
