#!/usr/bin/env python3
# DINOv3 PCA→RGB (crisper + canonical colors) — hard-coded paths

import os
import math
import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F

# ====== HARD-CODED PATHS ======
# your local clone of facebookresearch/dinov3
REPO_DIR = "dinov3"
WEIGHT_PATH = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
IMG_PATH = "image.png"
OUT_DIR = "out_paper_rgb_canonical"
# ==============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE = 16

# Visualization knobs
SUPER_SAMPLE = 2.0      # 1.0=no supersample; try 2.0 for much crisper output
POST_BLUR_SIGMA = 0.0   # optional mild blur after upsampling (0 disables)
UNSHARP = True
UNSHARP_RADIUS = 1.2
UNSHARP_PERCENT = 120
UNSHARP_THRESHOLD = 0

# -------------------- utils --------------------


def ensure_outdir(p): os.makedirs(p, exist_ok=True)


def pad_to_multiple_hw(img: Image.Image, m: int):
    H, W = img.height, img.width
    padH = (m - H % m) % m
    padW = (m - W % m) % m
    if padH == 0 and padW == 0:
        return img, (0, 0)
    arr = np.array(img)
    arr = np.pad(arr, ((0, padH), (0, 0), (0, 0)), mode="edge")
    arr = np.pad(arr, ((0, 0), (0, padW), (0, 0)), mode="edge")
    return Image.fromarray(arr), (padH, padW)


def to_model_tensor(img: Image.Image, device=DEVICE):
    x = torch.from_numpy(np.array(img)).float() / 255.0  # [H,W,3]
    x = x.permute(2, 0, 1).unsqueeze(0)                    # [1,3,H,W]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.to(device)


def save_image_uint8(path, arr_float01):
    arr = np.clip(arr_float01 * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

# -------------------- model --------------------


def load_model_from_repo(repo_dir, weight_path, device=DEVICE):
    model = torch.hub.load(repo_dir, 'dinov3_vits16',
                           source='local', pretrained=False)
    sd = torch.load(weight_path, map_location='cpu')
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    return model


@torch.no_grad()
def get_lastlayer_norm_patchtokens(model, x, patch_size=PATCH_SIZE):
    B, C, H, W = x.shape
    H_p, W_p = H // patch_size, W // patch_size
    P_expected = H_p * W_p

    if hasattr(model, "forward_features"):
        out = model.forward_features(x)
        if isinstance(out, dict) and "x_norm_patchtokens" in out:
            return out["x_norm_patchtokens"], (H_p, W_p)

    # Fallback: use last layer with norm=True
    assert hasattr(
        model, "get_intermediate_layers"), "Model lacks get_intermediate_layers"
    last = model.get_intermediate_layers(
        x, n=1, return_class_token=True, norm=True)[0]
    tokens = last[0] if isinstance(last, (tuple, list)) else last  # [B,N,D]
    B_, N, D = tokens.shape
    R = max(0, N - 1 - P_expected)
    patches = tokens[:, 1 + R:, :]                    # [B,P,D]
    patches = F.normalize(patches, dim=-1)
    return patches, (H_p, W_p)

# -------------------- PCA → RGB --------------------


def pca_canonical_rgb(feats_hwC: np.ndarray) -> np.ndarray:
    """
    PCA to 3 comps, then a deterministic color canonicalization:
      - sort by explained variance (PC1,PC2,PC3)
      - fix sign by correlating each component with (x,y,1) coordinate basis
        so colors are consistent image-to-image
    """
    H, W, C = feats_hwC.shape
    X = feats_hwC.reshape(-1, C).astype(np.float64)
    X = X - X.mean(0, keepdims=True)  # center

    # PCA via SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)  # X ≈ U S Vt
    PCs = Vt[:3, :]                                   # [3,C]
    proj = (X @ PCs.T).reshape(H, W, 3)               # [H,W,3]

    # Canonicalize sign using correlation with coords
    yy, xx = np.mgrid[0:H, 0:W]
    coords = [
        (xx - xx.mean()) / (xx.std() + 1e-8),
        (yy - yy.mean()) / (yy.std() + 1e-8),
        np.ones_like(xx)                              # bias direction
    ]
    proj_canon = proj.copy()
    for c in range(3):
        comp = proj[..., c]
        # pick the coordinate that has the largest abs correlation
        corrs = [np.corrcoef(comp.ravel(), z.ravel())[0, 1] for z in coords]
        sign = 1.0 if corrs[np.argmax(np.abs(corrs))] >= 0 else -1.0
        proj_canon[..., c] = comp * sign

    # Normalize each channel to [0,1] with robust percentiles
    out = np.zeros_like(proj_canon, dtype=np.float64)
    for c in range(3):
        lo, hi = np.percentile(proj_canon[..., c], 1), np.percentile(
            proj_canon[..., c], 99)
        if hi <= lo:
            hi = lo + 1e-6
        out[..., c] = np.clip((proj_canon[..., c] - lo) / (hi - lo), 0, 1)
    return out.astype(np.float32)

# -------------------- main --------------------


def main():
    ensure_outdir(OUT_DIR)

    # Load original and (optionally) super-sample before feeding the model
    img_orig = Image.open(IMG_PATH).convert("RGB")
    if SUPER_SAMPLE != 1.0:
        Hs = int(round(img_orig.height * SUPER_SAMPLE))
        Ws = int(round(img_orig.width * SUPER_SAMPLE))
        img_for_model = img_orig.resize((Ws, Hs), Image.BICUBIC)
    else:
        img_for_model = img_orig

    # pad to multiple of 16 to avoid dropping last row/col of patches
    img_pad, (padH, padW) = pad_to_multiple_hw(img_for_model, PATCH_SIZE)

    x = to_model_tensor(img_pad, DEVICE)

    print(f"Input for model: {img_for_model.width}x{img_for_model.height} "
          f"(padded to {img_pad.width}x{img_pad.height}), SUPER_SAMPLE={SUPER_SAMPLE}x")

    # Model + tokens
    model = load_model_from_repo(REPO_DIR, WEIGHT_PATH, DEVICE)
    print("Model loaded.")
    with torch.no_grad():
        pts, (H_p, W_p) = get_lastlayer_norm_patchtokens(model, x)

    # Build dense feature grid
    fmap = pts.reshape(1, H_p, W_p, -1)[0].cpu().numpy()  # [H_p,W_p,C]
    # Crop padding in patch space
    if padH or padW:
        H_p0 = (img_pad.height - padH) // PATCH_SIZE
        W_p0 = (img_pad.width - padW) // PATCH_SIZE
        fmap = fmap[:H_p0, :W_p0, :]

    # PCA→RGB with canonical colors
    rgb_patches = pca_canonical_rgb(fmap)  # [Hp0, Wp0, 3] in [0,1]
    save_image_uint8(os.path.join(OUT_DIR, "pca_rgb_patches.png"), rgb_patches)

    # Upsample to the (possibly super-sampled) image size (bicubic)
    up = torch.from_numpy(rgb_patches).permute(
        2, 0, 1).unsqueeze(0)  # [1,3,Hp0,Wp0]
    up = F.interpolate(up, size=(img_for_model.height, img_for_model.width),
                       mode="bicubic", align_corners=False)
    up = up.squeeze(0).permute(1, 2, 0).numpy()
    up = np.clip(up, 0, 1)
    up_img = Image.fromarray((up * 255).astype(np.uint8))

    # Optional slight blur to remove residual patchiness after bicubic
    if POST_BLUR_SIGMA > 0:
        up_img = up_img.filter(
            ImageFilter.GaussianBlur(radius=POST_BLUR_SIGMA))

    # If we super-sampled, Lanczos down to original resolution for extra crispness
    if SUPER_SAMPLE != 1.0:
        up_img = up_img.resize(
            (img_orig.width, img_orig.height), Image.LANCZOS)

    # Optional light unsharp mask for micro-contrast
    if UNSHARP:
        up_img = up_img.filter(ImageFilter.UnsharpMask(
            radius=UNSHARP_RADIUS, percent=UNSHARP_PERCENT, threshold=UNSHARP_THRESHOLD
        ))

    out_up = os.path.join(OUT_DIR, "pca_rgb_final.png")
    up_img.save(out_up)

    # Also save raw tokens for reuse
    torch.save({
        "patch_features_lastlayer_norm": pts.cpu(),  # [1,P,D]
        "grid": (H_p, W_p), "patch_size": PATCH_SIZE
    }, os.path.join(OUT_DIR, "dense_features_lastlayer.pt"))

    print("Saved:")
    print(" ", os.path.join(OUT_DIR, "pca_rgb_patches.png"))
    print(" ", out_up)
    print(" ", os.path.join(OUT_DIR, "dense_features_lastlayer.pt"))


if __name__ == "__main__":
    main()
