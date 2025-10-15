#!/usr/bin/env python3
"""
dinov3_mask.py
Role: DINO mask generator.
- Runs under WSL/Linux (preferred) or, when invoked by Windows Blender,
  automatically re‑executes inside WSL so it uses your WSL Python with Torch.

Note: The Windows Blender launcher mode has been removed. Use run_windows_blender_wsl.sh
to start Windows Blender; this script will be called by Blender at the DINO step.
"""

# ======================= CONFIG (edit these) =======================
import sys
import os
import subprocess
import platform
import re
from typing import Tuple

# Resolve project root from this script's directory (cross‑platform)
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

# your local clone of facebookresearch/dinov3
REPO_DIR = os.environ.get("SCDL_DINO_REPO", os.path.join(PROJ_DIR, "dinov3"))
WEIGHT_PATH = os.environ.get("SCDL_DINO_WEIGHTS", os.path.join(PROJ_DIR, "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"))
IMG_PATH = os.environ.get("SCDL_PREVIEW", os.path.join(PROJ_DIR, "out", "preview.png"))  # preview image rendered by Blender
OUT_DIR = os.environ.get("SCDL_OUT_DIR", os.path.join(PROJ_DIR, "out"))                    # where mask.npy and mask.png will be saved

AREA_BUDGET = 0.20   # fraction of pixels to keep as ROI (0.05–0.35 typical)
FEATHER = 8      # pixels: combined dilation+blur radius; 0=off
# ================================================================


def ensure_dir(p): os.makedirs(p, exist_ok=True)


def _load_image(path, device, patch=16):
    # Local imports to avoid requiring Torch/PIL in Windows launcher mode
    import torch
    import numpy as np
    from PIL import Image, ImageOps
    if not os.path.isfile(path):
        raise FileNotFoundError(f"IMG_PATH not found: {path}")
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    ow, oh = img.size
    pad_w = (patch - ow % patch) % patch
    pad_h = (patch - oh % patch) % patch
    if pad_w or pad_h:
        img = ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill=0)
    nw, nh = img.size

    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    t = (t - mean) / std
    return t.to(device), (oh, ow, nh, nw, pad_h, pad_w)


def _load_dinov3(repo_dir, weight_path, device):
    import torch
    if not os.path.isdir(repo_dir):
        raise FileNotFoundError(f"REPO_DIR not found: {repo_dir}")
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"WEIGHT_PATH not found: {weight_path}")
    sys.path.insert(0, repo_dir)
    from dinov3.models.vision_transformer import vit_small
    model = vit_small(patch_size=16, init_values=1.0, block_chunks=0)
    sd = torch.load(weight_path, map_location="cpu")
    state = sd.get("state_dict", sd.get("model", sd))
    # classifier heads may be missing; that's fine
    model.load_state_dict(state, strict=False)
    model.eval().to(device)

    def forward_tokens(x: torch.Tensor) -> torch.Tensor:
        # Return token sequence [B,T,D]
        if hasattr(model, "get_intermediate_layers"):
            return model.get_intermediate_layers(x, n=1)[0]
        if hasattr(model, "forward_features"):
            out = model.forward_features(x)
            if isinstance(out, dict):
                for k in ("x_norm_patchtokens", "tokens", "last_hidden_state"):
                    if k in out:
                        return out[k]
            return out
        y = model(x)
        if isinstance(y, (list, tuple)):
            y = y[0]
        if hasattr(y, "last_hidden_state"):
            return y.last_hidden_state
        return y
    return model, forward_tokens


# -------------------- Windows/WSL helpers --------------------
def _wslpath_to_windows(p: str) -> str:
    out = subprocess.check_output(["wslpath", "-w", p], text=True).strip()
    return out


def _windows_to_wsl_path(win_path: str) -> str:
    """Convert a Windows path to a WSL Linux path.
    Handles normal drive paths and UNC WSL paths like \\wsl.localhost\\<Distro>\\path\\to\\file.
    """
    if not win_path:
        return win_path
    p = win_path.replace("\\", "/")
    # UNC → Linux: //wsl.localhost/<Distro>/path or //wsl$/<Distro>/path
    m = re.match(r"^//wsl(?:\.localhost)?/([^/]+)/(.+)$", p, flags=re.IGNORECASE)
    if m:
        distro, rest = m.group(1), m.group(2)
        rest = rest.lstrip("/")
        return "/" + rest
    # Fallback to wslpath for drive letter and other cases
    try:
        out = subprocess.check_output(["wsl.exe", "wslpath", "-u", win_path], text=True).strip()
        if out:
            return out
    except Exception:
        pass
    # Last resort: if it already looks like a Linux path, return it
    if p.startswith("/"):
        return p
    raise RuntimeError(f"Unable to convert Windows path to WSL: {win_path}")


def _bounce_to_wsl_if_needed() -> None:
    """
    If we are running under Windows Python (e.g., called from Windows Blender),
    re-exec inside WSL so we use the WSL Python environment with Torch/weights.
    """
    if platform.system() != "Windows":
        return
    if os.environ.get("SCDL_BOUNCED") == "1":
        return
    # Choose WSL Python
    wsl_py = os.environ.get("SCDL_WSL_PYTHON", "/home/rubin/anaconda3/bin/python")
    # Resolve this script's Linux path
    this_win = os.path.abspath(__file__)
    try:
        this_linux = _windows_to_wsl_path(this_win)
    except Exception as e:
        print(f"[DINO] Failed to map path to WSL via wslpath: {e}", file=sys.stderr)
        sys.exit(1)

    # Convert key env paths to Linux for the WSL side
    env = dict(os.environ)
    env["SCDL_BOUNCED"] = "1"
    for key in ("SCDL_PREVIEW", "SCDL_OUT_DIR", "SCDL_DINO_REPO", "SCDL_DINO_WEIGHTS"):
        val = env.get(key)
        if val:
            try:
                env[key] = _windows_to_wsl_path(val)
            except Exception:
                pass

    cmd = ["wsl.exe", "-e", wsl_py, this_linux] + sys.argv[1:]
    print("[DINO] Re-exec in WSL:", " ".join(cmd))
    rc = subprocess.call(cmd, env=env)
    sys.exit(rc)


# (Launcher mode removed; use run_windows_blender_wsl.sh instead.)


def _dino_main():
    import torch
    import numpy as np
    from PIL import Image, ImageFilter
    ensure_dir(OUT_DIR)
    if not os.path.isfile(IMG_PATH):
        raise FileNotFoundError(
            f"Preview not found: {IMG_PATH}. Run from Blender (script_blender.py) after it renders the preview, or launch the pipeline via '--launch-blender'."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, (oh, ow, nh, nw, ph, pw) = _load_image(IMG_PATH, device)
    model, fwd_tok = _load_dinov3(REPO_DIR, WEIGHT_PATH, device)

    with torch.no_grad():
        toks = fwd_tok(x).float()  # [B,T,D]
    B, T, D = toks.shape
    gh, gw = nh // 16, nw // 16
    P = gh * gw

    # Heuristic: last P tokens are patch tokens; the rest (front) are special (CLS+registers)
    patch_feats = toks[:, T - P: T, :].squeeze(0)  # [P,D]
    special = toks[:, : T - P, :].squeeze(0)    # [S,D]
    cls = special[0:1, :] if special.numel(
    ) else patch_feats.mean(dim=0, keepdim=True)

    # Normalize
    patch_feats = torch.nn.functional.normalize(patch_feats, dim=1)
    cls = torch.nn.functional.normalize(cls, dim=1)

    # CLS saliency (patch ↔ CLS cosine), gridified
    sal = (patch_feats @ cls.T).squeeze(1).cpu().numpy().reshape(gh, gw)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)

    # Feature-edge energy (captures thin details/texture)
    F = patch_feats.cpu().numpy().reshape(gh, gw, D).astype(np.float32)
    dx = F[:, 1:, :] - F[:, :-1, :]
    dx = np.pad(dx, ((0, 0), (0, 1), (0, 0)), mode='edge')
    dy = F[1:, :, :] - F[:-1, :, :]
    dy = np.pad(dy, ((0, 1), (0, 0), (0, 0)), mode='edge')
    edge = np.sqrt((dx**2).sum(-1)) + np.sqrt((dy**2).sum(-1))
    edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)

    # Hybrid score (0.5/0.5)
    score_small = np.clip(0.5 * sal + 0.5 * edge, 0.0, 1.0)

    # Upsample to padded size → crop back to original
    from PIL import Image as _PIL_Image
    score = _PIL_Image.fromarray((score_small * 255).astype(np.uint8))
    score = score.resize((nw, nh), resample=_PIL_Image.BILINEAR)
    score = np.asarray(score).astype(np.float32) / 255.0
    if pw or ph:
        score = score[:oh, :ow]

    # Area budget → soft mask
    area = float(np.clip(AREA_BUDGET, 0.01, 0.95))
    thr = float(np.quantile(score, 1.0 - area))
    mask_soft = np.clip((score - thr) / max(1e-6, 1.0 - thr), 0.0, 1.0)

    # Feather (dilate then blur)
    feather = int(max(0, FEATHER))
    m_img = Image.fromarray((mask_soft * 255).astype(np.uint8), mode="L")
    if feather > 0:
        k = max(3, (feather | 1))  # odd kernel
        m_img = m_img.filter(ImageFilter.MaxFilter(size=k))
        m_img = m_img.filter(ImageFilter.GaussianBlur(radius=feather * 0.5))
    mask = np.asarray(m_img).astype(np.float32) / 255.0

    np.save(os.path.join(OUT_DIR, "mask.npy"), mask.astype(np.float32))
    Image.fromarray((mask * 255).astype(np.uint8)
                    ).save(os.path.join(OUT_DIR, "mask.png"))

    print(
        f"[OK] mask.npy  shape={mask.shape}  area≈{(mask >= 0.5).mean():.2%}")
    print("[OK] mask.png  saved")


if __name__ == "__main__":
    # If executed under Windows Python, re-run inside WSL Python
    _bounce_to_wsl_if_needed()
    # Run DINO locally (WSL/Linux)
    _dino_main()
