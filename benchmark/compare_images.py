#!/usr/bin/env python3
"""
Quick image comparer for SSIM, PSNR, and LPIPS.

Set IMG_REF and IMG_TEST to the two images you want to compare.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

# ==========
# Configure these two paths
# ==========
IMG_REF = Path("/home/rubin/uni/test/scdl3991_foveated_rendering/out_full_render/final.png")
IMG_TEST = Path("/home/rubin/uni/test/scdl3991_foveated_rendering/out/final.png")
OUT_DIR = Path(__file__).resolve().parent.parent / "compare_out"
OUT_MD = OUT_DIR / "compare_results.md"


def load_image(path: Path) -> np.ndarray:
    """Load an image as float32 in [0,1], strips alpha, expands grayscale."""

    if not path.exists():
        raise FileNotFoundError(path)

    with Image.open(path) as im:
        arr = np.array(im)

    is_16bit = arr.dtype == np.uint16
    arr = arr.astype(np.float32)

    if arr.ndim == 2:  # grayscale
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:  # drop alpha
        arr = arr[..., :3]

    return arr / 65535.0 if is_16bit else arr / 255.0


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0.0:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

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

    if model is None:
        device = torch.device("cpu")
        model = lpips.LPIPS(net="vgg").to(device).eval()
    else:
        device = next(model.parameters()).device

    a_t = a_t.to(device)
    b_t = b_t.to(device)
    with torch.no_grad():
        dist = model(a_t, b_t)
    return float(dist.item())


def compare_images(ref_path: Path, test_path: Path) -> Tuple[float, float, float]:
    ref = load_image(ref_path)
    test = load_image(test_path)
    if ref.shape != test.shape:
        raise ValueError(f"Shape mismatch: ref {ref.shape}, test {test.shape}")

    lp = lpips_distance(ref, test)
    ss = ssim_score(ref, test)
    ps = psnr(ref, test)
    return lp, ss, ps


def main() -> None:
    ref_path = IMG_REF
    test_path = IMG_TEST

    print(f"[Compare] Reference: {ref_path}")
    print(f"[Compare] Candidate: {test_path}")

    lp, ss, ps = compare_images(ref_path, test_path)

    print("\nMetrics:")
    print(f"- LPIPS: {lp:.6f} (lower is better)")
    print(f"- SSIM : {ss:.6f} (higher is better)")
    print(f"- PSNR : {ps:.2f} dB (higher is better)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = [
        "| Metric | Value | Note |",
        "|--------|-------|------|",
        f"| LPIPS | {lp:.6f} | lower is better |",
        f"| SSIM | {ss:.6f} | higher is better |",
        f"| PSNR | {ps:.2f} dB | higher is better |",
    ]
    print("\nMarkdown table:\n")
    print("\n".join(table))
    OUT_MD.write_text("\n".join(table), encoding="utf-8")
    print(f"\nWrote markdown table to {OUT_MD}")


if __name__ == "__main__":
    main()
