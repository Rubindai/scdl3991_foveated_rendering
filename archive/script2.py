import os
import math
from typing import Tuple, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T

# ============ USER CONFIG ============
REPO_DIR = "dinov3"  # <-- change: your local clone of facebookresearch/dinov3
WEIGHT_PATH = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"  # <-- change: the .pth file
IMG_PATH = "COCO_test2014_000000004520.jpg"  # <-- change: your input image
OUT_DIR = "out"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE = 16  # ViT-S/16
# Graph limits (for huge images). We solve graph on at most ~N nodes by
# average-pooling the feature grid first, then upsample mask back.
MAX_GRAPH_PATCHES = 12000  # ~12k nodes keeps memory + runtime reasonable
KNN_K = 10                 # neighbors per node in sparse graph
FAISS_TRY = True           # if faiss is installed, we'll use it for fast kNN
POWER_ITERS = 25           # spectral power iterations
LOW_RES_FACTOR = 8         # foveation strength (bigger => blurrier periphery)
MASK_GAMMA = 1.2           # contrast of final mask (>1 sharpens)
SMOOTH_KERNEL = 5          # average pooling size for mask smoothing (0/3/5/7...)
# ====================================

to_tensor = T.ToTensor()
norm = T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225])

def pad_to_multiple_hw(image: Image.Image, multiple: int) -> Tuple[Image.Image, Tuple[int,int]]:
    H, W = image.height, image.width
    H_pad = (multiple - H % multiple) % multiple
    W_pad = (multiple - W % multiple) % multiple
    if H_pad == 0 and W_pad == 0:
        return image, (0, 0)
    # pad on the bottom/right by replication
    arr = np.array(image)
    pad_h = np.pad(arr, ((0, H_pad), (0, 0), (0, 0)), mode="edge")
    pad_hw = np.pad(pad_h, ((0, 0), (0, W_pad), (0, 0)), mode="edge")
    return Image.fromarray(pad_hw), (H_pad, W_pad)

def load_model(repo_dir, weight_path, device=DEVICE):
    model = torch.hub.load(repo_dir, 'dinov3_vits16', source='local', pretrained=False)
    sd = torch.load(weight_path, map_location='cpu')
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    return model

@torch.no_grad()
def get_patch_tokens_and_cls(model, x, patch_size=PATCH_SIZE):
    """
    Returns:
      patch_tokens: [1, P, D]  (last-layer patch tokens, normalized if available)
      cls_token:    [1, D] or None
      H_p, W_p:     patch grid size
      R:            inferred #register tokens
    """
    B, C, H, W = x.shape
    H_p, W_p = H // patch_size, W // patch_size
    P_expected = H_p * W_p

    tokens_seq = None
    cls = None

    if hasattr(model, "forward_features"):
        out = model.forward_features(x)
        if isinstance(out, dict):
            # Common dinov2/3 naming:
            if "x_norm_patchtokens" in out:
                patch_tokens = out["x_norm_patchtokens"]  # [B, P, D]
                cls = out.get("x_norm_clstoken", None)    # [B, D] or None
                return patch_tokens, cls, H_p, W_p, 0
            if "tokens" in out and isinstance(out["tokens"], torch.Tensor) and out["tokens"].ndim == 3:
                tokens_seq = out["tokens"]  # [B, N, D]

    if tokens_seq is None:
        if hasattr(model, "get_intermediate_layers"):
            tokens_seq = model.get_intermediate_layers(x, n=1, return_class_token=True, norm=True)[0]  # [B, N, D]
        else:
            raise RuntimeError("Cannot obtain patch tokens. Need model.forward_features(dict) or get_intermediate_layers().")

    B, N, D = tokens_seq.shape
    R = max(0, N - 1 - P_expected)  # registers
    cls = tokens_seq[:, 0, :]
    patch_tokens = tokens_seq[:, 1 + R:, :]
    if patch_tokens.shape[1] != P_expected:
        raise RuntimeError(f"Patch count mismatch: got {patch_tokens.shape[1]}, expected {P_expected}.")
    return patch_tokens, cls, H_p, W_p, R

def grid_pool_features(fmap_hw_d: torch.Tensor, max_nodes: int) -> Tuple[torch.Tensor, Tuple[int,int], Tuple[int,int]]:
    """
    Downsample a [H_p, W_p, D] grid to <= max_nodes by average pooling.
    Returns:
      feats_ds: [N', D] (row-major flattened)
      (H_p, W_p): original grid
      (H_ds, W_ds): downsampled grid
    """
    H_p, W_p, D = fmap_hw_d.shape
    P = H_p * W_p
    if P <= max_nodes:
        return fmap_hw_d.reshape(-1, D), (H_p, W_p), (H_p, W_p)

    # Choose pooling strides to hit target nodes
    scale = math.sqrt(P / max_nodes)
    s_h = max(1, int(round(scale)))
    s_w = max(1, int(round(scale)))

    # Use avg_pool2d on channels-first
    x = fmap_hw_d.permute(2, 0, 1).unsqueeze(0)  # [1, D, H_p, W_p]
    pooled = F.avg_pool2d(x, kernel_size=(s_h, s_w), stride=(s_h, s_w), ceil_mode=True)  # [1, D, H_ds, W_ds]
    _, D, H_ds, W_ds = pooled.shape
    feats_ds = pooled.squeeze(0).permute(1, 2, 0).reshape(-1, D)  # [N', D]
    return feats_ds, (H_p, W_p), (H_ds, W_ds)

def try_build_faiss_index(feats_unit: np.ndarray):
    try:
        if not FAISS_TRY:
            return None
        import faiss  # type: ignore
        d = feats_unit.shape[1]
        index = faiss.IndexFlatIP(d)  # cosine if vectors unit-normalized
        index.add(feats_unit.astype(np.float32))
        return index
    except Exception:
        return None

def knn_torch_topk(feats_unit: torch.Tensor, k: int, row_chunk: int = 2048, col_chunk: int = 4096):
    """
    Compute for each row its top-k neighbors (cosine sims) using chunked matmul.
    Returns (idx: [N, k], sim: [N, k]); each row excludes self index if present.
    """
    N, D = feats_unit.shape
    device = feats_unit.device
    topk_vals = torch.full((N, k), -1.0, device=device)
    topk_idx = torch.full((N, k), -1, dtype=torch.long, device=device)

    for r0 in range(0, N, row_chunk):
        r1 = min(N, r0 + row_chunk)
        rows = feats_unit[r0:r1]  # [R,D]
        vals = torch.full((r1 - r0, k), -1.0, device=device)
        idxs = torch.full((r1 - r0, k), -1, dtype=torch.long, device=device)

        for c0 in range(0, N, col_chunk):
            c1 = min(N, c0 + col_chunk)
            cols = feats_unit[c0:c1]  # [C,D]
            sims = rows @ cols.t()    # [R,C]
            # Mask self when chunk overlaps diagonal
            if c0 <= r0 < c1:
                diag = torch.arange(r0, r1, device=device)
                sims[diag - r0, diag - c0] = -1.0

            # get top-k in this block
            block_vals, block_idx = torch.topk(sims, k=min(k, c1 - c0), dim=1)
            # merge with current best
            merged_vals = torch.cat([vals, block_vals], dim=1)
            merged_idx = torch.cat([idxs, block_idx + c0], dim=1)
            new_vals, new_pos = torch.topk(merged_vals, k=k, dim=1)
            new_idx = torch.gather(merged_idx, 1, new_pos)
            vals, idxs = new_vals, new_idx

        topk_vals[r0:r1] = vals
        topk_idx[r0:r1] = idxs

    return topk_idx, topk_vals

def build_sparse_adj_from_knn(idx: torch.Tensor, val: torch.Tensor, symmetric: bool = True):
    """
    Build a (symmetric) sparse adjacency from kNN indices/values.
    idx, val: [N, k]
    Returns A (torch.sparse_coo_tensor) with shape [N,N]
    """
    N, k = idx.shape
    rows = torch.arange(N, device=idx.device).unsqueeze(1).expand(N, k).reshape(-1)
    cols = idx.reshape(-1)
    vals = val.reshape(-1).clamp(min=0.0)  # non-negative weights

    # remove any -1 placeholders
    valid = cols >= 0
    rows, cols, vals = rows[valid], cols[valid], vals[valid]

    if symmetric:
        rows = torch.cat([rows, cols], dim=0)
        cols = torch.cat([cols, rows[:len(cols)]], dim=0)  # careful: use pre-concat rows
        vals = torch.cat([vals, vals], dim=0)

    A = torch.sparse_coo_tensor(
        torch.stack([rows, cols], dim=0), vals,
        size=(N, N), device=idx.device
    ).coalesce()
    return A

def normalize_adj_symmetric(A: torch.Tensor):
    """
    A: sparse adjacency (N,N), non-negative
    Return: \tilde{A} = D^{-1/2} A D^{-1/2} (symmetric normalization)
    """
    N = A.size(0)
    deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1e-12)  # [N]
    inv_sqrt = deg.pow(-0.5)
    # Multiply: inv_sqrt[i] * A_ij * inv_sqrt[j]
    rows, cols = A.indices()
    vals = A.values() * inv_sqrt[rows] * inv_sqrt[cols]
    return torch.sparse_coo_tensor(torch.stack([rows, cols], 0), vals, (N, N), device=A.device).coalesce()

@torch.no_grad()
def spectral_objectness_from_features(fmap_hw_d: torch.Tensor,
                                      max_nodes: int = MAX_GRAPH_PATCHES,
                                      k: int = KNN_K,
                                      power_iters: int = POWER_ITERS,
                                      use_faiss: bool = FAISS_TRY) -> torch.Tensor:
    """
    fmap_hw_d: [H_p, W_p, D] (float32 CPU or GPU)
    Returns sal_hw: [H_p, W_p] in [0,1] (on CPU)
    """
    dev = fmap_hw_d.device
    feats_flat, (H_p, W_p), (H_ds, W_ds) = grid_pool_features(fmap_hw_d, max_nodes=max_nodes)
    # unit-normalize
    feats = F.normalize(feats_flat, dim=1)

    # kNN (cosine)
    N = feats.size(0)
    if use_faiss:
        index = try_build_faiss_index(feats.cpu().numpy().astype(np.float32))
    else:
        index = None

    if index is not None:
        # Query self with top-(k+1) to discard self
        import faiss  # type: ignore
        sims, ids = index.search(feats.cpu().numpy().astype(np.float32), k + 1)  # [N,k+1]
        # drop self (assume first is self)
        ids = ids[:, 1:]
        sims = sims[:, 1:]
        idx = torch.from_numpy(ids).to(dev)
        val = torch.from_numpy(sims).to(dev)
    else:
        idx, val = knn_torch_topk(feats.to(dev), k=k)

    A = build_sparse_adj_from_knn(idx, val, symmetric=True)
    A = A.coalesce()

    # Symmetric normalization
    A_hat = normalize_adj_symmetric(A)  # sparse

    # Power iteration on A_hat to get dominant eigenvector
    v = torch.rand((N,), device=dev)
    v = v / (v.norm() + 1e-12)
    for _ in range(power_iters):
        v = torch.sparse.mm(A_hat, v.unsqueeze(1)).squeeze(1)
        nrm = v.norm() + 1e-12
        v = v / nrm

    # Normalize to [0,1]
    v = (v - v.min()) / (v.max() - v.min() + 1e-8)
    sal_ds = v.reshape(H_ds, W_ds)

    # Heuristic orientation: ensure center > border (flip if needed)
    border = torch.cat([
        sal_ds[0, :], sal_ds[-1, :], sal_ds[:, 0], sal_ds[:, -1]
    ])
    center = sal_ds[H_ds//4:3*H_ds//4, W_ds//4:3*W_ds//4]
    if border.mean() > center.mean():
        sal_ds = 1.0 - sal_ds

    # Smooth (optional)
    if SMOOTH_KERNEL and SMOOTH_KERNEL > 1:
        k = SMOOTH_KERNEL
        sal_ds = F.avg_pool2d(sal_ds.unsqueeze(0).unsqueeze(0), k, stride=1, padding=k//2)[0, 0]

    # Upsample back to original patch grid size
    sal_full = F.interpolate(
        sal_ds.unsqueeze(0).unsqueeze(0),
        size=(H_p, W_p),
        mode="bilinear",
        align_corners=False
    )[0, 0]

    # Final [0,1], contrast
    sal_full = sal_full.clamp(0, 1)
    if MASK_GAMMA != 1.0:
        sal_full = sal_full.pow(MASK_GAMMA)

    return sal_full.cpu()

def foveate_with_mask(image_pil: Image.Image, attn_mask_hw: torch.Tensor,
                      low_res_factor: int = LOW_RES_FACTOR, gamma: float = 1.0) -> Image.Image:
    H, W = image_pil.height, image_pil.width
    attn_full = F.interpolate(attn_mask_hw.unsqueeze(0).unsqueeze(0),
                              size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    if gamma != 1.0:
        attn_full = attn_full.clamp(0, 1).pow(gamma)

    hi = to_tensor(image_pil)
    lo = image_pil.resize((max(1, W // low_res_factor), max(1, H // low_res_factor)), Image.BILINEAR)\
                   .resize((W, H), Image.BILINEAR)
    lo = to_tensor(lo)
    mask = attn_full.unsqueeze(0)
    out = (mask * hi + (1 - mask) * lo).clamp(0, 1)
    return T.ToPILImage()(out)

@torch.no_grad()
def extract_dense_features(model, image_pil: Image.Image, device=DEVICE) -> Tuple[torch.Tensor, Tuple[int,int], Tuple[int,int]]:
    """
    Returns:
      fmap_hw_d: [H_p, W_p, D] (float32, on device)
      (H_p, W_p): patch grid
      (padH, padW): padding added
    """
    im_padded, (padH, padW) = pad_to_multiple_hw(image_pil, PATCH_SIZE)
    x = norm(to_tensor(im_padded)).unsqueeze(0).to(device)
    patch_tokens, cls, H_p, W_p, _ = get_patch_tokens_and_cls(model, x)
    fmap = patch_tokens.reshape(1, H_p, W_p, -1)[0]  # [H_p, W_p, D]
    return fmap, (H_p, W_p), (padH, padW)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load
    img_raw = Image.open(IMG_PATH).convert("RGB")
    print(f"Input image: {IMG_PATH}  size={img_raw.width}x{img_raw.height}")

    model = load_model(REPO_DIR, WEIGHT_PATH, DEVICE)
    print("Model loaded.")

    # Dense features
    fmap_hw_d, (H_p, W_p), (padH, padW) = extract_dense_features(model, img_raw, DEVICE)
    D = fmap_hw_d.shape[-1]
    print(f"Dense feature map: {(H_p, W_p, D)}  (patch size = {PATCH_SIZE})")

    # Feature-only objectness via sparse spectral graph
    print("Building k-NN graph + spectral objectness ...")
    sal_hw = spectral_objectness_from_features(fmap_hw_d, MAX_GRAPH_PATCHES, KNN_K, POWER_ITERS, FAISS_TRY)
    print("Objectness mask computed.")

    # Save mask (cropped if we padded)
    sal_np = sal_hw.numpy()
    if padH or padW:
        sal_np = sal_np[:H_p - padH // PATCH_SIZE, :W_p - padW // PATCH_SIZE]
    H_img, W_img = img_raw.height, img_raw.width
    mask_img = Image.fromarray((F.interpolate(torch.from_numpy(sal_np).unsqueeze(0).unsqueeze(0),
                                size=(H_img, W_img), mode="bilinear", align_corners=False)[0,0]
                                .clamp(0,1).numpy() * 255).astype(np.uint8))
    mask_path = os.path.join(OUT_DIR, "feature_objectness_mask.png")
    mask_img.save(mask_path)

    # Foveated rendering
    foveated = foveate_with_mask(img_raw, torch.from_numpy(sal_np).float(), low_res_factor=LOW_RES_FACTOR, gamma=1.0)
    foveated_path = os.path.join(OUT_DIR, "foveated_from_features.png")
    foveated.save(foveated_path)

    # Also save raw dense features
    np.save(os.path.join(OUT_DIR, "dense_feature_map.npy"), fmap_hw_d.cpu().numpy())

    print(f"Saved:\n  - {mask_path}\n  - {foveated_path}\n  - {os.path.join(OUT_DIR, 'dense_feature_map.npy')}")

if __name__ == "__main__":
    main()
