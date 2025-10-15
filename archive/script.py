import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

# ---------- CONFIG ----------
REPO_DIR = "dinov3"  # <-- change to your local clone
WEIGHT_PATH = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"  # <-- your .pth
IMG_PATH = "COCO_test2014_000000004520.jpg"  # <-- your image
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE = 16  # ViT-S/16
LOW_RES_FACTOR = 8  # foveation strength (bigger => blurrier periphery)
GAMMA = 1.0       # mask contrast (raise >1.0 to sharpen attention mask)

# ---------- UTILS ----------
to_tensor = T.ToTensor()
norm = T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225])

def load_model(repo_dir, weight_path, device=DEVICE):
    # torch.hub returns a model whose forward() gives pooled features.
    model = torch.hub.load(repo_dir, 'dinov3_vits16', source='local', pretrained=False)
    sd = torch.load(weight_path, map_location='cpu')
    # Some checkpoints store weights under "model"
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    return model

@torch.no_grad()
def get_patch_tokens_and_cls(model, x, patch_size=PATCH_SIZE):
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

    # Fallback: use get_intermediate_layers (returns list; take last).
    if tokens_seq is None:
        if hasattr(model, "get_intermediate_layers"):
            tokens_seq = model.get_intermediate_layers(x, n=1, return_class_token=True, norm=True)[0]  # [B, N, D]
        else:
            raise RuntimeError(
                "Cannot obtain patch tokens. Need model.forward_features (dict) or model.get_intermediate_layers."
            )

    # Split CLS / registers / patches from the sequence
    B, N, D = tokens_seq.shape
    R = max(0, N - 1 - P_expected)  # infer #register tokens
    cls = tokens_seq[:, 0, :]
    patch_tokens = tokens_seq[:, 1 + R:, :]  # [B, P_expected, D]
    if patch_tokens.shape[1] != P_expected:
        raise RuntimeError(
            f"Patch count mismatch: got {patch_tokens.shape[1]}, expected {P_expected}. "
            "Check patch size or how tokens are returned."
        )
    return patch_tokens, cls, H_p, W_p, R

@torch.no_grad()
def dense_feature_map(model, img_tensor_norm):
    """
    Returns:
      fmap: [H_p, W_p, D] dense features (last-layer patch tokens)
      cls:  [D] or None
      (H_p, W_p)
    """
    patch_tokens, cls, H_p, W_p, _ = get_patch_tokens_and_cls(model, img_tensor_norm)
    # reshape to a 2D grid for convenience
    fmap = patch_tokens.reshape(1, H_p, W_p, -1)[0].cpu()   # [H_p, W_p, D]
    cls = cls[0].cpu() if cls is not None else None
    return fmap, cls, (H_p, W_p)

@torch.no_grad()
def make_attention_map_from_cls_similarity(patch_tokens, cls_token, H_p, W_p):
    """
    patch_tokens: [B, P, D], cls_token: [B, D] or None
    Returns attn (0..1) of shape [H_p, W_p]
    """
    # Normalize for cosine similarity
    patches_n = F.normalize(patch_tokens, dim=-1)  # [B, P, D]
    if cls_token is None:
        # Fallback: use mean of patch tokens as pseudo-CLS
        cls_n = F.normalize(patches_n.mean(dim=1), dim=-1)  # [B, D]
    else:
        cls_n = F.normalize(cls_token, dim=-1)

    # cosine sim to CLS: [B, P]
    scores = torch.einsum("bpd,bd->bp", patches_n, cls_n)
    attn = scores.reshape(-1, H_p, W_p)[0]

    # normalize to [0,1]
    a_min, a_max = attn.min(), attn.max()
    attn = (attn - a_min) / (a_max - a_min + 1e-8)
    return attn.cpu()

def foveate_with_mask(image_pil, attn_mask_hw, low_res_factor=LOW_RES_FACTOR, gamma=GAMMA):
    """
    image_pil: original RGB PIL
    attn_mask_hw: torch.Tensor [H_p, W_p] in [0,1]
    Returns a PIL image (foveated)
    """
    H, W = image_pil.height, image_pil.width

    # Resize mask to full image, optionally add contrast (gamma)
    attn_full = F.interpolate(
        attn_mask_hw.unsqueeze(0).unsqueeze(0), size=(H, W),
        mode="bilinear", align_corners=False
    )[0, 0].clamp(0, 1)
    if gamma != 1.0:
        attn_full = attn_full.pow(gamma)

    # Low-res version
    lo_w = max(1, W // low_res_factor)
    lo_h = max(1, H // low_res_factor)
    low = image_pil.resize((lo_w, lo_h), Image.BILINEAR).resize((W, H), Image.BILINEAR)

    # Blend: high-res * attn + low-res * (1 - attn)
    hi = to_tensor(image_pil)
    lo = to_tensor(low)
    mask = attn_full.unsqueeze(0)  # [1,H,W]
    out = mask * hi + (1 - mask) * lo
    return T.ToPILImage()(out.clamp(0, 1))

# ---------- MAIN ----------
if __name__ == "__main__":
    # Load image (keep a copy unnormalized for rendering)
    image = Image.open(IMG_PATH).convert("RGB")
    img_for_model = norm(to_tensor(image)).unsqueeze(0).to(DEVICE)

    print("Input tensor shape:", img_for_model.shape)

    # Load model & extract dense features
    model = load_model(REPO_DIR, WEIGHT_PATH, DEVICE)
    print("Model loaded")

    fmap, cls_vec, (H_p, W_p) = dense_feature_map(model, img_for_model)
    print(f"Dense feature map: {tuple(fmap.shape)}  (H_p={H_p}, W_p={W_p}, D={fmap.shape[-1]})")

    # Also keep the raw patch tokens for attention computation
    with torch.no_grad():
        patch_tokens, cls_token, H_p, W_p, _ = get_patch_tokens_and_cls(model, img_for_model)

    # Attention map via CLS–patch cosine similarity
    attn_hw = make_attention_map_from_cls_similarity(patch_tokens, cls_token, H_p, W_p)
    print(f"Attention map (patch grid): {tuple(attn_hw.shape)}")

    # Save dense feature map to disk (optional)
    os.makedirs("out", exist_ok=True)
    np.save("out/dense_feature_map.npy", fmap.numpy())
    Image.fromarray((attn_hw.numpy() * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR).save("out/attention_mask.png")

    # Foveated rendering
    foveated = foveate_with_mask(image, attn_hw, low_res_factor=LOW_RES_FACTOR, gamma=GAMMA)
    foveated.save("out/foveated.png")
    print("Saved: out/dense_feature_map.npy, out/attention_mask.png, out/foveated.png")
