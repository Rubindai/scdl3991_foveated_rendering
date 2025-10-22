# step3_singlepass_foveated_v9.py
# Blender 4.5.2 — Single-pass material foveation, GPU-only, Principled v2-safe.
# Changes vs v8:
#   (1) Use ImageTexture Alpha (or Color fallback) for the mask factor.
#   (2) Enable Cycles filter_glossy and sample_clamp_indirect to stabilize denoising.
#   (3) Keep all v8 robustness (Principled v2 names, GPU-only, prohibited materials).

import bpy, os, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
OUT.mkdir(parents=True, exist_ok=True)

MASK_EXR = OUT / "user_importance_mask.exr"
NPY_PATH = OUT / "user_importance.npy"
FINAL = OUT / "final.png"

def log(msg): print(f"[Step3/Foveated] {msg}")

# ---------------- Render ----------------
def configure_cycles(scene):
    r = scene.render
    c = scene.cycles
    r.engine = 'CYCLES'

    device = os.environ.get("SCDL_CYCLES_DEVICE", "OPTIX").upper()
    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences
    use_gpu = False

    if device == "OPTIX":
        cprefs.compute_device_type = 'OPTIX'
        cprefs.get_devices()
        for d in cprefs.devices:
            d.use = (getattr(d, "type", "") == 'OPTIX')
            use_gpu |= d.use
        c.device = 'GPU'
        c.denoiser = 'OPTIX'
    elif device == "CUDA":
        cprefs.compute_device_type = 'CUDA'
        cprefs.get_devices()
        for d in cprefs.devices:
            d.use = (getattr(d, "type", "") == 'CUDA')
            use_gpu |= d.use
        c.device = 'GPU'
        # Prefer OptiX denoiser if available, else OIDN
        c.denoiser = 'OPTIX' if any(getattr(d, "type", "") == 'OPTIX' for d in cprefs.devices) else 'OPENIMAGEDENOISE'
    else:
        raise RuntimeError("SCDL_CYCLES_DEVICE must be OPTIX or CUDA; refusing CPU fallback.")

    if not use_gpu:
        raise RuntimeError("No compatible GPU device found; refusing CPU fallback.")

    # Sampling / perf
    c.samples = int(os.environ.get("SCDL_SAMPLES", "192"))
    c.use_adaptive_sampling = True
    c.adaptive_sampling_threshold = 0.01
    c.use_fast_gi = True
    c.max_bounces = 8
    c.transparent_max_bounces = 8

    # >>> v9: stabilize glossy energy before denoising (Blender manual recommendations)
    # Filter Glossy reduces noisy sharp glossy paths; Clamp Indirect limits rare bright samples (fireflies).
    c.filter_glossy = 0.8          # docs suggest ~1.0 as a good start; 0.8 preserves a bit more crispness. :contentReference[oaicite:6]{index=6}
    c.sample_clamp_indirect = 10.0 # clamp fireflies on indirect bounces only. :contentReference[oaicite:7]{index=7}
    # <<<

    r.image_settings.file_format = 'PNG'
    r.filepath = str(FINAL)

# -------------- Data ----------------
def thresholds(np_path: Path):
    arr = np.load(np_path).astype(np.float32)
    flat = arr[np.isfinite(arr)].ravel()
    if flat.size == 0:
        return 0.25, 0.75, 2.2, 0.0

    q20, q80 = np.quantile(flat, [0.2, 0.8])
    if not np.isfinite(q20): q20 = 0.0
    if not np.isfinite(q80): q80 = 0.0
    if q80 <= q20 + 1e-6:
        q20 = 0.0
        q80 = float(max(1e-3, np.quantile(flat, 0.6)))

    cov = float(np.mean(flat > q80))
    gamma = float(np.clip(2.0 + (0.9 - cov) * 2.0, 2.0, 4.0))
    return float(q20), float(q80), float(gamma), float(cov)

def load_mask_image(mask_path: Path):
    img = bpy.data.images.load(str(mask_path), check_existing=True)
    # Mask is data (not color) → mark Non-Color to avoid color-space transforms. :contentReference[oaicite:8]{index=8}
    img.colorspace_settings.name = 'Non-Color'
    img.alpha_mode = 'CHANNEL_PACKED'
    img.use_half_precision = True
    return img

# -------------- Node Group --------------
def ensure_foveation_group():
    name = "FoveationMix"
    if name in bpy.data.node_groups:
        return bpy.data.node_groups[name]
    g = bpy.data.node_groups.new(name=name, type='ShaderNodeTree')
    iface = g.interface
    # Principled v2 naming validated by manual (IOR Level / Coat). :contentReference[oaicite:9]{index=9}
    iface.new_socket(name="HQ Shader", in_out='INPUT',  socket_type='NodeSocketShader', description="Full-quality shader")
    iface.new_socket(name="LQ Shader", in_out='INPUT',  socket_type='NodeSocketShader', description="Simplified shader")
    iface.new_socket(name="Mask",      in_out='INPUT',  socket_type='NodeSocketFloat',  description="Foveation mask 0..1")
    iface.new_socket(name="LoThr",     in_out='INPUT',  socket_type='NodeSocketFloat',  description="Lower threshold")
    iface.new_socket(name="HiThr",     in_out='INPUT',  socket_type='NodeSocketFloat',  description="Upper threshold")
    iface.new_socket(name="Gamma",     in_out='INPUT',  socket_type='NodeSocketFloat',  description="Gamma shaping")
    iface.new_socket(name="Shader",    in_out='OUTPUT', socket_type='NodeSocketShader', description="Output")

    n_in  = g.nodes.new("NodeGroupInput")
    n_out = g.nodes.new("NodeGroupOutput")
    mapr  = g.nodes.new("ShaderNodeMapRange"); mapr.clamp = True; mapr.data_type = 'FLOAT'
    power = g.nodes.new("ShaderNodeMath"); power.operation = 'POWER'
    mix   = g.nodes.new("ShaderNodeMixShader")

    g.links.new(n_in.outputs["Mask"],   mapr.inputs["Value"])
    g.links.new(n_in.outputs["LoThr"],  mapr.inputs["From Min"])
    g.links.new(n_in.outputs["HiThr"],  mapr.inputs["From Max"])
    g.links.new(mapr.outputs["Result"], power.inputs[0])
    g.links.new(n_in.outputs["Gamma"],  power.inputs[1])

    g.links.new(n_in.outputs["LQ Shader"], mix.inputs[1])  # Fac=0 → LQ
    g.links.new(n_in.outputs["HQ Shader"], mix.inputs[2])  # Fac=1 → HQ
    g.links.new(power.outputs[0],          mix.inputs["Fac"])
    g.links.new(mix.outputs["Shader"],     n_out.inputs["Shader"])
    return g

# -------------- Safe socket helpers --------------
def _norm(s: str) -> str:
    return re.sub(r'\s+', '', s).lower()

def get_input(node: bpy.types.Node, *names):
    if not node: return None
    for nm in names:
        s = node.inputs.get(nm)
        if s: return s
    by_norm = {_norm(sock.name): sock for sock in node.inputs}
    for nm in names:
        s = by_norm.get(_norm(nm))
        if s: return s
    return None

def set_input_value(node: bpy.types.Node, names, value):
    s = get_input(node, *names)
    if s and hasattr(s, "default_value"):
        try: s.default_value = value
        except: pass

def link_or_copy(nt: bpy.types.NodeTree, src_node: bpy.types.Node, src_names, dst_node: bpy.types.Node, dst_name):
    s = get_input(src_node, *src_names)
    d = get_input(dst_node, dst_name)
    if not d: return
    if s and getattr(s, "is_linked", False) and s.links:
        nt.links.new(s.links[0].from_socket, d)
    elif s and hasattr(s, "default_value"):
        try: d.default_value = s.default_value
        except: pass

# -------------- Material helpers --------------
PROHIBITED_TYPES = {"EMISSION", "BSDF_GLASS", "BSDF_TRANSPARENT", "BSDF_REFRACTION", "VOL_ABSORPTION", "VOL_SCATTER"}

def material_is_prohibited(mat: bpy.types.Material) -> bool:
    nt = mat.node_tree
    if not nt: return False
    for n in nt.nodes:
        if n.type in PROHIBITED_TYPES:
            return True
        if n.type == 'BSDF_PRINCIPLED':
            t = get_input(n, "Transmission", "Transmission Weight")
            if t and (t.is_linked or (hasattr(t, "default_value") and float(t.default_value) > 0.1)):
                return True
    return False

def find_principled_bfs(nt: bpy.types.NodeTree):
    out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output), None)
    if not out or not out.inputs.get("Surface") or not out.inputs["Surface"].links:
        return None
    start = [out.inputs["Surface"].links[0].from_node]
    seen = set()
    steps = 0
    while start and steps < 64:
        steps += 1
        node = start.pop(0)
        ptr = getattr(node, "as_pointer", lambda: id(node))()
        if ptr in seen: 
            continue
        seen.add(ptr)
        if node.type == 'BSDF_PRINCIPLED':
            return node
        for inp in getattr(node, "inputs", []):
            for l in getattr(inp, "links", []):
                if hasattr(l, "from_node"):
                    start.append(l.from_node)
    return None

def build_lq_principled(nt: bpy.types.NodeTree, ref_p: bpy.types.Node):
    lq = nt.nodes.new("ShaderNodeBsdfPrincipled")
    lq.label = "LQ_Principled"

    # Base defaults for cheap shading
    set_input_value(lq, ["Roughness"], 0.6)
    set_input_value(lq, ["Specular", "Specular IOR Level", "IOR Level"], 0.2)

    # Mirror selected inputs if available
    link_or_copy(nt, ref_p, ["Base Color"], lq, "Base Color")
    link_or_copy(nt, ref_p, ["Roughness"],  lq, "Roughness")
    link_or_copy(nt, ref_p, ["Normal"],     lq, "Normal")

    # Metallic attenuated
    src_met = get_input(ref_p, "Metallic")
    if src_met and not src_met.is_linked:
        try: set_input_value(lq, ["Metallic"], max(0.0, min(1.0, float(src_met.default_value) * 0.25)))
        except: pass
    else:
        set_input_value(lq, ["Metallic"], 0.1)

    # Coat/Clearcoat attenuated if present (v2 and v1 names)
    set_input_value(lq, ["Coat Weight", "Clearcoat", "Coat"], 0.15)
    set_input_value(lq, ["Coat Roughness", "Clearcoat Roughness"], 0.5)

    # Disable expensive lobes when present
    for nm in ("Transmission", "Subsurface", "Emission Strength", "Anisotropic", "Sheen", "Specular Tint"):
        set_input_value(lq, [nm], 0.0)

    return lq

def inject_group_into_material(mat: bpy.types.Material, mask_img: bpy.types.Image,
                               group: bpy.types.NodeTree, lo, hi, gamma) -> bool:
    if not mat.use_nodes or not mat.node_tree:
        return False
    if material_is_prohibited(mat):
        return False

    nt = mat.node_tree
    out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output), None)
    if not out or not out.inputs.get("Surface") or not out.inputs["Surface"].links:
        return False

    orig_link = out.inputs["Surface"].links[0]
    orig_socket = orig_link.from_socket

    # Screen-space sampler: Window coords → Image Texture (mask as Non-Color data)
    tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = mask_img; tex.label = "FoveaMask"
    tex.interpolation = 'Linear'
    tex.extension = 'CLIP'
    tex.projection = 'FLAT'
    tex.image.colorspace_settings.name = 'Non-Color'  # data, not color. :contentReference[oaicite:10]{index=10}
    tc = nt.nodes.new("ShaderNodeTexCoord"); tc.label = "Coords"
    nt.links.new(tc.outputs["Window"], tex.inputs["Vector"])  # Window = 0..1 screen coords. :contentReference[oaicite:11]{index=11}

    # Group instance + LQ shader
    gi = nt.nodes.new("ShaderNodeGroup"); gi.node_tree = group; gi.label = "FoveationMix"
    ref_p = find_principled_bfs(nt)
    lq = build_lq_principled(nt, ref_p)

    # >>> v9: drive Mask with Alpha when available (scalar), else Color fallback. :contentReference[oaicite:12]{index=12}
    mask_out = tex.outputs.get("Alpha") or tex.outputs.get("Color")
    # <<<

    # Wire
    nt.links.new(orig_socket,        gi.inputs["HQ Shader"])
    nt.links.new(lq.outputs["BSDF"], gi.inputs["LQ Shader"])
    nt.links.new(mask_out,           gi.inputs["Mask"])
    gi.inputs["LoThr"].default_value = float(lo)
    gi.inputs["HiThr"].default_value = float(hi)
    gi.inputs["Gamma"].default_value = float(gamma)

    # Replace original surface link with foveation mix
    nt.links.remove(orig_link)
    nt.links.new(gi.outputs["Shader"], out.inputs["Surface"])
    return True

# -------------- Main --------------
def main():
    scene = bpy.context.scene
    lo, hi, gamma, cov = thresholds(NPY_PATH)
    log(f"Mask thresholds: lo={lo:.4f}, hi={hi:.4f}, gamma={gamma:.2f}, cov≈{cov:.3f}")

    configure_cycles(scene)
    if not MASK_EXR.exists():
        raise FileNotFoundError(f"Mask image not found: {MASK_EXR}")
    mask = load_mask_image(MASK_EXR)
    group = ensure_foveation_group()

    injected = 0
    for mat in bpy.data.materials:
        try:
            if inject_group_into_material(mat, mask, group, lo, hi, gamma):
                injected += 1
        except Exception as e:
            log(f"WARN: material '{mat.name}' injection failed: {e}")
    log(f"Injected into {injected} materials")

    bpy.ops.render.render(write_still=True)
    log(f"Saved {FINAL}")

if __name__ == "__main__":
    main()
