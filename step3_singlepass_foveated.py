# step3_singlepass_foveated_v9.py
# Blender 4.5.2 — Single-pass material foveation, GPU-only, Principled v2-safe.

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
        # ensure denoising is actually enabled for final render
        scene.cycles.use_denoising = True
        for vl in scene.view_layers:
            vl.cycles.use_denoising = True
    elif device == "CUDA":
        cprefs.compute_device_type = 'CUDA'
        cprefs.get_devices()
        for d in cprefs.devices:
            d.use = (getattr(d, "type", "") == 'CUDA')
            use_gpu |= d.use
        c.device = 'GPU'
        # Prefer OptiX denoiser if any OptiX device exists; else OIDN
        c.denoiser = 'OPTIX' if any(getattr(d, "type", "") == 'OPTIX' for d in cprefs.devices) else 'OPENIMAGEDENOISE'
        # ensure denoising is actually enabled for final render
        scene.cycles.use_denoising = True
        for vl in scene.view_layers:
            vl.cycles.use_denoising = True
    else:
        raise RuntimeError("SCDL_CYCLES_DEVICE must be OPTIX or CUDA; refusing CPU fallback.")

    if not use_gpu:
        raise RuntimeError("No compatible GPU device found; refusing CPU fallback.")

    # Sampling / perf
    c.samples = int(os.environ.get("SCDL_SAMPLES", "192"))
    c.use_adaptive_sampling = True
    # FIX: correct property name and add a small floor so fine features aren’t missed.
    c.adaptive_threshold = float(os.environ.get("SCDL_ADAPTIVE_THRESHOLD", "0.02"))
    c.adaptive_min_samples = int(os.environ.get("SCDL_ADAPTIVE_MIN_SAMPLES", "32"))

    # Optional fast-GI / bounces you already chose
    c.use_fast_gi = True
    c.max_bounces = 8
    c.transparent_max_bounces = 8

    # Glossy stabilization + firefly clamp (your intent retained)
    # FIX: property name is blur_glossy (Filter Glossy).
    c.blur_glossy = float(os.environ.get("SCDL_FILTER_GLOSSY", "0.8"))
    c.sample_clamp_indirect = float(os.environ.get("SCDL_CLAMP_INDIRECT", "10"))

    r.image_settings.file_format = 'PNG'
    r.filepath = str(FINAL)

# -------------- Data ----------------
def thresholds(np_path: Path):
    arr = np.load(np_path).astype(np.float32)
    flat = arr[np.isfinite(arr)].ravel()
    if flat.size == 0:
        return 0.25, 0.75, 2.2, 0.0

    q20, q80 = np.quantile(flat, [0.2, 0.8])
    lo = float(q20)
    hi = float(q80)
    gamma = 2.2
    cov = float(flat.mean())
    return lo, hi, gamma, cov

# -------------- Nodes: foveation group --------------
def ensure_foveation_group() -> bpy.types.NodeTree:
    name = "SCDL_FoveationMix"
    if name in bpy.data.node_groups:
        return bpy.data.node_groups[name]

    g = bpy.data.node_groups.new(name, "ShaderNodeTree")
    iface = g.interface
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
    for nm in names:
        sock = node.inputs.get(nm)
        if sock is not None:
            return sock
    # try normalized names (Principled v2 compatibility)
    by_norm = { _norm(s.name): s for s in node.inputs }
    for nm in names:
        s = by_norm.get(_norm(nm))
        if s is not None:
            return s
    return None

def link_or_copy(nt: bpy.types.NodeTree, src_node: bpy.types.Node, src_names, dst_node: bpy.types.Node, dst_name: str):
    src = get_input(src_node, *src_names)
    dst = get_input(dst_node, dst_name)
    if not dst:
        return
    if src and src.is_linked:
        nt.links.new(src.links[0].from_socket, dst)
    elif src:
        dst.default_value = getattr(src, "default_value", dst.default_value)

# -------------- LQ Principled (cheap) --------------
def build_lq_principled(nt: bpy.types.NodeTree, ref_p: bpy.types.Node):
    lq = nt.nodes.new("ShaderNodeBsdfPrincipled")
    lq.label = "LQ_Principled"

    # Base defaults for cheap shading
    get_input(lq, "Roughness").default_value = 0.6
    for nm in ("Specular", "Specular IOR Level", "IOR Level"):
        s = get_input(lq, nm)
        if s: s.default_value = 0.2

    # Mirror selected inputs if available
    link_or_copy(nt, ref_p, ["Base Color"], lq, "Base Color")
    link_or_copy(nt, ref_p, ["Roughness"],  lq, "Roughness")
    link_or_copy(nt, ref_p, ["Normal"],     lq, "Normal")

    # Metallic attenuated
    src_met = get_input(ref_p, "Metallic")
    dst_met = get_input(lq, "Metallic")
    if src_met and dst_met:
        if src_met.is_linked:
            nt.links.new(src_met.links[0].from_socket, dst_met)
        else:
            dst_met.default_value = 0.5 * src_met.default_value

    # Kill SSS / Transmission / Coat outside fovea at the group level, so no changes here
    return lq

def find_principled_bfs(nt: bpy.types.NodeTree):
    q = [n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED']
    return q[0] if q else nt.nodes.new("ShaderNodeBsdfPrincipled")

# -------------- Mask image --------------
def load_mask_image(path: Path) -> bpy.types.Image:
    img = bpy.data.images.load(str(path), check_existing=True)
    img.colorspace_settings.name = "Non-Color"  # treat as data, not color-managed
    return img

# -------------- Material injection --------------
def material_is_prohibited(mat: bpy.types.Material) -> bool:
    n = (mat.name or "").lower()
    return any(k in n for k in ("volume", "holdout", "toon"))

def inject_group_into_material(mat: bpy.types.Material, mask_img: bpy.types.Image,
                               group: bpy.types.NodeTree, lo: float, hi: float, gamma: float) -> bool:
    if not mat or not mat.use_nodes or material_is_prohibited(mat):
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
    tex.image.colorspace_settings.name = 'Non-Color'
    tc = nt.nodes.new("ShaderNodeTexCoord"); tc.label = "Coords"
    nt.links.new(tc.outputs["Window"], tex.inputs["Vector"])  # Window = 0..1 screen coords

    # Group instance + LQ shader
    gi = nt.nodes.new("ShaderNodeGroup"); gi.node_tree = group; gi.label = "FoveationMix"
    ref_p = find_principled_bfs(nt)
    lq = build_lq_principled(nt, ref_p)

    # Drive Mask with Alpha when available (scalar), else Color fallback.
    mask_out = tex.outputs.get("Alpha") or tex.outputs.get("Color")

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

# -------------- Main ----------------
def main():
    scene = bpy.context.scene
    lo, hi, gamma, cov = thresholds(NPY_PATH)
    log(f"Mask thresholds: lo={lo:.4f}, hi={hi:.4f}, gamma={gamma:.2f}, cov≈{cov:.3f}")

    configure_cycles(scene)
    # Coverage-aware samples: bias budget toward large fovea; keep bounded
    try:
        base = int(os.environ.get("SCDL_SAMPLES", "192"))
        scene.cycles.samples = max(64, min(768, int(base * (0.5 + 0.5*cov))))
    except Exception:
        pass

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
