# step3_singlepass_foveated.py
# Single-pass foveated render for Blender 4.5 LTS+

import bpy, os, sys
from pathlib import Path

# ---------- config ----------
ROOT = Path(__file__).resolve().parent

OUT  = ROOT / "out"
MASK_EXR = Path(os.environ.get("SCDL_MASK_EXR", str(OUT / "user_importance_mask.exr")))
FINAL = Path(os.environ.get("SCDL_FINAL_PATH", str(OUT / "final.png")))
NOISE_T = float(os.environ.get("SCDL_ADAPTIVE_THRESHOLD", "0.03"))
MIN_SPP = int(os.environ.get("SCDL_MIN_SAMPLES", "0"))
MAX_SPP = int(os.environ.get("SCDL_MAX_SAMPLES", "256"))
DENOISER = os.environ.get("SCDL_DENOISER", "OPTIX").upper()    # OPTIX|AUTO|OIDN
CAMERA_RAYS_ONLY = os.environ.get("SCDL_CAMERA_RAYS_ONLY", "0") == "1"

def log(msg): print(f"[Step3/Foveated] {msg}")

def ensure_image_noncolor(path: Path) -> bpy.types.Image:
    img = bpy.data.images.get(path.name)
    if not img:
        img = bpy.data.images.load(str(path))
    img.colorspace_settings.name = "Non-Color"  # masks = data, not color-managed
    img.alpha_mode = 'CHANNEL_PACKED'
    return img

def get_or_create_group(name="FoveationMix"):
    nt = bpy.data.node_groups.get(name)
    if nt:
        return nt
    nt = bpy.data.node_groups.new(name=name, type='ShaderNodeTree')

    # --- Group interface (Blender 4.x requires keyword-only args here) ---
    nt.interface.new_socket(
        "HQ Shader", description="", in_out='INPUT', socket_type='NodeSocketShader'
    )
    nt.interface.new_socket(
        "LQ Shader", description="", in_out='INPUT', socket_type='NodeSocketShader'
    )
    nt.interface.new_socket(
        "Mask", description="", in_out='INPUT', socket_type='NodeSocketFloat'
    )
    nt.interface.new_socket(
        "Hardness", description="", in_out='INPUT', socket_type='NodeSocketFloat'
    )
    nt.interface.new_socket(
        "OnlyCam", description="", in_out='INPUT', socket_type='NodeSocketBool'
    )
    nt.interface.new_socket(
        "Shader", description="", in_out='OUTPUT', socket_type='NodeSocketShader'
    )
    # ---------------------------------------------------------------------

    n_in  = nt.nodes.new("NodeGroupInput");  n_in.location  = (-500, 0)
    n_out = nt.nodes.new("NodeGroupOutput"); n_out.location = (300, 0)

    # Shape a narrow, smooth rim so Mix Shader often sees exact 0/1
    mapr = nt.nodes.new("ShaderNodeMapRange"); mapr.location = (-250, 150)
    mapr.inputs["From Min"].default_value = 0.45
    mapr.inputs["From Max"].default_value = 0.55
    mapr.clamp = True

    # Optional camera-ray gating
    mix_cam = nt.nodes.new("ShaderNodeMath"); mix_cam.location = (0, 150)
    mix_cam.operation = 'MULTIPLY'
    lp = nt.nodes.new("ShaderNodeLightPath"); lp.location = (-250, 300)

    # Switch between (Mask * IsCam) and Mask
    switch = nt.nodes.new("ShaderNodeMix"); switch.data_type='FLOAT'; switch.location = (150, 150)
    switch.inputs["A"].default_value = 0.0

    mix = nt.nodes.new("ShaderNodeMixShader"); mix.location = (100, 0)

    # Links
    nt.links.new(n_in.outputs["Hardness"], mapr.inputs["To Max"])
    nt.links.new(n_in.outputs["Mask"],     mapr.inputs["Value"])
    nt.links.new(mapr.outputs["Result"],   mix_cam.inputs[0])
    nt.links.new(lp.outputs["Is Camera Ray"], mix_cam.inputs[1])

    nt.links.new(mix_cam.outputs[0],       switch.inputs["B"])
    nt.links.new(n_in.outputs["OnlyCam"],  switch.inputs["Factor"])
    nt.links.new(n_in.outputs["Mask"],     switch.inputs["A"])

    nt.links.new(n_in.outputs["LQ Shader"], mix.inputs[1])
    nt.links.new(n_in.outputs["HQ Shader"], mix.inputs[2])
    nt.links.new(switch.outputs["Result"],  mix.inputs["Fac"])
    nt.links.new(mix.outputs[0],            n_out.inputs["Shader"])
    return nt

def build_lq_diffuse(nodes) -> bpy.types.Node:
    diff = nodes.new("ShaderNodeBsdfDiffuse")
    diff.inputs["Roughness"].default_value = 1.0  # Oren–Nayar
    return diff

def inject_group_into_material(mat: bpy.types.Material, mask_img: bpy.types.Image, group: bpy.types.NodeTree):
    if not mat.use_nodes:
        return
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links

    out = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
    if not out or not out.inputs["Surface"].is_linked:
        return

    surf_link = out.inputs["Surface"].links[0]
    upstream = surf_link.from_node

    # LQ branch
    lq = build_lq_diffuse(nodes)
    principled = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if principled and principled.inputs["Base Color"].is_linked:
        links.new(principled.inputs["Base Color"].links[0].from_socket, lq.inputs["Color"])
    elif principled:
        lq.inputs["Color"].default_value = principled.inputs["Base Color"].default_value

    # Mask sampler (Non-Color) in screen space (Window coords)
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = mask_img
    tex.interpolation = 'Linear'
    tex.projection = 'FLAT'
    tex.extension = 'CLIP'
    tc = nodes.new("ShaderNodeTexCoord")
    links.new(tc.outputs["Window"], tex.inputs["Vector"])

    # Group
    g = nodes.new("ShaderNodeGroup"); g.node_tree = group
    g.inputs["Hardness"].default_value = 0.35
    g.inputs["OnlyCam"].default_value = CAMERA_RAYS_ONLY

    links.new(tex.outputs["Color"], g.inputs["Mask"])
    links.new(lq.outputs["BSDF"],   g.inputs["LQ Shader"])
    links.new(upstream.outputs[0],  g.inputs["HQ Shader"])

    links.remove(surf_link)
    links.new(g.outputs["Shader"], out.inputs["Surface"])

def configure_render(scene: bpy.types.Scene):
    c = scene.cycles
    c.use_adaptive_sampling = True
    c.adaptive_threshold = NOISE_T
    c.samples = MAX_SPP
    c.preview_samples = min(64, MAX_SPP)
    c.sample_clamp_direct = 0.0
    c.sample_clamp_indirect = 2.0
    c.transparent_max_bounces = 8
    c.adaptive_min_samples = max(0, MIN_SPP)

    # Explicit denoiser selection
    valid = {"OPTIX", "AUTO", "OIDN"}
    algo = DENOISER if DENOISER in valid else "OPTIX"
    scene.cycles.denoiser = algo    # Automatic prefers OIDN and uses GPU when supported
    bpy.context.view_layer.cycles.use_denoising = True

def main():
    if not MASK_EXR.exists():
        raise FileNotFoundError(f"Mask not found: {MASK_EXR}")

    FINAL.parent.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    configure_render(scene)

    mask_img = ensure_image_noncolor(MASK_EXR)
    group = get_or_create_group("FoveationMix")

    for mat in bpy.data.materials:
        try:
            inject_group_into_material(mat, mask_img, group)
        except Exception as e:
            log(f"Skip material '{mat.name}': {e}")

    scene.render.filepath = str(FINAL)
    log(f"Render device: {scene.cycles.device}, denoiser: {scene.cycles.denoiser}, threshold: {scene.cycles.adaptive_threshold}")
    bpy.ops.render.render(write_still=True)
    log(f"Saved: {FINAL}")

if __name__ == "__main__":
    main()
