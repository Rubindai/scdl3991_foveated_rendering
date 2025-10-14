#!/usr/bin/env python3
# step3_singlepass_foveated.py
# Purpose: Render the final image in a single pass using a foveation mask
# to guide adaptive sampling. Replaces the old composite-based step3.

import os
import sys
from pathlib import Path
import bpy
from mathutils import Vector

# Add project dir to path to import local modules
project_dir = Path(__file__).resolve().parent
if str(project_dir) not in sys.path:
    sys.path.append(str(project_dir))

from logging_utils import get_logger
from scdl_config import get_pipeline_paths

try:
    # Optional: reuse device configuration helper if available
    import blender_steps as _bs  # type: ignore
except Exception:
    _bs = None  # type: ignore

# --- Configuration ---
PATHS = get_pipeline_paths(project_dir)
LOGGER = get_logger("scdl_step3_singlepass", PATHS.log_file)

# Read config from environment variables, consistent with other scripts
CYCLES_DEVICE = os.environ.get("SCDL_CYCLES_DEVICE", "CPU")
ADAPTIVE_THRESHOLD = float(os.environ.get("SCDL_ROI_CYCLES_THRESHOLD", "0.01"))
MAX_SAMPLES = int(os.environ.get("SCDL_BASE_CYCLES_SPP", "1024"))
MIN_SAMPLES = int(os.environ.get("SCDL_ROI_ADAPTIVE_MIN_SAMPLES", "32"))

# --- Constants ---
FOVEATION_GROUP_NAME = "FoveationMixer"
SIMPLIFIED_SUFFIX = "_simplified"

def log_info(msg: str): LOGGER.info(msg)
def log_warn(msg: str): LOGGER.warning(msg)
def fail(msg: str, code: int = 1):
    LOGGER.error(msg)
    bpy.ops.wm.quit_blender()
    sys.exit(code)

def create_foveated_mixer_group(mask_image):
    """
    Creates the 'FoveatedMixer' node group that blends between two shaders
    based on the screen-space coordinates of the provided mask image.
    Includes aspect ratio correction.
    """
    if FOVEATION_GROUP_NAME in bpy.data.node_groups:
        log_info(f"Node group '{FOVEATION_GROUP_NAME}' already exists.")
        return bpy.data.node_groups[FOVEATION_GROUP_NAME]

    log_info(f"Creating new node group: '{FOVEATION_GROUP_NAME}'")
    g = bpy.data.node_groups.new(name=FOVEATION_GROUP_NAME, type='ShaderNodeTree')

    # Create interface sockets for the group
    g.interface.new_socket(name="Full Quality", in_out='INPUT', socket_type='NodeSocketShader')
    g.interface.new_socket(name="Low Quality", in_out='INPUT', socket_type='NodeSocketShader')
    g.interface.new_socket(name="Shader", in_out='OUTPUT', socket_type='NodeSocketShader')

    # Place the Group Input and Output nodes
    inp = g.nodes.new('NodeGroupInput')
    inp.location = Vector((-600, 0))

    out = g.nodes.new('NodeGroupOutput')
    out.location = Vector((200, 0))

    # Core nodes
    mix = g.nodes.new('ShaderNodeMixShader')
    mix.location = Vector((0, 0))

    tex_coord = g.nodes.new('ShaderNodeTexCoord')
    tex_coord.location = Vector((-600, -200))

    mapping = g.nodes.new('ShaderNodeMapping')
    mapping.location = Vector((-400, -200))

    img_tex = g.nodes.new('ShaderNodeTexImage')
    img_tex.image = mask_image
    img_tex.interpolation = 'Linear'
    img_tex.extension = 'EXTEND'
    img_tex.projection = 'FLAT'
    img_tex.location = Vector((-200, -200))

    # Screen-space mapping (Window) with Y flip to account for image row origin
    # Equivalent to: y' = 1 - y
    mapping.inputs['Scale'].default_value[0] = 1.0
    mapping.inputs['Scale'].default_value[1] = -1.0
    mapping.inputs['Location'].default_value[1] = 1.0

    # Links
    g.links.new(tex_coord.outputs['Window'], mapping.inputs['Vector'])
    g.links.new(mapping.outputs['Vector'], img_tex.inputs['Vector'])
    g.links.new(img_tex.outputs['Color'], mix.inputs['Fac'])
    # Important: Factor=1 should pick Full Quality at fovea.
    # The mask is 1 for the fovea, which corresponds to the *second* shader input.
    g.links.new(inp.outputs['Low Quality'], mix.inputs[1])
    g.links.new(inp.outputs['Full Quality'], mix.inputs[2])
    g.links.new(mix.outputs['Shader'], out.inputs['Shader'])

    return g

def main():
    """Main execution function"""
    log_info("--- Step 3: Single-Pass Foveated Render ---")

    # 1. Setup scene and renderer
    scene = bpy.context.scene
    log_info(f"Processing blend file: {bpy.data.filepath}")
    scene.render.engine = 'CYCLES'
    cycles = scene.cycles

    # Configure device sensibly
    try:
        if _bs is not None and hasattr(_bs, "_configure_cycles_device"):
            _bs._ensure_cycles_addon()
            _bs._configure_cycles_device(scene)
            log_info("Configured Cycles device via helper.")
        else:
            # Fallback: map env to CPU/GPU when possible
            dev = CYCLES_DEVICE.strip().upper()
            scene.cycles.device = 'CPU' if dev == 'CPU' else 'GPU'
            log_info(f"Set Cycles device to: {scene.cycles.device} (raw='{CYCLES_DEVICE}')")
    except Exception as exc:
        log_warn(f"Could not configure Cycles device ('{CYCLES_DEVICE}'): {exc}")

    # 2. Configure Cycles for single-pass adaptive sampling
    cycles.use_adaptive_sampling = True
    cycles.adaptive_threshold = ADAPTIVE_THRESHOLD
    cycles.samples = MAX_SAMPLES
    cycles.adaptive_min_samples = MIN_SAMPLES

    cycles.use_denoising = True
    try:
        cycles.denoiser = 'OPENIMAGEDENOISE'  # OIDN generally good and CPU-friendly
    except Exception:
        pass

    # Enable denoising on view layers and store guiding passes (albedo/normal)
    for vl in scene.view_layers:
        try:
            vl.cycles.use_denoising = True
        except Exception:
            pass
        try:
            vl.cycles.denoising_store_passes = True
        except Exception:
            pass

    # Global noise reduction settings
    cycles.caustics_reflective = False
    cycles.caustics_refractive = False
    cycles.blur_glossy = 0.5 # Filter Glossy

    log_info(f"Configured Cycles for adaptive sampling (threshold={cycles.adaptive_threshold}, min={cycles.adaptive_min_samples}, max={cycles.samples})")

    # 3. Load fovea mask
    # Prefer EXR; gracefully fallback to PNG variants if EXR is unavailable
    mask_candidates = [
        PATHS.out_dir / "fovea_mask.exr",
        PATHS.out_dir / "fovea_mask.png",
        PATHS.mask_preview,  # user_importance_preview.png (8-bit)
    ]

    mask_image = None
    for mpath in mask_candidates:
        if not mpath.exists():
            continue
        log_info(f"Loading fovea mask: {mpath}")
        try:
            img = bpy.data.images.load(str(mpath))
            # Treat as data to avoid color management
            try:
                img.colorspace_settings.name = 'Non-Color'
            except Exception:
                pass
            mask_image = img
            if mpath.suffix.lower() != ".exr":
                log_warn("Using non-EXR mask; precision may be reduced.")
            break
        except Exception as e:
            log_warn(f"Failed to load mask at {mpath}: {e}")

    if mask_image is None:
        fail("No usable foveation mask found. Expected fovea_mask.exr or PNG fallback from Step 2.")

    # 4. Create or get the Foveated Mixer node group
    foveation_group = create_foveated_mixer_group(mask_image)

    # 5. Process all materials in the scene
    log_info("Processing materials with high-quality simplification...")
    materials_to_process = [m for m in bpy.data.materials if m.use_nodes]

    for mat in materials_to_process:
        tree = mat.node_tree
        output_node = next((n for n in tree.nodes if isinstance(n, bpy.types.ShaderNodeOutputMaterial) and n.is_active_output), None)

        if not output_node or not output_node.inputs['Surface'].is_linked:
            log_warn(f"Skipping material '{mat.name}': no active, linked output node.")
            continue

        original_shader_link = output_node.inputs['Surface'].links[0]
        original_shader_node = original_shader_link.from_node
        original_shader_socket = original_shader_link.from_socket

        # Skip if a FoveationMixer already feeds the output
        if isinstance(original_shader_node, bpy.types.ShaderNodeGroup) and \
           original_shader_node.node_tree and \
           original_shader_node.node_tree.name == FOVEATION_GROUP_NAME:
            log_info(f"Material '{mat.name}' already foveation-mixed; skipping rewire.")
            continue

        # --- Create a simplified version of the shader network ---
        simplified_shader_socket = None
        try:
            # 1. Duplicate the final shader node as a starting point
            simplified_node = tree.nodes.new(type=original_shader_node.bl_idname)
            simplified_node.location = original_shader_node.location + Vector((0, -300))

            # 2. Copy scalar/default inputs; preserve links where possible
            for inp_name, inp in original_shader_node.inputs.items():
                try:
                    if inp.is_linked:
                        for link in inp.links:
                            tree.links.new(link.from_socket, simplified_node.inputs[inp_name])
                    else:
                        simplified_node.inputs[inp_name].default_value = inp.default_value
                except Exception:
                    pass

            # 3. Aggressively reduce variance for faster convergence and clearer effect
            node_modified = False
            if hasattr(bpy.types, 'ShaderNodeBsdfPrincipled') and isinstance(simplified_node, bpy.types.ShaderNodeBsdfPrincipled):
                def set_in(node, name, value):
                    try:
                        if name in node.inputs:
                            node.inputs[name].default_value = value
                            return True
                    except Exception:
                        pass
                    return False

                # Push to diffuse-like
                set_in(simplified_node, 'Roughness', 1.0)
                # Zero variance-heavy features (robust to Blender 4.x renames)
                for sock in (
                    'Specular', 'Specular IOR Level', 'Metallic',
                    'Transmission', 'Transmission Weight',
                    'Clearcoat', 'Coat Weight',
                    'Subsurface', 'SSS Weight', 'Sheen', 'Anisotropic',
                ):
                    set_in(simplified_node, sock, 0.0)
                node_modified = True

            # If not principled, create a Diffuse fallback for a strong visual difference
            if not node_modified:
                try:
                    diffuse = tree.nodes.new('ShaderNodeBsdfDiffuse')
                    diffuse.location = simplified_node.location
                    # Neutral mid-grey to reduce lighting complexity
                    diffuse.inputs['Color'].default_value = (0.5, 0.5, 0.5, 1.0)
                    simplified_node = diffuse
                    node_modified = True
                except Exception:
                    pass

            if not node_modified:
                log_warn(f"Could not simplify node type {simplified_node.bl_idname} for material '{mat.name}'. Using original for both paths.")
                simplified_shader_socket = original_shader_socket
            else:
                # Find the correct output socket on the new node
                out_name = original_shader_socket.name if original_shader_socket.name in [s.name for s in simplified_node.outputs] else 'BSDF'
                simplified_shader_socket = simplified_node.outputs[out_name]

        except Exception as e:
            log_warn(f"Failed to create simplified shader for '{mat.name}', using original for both paths. Error: {e}")
            simplified_shader_socket = original_shader_socket

        # Insert the foveation mixer group
        mixer_node = tree.nodes.new('ShaderNodeGroup')
        mixer_node.node_tree = foveation_group
        mixer_node.location = output_node.location + Vector((-200, 0))

        # Rewire everything
        # Unlink the original connection to the output
        tree.links.remove(original_shader_link)
        
        # Link original and simplified shaders to the mixer
        tree.links.new(original_shader_socket, mixer_node.inputs['Full Quality'])
        tree.links.new(simplified_shader_socket, mixer_node.inputs['Low Quality'])
        
        # Link mixer to the final output
        tree.links.new(mixer_node.outputs['Shader'], output_node.inputs['Surface'])
        log_info(f"Applied foveation mix to '{mat.name}'")

    # 6. Optional compositing: blur periphery guided by mask (defocus mode)
    focus_mode = os.environ.get("SCDL_FOCUS_MODE", "defocus").strip().lower()
    if focus_mode in ("defocus", "blur"):
        try:
            scene.use_nodes = True
            nt = scene.node_tree
            nt.nodes.clear(); nt.links.clear()

            rw = int(scene.render.resolution_x)
            rh = int(scene.render.resolution_y)

            n_rl = nt.nodes.new('CompositorNodeRLayers')
            n_rl.location = Vector((-600, 200))

            n_blur = nt.nodes.new('CompositorNodeBlur')
            n_blur.location = Vector((-200, 200))
            try:
                n_blur.filter_type = 'GAUSS'
            except Exception:
                pass
            n_blur.use_relative = False
            blur_px = int(float(os.environ.get("SCDL_DEFOCUS_MAXBLUR", "32")))
            blur_px = max(0, blur_px)
            n_blur.size_x = blur_px
            n_blur.size_y = blur_px

            n_mask = nt.nodes.new('CompositorNodeImage')
            n_mask.location = Vector((-650, -80))
            n_mask.image = mask_image

            # Ensure factor is a single channel
            n_bw = None
            try:
                n_bw = nt.nodes.new('CompositorNodeRGBToBW')
            except Exception:
                try:
                    n_bw = nt.nodes.new('CompositorNodeSeparateColor')
                except Exception:
                    n_bw = None
            if n_bw is not None:
                n_bw.location = Vector((-450, -80))

            n_scale = nt.nodes.new('CompositorNodeScale')
            n_scale.location = Vector((-250, -80))
            # Scale the mask from its native size to render size using relative factors
            try:
                mw, mh = int(mask_image.size[0]), int(mask_image.size[1])
            except Exception:
                mw, mh = rw, rh
            sx = float(rw) / max(1.0, float(mw))
            sy = float(rh) / max(1.0, float(mh))
            try:
                n_scale.space = 'RELATIVE'
            except Exception:
                pass
            n_scale.inputs['X'].default_value = sx
            n_scale.inputs['Y'].default_value = sy

            n_mix = nt.nodes.new('CompositorNodeMixRGB')
            n_mix.location = Vector((0, 100))
            n_mix.blend_type = 'MIX'

            n_comp = nt.nodes.new('CompositorNodeComposite')
            n_comp.location = Vector((200, 100))

            # Links: output = mask*sharp + (1-mask)*blur
            nt.links.new(n_rl.outputs['Image'], n_blur.inputs['Image'])
            nt.links.new(n_blur.outputs['Image'], n_mix.inputs[1])  # A = blur
            nt.links.new(n_rl.outputs['Image'], n_mix.inputs[2])    # B = sharp
            if n_bw is not None:
                # Route through grayscale
                in_name = 'Image' if 'Image' in [s.name for s in n_bw.inputs] else list(n_bw.inputs)[0].name
                nt.links.new(n_mask.outputs['Image'], n_bw.inputs[in_name])
                out_name = 'Val'
                names = [s.name for s in n_bw.outputs]
                if out_name not in names and names:
                    out_name = names[0]
                nt.links.new(n_bw.outputs[out_name], n_scale.inputs['Image'])
            else:
                # Fallback: feed color directly; Blender will implicitly convert to value
                nt.links.new(n_mask.outputs['Image'], n_scale.inputs['Image'])
            nt.links.new(n_scale.outputs['Image'], n_mix.inputs[0]) # Fac = mask (scalar)
            nt.links.new(n_mix.outputs['Image'], n_comp.inputs['Image'])

            # Optional debug outputs of mask/blur/mix to out/ (disabled by default)
            if int(os.environ.get("SCDL_SAVE_COMPOSITOR_DEBUG", "0")):
                try:
                    n_out = nt.nodes.new('CompositorNodeOutputFile')
                    n_out.location = Vector((200, -100))
                    n_out.format.file_format = 'PNG'
                    n_out.base_path = str(PATHS.out_dir)
                    # Clear default slot and add named slots
                    while len(n_out.file_slots) > 0:
                        n_out.file_slots.remove(n_out.file_slots[0])
                    s_mask = n_out.file_slots.new('mask')
                    s_blur = n_out.file_slots.new('blur')
                    s_mix  = n_out.file_slots.new('mix')
                    s_mask.path = 'debug_mask'
                    s_blur.path = 'debug_blur'
                    s_mix.path  = 'debug_mix'
                    nt.links.new(n_scale.outputs['Image'], s_mask)
                    nt.links.new(n_blur.outputs['Image'],  s_blur)
                    nt.links.new(n_mix.outputs['Image'],   s_mix)
                    log_info("Compositor debug outputs enabled (mask/blur/mix).")
                except Exception as exc:
                    log_warn(f"Could not enable compositor debug outputs: {exc}")

            log_info("Compositor set: periphery blur enabled (defocus mode).")
        except Exception as exc:
            log_warn(f"Could not set compositor defocus: {exc}")
    elif focus_mode == "dof":
        # Optional: enable camera DOF if requested by env (depth-based bokeh)
        try:
            cam = scene.camera.data if scene.camera else None
            if cam and hasattr(cam, 'dof'):
                cam.dof.use_dof = True
                fstop = float(os.environ.get("SCDL_FOCUS_FSTOP", "2.8"))
                cam.dof.aperture_fstop = fstop
                dist = float(os.environ.get("SCDL_FOCUS_DISTANCE", "2.0"))
                cam.dof.focus_distance = dist
                log_info(f"Camera DOF enabled (fstop={fstop}, distance={dist}).")
        except Exception as exc:
            log_warn(f"Could not configure camera DOF: {exc}")
    else:
        log_info("Focus mode off; skipping compositor blur.")

    # 7. Render and Save
    log_info("Starting final render...")
    scene.render.filepath = str(PATHS.final)
    scene.render.image_settings.file_format = 'PNG'
    bpy.ops.render.render(write_still=True)
    log_info(f"Finished. Final image saved to {PATHS.final}")

    bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        LOGGER.error(f"An unexpected error occurred: {e}")
        bpy.ops.wm.quit_blender()
        sys.exit(1)
