This document describes the stages of the foveated rendering pipeline.

## Pipeline Stages

### 1. Preview Render → `out/preview.png`
This step is unchanged. It runs in Windows Blender and produces a quick, low-sample render that preserves the final aspect ratio. This image serves as the input for the saliency analysis.

-   **Inputs**: Scene `.blend` file, runtime overrides such as `SCDL_CYCLES_DEVICE`.
-   **Outputs**: `out/preview.png`.
-   **Script**: `step1_preview_blender.py`.

### 2. Saliency Analysis → Importance Mask
This step runs in the WSL Conda environment. It uses the DINOv3 model to analyze the preview image and determine which regions are most salient or important.

-   **Inputs**: `out/preview.png`, DINO config/weights.
-   **Outputs**:
    -   `out/user_importance.npy`: The raw, numeric importance mask. Used for debugging.
    -   `out/user_importance_preview.png`: A visual representation of the raw mask for sanity checks.
    -   `out/fovea_mask.exr`: **(New)** A blurred, 32-bit float EXR version of the mask. This is the primary input for the final rendering step.
    -   `out/roi_bbox.txt`: The bounding box of the most salient region. (Note: This is no longer used by the rendering pipeline but is still generated).
-   **Script**: `step2_dino_mask.py`.

### 3. Single-Pass Foveated Render → `out/final.png`
This is the new, highly efficient final rendering step. It replaces the old method of rendering and compositing two separate images. It works by using the importance mask to intelligently guide the renderer within a single pass.

-   **Technique**: Instead of varying the sample count directly, we vary the *complexity* of the materials based on the importance mask. Cycles' built-in Adaptive Sampling feature then naturally allocates more render time to the complex, important regions.
-   **Mechanism**:
    1.  The `fovea_mask.exr` is loaded into Blender as a texture.
    2.  A special node group (`FoveationMixer`) is programmatically inserted into every material in the scene.
    3.  This node group uses the mask (mapped to the screen) to blend between the material's original, high-quality shader and a simplified, low-variance version.
    4.  The simplified shader (e.g., with higher roughness and no transmission) produces less noise. The adaptive sampler detects this, stops sampling the unimportant pixels early, and focuses effort on the important regions.
-   **Inputs**: Scene `.blend` file, `out/fovea_mask.exr`.
-   **Output**: `out/final.png`.
-   **Script**: `step3_singlepass_foveated.py`.

## Artifacts Summary
-   `out/preview.png`: Low-quality render for analysis.
-   `out/fovea_mask.exr`: **Primary mask used for rendering.**
-   `out/final.png`: The final, high-quality foveated render.
-   `out/user_importance.npy`, `out/user_importance_preview.png`, `out/roi_bbox.txt`: Debugging and visualization artifacts.