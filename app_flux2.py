"""
app_flux2.py — Standalone Gradio app for FLUX.2 Klein 9B
Exposes a full REST API that can be called from any local Python script.

API endpoint (Gradio native):
    POST http://127.0.0.1:7865/api/predict
    Or use the gradio_client library (see bottom of this file for example).
"""

import os
import sys
import json
import time
import torch
import gradio as gr
from PIL import Image

# ── Path Setup ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# wgp shim so internal imports of `from wgp import ...` don't crash
sys.modules.setdefault("wgp", sys.modules[__name__])

from shared.utils import files_locator as fl
from shared.utils.loras_mutipliers import parse_loras_multipliers
from models.flux.flux_main import model_factory
from models.flux.flux_handler import family_handler
from mmgp import offload

# ── Config ───────────────────────────────────────────────────────────────────
fl.set_checkpoints_paths(["ckpts", "models", "."])

config_path = os.path.join(ROOT, "defaults", "flux2_klein_9b.json")
try:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
except FileNotFoundError:
    config = {
        "model": {"name": "Flux 2 Klein 9B", "flux-model": "flux2_klein_9b"},
        "prompt": "",
        "resolution": "1024x1024",
        "num_inference_steps": 4,
    }

model_def = config.get("model", {})
model_def.update(family_handler.query_model_def("flux2_klein_9b", model_def))

_pipeline = None
_output_dir = os.path.join(ROOT, "outputs_flux2")
os.makedirs(_output_dir, exist_ok=True)

# ── Model Loading ─────────────────────────────────────────────────────────────
def get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    print("[flux2] Initializing FLUX.2 Klein 9B…")
    text_encoder_filename = family_handler.get_text_encoder_name("flux2_klein_9b", "bf16")
    _pipeline = model_factory(
        checkpoint_dir="ckpts",
        model_filename=["flux-2-klein-9b.safetensors"],
        model_type="flux2_klein_9b",
        model_def=model_def,
        base_model_type="flux2_klein_9b",
        text_encoder_filename=text_encoder_filename,
        dtype=torch.bfloat16,
        VAE_dtype=torch.float32,
        quantizeTransformer=False,
        mixed_precision_transformer=False,
    )
    print("[flux2] Model ready.")
    return _pipeline


# ── Core Generate Function ────────────────────────────────────────────────────
def generate(
    prompt: str,
    negative_prompt: str,
    resolution: str,
    steps: int,
    seed: int,
    lora_config: str,
    ref_image: Image.Image | None,
    image_ref_strength: float,
):
    """
    Core inference function — this is ALSO the Gradio API endpoint.

    Parameters (callable from Python via gradio_client):
        prompt            : str       — main generation prompt
        negative_prompt   : str       — negative prompt
        resolution        : str       — e.g. "1024x1024"
        steps             : int       — number of denoising steps
        seed              : int       — -1 for random
        lora_config       : str       — one lora per line: "path/to/lora.safetensors:0.8"
        ref_image         : PIL Image — optional reference/input image
        image_ref_strength: float     — strength of reference image conditioning (0.0–1.0)

    Returns:
        tuple: (output_image_path: str, api_info: str)
    """
    pipeline = get_pipeline()
    width, height = map(int, resolution.split("x"))
    actual_seed = seed if seed >= 0 else int(time.time() * 1000) % (2**32)

    # ── Parse LoRAs ────────────────────────────────────────────────────────
    lora_paths, lora_mults = [], []
    for line in (lora_config or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            path, mult = line.rsplit(":", 1)
        else:
            path, mult = line, "1.0"
        lora_paths.append(path.strip())
        lora_mults.append(mult.strip())

    loras_slists = []
    if lora_paths:
        resolved = [fl.locate_file(p, error_if_none=False) or p for p in lora_paths]
        print(f"[flux2] Loading LoRAs: {resolved}")
        offload.load_loras_into_model(pipeline.model, resolved, activate_all_loras=False)
        mult_str = " ".join(lora_mults)
        loras_slists, _, err = parse_loras_multipliers(mult_str, len(resolved), steps)
        if err:
            raise gr.Error(f"LoRA multiplier parse error: {err}")

    # ── Reference Image ────────────────────────────────────────────────────
    input_ref_images = None
    if ref_image is not None:
        input_ref_images = [ref_image]

    # ── Inference ──────────────────────────────────────────────────────────
    print(f"[flux2] Generating — seed={actual_seed}, steps={steps}, size={width}x{height}")
    result = pipeline.generate(
        seed=actual_seed,
        input_prompt=prompt,
        n_prompt=negative_prompt,
        sampling_steps=steps,
        width=width,
        height=height,
        embedded_guidance_scale=1.0,
        guide_scale=1.0,
        batch_size=1,
        loras_slists=loras_slists,
        input_ref_images=input_ref_images,
        image_refs_relative_size=int(image_ref_strength * 100),
    )

    if result is None:
        raise gr.Error("Generation was interrupted or failed.")

    # ── Decode Tensor → PIL Image ──────────────────────────────────────────
    frame = result[0] if result.ndim == 4 else result
    frame = frame.cpu().clamp(-1, 1).add(1).mul(127.5).clamp(0, 255).to(torch.uint8)
    if frame.shape[0] == 3:
        frame = frame.permute(1, 2, 0)
    img = Image.fromarray(frame.numpy(), mode="RGB")

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = os.path.join(_output_dir, f"flux2_{actual_seed}.png")
    img.save(out_path)
    print(f"[flux2] Saved → {out_path}")

    api_info = json.dumps({
        "seed": actual_seed,
        "output_path": out_path,
        "resolution": resolution,
        "steps": steps,
        "loras": lora_paths,
    }, indent=2)

    return out_path, api_info


# ── Gradio UI ─────────────────────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
* { font-family: 'Inter', sans-serif; box-sizing: border-box; }
body { background: #0d1117; }
.gradio-container { background: #0d1117 !important; max-width: 1200px !important; margin: auto !important; }
.panel { background: #161b22 !important; border: 1px solid #30363d !important; border-radius: 12px !important; }
h1 { background: linear-gradient(135deg, #58a6ff, #bc8cff); -webkit-background-clip: text;
     -webkit-text-fill-color: transparent; font-size: 2rem !important; font-weight: 600 !important; margin-bottom: 4px !important; }
.api-box textarea { font-family: 'Courier New', monospace !important; font-size: 12px !important; color: #8b949e !important; }
footer { display: none !important; }
"""

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="blue",
        secondary_hue="purple",
        neutral_hue="slate",
    ),
    css=css,
    title="FLUX.2 Klein 9B Studio",
    analytics_enabled=False,
) as app:

    gr.Markdown("# ✦ FLUX.2 Klein 9B Studio")
    gr.Markdown("Standalone image generator with LoRA support · API exposed at `/api/predict`")

    with gr.Row():
        # ── Left column: inputs ──────────────────────────────────────────
        with gr.Column(scale=5, elem_classes="panel"):
            prompt_box = gr.Textbox(
                label="Prompt",
                lines=4,
                placeholder="A cinematic portrait of an astronaut on Mars at golden hour…",
                value=config.get("prompt", ""),
            )
            neg_prompt_box = gr.Textbox(
                label="Negative Prompt",
                lines=2,
                value="low quality, blurred, distorted, watermark, text",
            )

            with gr.Row():
                resolution = gr.Dropdown(
                    label="Resolution",
                    choices=[
                        "512x512", "768x768", "832x480",
                        "1024x1024", "1280x720", "1920x1080",
                    ],
                    value=config.get("resolution", "1024x1024"),
                )
                steps = gr.Slider(
                    label="Steps", minimum=1, maximum=25, step=1,
                    value=config.get("num_inference_steps", 4),
                )
                seed = gr.Number(
                    label="Seed (-1 = random)", value=-1, precision=0,
                )

            with gr.Accordion("🔗 LoRA Configuration", open=False):
                gr.Markdown(
                    "One LoRA per line: `path/to/lora.safetensors:multiplier`  \n"
                    "Example: `my_style.safetensors:0.8`"
                )
                lora_config = gr.Textbox(
                    label="LoRA Config",
                    lines=4,
                    placeholder="my_lora.safetensors:1.0\nanother_lora.safetensors:0.5",
                )

            with gr.Accordion("🖼️ Reference / Input Image", open=False):
                gr.Markdown(
                    "Upload an optional reference image for image-conditioned generation."
                )
                ref_image = gr.Image(
                    label="Reference Image",
                    type="pil",
                    sources=["upload", "clipboard"],
                )
                ref_strength = gr.Slider(
                    label="Reference Strength",
                    minimum=0.0, maximum=1.0, step=0.05, value=1.0,
                )

            gen_btn = gr.Button("🚀 Generate", variant="primary", size="lg")

        # ── Right column: outputs ────────────────────────────────────────
        with gr.Column(scale=7, elem_classes="panel"):
            output_image = gr.Image(
                label="Generated Image",
                type="filepath",
                interactive=False,
                height=580,
                show_download_button=True,
            )
            api_output = gr.Textbox(
                label="📡 API Response (JSON)",
                lines=8,
                interactive=False,
                elem_classes="api-box",
            )

    gen_btn.click(
        fn=generate,
        inputs=[prompt_box, neg_prompt_box, resolution, steps,
                seed, lora_config, ref_image, ref_strength],
        outputs=[output_image, api_output],
        api_name="generate",   # exposes as /api/generate
    )

    gr.Markdown(
        """
---
**📡 API Usage from local Python scripts:**
```python
from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:7865")
result = client.predict(
    prompt="A futuristic city at night",
    negative_prompt="blurry, low quality",
    resolution="1024x1024",
    steps=4,
    seed=-1,
    lora_config="my_lora.safetensors:0.8",
    ref_image=handle_file("/path/to/ref.png"),  # or None
    image_ref_strength=1.0,
    api_name="/generate"
)
print(result)  # (image_path, json_info)
```
        """
    )


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7865,
        share=True,
        inbrowser=True,
        show_api=True,          # shows the built-in API docs at /docs
    )
