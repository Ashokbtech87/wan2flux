import os
import sys
import json
import torch
import gradio as gr
from PIL import Image

# Setup Paths for Wan2GP
p = os.path.dirname(os.path.abspath(__file__))
if p not in sys.path:
    sys.path.insert(0, p)

if sys.modules.get("wgp") is not sys.modules.get(__name__):
    sys.modules["wgp"] = sys.modules[__name__]

from shared.utils import files_locator as fl
from shared.utils.loras_mutipliers import parse_loras_multipliers
from models.flux.flux_main import model_factory
from models.flux.flux_handler import family_handler
from mmgp import offload

# Configuration
fl.set_checkpoints_paths(["ckpts", "models", "."])
config_path = os.path.join(p, "defaults", "flux2_klein_9b.json")
try:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
except FileNotFoundError:
    config = {
        "model": {
            "name": "Flux 2 Klein 9B",
            "architecture": "flux2_klein_9b",
            "flux-model": "flux2_klein_9b"
        },
        "prompt": "a futuristic city skyline with flying cars at sunset",
        "resolution": "1024x1024",
        "num_inference_steps": 4
    }

model_def = config.get("model", {})
extra_model_def = family_handler.query_model_def("flux2_klein_9b", model_def)
model_def.update(extra_model_def)

flux_pipeline = None

# UI Defaults
default_prompt = config.get("prompt", "")
default_res = config.get("resolution", "1024x1024")
default_steps = config.get("num_inference_steps", 4)


def load_model():
    global flux_pipeline
    if flux_pipeline is not None:
        return flux_pipeline

    print("Initializing FLUX.2 Klein 9B...")
    
    # Normally Wan2GP expects files to be downloaded. 
    # For a standalone app we assume the ckpts are already present in standard paths.
    model_filename = "flux-2-klein-9b.safetensors"
    resolved_model_path = fl.locate_file(model_filename, error_if_none=False)
    if not resolved_model_path:
        # Fallback to direct path or let model_factory handle it assuming HF download script ran
        model_filename = "flux-2-klein-9b.safetensors"

    text_encoder_filename = family_handler.get_text_encoder_name("flux2_klein_9b", "bf16")

    flux_pipeline = model_factory(
        checkpoint_dir="ckpts",
        model_filename=[model_filename],
        model_type="flux2_klein_9b",
        model_def=model_def,
        base_model_type="flux2_klein_9b",
        text_encoder_filename=text_encoder_filename,
        dtype=torch.bfloat16,
        VAE_dtype=torch.float32,
        quantizeTransformer=False,
        mixed_precision_transformer=False
    )
    
    # Apply splitting if required for offload
    if hasattr(flux_pipeline.model, "split_linear_modules_map"):
        from models.flux.modules.layers import get_linear_split_map
        split_map = get_linear_split_map(
            flux_pipeline.model.hidden_size,
            getattr(flux_pipeline.model.params, "mlp_ratio", 4.0),
            getattr(flux_pipeline.model.params, "single_linear1_mlp_ratio", None),
            getattr(flux_pipeline.model.params, "double_linear1_mlp_ratio", None),
        )
        offload.split_linear_modules(flux_pipeline.model, split_map)

    print("Model initialized.")
    return flux_pipeline


def generate_image(prompt, n_prompt, resolution, sampling_steps, lora_list_text):
    try:
        pipeline = load_model()
        width, height = map(int, resolution.split("x"))
        
        # Parse LoRAs
        lora_lines = [x.strip() for x in lora_list_text.splitlines() if x.strip()]
        lora_paths = []
        lora_mults = []
        for lora in lora_lines:
            if ":" in lora:
                path, mult = lora.rsplit(":", 1)
            else:
                path, mult = lora, "1.0"
            lora_paths.append(path)
            lora_mults.append(mult)
            
        mult_str = " ".join(lora_mults)
        loras_slists = []
        
        if lora_paths:
            print(f"Loading LoRAs: {lora_paths}")
            # Ensure they are absolute paths or locatable
            resolved_loras = [fl.locate_file(p, error_if_none=False) or p for p in lora_paths]
            
            # Load LoRAs into the model
            offload.load_loras_into_model(
                pipeline.model, 
                resolved_loras, 
                activate_all_loras=False
            )
            
            # Parse multipliers mapping phase to step distribution
            loras_slists, _, err = parse_loras_multipliers(mult_str, len(resolved_loras), int(sampling_steps))
            if err:
                raise ValueError(f"LoRA Parse Error: {err}")
                
        # Generate Execution
        print("Starting generation sequence...")
        generated_latents = pipeline.generate(
            seed=None,
            input_prompt=prompt,
            n_prompt=n_prompt,
            sampling_steps=int(sampling_steps),
            width=width,
            height=height,
            embedded_guidance_scale=1.0, # Flux 2 embedded guidance defaults to 1.0 for Klein
            guide_scale=1.0,  
            batch_size=1,
            loras_slists=loras_slists
        )
        
        if generated_latents is None:
            raise RuntimeError("Generation returned None. Interrupted or failed.")
            
        # The output is directly tensor representing pixel values in range [-1, 1], shape [T, C, H, W] for videos or [1, C, H, W] for images.
        if generated_latents.ndim == 4:
            frame = generated_latents[0] # Grab first batch frame
        else:
            frame = generated_latents
            
        # Convert [-1, 1] mapped tensor -> [0, 255] RGB Image
        frame = frame.cpu().clamp(-1, 1)
        frame = frame.add(1).mul(127.5).clamp(0, 255).to(torch.uint8)
        
        # Determine shape arrangement
        if frame.shape[0] == 3: # C, H, W
            frame = frame.permute(1, 2, 0)
        img = Image.fromarray(frame.numpy(), mode="RGB")
        return img
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(str(e))


# UI Definition
custom_css = """
body { background-color: #0b0f19; font-family: 'Inter', sans-serif; }
#app-container { max-width: 1100px; margin: auto; padding-top: 2rem; }
.gradio-container { background: transparent !important; }
"""

with gr.Blocks(theme=gr.themes.Monochrome(), css=custom_css, title="Flux 2 Klein 9B Studio") as app:
    with gr.Column(elem_id="app-container"):
        gr.Markdown("# 🌌 FLUX.2 Klein 9B • Standalone Generator")
        gr.Markdown("A specialized headless wrapper focusing entirely on Flux 2 Klein 9B with LoRA support.")
        
        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(label="Prompt", lines=3, value=default_prompt, placeholder="Type your creative prompt here...")
                n_prompt = gr.Textbox(label="Negative Prompt", lines=2, value="low quality, blurred, distorted")
                
                with gr.Row():
                    res = gr.Dropdown(label="Resolution", choices=["512x512", "768x768", "832x480", "1024x1024", "1280x720", "1920x1080"], value=default_res)
                    steps = gr.Slider(label="Inference Steps", minimum=1, maximum=20, step=1, value=default_steps)
                    
                gr.Markdown("### 🔗 LoRAs Configuration")
                gr.Markdown("Enter paths and multipliers. Format: `[absolute_path_or_name]:[multiplier]`. One per line.\nExample: `my_lora.safetensors:0.8`")
                loras = gr.Textbox(label="LoRA Config", lines=3, placeholder="lora_file.safetensors:1.0")

                gen_btn = gr.Button("🚀 Generate Image", variant="primary", size="lg")
                
            with gr.Column(scale=3):
                output_image = gr.Image(label="Output", type="filepath", interactive=False, height=600)

        gen_btn.click(
            fn=generate_image,
            inputs=[prompt, n_prompt, res, steps, loras],
            outputs=[output_image]
        )

def start_cloudflare_tunnel(port):
    import subprocess
    import threading
    import urllib.request
    import platform
    import os

    print("Starting Cloudflare Tunnel...")
    system = platform.system().lower()
    
    executable = "cloudflared"
    if system == "windows":
        executable = "cloudflared.exe"
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    elif system == "linux":
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    else:
        print("Cloudflare Tunnel auto-setup not supported on this OS.")
        return

    exe_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), executable)
    if not os.path.exists(exe_path):
        print("Downloading cloudflared...")
        try:
            urllib.request.urlretrieve(url, exe_path)
            if system != "windows":
                os.chmod(exe_path, 0o755)
        except Exception as e:
            print(f"Failed to download cloudflared: {e}")
            return

    def run_tunnel():
        cmd = [exe_path, "tunnel", "--url", f"http://127.0.0.1:{port}"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while True:
            line = process.stderr.readline()
            if not line:
                break
            if "trycloudflare.com" in line:
                for part in line.split():
                    if "trycloudflare.com" in part:
                        print("\n" + "="*60)
                        print(f"☁️ CLOUDFLARE TUNNEL URL: {part}")
                        print("Click the link above to access your Gradio App securely (bypasses Gradio Share errors).")
                        print("="*60 + "\n")
                        return

    threading.Thread(target=run_tunnel, daemon=True).start()

if __name__ == "__main__":
    import os
    server_name = "0.0.0.0" if "COLAB_GPU" in os.environ or "COLAB_JUPYTER_IP" in os.environ else "127.0.0.1"
    port = 7865
    
    # Fire up the Cloudflare Tunnel fallback
    start_cloudflare_tunnel(port)
    
    # Colab Native Proxy
    try:
        import google.colab
        from google.colab.output import eval_js
        public_url = eval_js(f"google.colab.kernel.proxyPort({port})")
        print("\n" + "="*60)
        print("🟢 In Colab Environment Detected!")
        print(f"🔗 NATIVE COLAB URL: {public_url}")
        print("="*60 + "\n")
    except ImportError:
        pass

    try:
        app.launch(server_name=server_name, server_port=port, inbrowser=True, share=True)
    except Exception as e:
        print(f"Gradio share failed due to internet or certificate error: {e}")
        app.launch(server_name=server_name, server_port=port, inbrowser=True, share=False)
