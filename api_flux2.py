"""
api_flux2.py — Pure FastAPI REST server for FLUX.2 Klein 9B
No Gradio, no UI — clean HTTP API for use from any local/remote Python script.

Start server:
    python api_flux2.py

Endpoints:
    GET  /health          → server + model status
    POST /generate        → generate image (returns image file or JSON with path)
    POST /generate/base64 → generate image (returns base64 in JSON)
    GET  /outputs/{name}  → download a saved image directly

Example client call — see bottom of this file or client_flux2.py
"""

import os
import sys
import json
import time
import uuid
import base64
import io
import torch
from typing import Optional, List
from pathlib import Path

# ── Path Setup ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# wgp shim required by internal imports
sys.modules.setdefault("wgp", sys.modules[__name__])

from shared.utils import files_locator as fl
from shared.utils.loras_mutipliers import parse_loras_multipliers
from models.flux.flux_main import model_factory
from contextlib import asynccontextmanager
from models.flux.flux_handler import family_handler, get_text_encoder_name
from mmgp import offload
from PIL import Image

from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# ── Config ────────────────────────────────────────────────────────────────────
fl.set_checkpoints_paths(["ckpts", "models", "."])

config_path = os.path.join(ROOT, "defaults", "flux2_klein_9b.json")
try:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
except FileNotFoundError:
    config = {
        "model": {"name": "Flux 2 Klein 9B", "flux-model": "flux2_klein_9b"},
        "num_inference_steps": 4,
    }

model_def = config.get("model", {})
model_def.update(family_handler.query_model_def("flux2_klein_9b", model_def))

OUTPUT_DIR = Path(ROOT) / "outputs_flux2"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── FastAPI App ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load model at startup."""
    print("[flux2-api] Pre-loading model on startup…")
    get_pipeline()
    yield

app = FastAPI(
    title="FLUX.2 Klein 9B API",
    description="REST API for FLUX.2 Klein 9B image generation with LoRA + image conditioning.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global pipeline ───────────────────────────────────────────────────────────
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    print("[flux2-api] Loading FLUX.2 Klein 9B model…")
    text_enc = get_text_encoder_name("flux2_klein_9b", "bf16")
    _pipeline = model_factory(
        checkpoint_dir="ckpts",
        model_filename=["flux-2-klein-9b.safetensors"],
        model_type="flux2_klein_9b",
        model_def=model_def,
        base_model_type="flux2_klein_9b",
        text_encoder_filename=text_enc,
        dtype=torch.bfloat16,
        VAE_dtype=torch.float32,
        quantizeTransformer=False,
        mixed_precision_transformer=False,
    )
    print("[flux2-api] Model ready.")
    return _pipeline


def tensor_to_pil(tensor) -> Image.Image:
    frame = tensor[0] if tensor.ndim == 4 else tensor
    frame = frame.cpu().clamp(-1, 1).add(1).mul(127.5).clamp(0, 255).to(torch.uint8)
    if frame.shape[0] == 3:
        frame = frame.permute(1, 2, 0)
    return Image.fromarray(frame.numpy(), mode="RGB")


def run_inference(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    seed: int,
    loras: List[str],          # ["path:mult", ...]
    ref_image: Optional[Image.Image],
    ref_strength: float,
) -> tuple[Image.Image, int]:
    """Shared inference logic used by all endpoints."""
    pipeline = get_pipeline()
    actual_seed = seed if seed >= 0 else int(time.time() * 1000) % (2**32)

    # Parse LoRAs
    loras_slists = []
    if loras:
        paths, mults = [], []
        for entry in loras:
            if ":" in entry:
                p, m = entry.rsplit(":", 1)
            else:
                p, m = entry, "1.0"
            paths.append(p.strip())
            mults.append(m.strip())

        resolved = [fl.locate_file(p, error_if_none=False) or p for p in paths]
        offload.load_loras_into_model(pipeline.model, resolved, activate_all_loras=False)
        loras_slists, _, err = parse_loras_multipliers(" ".join(mults), len(resolved), steps)
        if err:
            raise ValueError(f"LoRA parse error: {err}")

    input_ref_images = [ref_image] if ref_image is not None else None

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
        image_refs_relative_size=int(ref_strength * 100),
    )

    if result is None:
        raise RuntimeError("Generation returned None — interrupted or failed.")

    return tensor_to_pil(result), actual_seed


# ── Pydantic Request / Response Models ────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Image generation prompt")
    negative_prompt: str = Field(
        default="low quality, blurred, distorted",
        description="Negative prompt"
    )
    width: int = Field(default=1024, description="Output width in pixels")
    height: int = Field(default=1024, description="Output height in pixels")
    steps: int = Field(default=4, ge=1, le=50, description="Denoising steps")
    seed: int = Field(default=-1, description="RNG seed (-1 = random)")
    loras: List[str] = Field(
        default=[],
        description='LoRA paths with optional multiplier: ["path/to.safetensors:0.8"]'
    )
    ref_image_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded reference image (PNG/JPG). Optional."
    )
    ref_strength: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Strength of reference image conditioning"
    )
    return_base64: bool = Field(
        default=False,
        description="If true, return image as base64 JSON. If false, save and return file path."
    )


class GenerateResponse(BaseModel):
    seed: int
    output_path: Optional[str] = None
    base64_image: Optional[str] = None
    width: int
    height: int
    steps: int
    elapsed_seconds: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", summary="Server & model health check")
def health():
    return {
        "status": "ok",
        "model_loaded": _pipeline is not None,
        "model": "FLUX.2 Klein 9B",
        "output_dir": str(OUTPUT_DIR),
    }


@app.post("/generate", response_model=GenerateResponse, summary="Generate image (JSON body)")
def generate_json(req: GenerateRequest):
    """
    Generate an image from a JSON body. Supports LoRAs and base64 reference image.
    Returns either a saved file path or inline base64, depending on `return_base64`.
    """
    t0 = time.time()
    try:
        # Decode optional ref image
        ref_pil = None
        if req.ref_image_base64:
            img_bytes = base64.b64decode(req.ref_image_base64)
            ref_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        img, seed = run_inference(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            steps=req.steps,
            seed=req.seed,
            loras=req.loras,
            ref_image=ref_pil,
            ref_strength=req.ref_strength,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = round(time.time() - t0, 2)

    if req.return_base64:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return GenerateResponse(
            seed=seed, base64_image=b64,
            width=req.width, height=req.height,
            steps=req.steps, elapsed_seconds=elapsed,
        )

    out_path = OUTPUT_DIR / f"flux2_{seed}_{uuid.uuid4().hex[:6]}.png"
    img.save(out_path)
    return GenerateResponse(
        seed=seed, output_path=str(out_path),
        width=req.width, height=req.height,
        steps=req.steps, elapsed_seconds=elapsed,
    )


@app.post("/generate/upload", summary="Generate image with multipart ref image upload")
async def generate_upload(
    prompt: str = Form(...),
    negative_prompt: str = Form(default="low quality, blurred, distorted"),
    width: int = Form(default=1024),
    height: int = Form(default=1024),
    steps: int = Form(default=4),
    seed: int = Form(default=-1),
    loras: str = Form(default="", description='JSON array string: ["path:0.8"]'),
    ref_strength: float = Form(default=1.0),
    ref_image: Optional[UploadFile] = File(default=None),
):
    """
    Generate with multipart/form-data — useful when sending an image file directly
    instead of base64. Returns the generated image file directly.
    """
    t0 = time.time()
    try:
        lora_list = json.loads(loras) if loras.strip() else []
        ref_pil = None
        if ref_image is not None:
            data = await ref_image.read()
            ref_pil = Image.open(io.BytesIO(data)).convert("RGB")

        img, actual_seed = run_inference(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width, height=height,
            steps=steps, seed=seed,
            loras=lora_list,
            ref_image=ref_pil,
            ref_strength=ref_strength,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    out_path = OUTPUT_DIR / f"flux2_{actual_seed}_{uuid.uuid4().hex[:6]}.png"
    img.save(out_path)
    elapsed = round(time.time() - t0, 2)
    print(f"[flux2-api] Done in {elapsed}s → {out_path}")
    return FileResponse(str(out_path), media_type="image/png", filename=out_path.name)


@app.get("/outputs/{filename}", summary="Download a previously generated image")
def get_output(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(path), media_type="image/png")



# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "api_flux2:app",
        host="0.0.0.0",
        port=7866,
        reload=False,
        log_level="info",
    )
