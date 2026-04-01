"""
client_flux2.py — Example Python client for api_flux2.py
Run the server first: python api_flux2.py
Then run this:       python client_flux2.py
"""

import requests
import base64
import json
from pathlib import Path

SERVER = "http://127.0.0.1:7866"


# ─── Example 1: Simple text-to-image (saves to server's outputs_flux2/) ──────
def example_text_to_image():
    payload = {
        "prompt": "A cinematic portrait of an astronaut on Mars at golden hour, photorealistic",
        "negative_prompt": "blurry, low quality, cartoon",
        "width": 1024,
        "height": 1024,
        "steps": 4,
        "seed": -1,
        "loras": [],
        "return_base64": False,
    }
    resp = requests.post(f"{SERVER}/generate", json=payload)
    resp.raise_for_status()
    data = resp.json()
    print("✅ Text-to-image done!")
    print(f"   Seed       : {data['seed']}")
    print(f"   Output path: {data['output_path']}")
    print(f"   Time       : {data['elapsed_seconds']}s")


# ─── Example 2: Get image as base64 (keep everything in memory) ───────────────
def example_get_base64():
    payload = {
        "prompt": "A futuristic neon-lit Tokyo street, cyberpunk, rain",
        "width": 832,
        "height": 480,
        "steps": 4,
        "seed": 42,
        "return_base64": True,
    }
    resp = requests.post(f"{SERVER}/generate", json=payload)
    resp.raise_for_status()
    data = resp.json()
    img_bytes = base64.b64decode(data["base64_image"])
    out = Path("local_output.png")
    out.write_bytes(img_bytes)
    print("✅ Base64 image received and saved locally → local_output.png")


# ─── Example 3: With LoRA ─────────────────────────────────────────────────────
def example_with_lora():
    payload = {
        "prompt": "A samurai warrior standing in a bamboo forest, anime style",
        "width": 1024,
        "height": 1024,
        "steps": 4,
        "seed": -1,
        "loras": [
            "anime_style_lora.safetensors:0.8",   # update to your actual LoRA path
        ],
        "return_base64": False,
    }
    resp = requests.post(f"{SERVER}/generate", json=payload)
    resp.raise_for_status()
    data = resp.json()
    print("✅ LoRA generation done!")
    print(f"   Output path: {data['output_path']}")


# ─── Example 4: Multipart upload with a reference image ──────────────────────
def example_with_ref_image(image_path: str):
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{SERVER}/generate/upload",
            data={
                "prompt": "Transform this into a cyberpunk style painting",
                "width": 1024,
                "height": 1024,
                "steps": 4,
                "seed": -1,
                "loras": json.dumps([]),
                "ref_strength": 0.85,
            },
            files={"ref_image": ("ref.png", f, "image/png")},
        )
    resp.raise_for_status()
    out = Path("output_with_ref.png")
    out.write_bytes(resp.content)
    print(f"✅ Image-conditioned generation done → {out}")


# ─── Example 5: With base64-encoded reference image ──────────────────────────
def example_with_base64_ref(image_path: str):
    img_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    payload = {
        "prompt": "Oil painting portrait version of the input image",
        "width": 1024,
        "height": 1024,
        "steps": 4,
        "seed": -1,
        "ref_image_base64": img_b64,
        "ref_strength": 0.9,
        "return_base64": False,
    }
    resp = requests.post(f"{SERVER}/generate", json=payload)
    resp.raise_for_status()
    data = resp.json()
    print("✅ Base64 ref image generation done!")
    print(f"   Output path: {data['output_path']}")


if __name__ == "__main__":
    # Check server health first
    health = requests.get(f"{SERVER}/health").json()
    print(f"🟢 Server health: {health}\n")

    example_text_to_image()
    # example_get_base64()
    # example_with_lora()
    # example_with_ref_image("path/to/your/image.png")
    # example_with_base64_ref("path/to/your/image.png")
