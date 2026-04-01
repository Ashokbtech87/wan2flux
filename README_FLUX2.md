# FLUX.2 Klein 9B — Standalone Apps

A standalone image generation suite extracted from [Wan2GP](https://github.com/deepbeepmeep/Wan2GP), focused exclusively on the **FLUX.2 Klein 9B** model with full **LoRA** and **reference image** support.

---

## 📁 Files

| File | Purpose |
|---|---|
| `app_flux2.py` | Gradio web UI + API server (port `7865`) |
| `api_flux2.py` | Pure FastAPI REST server, no UI (port `7866`) |
| `client_flux2.py` | Example Python client for `api_flux2.py` |

---

## ⚙️ Requirements

```bash
pip install gradio gradio_client fastapi uvicorn python-multipart torch transformers pillow requests
```

> The model also requires `mmgp` and the rest of the Wan2GP dependencies to be installed.

---

## 🖥️ Option 1 — Gradio Web UI (`app_flux2.py`)

A browser-based interface with a built-in REST API.

### Start

```bash
python app_flux2.py
```

- **Local UI** → `http://127.0.0.1:7865`
- **API Docs** → `http://127.0.0.1:7865/docs`
- **Public link** → printed in terminal when `share=True` (requires Gradio servers to be up)

### UI Features

- ✏️ Prompt & Negative Prompt
- 📐 Resolution selector (512×512 up to 1920×1080)
- 🔢 Steps slider & Seed control
- 🔗 LoRA configuration (multiple LoRAs with per-lora multipliers)
- 🖼️ Reference image upload with strength slider
- 💾 Auto-saves output to `outputs_flux2/`
- 📡 JSON API response shown in UI

### Call from Python (via `gradio_client`)

```python
from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:7865")

# Text-to-image
result = client.predict(
    prompt="A cinematic portrait of an astronaut on Mars",
    negative_prompt="blurry, low quality",
    resolution="1024x1024",
    steps=4,
    seed=-1,
    lora_config="",           # empty = no LoRA
    ref_image=None,
    image_ref_strength=1.0,
    api_name="/generate"
)
image_path, json_info = result
print(image_path)

# With LoRA
result = client.predict(
    prompt="A samurai warrior, anime style",
    negative_prompt="blurry",
    resolution="1024x1024",
    steps=4,
    seed=42,
    lora_config="anime.safetensors:0.8\ndetail_lora.safetensors:0.5",
    ref_image=None,
    image_ref_strength=1.0,
    api_name="/generate"
)

# With reference image
result = client.predict(
    prompt="Cyberpunk version of this person",
    negative_prompt="low quality",
    resolution="1024x1024",
    steps=4,
    seed=-1,
    lora_config="",
    ref_image=handle_file("/path/to/portrait.png"),
    image_ref_strength=0.85,
    api_name="/generate"
)
```

---

## 🚀 Option 2 — FastAPI REST Server (`api_flux2.py`)

A pure REST API — no Gradio, no browser required. Ideal for batch scripts and pipelines.

### Start

```bash
python api_flux2.py
```

- **API Base URL** → `http://127.0.0.1:7866`
- **Swagger UI** → `http://127.0.0.1:7866/docs`
- **ReDoc** → `http://127.0.0.1:7866/redoc`

---

### Endpoints

#### `GET /health`
Check if the server and model are loaded.

```bash
curl http://127.0.0.1:7866/health
```
```json
{ "status": "ok", "model_loaded": true, "model": "FLUX.2 Klein 9B" }
```

---

#### `POST /generate`
Generate an image via JSON body. Returns a saved file path or base64.

**Request body:**

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | str | required | Generation prompt |
| `negative_prompt` | str | `"low quality, blurred..."` | Negative prompt |
| `width` | int | `1024` | Output width |
| `height` | int | `1024` | Output height |
| `steps` | int | `4` | Denoising steps (1–50) |
| `seed` | int | `-1` | Seed (-1 = random) |
| `loras` | list[str] | `[]` | LoRA paths: `["path.safetensors:0.8"]` |
| `ref_image_base64` | str | `null` | Base64-encoded reference image |
| `ref_strength` | float | `1.0` | Reference conditioning strength (0–1) |
| `return_base64` | bool | `false` | Return base64 image instead of file path |

**Example — text-to-image:**

```python
import requests

resp = requests.post("http://127.0.0.1:7866/generate", json={
    "prompt": "A futuristic city skyline at night, photorealistic",
    "width": 1024,
    "height": 1024,
    "steps": 4,
    "seed": -1,
})
data = resp.json()
print(data["output_path"])     # path to saved PNG
print(data["seed"])            # actual seed used
print(data["elapsed_seconds"]) # generation time
```

**Example — with LoRA:**

```python
resp = requests.post("http://127.0.0.1:7866/generate", json={
    "prompt": "A portrait in watercolor style",
    "width": 1024, "height": 1024,
    "steps": 4, "seed": 42,
    "loras": [
        "watercolor_style.safetensors:0.9",
        "detail_enhancer.safetensors:0.5",
    ],
})
```

**Example — with base64 reference image + get base64 back:**

```python
import base64, requests
from pathlib import Path

img_b64 = base64.b64encode(Path("portrait.png").read_bytes()).decode()

resp = requests.post("http://127.0.0.1:7866/generate", json={
    "prompt": "Oil painting portrait version",
    "width": 1024, "height": 1024,
    "steps": 4, "seed": -1,
    "ref_image_base64": img_b64,
    "ref_strength": 0.85,
    "return_base64": True,
})
data = resp.json()
Path("output.png").write_bytes(base64.b64decode(data["base64_image"]))
```

---

#### `POST /generate/upload`
Generate with a **multipart file upload** — send the image as a file, receive PNG directly.

```python
import requests, json

with open("portrait.png", "rb") as f:
    resp = requests.post(
        "http://127.0.0.1:7866/generate/upload",
        data={
            "prompt": "Transform into cyberpunk style",
            "width": 1024, "height": 1024,
            "steps": 4, "seed": -1,
            "loras": json.dumps(["anime.safetensors:0.8"]),
            "ref_strength": 0.85,
        },
        files={"ref_image": ("portrait.png", f, "image/png")},
    )

with open("output.png", "wb") as out:
    out.write(resp.content)
```

---

#### `GET /outputs/{filename}`
Download a previously generated image by filename.

```python
resp = requests.get("http://127.0.0.1:7866/outputs/flux2_1234567890_abc123.png")
open("download.png", "wb").write(resp.content)
```

---

### LoRA Format

LoRAs are passed as a list of strings: `"path/to/lora.safetensors:multiplier"`

- Path can be an **absolute path** or just a **filename** (searched in `ckpts/` and `models/`)
- Multiplier is a float (e.g. `0.8`). Omitting `:multiplier` defaults to `1.0`
- Multiple LoRAs: just add more items to the list

```json
"loras": [
    "anime_style.safetensors:0.8",
    "C:/path/to/detail_lora.safetensors:0.5"
]
```

---

## 📂 Output

All generated images are saved automatically to `outputs_flux2/` in the Wan2GP directory.  
Filenames include the seed and a short random ID: `flux2_<seed>_<id>.png`

---

## 🔁 Running in Google Colab

```python
# In a Colab cell
import subprocess, threading

def run():
    subprocess.run(["python", "api_flux2.py"])

threading.Thread(target=run, daemon=True).start()

# Then expose with SSH tunnel
!ssh -p 443 -R0:localhost:7866 -o StrictHostKeyChecking=no a.pinggy.io
```
