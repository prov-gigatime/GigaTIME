# Quick Setup with UV

[UV](https://github.com/astral-sh/uv) is a fast Python package manager that resolves and installs dependencies significantly faster than pip/conda.

## 1. Install UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Clone and set up

```bash
git clone https://github.com/prov-gigatime/GigaTIME.git
cd GigaTIME

# Create virtual environment (Python 3.11 recommended)
uv venv --python 3.11
source .venv/bin/activate
```

## 3. Install dependencies

**macOS / CPU:**
```bash
uv pip install -r requirements.txt
```

**Linux + NVIDIA CUDA:**
```bash
uv pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  'torch==2.4.1+cu124' \
  'torchvision==0.19.1+cu124'
```

## 4. Authenticate with HuggingFace

The GigaTIME model weights require accepting the license at [huggingface.co/prov-gigatime/GigaTIME](https://huggingface.co/prov-gigatime/GigaTIME).

```bash
# Option A: environment variable
export HF_TOKEN=hf_your_token_here

# Option B: CLI login
uv pip install huggingface_hub[cli]
huggingface-cli login
```

## 5. Run the 3D viewer

```bash
python scripts/gigatime_3d_integrated.py
```

This starts two servers:
- **http://127.0.0.1:7860** — Gradio app (upload H&E, run inference, 2D gallery)
- **http://127.0.0.1:7861** — 3D Three.js viewer (embedded via iframe)

Both servers bind to loopback by default. To opt into LAN access, bind both
servers explicitly and use an externally reachable viewer URL if a reverse
proxy or different public port is involved:

```bash
export GIGATIME_HOST=0.0.0.0
export GIGATIME_VIEWER_HOST=0.0.0.0
# Optional behind a proxy: export GIGATIME_VIEWER_PUBLIC_URL=https://example.org/viewer
# Optional retention period: export GIGATIME_PAYLOAD_TTL_SECONDS=3600
python scripts/gigatime_3d_integrated.py
```

## 6. Run the original notebook (optional)

```bash
cd scripts/
jupyter lab gigatime_testing.ipynb
```
