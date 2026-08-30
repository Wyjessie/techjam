"""Thin inference server wrapping SQuaDE for the Jamlai backend.

Loads the model once at startup (heavy: DINOv2-giant + both expert banks +
gate), then serves everything after that from memory. Exposes exactly the
contract backend/model_client.py already expects from MODEL_API_URL:
multipart field "file" in, {"label", "confidence", "model_version"} out —
see that file's docstring for the contract. Nothing on the backend side
needs to change; only MODEL_API_URL needs to point here.

Run (from this directory):
    .venv/bin/uvicorn serve:app --port 8100

Requests are serialized through a single lock — this wraps one model
instance, not a batch-serving setup, and concurrent forward passes through
the same nn.Module aren't something this is set up to do safely. Batches
from the frontend arrive as N concurrent single-image HTTP calls (see
backend/main.py's predict_batch), so they just queue up here and run one
at a time (~1-2s/image on this Mac's MPS backend) rather than failing.
"""

import asyncio
import io
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from Inference import SQuaDE, confidence, pick_kernel
from utils.preprocess import normalize

CKPT_DIR = Path(__file__).parent / "ckpt"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_VERSION = "squade-vitg"
# Matches Inference.py's own --threshold default: the fused logit is compared
# against this, not 0.5 — the teammate's update moved the decision (and the
# confidence readout) into logit space. See Inference.py's module docstring.
THRESHOLD = -8.7

app = FastAPI(title="SQuaDE inference server")
_lock = asyncio.Lock()
_net: SQuaDE | None = None


@app.on_event("startup")
def _load_model():
    global _net
    _net = SQuaDE(
        CKPT_DIR / "vg_shallow_full",
        CKPT_DIR / "vg_deep_full",
        CKPT_DIR / "gate_full.pt",
        "facebook/dinov2-giant",
        device=DEVICE,
    )


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model_loaded": _net is not None}


def _load_image_from_bytes(data: bytes, name: str, size: int = 512) -> Image.Image:
    """Same normalization Inference.load_image() applies to a file on disk,
    just starting from in-memory bytes instead of a path."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    kernel = pick_kernel(name) if min(img.size) < size else None
    return normalize(img, size, kernel, fit="crop")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if _net is None:
        raise HTTPException(status_code=503, detail="Model still loading")

    raw = await file.read()
    name = file.filename or "upload.jpg"
    try:
        img = _load_image_from_bytes(raw, name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")

    async with _lock:
        loop = asyncio.get_event_loop()
        # predict() now returns raw fused/expert logits, not sigmoid'd
        # probabilities — the threshold and confidence must both be computed
        # in logit space (see Inference.py's module docstring: sigmoid
        # saturates at both ends, which is exactly why this used to always
        # come back as confidence 1.0).
        zlog, _clean, _gz, votes = await loop.run_in_executor(None, _net.predict, [img], [name])

    z = float(zlog[0])
    is_ai = z > THRESHOLD
    conf = confidence(z, votes[0], THRESHOLD)  # 0-100 reliability index
    return {
        "label": "ai_generated" if is_ai else "authentic",
        "confidence": round(conf / 100, 4),
        "model_version": MODEL_VERSION,
    }
