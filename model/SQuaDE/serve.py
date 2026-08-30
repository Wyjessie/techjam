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
import os
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from Inference import SQuaDE, pick_kernel
from utils.preprocess import normalize

# 版本可用 SQUADE_CKPT 切换(v1/v2/v3),默认 v3。三版的抽头层、骨干、精度、
# 输入窗口完全相同,只是权重不同,所以换版本只需换路径。
CKPT_VER = os.getenv("SQUADE_CKPT", "v3")
_NAMES = {"v1": ("vg_shallow_full", "vg_deep_full", "gate_full.pt"),
          "v2": ("mix_shallow", "mix_deep", "mix_gate.pt"),
          "v3": ("mix2_shallow", "mix2_deep", "mix2_gate.pt")}
if CKPT_VER not in _NAMES:
    raise SystemExit(f"SQUADE_CKPT={CKPT_VER!r}: expected one of {sorted(_NAMES)}")
_SH, _DP, _GT = _NAMES[CKPT_VER]
# v1 的三个文件平铺在 ckpt/ 根下(早于 v1/v2/v3 的目录划分),不在 ckpt/v1/
_ROOT = Path(__file__).parent / "ckpt"
CKPT_DIR = _ROOT if CKPT_VER == "v1" else _ROOT / CKPT_VER
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_VERSION = f"squade-vitg-{CKPT_VER}"

app = FastAPI(title="SQuaDE inference server")
_lock = asyncio.Lock()
_net: SQuaDE | None = None


@app.on_event("startup")
def _load_model():
    global _net
    # Only v3 ships in this repo (it is the default). v1 is still here from
    # before; v2 has to be fetched. Fail with the fix rather than a bare
    # FileNotFoundError three frames deep inside torch.load.
    missing = [str(CKPT_DIR / n) for n in (_SH, _DP, _GT)
               if not (CKPT_DIR / n).exists()]
    if missing:
        raise SystemExit(
            f"SQUADE_CKPT={CKPT_VER} but these are missing:\n  "
            + "\n  ".join(missing)
            + f'\n\nFetch them with:\n  hf download kelvinchua/squade-vitg '
              f'--include "{CKPT_VER}/*" --local-dir '
              f'{Path(__file__).parent / "ckpt"}')
    _net = SQuaDE(
        CKPT_DIR / _SH,
        CKPT_DIR / _DP,
        CKPT_DIR / _GT,
        "facebook/dinov2-giant",
        device=DEVICE,
    )


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model_loaded": _net is not None,
            "model_version": MODEL_VERSION}


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
        prob, _clean, _gz, _votes = await loop.run_in_executor(None, _net.predict, [img], [name])

    score = float(prob[0])
    is_ai = score > 0.5
    return {
        "label": "ai_generated" if is_ai else "authentic",
        "confidence": round(score if is_ai else 1 - score, 4),
        "model_version": MODEL_VERSION,
    }
