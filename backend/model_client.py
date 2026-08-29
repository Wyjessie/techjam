"""
Bridge to the image-authenticity model.

Today: no model exists yet, so this returns a mocked verdict so the rest
of the app can be built and demoed end-to-end.

Later: set MODEL_API_URL (and MODEL_API_KEY if it needs auth) in the
environment to point at wherever the model ends up running — a cloud
inference endpoint (Modal / Replicate / a hosted FastAPI service / HF
Inference Endpoints / etc). This file is the ONLY place that needs to
change. main.py and the whole frontend are unaffected.

Response contract (keep stable so the frontend never needs to change):

{
  "label": "ai_generated" | "authentic",
  "confidence": float in [0, 1],   # confidence in `label`
  "model_version": str,
}

If the real model's robustness track ends up returning more (e.g. a
breakdown per transform), extend this dict rather than replacing keys,
so older frontend code keeps working.
"""

import asyncio
import os
import random

import httpx
from dotenv import load_dotenv

load_dotenv()  # picks up .env alongside this file, per .env.example's convention

MODEL_API_URL = os.getenv("MODEL_API_URL")
MODEL_API_KEY = os.getenv("MODEL_API_KEY")


async def predict_image(image_bytes: bytes, filename: str) -> dict:
    if MODEL_API_URL:
        return await _call_cloud_model(image_bytes, filename)
    return await _mock_predict(image_bytes)


async def _call_cloud_model(image_bytes: bytes, filename: str) -> dict:
    headers = {"Authorization": f"Bearer {MODEL_API_KEY}"} if MODEL_API_KEY else {}
    files = {"file": (filename, image_bytes)}
    # A batch arrives here as N concurrent single-image calls (see main.py's
    # predict_batch), and at least the current local model server serializes
    # them through one model instance — the last image in a full 30-image
    # batch can sit queued for a while before it even starts. 30s was tuned
    # for a single call, not a queued one, so give it more room.
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(MODEL_API_URL, headers=headers, files=files)
        resp.raise_for_status()
        return resp.json()


async def _mock_predict(image_bytes: bytes) -> dict:
    # Seeded on the file's bytes so re-uploading the same image gives the
    # same demo verdict instead of flapping randomly on every click.
    seed = sum(image_bytes[:1024]) if image_bytes else 0
    rng = random.Random(seed)
    await asyncio.sleep(0.6)  # pretend inference takes a moment; non-blocking so a batch overlaps
    label = rng.choice(["ai_generated", "authentic"])
    confidence = round(rng.uniform(0.62, 0.97), 3)
    return {
        "label": label,
        "confidence": confidence,
        "model_version": "mock-v0",
    }
