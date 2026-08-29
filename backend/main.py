import asyncio
import os
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from model_client import predict_image

app = FastAPI(title="Jamlai AI Image Detector API")

# Wide open for local dev. Tighten allow_origins before any public deploy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_BATCH_SIZE = 30  # keep a demo batch fast and the payload sane


@app.get("/api/health")
def health():
    return {"status": "ok", "mode": "cloud" if os.getenv("MODEL_API_URL") else "mock"}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")

    return await predict_image(image_bytes, filename=file.filename)


@app.post("/api/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Score a batch of images (e.g. a representative set spanning different
    generators, quality levels, image types) in one request. Each image is
    scored independently; one bad file doesn't fail the whole batch — it
    just comes back with an "error" field instead of a verdict.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"Too many files (max {MAX_BATCH_SIZE})")

    async def run_one(f: UploadFile) -> dict:
        if f.content_type not in ALLOWED_CONTENT_TYPES:
            return {"filename": f.filename, "error": f"Unsupported file type: {f.content_type}"}
        image_bytes = await f.read()
        if not image_bytes:
            return {"filename": f.filename, "error": "Empty file"}
        if len(image_bytes) > MAX_FILE_SIZE_BYTES:
            return {"filename": f.filename, "error": "File too large (max 10MB)"}
        try:
            result = await predict_image(image_bytes, filename=f.filename)
        except Exception as e:  # a single cloud-call failure shouldn't sink the batch
            return {"filename": f.filename, "error": str(e)}
        return {"filename": f.filename, **result}

    results = await asyncio.gather(*(run_one(f) for f in files))
    return {"results": results}
