# Jamlai — AI Image Detector Demo

TechJam 2026, Track 5: Robust Detection of AI-Generated Images Under
Real-World Transformations. This is the demo webapp — a "detection
lightbox": a generator × transformation-condition contact sheet.

- **Columns (generators)** are open-ended — add any generator/source by
  name any time, and set its ground truth (authentic / AI-generated /
  unknown) right there in the header. Ground truth lives on the
  generator, not the cell, since a transform never changes whether a
  photo is real.
- **Rows (conditions)** are fixed, because they're real code, not just a
  label: Original, Blurred, Compressed, Cropped, Color-shifted, Rescaled.
- **You only ever upload into the Original cell.** Hitting "Develop"
  scores the originals, then derives each of the other five conditions
  from those same files via client-side canvas transforms and scores
  each — so "SDXL × Blurred" is provably the same photos as
  "SDXL × Original," not a separately-curated set, and a whole
  robustness row comes from one upload.
- Each cell in the sheet shows a cover thumbnail and its accuracy at a
  glance (border color = accuracy tier), so patterns are visible without
  opening anything: a red column is a generalization gap, a red row is a
  robustness gap.
- Opening a cell shows the model's call first; a beat later, a stamped
  "Confirmed"/"Missed" lands on each thumbnail against the generator's
  ground truth.

## Architecture

```
frontend/ (static HTML/CSS/JS, no build step)
   │  POST /api/predict/batch (multipart, multiple images)
   ▼
backend/  (FastAPI)
   │  scores each image concurrently; one bad file doesn't fail the batch
   │  model_client.py: calls MODEL_API_URL if set, else returns a mock verdict
   ▼
(future) cloud-hosted model
```

`POST /api/predict` (single image) still exists too, for when a one-off
check is all that's needed.

The model doesn't exist yet, so `backend/model_client.py` returns a
seeded mock verdict by default. Once the model is deployed somewhere in
the cloud, set `MODEL_API_URL` (and `MODEL_API_KEY` if needed) in
`backend/.env` and the backend starts calling it for real — nothing else
in the app needs to change. See the contract documented at the top of
`model_client.py`.

## Run it

Backend:

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Frontend (any static server works; simplest is Python's built-in one):

```bash
cd frontend
python3 -m http.server 5500
```

Then open http://localhost:5500 in a browser, drop an image, and click
Analyze.

## Next steps

- Swap the mock in `model_client.py` for a real call once the model has
  a cloud endpoint.
- Consider showing the verdict before/after a transform (blur/compress/
  crop) side by side in the UI — it's a direct, visual demonstration of
  the robustness the track is being judged on.
- Deploying a public link (frontend + backend) is a later step, not
  needed yet — localhost is fine while iterating.
