"""
main.py — AccessScan FastAPI Backend
Sprint 4 | AccessScan

Endpoints:
  POST /audit          — upload image → returns PDF report
  GET  /audit/{job_id} — re-download a previously generated report
  GET  /health         — health check

Run with:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import sys
import shutil
import uuid
import tempfile
from pathlib import Path

# ── Ensure project folder is on the Python path so sibling modules are found ──
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ultralytics import YOLO
from compliance import classify_compliance
from score import compute_score, score_label
from report import generate_report

# ── Use a platform-safe temp directory (works on Windows AND Linux/Mac) ────────
TMP_DIR = Path(tempfile.gettempdir())

# ── Depth model (optional — loaded lazily to avoid crash if torch unavailable)
_depth_loaded = False
_midas_depth = None

def get_depth_map(image_path: str):
    global _depth_loaded, _midas_depth
    try:
        if not _depth_loaded:
            from depth import estimate_depth
            _midas_depth = estimate_depth
            _depth_loaded = True
        return _midas_depth(image_path)
    except Exception as e:
        print(f"[main] Depth estimation unavailable: {e}")
        return None


# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "runs/detect/accessscan-v1/weights/best.pt")
REPORTS_DIR = TMP_DIR / "accessscan_reports"
REPORTS_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="AccessScan API",
    description="CV-powered DSAPT accessibility audit tool — La Trobe × DITRDCA",
    version="1.0.0",
)

# CORS — allow the frontend (index.html) served from any origin during dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load YOLO model once at startup ───────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print(f"[main] WARNING: Model not found at {MODEL_PATH}")
    print("[main] Set MODEL_PATH env var or place weights at the default path.")
    model = None
else:
    print(f"[main] Loading model from {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("[main] Model loaded.")

# Serve the frontend static files from ./static/
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "AccessScan API running. POST /audit with an image file."}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
    }


@app.post("/audit")
async def audit(
    file: UploadFile = File(...),
    site_name: str = Form(default="Unknown Site"),
):
    """
    Upload an image → receive a PDF accessibility compliance report.

    Form fields:
      file      (required): image file (jpg/png)
      site_name (optional): label for the report header
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Expected at: {MODEL_PATH}. "
                   "Train the model in Sprint 2 first."
        )

    # ── Save uploaded file ────────────────────────────────────────────────────
    job_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix or ".jpg"
    image_path = str(TMP_DIR / f"{job_id}{ext}")

    with open(image_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    print(f"[main] Job {job_id}: saved upload to {image_path}")

    # ── YOLOv8 detection ──────────────────────────────────────────────────────
    results = model(image_path)
    detections = [
        {
            "class": results[0].names[int(b.cls)],
            "confidence": float(b.conf),
            "bbox": b.xyxy[0].tolist(),
        }
        for b in results[0].boxes
    ]
    print(f"[main] Job {job_id}: {len(detections)} detections")

    # ── Depth estimation ──────────────────────────────────────────────────────
    depth_map = get_depth_map(image_path)

    # ── Compliance engine ─────────────────────────────────────────────────────
    compliance = classify_compliance(detections, depth_map=depth_map)
    score = compute_score(compliance)
    label = score_label(score)
    print(f"[main] Job {job_id}: score={score}% [{label}]")

    # ── Generate PDF report ───────────────────────────────────────────────────
    report_path = generate_report(job_id, image_path, compliance, score, site_name)

    # Copy to persistent reports dir so /audit/{job_id} can re-download
    stored = REPORTS_DIR / f"report_{job_id}.pdf"
    shutil.copy(report_path, stored)

    return FileResponse(
        str(stored),
        media_type="application/pdf",
        filename=f"AccessScan-Report-{job_id}.pdf",
        headers={
            "X-Job-Id": job_id,
            "X-Score": str(score),
            "X-Score-Label": label,
            "X-Detections": str(len(detections)),
        }
    )


@app.get("/audit/{job_id}")
def download_report(job_id: str):
    """Re-download a previously generated report."""
    path = REPORTS_DIR / f"report_{job_id}.pdf"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No report found for job_id={job_id}")
    return FileResponse(str(path), media_type="application/pdf",
                        filename=f"AccessScan-Report-{job_id}.pdf")


@app.get("/detections/{job_id}")
def get_detections_json(job_id: str):
    """Return raw detection JSON (for debugging)."""
    # In production you'd store this; for PoC just note the endpoint exists
    raise HTTPException(status_code=501, detail="JSON storage not implemented in PoC")
