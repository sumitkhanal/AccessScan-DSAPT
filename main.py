"""
main.py — AccessScan FastAPI Backend
Sprint 4 | AccessScan

Endpoints:
  POST /login          — authenticate → returns session token
  GET  /me             — return current user info (requires token)
  POST /logout         — invalidate token
  POST /audit          — upload image → returns PDF report
  GET  /audit/{job_id} — re-download a previously generated report
  GET  /health         — health check

Run with:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import sys
import json
import re
import shutil
import uuid
import tempfile
import secrets
import hashlib
from typing import Optional, Dict, List
from datetime import datetime
from pathlib import Path

# ── Ensure project folder is on the Python path so sibling modules are found ──
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
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
# Resolve model path: honour env var first, then fall back through known files.
def _resolve_model_path() -> str:
    candidates = [
        os.getenv("MODEL_PATH", ""),
        "runs/detect/accessscan-v1/weights/best.pt",
        "yolov8s.pt",
        "yolo26n.pt",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return candidates[1]   # return the default path so the warning message is informative

MODEL_PATH = _resolve_model_path()
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
    expose_headers=["X-Job-Id", "X-Score", "X-Score-Label", "X-Detections"],
)

# ── Auth: persistent user store + in-memory session tokens ───────────────────
USERS_FILE = Path(__file__).parent / "users.json"

def _hash_pw(password: str) -> str:
    """SHA-256 hash of the password (hex digest)."""
    return hashlib.sha256(password.encode()).hexdigest()

def _load_users() -> Dict[str, dict]:
    """Load users from users.json, seeding demo accounts if missing."""
    _defaults = {
        "admin":   {"password_hash": _hash_pw("admin123"), "role": "admin",   "name": "Admin User",    "email": "admin@accessscan.local",   "created_at": "2025-01-01T00:00:00"},
        "auditor": {"password_hash": _hash_pw("audit123"), "role": "auditor", "name": "Field Auditor", "email": "auditor@accessscan.local", "created_at": "2025-01-01T00:00:00"},
        "demo":    {"password_hash": _hash_pw("demo"),     "role": "viewer",  "name": "Demo User",     "email": "demo@accessscan.local",    "created_at": "2025-01-01T00:00:00"},
    }
    if USERS_FILE.exists():
        with open(USERS_FILE) as f:
            data = json.load(f)
        # Merge in any missing defaults
        changed = False
        for k, v in _defaults.items():
            if k not in data:
                data[k] = v
                changed = True
        if changed:
            _save_users(data)
        return data
    _save_users(_defaults)
    return _defaults

def _save_users(users: Dict[str, dict]) -> None:
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def _validate_password(pw: str) -> Optional[str]:
    """Return error string if password is too weak, else None."""
    if len(pw) < 8:
        return "Password must be at least 8 characters."
    if not re.search(r"[A-Za-z]", pw):
        return "Password must contain at least one letter."
    if not re.search(r"[0-9]", pw):
        return "Password must contain at least one number."
    return None

# token → {username, role, name, issued_at}
_SESSIONS: Dict[str, dict] = {}

def _require_auth(authorization: Optional[str]) -> dict:
    """Parse Bearer token from Authorization header, raise 401 if invalid."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1]
    session = _SESSIONS.get(token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return session

def _require_role(session: dict, roles: List[str]) -> None:
    if session.get("role") not in roles:
        raise HTTPException(status_code=403, detail="Insufficient permissions")


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


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    """Authenticate with username + password → Bearer token."""
    users = _load_users()
    user = users.get(username)
    if not user or user.get("password_hash") != _hash_pw(password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = secrets.token_hex(32)
    _SESSIONS[token] = {
        "username": username,
        "role":     user["role"],
        "name":     user["name"],
        "email":    user.get("email", ""),
        "issued_at": datetime.now().isoformat(),
    }
    return JSONResponse({
        "token":    token,
        "username": username,
        "name":     user["name"],
        "role":     user["role"],
        "email":    user.get("email", ""),
    })


@app.post("/register")
def register(
    username:  str = Form(...),
    password:  str = Form(...),
    name:      str = Form(...),
    email:     str = Form(default=""),
    role:      str = Form(default="auditor"),
):
    """
    Create a new user account.
    Allowed roles for self-registration: auditor, viewer.
    Admin accounts can only be created by an existing admin.
    """
    users = _load_users()

    # Validate username
    if not re.match(r"^[a-z0-9_]{3,32}$", username):
        raise HTTPException(status_code=422, detail="Username must be 3–32 lowercase letters, numbers, or underscores.")
    if username in users:
        raise HTTPException(status_code=409, detail="Username already taken.")

    # Validate password
    pw_err = _validate_password(password)
    if pw_err:
        raise HTTPException(status_code=422, detail=pw_err)

    # Only allow safe self-registration roles
    if role not in ("auditor", "viewer"):
        role = "auditor"

    users[username] = {
        "password_hash": _hash_pw(password),
        "role":       role,
        "name":       name.strip() or username,
        "email":      email.strip(),
        "created_at": datetime.now().isoformat(),
    }
    _save_users(users)
    print(f"[auth] New user registered: {username} ({role})")
    return JSONResponse({"status": "registered", "username": username, "role": role})


@app.get("/me")
def me(authorization: Optional[str] = Header(default=None)):
    """Return info about the currently authenticated user."""
    session = _require_auth(authorization)
    return JSONResponse(session)


@app.post("/change-password")
def change_password(
    current_password: str = Form(...),
    new_password:     str = Form(...),
    authorization:    Optional[str] = Header(default=None),
):
    """Change the authenticated user's password."""
    session = _require_auth(authorization)
    users   = _load_users()
    uname   = session["username"]
    user    = users[uname]

    if user.get("password_hash") != _hash_pw(current_password):
        raise HTTPException(status_code=401, detail="Current password is incorrect.")
    pw_err = _validate_password(new_password)
    if pw_err:
        raise HTTPException(status_code=422, detail=pw_err)

    user["password_hash"] = _hash_pw(new_password)
    _save_users(users)
    return JSONResponse({"status": "password updated"})


@app.get("/users")
def list_users(authorization: Optional[str] = Header(default=None)):
    """Admin only — list all registered users (passwords excluded)."""
    session = _require_auth(authorization)
    _require_role(session, ["admin"])
    users = _load_users()
    safe = [
        {
            "username":   uname,
            "name":       u.get("name"),
            "email":      u.get("email", ""),
            "role":       u.get("role"),
            "created_at": u.get("created_at", ""),
        }
        for uname, u in users.items()
    ]
    return JSONResponse(safe)


@app.get("/stats")
def stats(authorization: Optional[str] = Header(default=None)):
    """Return aggregate audit statistics from stored result files."""
    _require_auth(authorization)
    result_files = list(REPORTS_DIR.glob("results_*.json"))
    scores, labels, feature_counts = [], [], {}

    for rf in result_files:
        try:
            with open(rf) as f:
                data = json.load(f)
            scores.append(data.get("score", 0))
            labels.append(data.get("label", ""))
            for item in data.get("compliance", []):
                feat = item.get("feature", "unknown")
                feature_counts[feat] = feature_counts.get(feat, 0) + 1
        except Exception:
            pass

    top_issue = None
    if feature_counts:
        top_issue = max(feature_counts, key=feature_counts.get)

    return JSONResponse({
        "total_audits":    len(scores),
        "average_score":   round(sum(scores) / len(scores), 1) if scores else 0,
        "good_count":      sum(1 for l in labels if l == "GOOD"),
        "moderate_count":  sum(1 for l in labels if l == "MODERATE"),
        "poor_count":      sum(1 for l in labels if l == "POOR"),
        "critical_count":  sum(1 for l in labels if l == "CRITICAL"),
        "most_audited_feature": top_issue,
        "feature_counts":  feature_counts,
    })


@app.post("/logout")
def logout(authorization: Optional[str] = Header(default=None)):
    """Invalidate the current session token."""
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
        _SESSIONS.pop(token, None)
    return JSONResponse({"status": "logged out"})


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
    lat: float = Form(default=None),
    lon: float = Form(default=None),
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

    # ── Save annotated image (YOLO bounding boxes drawn) ─────────────────────
    try:
        import cv2 as _cv2ann
        annotated_arr = results[0].plot()          # numpy array, BGR
        annotated_path = str(REPORTS_DIR / f"annotated_{job_id}.jpg")
        _cv2ann.imwrite(annotated_path, annotated_arr)
    except Exception as _e:
        print(f"[main] Could not save annotated image: {_e}")
        annotated_path = None

    # ── Image shape (for width estimation) ───────────────────────────────────
    import cv2 as _cv2
    _img = _cv2.imread(image_path)
    image_shape = _img.shape if _img is not None else None

    # ── Depth estimation ──────────────────────────────────────────────────────
    depth_map = get_depth_map(image_path)

    # ── Compliance engine ─────────────────────────────────────────────────────
    compliance = classify_compliance(detections, depth_map=depth_map, image_shape=image_shape)
    score = compute_score(compliance)
    label = score_label(score)
    print(f"[main] Job {job_id}: score={score}% [{label}]")

    # ── Generate PDF report ───────────────────────────────────────────────────
    report_path = generate_report(job_id, image_path, compliance, score, site_name)

    # Copy to persistent reports dir so /audit/{job_id} can re-download
    stored = REPORTS_DIR / f"report_{job_id}.pdf"
    shutil.copy(report_path, stored)

    # ── Persist results JSON for /results/{job_id} ────────────────────────────
    results_data = {
        "job_id": job_id,
        "site_name": site_name,
        "timestamp": datetime.now().isoformat(),
        "score": score,
        "label": label,
        "detections_count": len(detections),
        "compliance": compliance,
        "lat": lat,
        "lon": lon,
        "annotated_image_url": f"/annotated/{job_id}" if annotated_path else None,
    }
    results_path = REPORTS_DIR / f"results_{job_id}.json"
    with open(results_path, "w") as _f:
        json.dump(results_data, _f, indent=2)

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


@app.get("/annotated/{job_id}")
def get_annotated_image(job_id: str):
    """Return the YOLO bounding-box annotated image for a job."""
    path = REPORTS_DIR / f"annotated_{job_id}.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No annotated image for job_id={job_id}")
    return FileResponse(str(path), media_type="image/jpeg")


@app.get("/results/{job_id}")
def get_results_json(job_id: str):
    """Return full compliance results JSON for a previously run audit."""
    path = REPORTS_DIR / f"results_{job_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No results found for job_id={job_id}")
    with open(path) as f:
        return JSONResponse(content=json.load(f))


@app.get("/detections/{job_id}")
def get_detections_json(job_id: str):
    """Return raw detection JSON (for debugging). Alias to /results/{job_id}."""
    return get_results_json(job_id)
