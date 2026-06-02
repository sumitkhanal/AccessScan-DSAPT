"""
tests/test_api.py — Integration tests for the FastAPI endpoints
Uses TestClient (no live server required) with the YOLO model mocked out.
"""
import io
import sys
import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


def _build_mock_yolo():
    """Build a fully-mocked YOLO callable that returns realistic detections."""
    import numpy as np

    mock_yolo_instance = MagicMock()
    mock_result = MagicMock()
    mock_box = MagicMock()
    mock_box.cls = MagicMock()
    mock_box.cls.__int__ = lambda self: 8  # class index 8 = 'tgsi'
    mock_box.conf = MagicMock(return_value=0.9)
    mock_box.conf.__float__ = lambda self: 0.9
    # xyxy[0].tolist() is called by main.py — use a numpy row so .tolist() works
    mock_box.xyxy = np.array([[10.0, 10.0, 100.0, 50.0]])
    mock_result.boxes = [mock_box]
    mock_result.names = {8: "tgsi"}
    mock_result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_yolo_instance.return_value = [mock_result]
    return mock_yolo_instance


def _inject_ultralytics_mock(mock_yolo_instance):
    """
    Inject a fake 'ultralytics' package into sys.modules so that
    'from ultralytics import YOLO' inside main.py succeeds even when
    the real package is not installed.
    """
    mock_ultralytics = MagicMock()
    mock_ultralytics.YOLO = MagicMock(return_value=mock_yolo_instance)
    sys.modules.setdefault("ultralytics", mock_ultralytics)
    return mock_ultralytics


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI TestClient with ultralytics and cv2 mocked."""
    mock_yolo_instance = _build_mock_yolo()

    # ── Inject ultralytics BEFORE main.py is imported ────────────────────────
    mock_ultralytics = _inject_ultralytics_mock(mock_yolo_instance)

    # ── Also stub cv2 so OpenCV is not required in CI ────────────────────────
    mock_cv2 = MagicMock()
    mock_cv2.imread.return_value = MagicMock(shape=(100, 100, 3))
    mock_cv2.imwrite.return_value = True
    sys.modules.setdefault("cv2", mock_cv2)

    # ── Force-reimport main with mocks already in sys.modules ─────────────────
    for mod in list(sys.modules.keys()):
        if mod in ("main",):
            del sys.modules[mod]

    with patch("os.path.exists", return_value=True):
        from fastapi.testclient import TestClient
        import main as app_module

    # Override the loaded model with our mock
    app_module.model = mock_yolo_instance
    yield TestClient(app_module.app)


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_endpoint_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_health_endpoint_has_model_loaded_field(client):
    response = client.get("/health")
    assert "model_loaded" in response.json()


# ── / (root) ──────────────────────────────────────────────────────────────────

def test_root_endpoint_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200


# ── /audit ────────────────────────────────────────────────────────────────────

def _make_image_bytes():
    """Create a minimal valid JPEG in memory using PIL."""
    try:
        from PIL import Image
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()
    except ImportError:
        # Fallback: use a real image from the static folder if PIL unavailable
        p = Path("static/raw_9c67c242.jpg")
        if p.exists():
            return p.read_bytes()
        pytest.skip("PIL not available and no test image found")


def _audit_post(client):
    """Helper: POST a synthetic image to /audit with depth and cv2 mocked."""
    img_bytes = _make_image_bytes()
    # cv2 is already injected as a mock in sys.modules by the client fixture;
    # we just patch get_depth_map to skip the MiDaS network call.
    with patch("main.get_depth_map", return_value=None):
        return client.post(
            "/audit",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            data={"site_name": "Test Site"},
        )


def test_audit_returns_pdf(client):
    response = _audit_post(client)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"


def test_audit_response_has_job_id_header(client):
    response = _audit_post(client)
    assert "x-job-id" in response.headers


def test_audit_response_has_score_header(client):
    response = _audit_post(client)
    assert "x-score" in response.headers
    score = float(response.headers["x-score"])
    assert 0.0 <= score <= 100.0


# ── /results/{job_id} ────────────────────────────────────────────────────────

def test_results_endpoint_404_for_unknown_job(client):
    response = client.get("/results/nonexistent00")
    assert response.status_code == 404


def test_results_endpoint_returns_json_after_audit(client):
    audit_resp = _audit_post(client)
    job_id = audit_resp.headers.get("x-job-id")
    if not job_id:
        pytest.skip("Audit did not return job id (model may not be loaded)")
    results_resp = client.get(f"/results/{job_id}")
    assert results_resp.status_code == 200
    data = results_resp.json()
    assert "compliance" in data
    assert "score" in data
    assert "job_id" in data


# ── /audit/{job_id} re-download ───────────────────────────────────────────────

def test_report_download_404_for_unknown_job(client):
    response = client.get("/audit/nonexistent00")
    assert response.status_code == 404
