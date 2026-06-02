"""
tests/test_compliance.py — Unit tests for compliance.py
"""
import pytest
import numpy as np
from compliance import classify_compliance, is_adjacent


# ── is_adjacent ───────────────────────────────────────────────────────────────

def test_identical_bboxes_are_adjacent():
    bbox = [0, 0, 100, 100]
    assert is_adjacent(bbox, bbox) is True


def test_nearby_bboxes_are_adjacent():
    a = [0, 0, 100, 100]
    b = [120, 0, 220, 100]   # just to the right, within threshold
    assert is_adjacent(a, b) is True


def test_far_bboxes_are_not_adjacent():
    a = [0, 0, 10, 10]       # small box
    b = [5000, 5000, 5010, 5010]  # far away
    assert is_adjacent(a, b) is False


# ── classify_compliance — present check ───────────────────────────────────────

def test_tgsi_detected_is_compliant():
    detections = [{"class": "tgsi", "confidence": 0.9, "bbox": [10, 10, 100, 50]}]
    results = classify_compliance(detections)
    assert len(results) == 1
    assert results[0]["status"] == "COMPLIANT"


def test_handrail_detected_is_compliant():
    detections = [{"class": "handrail", "confidence": 0.85, "bbox": [0, 0, 50, 200]}]
    results = classify_compliance(detections)
    assert results[0]["status"] == "COMPLIANT"


def test_signage_detected_is_compliant():
    detections = [{"class": "signage", "confidence": 0.75, "bbox": [0, 0, 80, 80]}]
    results = classify_compliance(detections)
    assert results[0]["status"] == "COMPLIANT"


def test_stairs_detected_is_compliant():
    detections = [{"class": "stairs", "confidence": 0.80, "bbox": [0, 0, 200, 300]}]
    results = classify_compliance(detections)
    assert results[0]["status"] == "COMPLIANT"


# ── classify_compliance — non_compliant check ─────────────────────────────────

def test_obstruction_always_non_compliant():
    detections = [{"class": "obstruction", "confidence": 0.95, "bbox": [100, 100, 200, 200]}]
    results = classify_compliance(detections)
    assert results[0]["status"] == "NON-COMPLIANT"


# ── classify_compliance — tgsi_adjacent check ────────────────────────────────

def test_platform_edge_with_adjacent_tgsi_is_compliant():
    detections = [
        {"class": "platform_edge", "confidence": 0.85, "bbox": [100, 10, 500, 60]},
        {"class": "tgsi",          "confidence": 0.90, "bbox": [110, 10, 300, 50]},
    ]
    results = classify_compliance(detections)
    edge = next(r for r in results if r["class"] == "platform_edge")
    assert edge["status"] == "COMPLIANT"


def test_platform_edge_without_tgsi_is_non_compliant():
    detections = [
        {"class": "platform_edge", "confidence": 0.85, "bbox": [100, 10, 500, 60]},
    ]
    results = classify_compliance(detections)
    assert results[0]["status"] == "NON-COMPLIANT"


def test_kerb_cut_with_adjacent_tgsi_is_compliant():
    detections = [
        {"class": "kerb_cut", "confidence": 0.80, "bbox": [200, 200, 400, 350]},
        {"class": "tgsi",     "confidence": 0.88, "bbox": [210, 210, 390, 340]},
    ]
    results = classify_compliance(detections)
    kc = next(r for r in results if r["class"] == "kerb_cut")
    assert kc["status"] == "COMPLIANT"


def test_kerb_cut_without_tgsi_is_non_compliant():
    detections = [
        {"class": "kerb_cut", "confidence": 0.80, "bbox": [200, 200, 400, 350]},
    ]
    results = classify_compliance(detections)
    assert results[0]["status"] == "NON-COMPLIANT"


# ── classify_compliance — gradient check ─────────────────────────────────────

def test_ramp_without_depth_needs_review():
    detections = [{"class": "ramp", "confidence": 0.85, "bbox": [50, 50, 300, 400]}]
    results = classify_compliance(detections, depth_map=None)
    assert results[0]["status"] == "NEEDS REVIEW"


def test_ramp_compliant_with_low_gradient():
    """Flat depth map → near-zero gradient → COMPLIANT. Mocks depth module."""
    import sys
    from unittest.mock import MagicMock, patch
    depth_map = np.ones((480, 640), dtype=np.float32) * 100.0
    # estimate_gradient returns 0.0 for flat surface
    mock_depth = MagicMock()
    mock_depth.estimate_gradient.return_value = 0.0
    with patch.dict(sys.modules, {"depth": mock_depth}):
        detections = [{"class": "ramp", "confidence": 0.85, "bbox": [100, 100, 300, 400]}]
        results = classify_compliance(detections, depth_map=depth_map)
    assert results[0]["status"] == "COMPLIANT"


def test_ramp_non_compliant_with_steep_gradient():
    """Steep depth gradient → exceeds 4° threshold → NON-COMPLIANT."""
    import sys
    from unittest.mock import MagicMock, patch
    depth_map = np.zeros((480, 640), dtype=np.float32)
    depth_map[:240, :] = 900.0
    depth_map[240:, :] = 10.0
    mock_depth = MagicMock()
    mock_depth.estimate_gradient.return_value = 12.5  # well above 4° threshold
    with patch.dict(sys.modules, {"depth": mock_depth}):
        detections = [{"class": "ramp", "confidence": 0.85, "bbox": [100, 50, 500, 430]}]
        results = classify_compliance(detections, depth_map=depth_map)
    assert results[0]["status"] == "NON-COMPLIANT"


# ── classify_compliance — width check ────────────────────────────────────────

def test_clear_path_without_depth_needs_review():
    detections = [{"class": "clear_path", "confidence": 0.80, "bbox": [0, 0, 640, 480]}]
    results = classify_compliance(detections, depth_map=None, image_shape=None)
    assert results[0]["status"] == "NEEDS REVIEW"


def test_clear_path_wide_is_compliant():
    """Wide path (>= 1500mm estimated) should be COMPLIANT."""
    import sys
    from unittest.mock import MagicMock, patch
    depth_map = np.ones((480, 640), dtype=np.float32) * 50.0
    mock_depth = MagicMock()
    mock_depth.estimate_width_mm.return_value = 2400.0  # 2400mm > 1500mm threshold
    with patch.dict(sys.modules, {"depth": mock_depth}):
        detections = [{"class": "clear_path", "confidence": 0.80, "bbox": [0, 100, 640, 380]}]
        results = classify_compliance(detections, depth_map=depth_map, image_shape=(480, 640, 3))
    assert results[0]["status"] == "COMPLIANT"


def test_clear_path_narrow_is_non_compliant():
    """Narrow path (< 1500mm estimated) should be NON-COMPLIANT."""
    import sys
    from unittest.mock import MagicMock, patch
    depth_map = np.ones((480, 640), dtype=np.float32) * 50.0
    mock_depth = MagicMock()
    mock_depth.estimate_width_mm.return_value = 900.0  # 900mm < 1500mm threshold
    with patch.dict(sys.modules, {"depth": mock_depth}):
        detections = [{"class": "clear_path", "confidence": 0.80, "bbox": [0, 100, 100, 380]}]
        results = classify_compliance(detections, depth_map=depth_map, image_shape=(480, 640, 3))
    assert results[0]["status"] == "NON-COMPLIANT"


# ── classify_compliance — output structure ────────────────────────────────────

def test_result_contains_required_keys():
    detections = [{"class": "tgsi", "confidence": 0.9, "bbox": [10, 10, 100, 50]}]
    results = classify_compliance(detections)
    for key in ("class", "clause", "requirement", "confidence", "status", "note"):
        assert key in results[0], f"Missing key '{key}' in result"


def test_unknown_class_skipped():
    detections = [{"class": "unknown_feature", "confidence": 0.9, "bbox": [0, 0, 100, 100]}]
    results = classify_compliance(detections)
    assert results == []


def test_confidence_rounded_to_two_decimal_places():
    detections = [{"class": "tgsi", "confidence": 0.91234, "bbox": [0, 0, 50, 50]}]
    results = classify_compliance(detections)
    assert results[0]["confidence"] == 0.91


def test_multiple_detections_all_processed():
    detections = [
        {"class": "tgsi",       "confidence": 0.9,  "bbox": [0, 0, 50, 50]},
        {"class": "handrail",   "confidence": 0.85, "bbox": [0, 0, 50, 200]},
        {"class": "obstruction","confidence": 0.7,  "bbox": [100, 100, 200, 200]},
    ]
    results = classify_compliance(detections)
    assert len(results) == 3
