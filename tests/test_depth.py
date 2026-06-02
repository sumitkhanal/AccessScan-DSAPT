"""
tests/test_depth.py — Unit tests for depth.py helper functions.

torch and cv2 are mocked so these tests run without GPU/model weights.
The MiDaS model itself is NOT tested here — only the pure numpy helper logic
(estimate_gradient, estimate_width_mm) and the lazy-loading guard.
"""
import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


def _import_depth_with_mock_torch():
    """Import depth.py with torch and cv2 replaced by MagicMocks."""
    mock_torch = MagicMock()
    mock_cv2  = MagicMock()
    # Remove cached real module if present
    for mod in list(sys.modules.keys()):
        if mod in ("depth", "torch", "torchvision"):
            del sys.modules[mod]
    with patch.dict(sys.modules, {"torch": mock_torch, "cv2": mock_cv2}):
        import depth as _d
    return _d


@pytest.fixture(scope="module")
def depth_mod():
    return _import_depth_with_mock_torch()


# ── estimate_gradient ─────────────────────────────────────────────────────────

def test_estimate_gradient_flat_surface_returns_near_zero(depth_mod):
    depth_map = np.ones((480, 640), dtype=np.float32) * 100.0
    gradient = depth_mod.estimate_gradient([100, 50, 500, 430], depth_map)
    assert gradient == pytest.approx(0.0, abs=0.1)


def test_estimate_gradient_steep_ramp(depth_mod):
    depth_map = np.zeros((480, 640), dtype=np.float32)
    depth_map[:240, :] = 900.0
    depth_map[240:, :] = 10.0
    gradient = depth_mod.estimate_gradient([0, 0, 640, 480], depth_map)
    assert gradient > 4.0  # exceeds the 4° DSAPT threshold


def test_estimate_gradient_empty_bbox_returns_zero(depth_mod):
    depth_map = np.ones((480, 640), dtype=np.float32)
    assert depth_mod.estimate_gradient([100, 100, 100, 100], depth_map) == 0.0


def test_estimate_gradient_out_of_bounds_bbox_clamped(depth_mod):
    depth_map = np.ones((480, 640), dtype=np.float32) * 100.0
    # Bbox extends beyond image dimensions — should clamp without error
    gradient = depth_mod.estimate_gradient([-50, -50, 9999, 9999], depth_map)
    assert isinstance(gradient, float)


def test_estimate_gradient_returns_float(depth_mod):
    depth_map = np.ones((480, 640), dtype=np.float32) * 50.0
    result = depth_mod.estimate_gradient([10, 10, 200, 300], depth_map)
    assert isinstance(result, float)


# ── estimate_width_mm ─────────────────────────────────────────────────────────

def test_estimate_width_mm_full_image_width_reasonable(depth_mod):
    # Two distinct depth values so normalisation doesn't collapse
    depth_map = np.zeros((480, 640), dtype=np.float32)
    depth_map[:, :320] = 50.0
    depth_map[:, 320:] = 100.0
    width = depth_mod.estimate_width_mm([0, 100, 640, 380], depth_map, (480, 640, 3))
    assert width > 0
    assert isinstance(width, float)


def test_estimate_width_mm_empty_bbox_returns_zero(depth_mod):
    depth_map = np.ones((480, 640), dtype=np.float32) * 50.0
    assert depth_mod.estimate_width_mm([100, 100, 100, 100], depth_map, (480, 640, 3)) == 0.0


def test_estimate_width_mm_uniform_depth_map_returns_zero(depth_mod):
    # Uniform depth → d_max == d_min → cannot normalise → returns 0.0
    depth_map = np.ones((480, 640), dtype=np.float32) * 50.0
    result = depth_mod.estimate_width_mm([0, 0, 320, 480], depth_map, (480, 640, 3))
    assert result == 0.0


def test_estimate_width_mm_narrow_vs_wide_bbox(depth_mod):
    depth_map = np.zeros((480, 640), dtype=np.float32)
    depth_map[:, :320] = 50.0
    depth_map[:, 320:] = 100.0
    image_shape = (480, 640, 3)
    narrow = depth_mod.estimate_width_mm([0, 0, 100, 480], depth_map, image_shape)
    wide   = depth_mod.estimate_width_mm([0, 0, 600, 480], depth_map, image_shape)
    assert wide > narrow


def test_estimate_width_mm_returns_float(depth_mod):
    depth_map = np.zeros((480, 640), dtype=np.float32)
    depth_map[:, :320] = 30.0
    depth_map[:, 320:] = 80.0
    result = depth_mod.estimate_width_mm([0, 0, 200, 400], depth_map, (480, 640, 3))
    assert isinstance(result, float)


# ── lazy loading guard ────────────────────────────────────────────────────────

def test_depth_module_does_not_load_midas_on_import():
    """Importing depth.py must NOT call torch.hub.load at import time."""
    # Remove any cached depth module
    for key in list(sys.modules.keys()):
        if key == "depth":
            del sys.modules[key]

    hub_calls = []
    mock_torch = MagicMock()
    mock_torch.hub.load.side_effect = lambda *a, **kw: hub_calls.append(a) or MagicMock()
    mock_cv2 = MagicMock()

    with patch.dict(sys.modules, {"torch": mock_torch, "cv2": mock_cv2}):
        import depth  # noqa: F401

    assert hub_calls == [], (
        f"torch.hub.load was called at import time — lazy loading broken. Calls: {hub_calls}"
    )
