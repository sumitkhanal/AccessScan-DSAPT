"""
depth.py — MiDaS depth estimation module

"""

import math
import torch
import cv2
import numpy as np

# ── Lazy model cache — loaded on first call, not at import time ───────────────
_midas = None
_transform = None


def _load_midas():
    """Load MiDaS model on first use and cache it."""
    global _midas, _transform
    if _midas is None:
        print("[depth] Loading MiDaS model (first use)...")
        _midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
        _midas.eval()
        _transform = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True).small_transform
        print("[depth] MiDaS model loaded.")
    return _midas, _transform


def estimate_depth(image_path: str) -> np.ndarray:
    """
    Run MiDaS depth estimation on any image.

    Args:
        image_path: Path to the image file (jpg, png, etc.)
                    Previously this was hardcoded to 'ramp_photo.jpg'.
                    Now it accepts whatever path is passed — e.g. the
                    uploaded file saved to /tmp/<job_id>.jpg by main.py.

    Returns:
        depth_map: 2D numpy array of relative depth values (float32).
                   Values are NOT in metres — they are relative.
                   Higher value = closer to camera.
    """
    midas, transform = _load_midas()
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb)

    with torch.no_grad():
        depth = midas(input_tensor)

    depth_map = depth.squeeze().numpy()
    print(f"[depth] Depth range for {image_path}: {depth_map.min():.2f} to {depth_map.max():.2f}")
    return depth_map


def estimate_gradient(bbox: list, depth_map: np.ndarray) -> float:
    """
    Estimate ramp gradient (degrees) from depth variation across a bounding box.

    Args:
        bbox: [x1, y1, x2, y2] bounding box of the detected ramp.
        depth_map: 2D numpy depth array from estimate_depth().

    Returns:
        Estimated gradient in degrees. Returns 0.0 if bbox is invalid.

    Note:
        MiDaS gives relative depth — not metres. This estimate is a proxy.
        Use a known reference object (e.g., standard brick = 65mm) in frame
        to calibrate scale if absolute gradient is needed for compliance.
    """
    h, w = depth_map.shape
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Clamp to image bounds
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi = depth_map[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    # Depth gradient along vertical axis (top of ramp vs bottom)
    top_depth = float(np.mean(roi[:max(1, roi.shape[0] // 4), :]))
    bot_depth = float(np.mean(roi[-(roi.shape[0] // 4):, :]))
    depth_diff = abs(top_depth - bot_depth)

    # Estimate angle from depth difference relative to bbox height
    bbox_height_px = y2 - y1
    if bbox_height_px == 0:
        return 0.0

    # Rough proxy: depth_diff / bbox_height → tan(angle)
    gradient_deg = float(np.degrees(np.arctan(depth_diff / (bbox_height_px + 1e-6))))
    return round(gradient_deg, 2)


def estimate_width_mm(
    bbox: list,
    depth_map: np.ndarray,
    image_shape: tuple,
    hfov_deg: float = 60.0,
) -> float:
    """
    Estimate the real-world width (mm) of a detected bbox region.

    Uses the mean MiDaS depth value within the bbox as a relative distance
    proxy, combined with an assumed horizontal field-of-view to convert pixel
    width to approximate millimetres.

    Args:
        bbox:        [x1, y1, x2, y2] bounding box in pixels.
        depth_map:   2D numpy depth array from estimate_depth().
        image_shape: (height, width[, channels]) of the original image.
        hfov_deg:    Assumed horizontal FOV of the camera (default 60°).

    Returns:
        Estimated width in mm, or 0.0 if estimation is not possible.

    Note:
        MiDaS gives relative inverse depth — not metres. This is a proxy
        estimate only; results require on-site verification for compliance.
    """
    h, w = depth_map.shape
    img_h, img_w = image_shape[:2]

    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi = depth_map[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    mean_depth = float(np.mean(roi))
    d_min, d_max = float(depth_map.min()), float(depth_map.max())
    if d_max == d_min or mean_depth <= 0:
        return 0.0

    # MiDaS: higher value = closer. Normalise to [0,1] then invert for distance.
    norm_depth = (mean_depth - d_min) / (d_max - d_min)
    rel_distance = 1.0 - norm_depth          # 0 = very close, 1 = far

    # Map to a plausible physical range for public transport sites (0.5 m – 10 m)
    est_distance_m = 0.5 + rel_distance * 9.5

    # Real-world width per pixel using pinhole camera model
    hfov_rad = math.radians(hfov_deg)
    real_width_per_px_m = (2.0 * est_distance_m * math.tan(hfov_rad / 2.0)) / img_w

    bbox_width_px = x2 - x1
    real_width_mm = bbox_width_px * real_width_per_px_m * 1000.0
    return round(real_width_mm, 0)


# ── Quick test (run: python depth.py <image_path>) ───────────────────────────
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python depth.py <image_path>")
        print("Example: python depth.py ramp_photo.jpg")
        sys.exit(1)

    path = sys.argv[1]
    depth_map = estimate_depth(path)
    print(f"Depth shape: {depth_map.shape}")
    print(f"Min: {depth_map.min():.3f}  Max: {depth_map.max():.3f}  Mean: {depth_map.mean():.3f}")
