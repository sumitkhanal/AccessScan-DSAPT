"""
compliance.py — DSAPT compliance classifier
Sprint 3 | AccessScan
"""

from dsapt_rules import DSAPT_RULES


def is_adjacent(bbox_a: list, bbox_b: list, tolerance: float = 0.15) -> bool:
    """
    Check whether two bounding boxes are spatially adjacent.
    tolerance is a fraction of the image — boxes within 15% of each other
    are considered adjacent.
    """
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    a_cx = (ax1 + ax2) / 2
    a_cy = (ay1 + ay2) / 2
    b_cx = (bx1 + bx2) / 2
    b_cy = (by1 + by2) / 2

    # Approximate: use fraction of bbox diagonal as tolerance
    diag = ((ax2 - ax1) ** 2 + (ay2 - ay1) ** 2) ** 0.5
    dist = ((a_cx - b_cx) ** 2 + (a_cy - b_cy) ** 2) ** 0.5

    return dist < max(diag * 2, 50)  # at least 50px tolerance


def classify_compliance(detections: list, depth_map=None, image_shape=None) -> list:
    """
    Classify each detected feature against DSAPT rules.

    Args:
        detections:   List of dicts with keys:
                        'class', 'confidence', 'bbox' ([x1,y1,x2,y2])
        depth_map:    2D numpy array from depth.py (optional).
                      Required for gradient-based checks (ramps) and
                      depth-assisted width estimation (clear_path).
        image_shape:  (height, width[, channels]) of the original image.
                      Required for width estimation.

    Returns:
        List of compliance result dicts, one per detection.
    """
    results = []

    for detection in detections:
        cls = detection['class']
        rule = DSAPT_RULES.get(cls)
        if not rule:
            continue

        check = rule['check']

        # ── Present / absence checks ─────────────────────────────────────────
        if check == 'present':
            status = 'COMPLIANT'
            note = f'{cls} detected — presence requirement met'

        elif check == 'non_compliant':
            status = 'NON-COMPLIANT'
            note = f'{cls} detected — this is an access barrier that must be resolved'

        # ── Gradient check (ramps) ───────────────────────────────────────────
        elif check == 'gradient':
            if depth_map is not None:
                from depth import estimate_gradient
                gradient = estimate_gradient(detection['bbox'], depth_map)
                threshold = rule.get('threshold', 4.0)
                if gradient <= threshold:
                    status = 'COMPLIANT'
                    note = f'Estimated gradient {gradient:.1f}° — within 1:14 limit'
                else:
                    status = 'NON-COMPLIANT'
                    note = f'Estimated gradient {gradient:.1f}° — exceeds 1:14 (4°) limit'
            else:
                status = 'NEEDS REVIEW'
                note = 'Gradient estimation unavailable — manual verification required'

        # ── Width check (clear_path) ─────────────────────────────────────────
        elif check == 'width':
            if depth_map is not None and image_shape is not None:
                from depth import estimate_width_mm
                width_mm = estimate_width_mm(detection['bbox'], depth_map, image_shape)
                threshold = rule.get('threshold', 1500)
                if width_mm >= threshold:
                    status = 'COMPLIANT'
                    note = (f'Estimated path width ~{width_mm:.0f} mm — '
                            f'meets {threshold} mm minimum (depth-assisted proxy)')
                elif width_mm > 0:
                    status = 'NON-COMPLIANT'
                    note = (f'Estimated path width ~{width_mm:.0f} mm — '
                            f'below {threshold} mm minimum (depth-assisted proxy; verify on-site)')
                else:
                    status = 'NEEDS REVIEW'
                    note = 'Width estimation inconclusive — manual on-site verification required'
            else:
                status = 'NEEDS REVIEW'
                note = 'Path width measurement requires depth data — manual verification required'

        # ── TGSI adjacency check (platform_edge, kerb_cut) ───────────────────
        elif check == 'tgsi_adjacent':
            tgsi_detections = [d for d in detections if d['class'] == 'tgsi']
            tgsi_near = any(
                is_adjacent(detection['bbox'], d['bbox'])
                for d in tgsi_detections
            )
            if tgsi_near:
                status = 'COMPLIANT'
                note = f'TGSI detected adjacent to {cls}'
            else:
                status = 'NON-COMPLIANT'
                note = f'No TGSI detected at {cls} — tactile indicator required'

        else:
            status = 'NEEDS REVIEW'
            note = 'Check type not implemented — manual verification required'

        results.append({
            'class': cls,
            'clause': rule['clause'],
            'requirement': rule['requirement'],
            'confidence': round(detection.get('confidence', 0.0), 2),
            'status': status,
            'note': note,
        })

    return results
