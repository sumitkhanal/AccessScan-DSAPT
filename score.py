"""
score.py — Compliance score calculator
Sprint 3 | AccessScan
"""


def compute_score(compliance_results: list) -> float:
    """
    Compute overall compliance score (0–100%).

    Scoring:
      COMPLIANT     = 2 points
      NEEDS REVIEW  = 1 point (partial credit)
      NON-COMPLIANT = 0 points
    """
    if not compliance_results:
        return 0.0

    points = 0
    max_points = len(compliance_results) * 2

    for result in compliance_results:
        if result['status'] == 'COMPLIANT':
            points += 2
        elif result['status'] == 'NEEDS REVIEW':
            points += 1
        # NON-COMPLIANT = 0

    return round((points / max_points) * 100, 1)


def score_label(score: float) -> str:
    """Return a human-readable label for the score."""
    if score >= 80:
        return 'GOOD'
    elif score >= 60:
        return 'MODERATE'
    elif score >= 40:
        return 'POOR'
    else:
        return 'CRITICAL'
