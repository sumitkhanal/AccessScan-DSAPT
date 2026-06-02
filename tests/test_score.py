"""
tests/test_score.py — Unit tests for score.py
"""
import pytest
from score import compute_score, score_label


# ── compute_score ─────────────────────────────────────────────────────────────

def test_empty_results_returns_zero():
    assert compute_score([]) == 0.0


def test_all_compliant_returns_100():
    results = [{"status": "COMPLIANT"}, {"status": "COMPLIANT"}]
    assert compute_score(results) == 100.0


def test_all_non_compliant_returns_zero():
    results = [{"status": "NON-COMPLIANT"}, {"status": "NON-COMPLIANT"}]
    assert compute_score(results) == 0.0


def test_all_needs_review_returns_50():
    results = [{"status": "NEEDS REVIEW"}, {"status": "NEEDS REVIEW"}]
    assert compute_score(results) == 50.0


def test_mixed_results_correct_score():
    # 1 COMPLIANT (2pts) + 1 NEEDS REVIEW (1pt) + 1 NON-COMPLIANT (0pts) = 3/6 = 50%
    results = [
        {"status": "COMPLIANT"},
        {"status": "NEEDS REVIEW"},
        {"status": "NON-COMPLIANT"},
    ]
    assert compute_score(results) == 50.0


def test_single_compliant():
    assert compute_score([{"status": "COMPLIANT"}]) == 100.0


def test_single_non_compliant():
    assert compute_score([{"status": "NON-COMPLIANT"}]) == 0.0


def test_score_is_rounded_to_one_decimal():
    # 2 COMPLIANT (4pts) + 1 NEEDS REVIEW (1pt) = 5/6 = 83.3%
    results = [{"status": "COMPLIANT"}, {"status": "COMPLIANT"}, {"status": "NEEDS REVIEW"}]
    score = compute_score(results)
    assert score == 83.3


# ── score_label ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("score,expected", [
    (100.0, "GOOD"),
    (80.0,  "GOOD"),
    (79.9,  "MODERATE"),
    (60.0,  "MODERATE"),
    (59.9,  "POOR"),
    (40.0,  "POOR"),
    (39.9,  "CRITICAL"),
    (0.0,   "CRITICAL"),
])
def test_score_label_thresholds(score, expected):
    assert score_label(score) == expected
