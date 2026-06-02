"""
tests/test_dsapt_rules.py — Unit tests for dsapt_rules.py
"""
import pytest
from dsapt_rules import DSAPT_RULES

REQUIRED_KEYS = {"clause", "requirement", "check"}
VALID_CHECKS   = {"present", "non_compliant", "gradient", "width", "tgsi_adjacent"}
KNOWN_CLASSES  = {"tgsi", "ramp", "handrail", "clear_path", "signage",
                  "platform_edge", "stairs", "kerb_cut", "obstruction"}


def test_all_expected_classes_present():
    assert KNOWN_CLASSES == set(DSAPT_RULES.keys())


def test_each_rule_has_required_keys():
    for cls, rule in DSAPT_RULES.items():
        missing = REQUIRED_KEYS - set(rule.keys())
        assert not missing, f"Rule '{cls}' missing keys: {missing}"


def test_each_check_type_is_valid():
    for cls, rule in DSAPT_RULES.items():
        assert rule["check"] in VALID_CHECKS, \
            f"Rule '{cls}' has unknown check type: '{rule['check']}'"


def test_gradient_check_has_threshold():
    assert "threshold" in DSAPT_RULES["ramp"]
    assert isinstance(DSAPT_RULES["ramp"]["threshold"], (int, float))


def test_width_check_has_threshold():
    assert "threshold" in DSAPT_RULES["clear_path"]
    assert DSAPT_RULES["clear_path"]["threshold"] == 1500


def test_ramp_gradient_threshold_is_4_degrees():
    assert DSAPT_RULES["ramp"]["threshold"] == 4.0


def test_obstruction_is_non_compliant():
    assert DSAPT_RULES["obstruction"]["check"] == "non_compliant"


def test_platform_edge_requires_tgsi_adjacent():
    assert DSAPT_RULES["platform_edge"]["check"] == "tgsi_adjacent"


def test_kerb_cut_requires_tgsi_adjacent():
    assert DSAPT_RULES["kerb_cut"]["check"] == "tgsi_adjacent"


def test_all_clauses_reference_dsapt():
    for cls, rule in DSAPT_RULES.items():
        assert rule["clause"].startswith("DSAPT"), \
            f"Rule '{cls}' clause does not reference DSAPT: '{rule['clause']}'"


def test_all_requirements_are_non_empty():
    for cls, rule in DSAPT_RULES.items():
        assert len(rule["requirement"].strip()) > 10, \
            f"Rule '{cls}' has suspiciously short requirement text"
