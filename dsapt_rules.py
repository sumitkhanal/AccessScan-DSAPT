"""
dsapt_rules.py — DSAPT compliance rules dictionary
Sprint 3 | AccessScan
"""

DSAPT_RULES = {
    'tgsi': {
        'clause': 'DSAPT Part 3.3',
        'requirement': 'Tactile ground surface indicators must be present at platform edges, ramp tops, and pedestrian crossings',
        'check': 'present'
    },
    'ramp': {
        'clause': 'DSAPT Part 5.2',
        'requirement': 'Ramp gradient must not exceed 1:14 (approx 4 degrees)',
        'check': 'gradient',
        'threshold': 4.0  # degrees
    },
    'handrail': {
        'clause': 'DSAPT Part 5.4',
        'requirement': 'Handrails required on both sides of ramps and stairs, height 865-1000mm',
        'check': 'present'
    },
    'clear_path': {
        'clause': 'DSAPT Part 4.1',
        'requirement': 'Clear path of travel minimum 1500mm wide',
        'check': 'width',
        'threshold': 1500  # mm
    },
    'signage': {
        'clause': 'DSAPT Part 6.1',
        'requirement': 'Accessible route signage must be present and visible',
        'check': 'present'
    },
    'platform_edge': {
        'clause': 'DSAPT Part 5.6',
        'requirement': 'Platform edge must have TGSI warning strip and contrast marking',
        'check': 'tgsi_adjacent'
    },
    'stairs': {
        'clause': 'DSAPT Part 5.3',
        'requirement': 'Stairs must have tactile indicators on step nosings and handrails on both sides',
        'check': 'present'
    },
    'kerb_cut': {
        'clause': 'DSAPT Part 4.2',
        'requirement': 'Kerb cuts must be present at pedestrian crossings with TGSI warning strip at base',
        'check': 'tgsi_adjacent'
    },
    'obstruction': {
        'clause': 'DSAPT Part 4.1',
        'requirement': 'Clear path must be free of obstructions — minimum 1500mm width',
        'check': 'non_compliant'
    },
}
