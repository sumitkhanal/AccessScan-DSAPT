"""
report.py — PDF compliance report generator
Sprint 4 | AccessScan

Generates a structured PDF report using ReportLab.
Includes: cover info, annotated image, per-feature compliance table, score.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path

from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph,
    Image as RLImage, Spacer, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm

# ── Colour palette ────────────────────────────────────────────────────────────
COMPLIANT_COLOR     = colors.HexColor('#1a7a4a')
NON_COMPLIANT_COLOR = colors.HexColor('#c0392b')
NEEDS_REVIEW_COLOR  = colors.HexColor('#e67e22')
HEADER_COLOR        = colors.HexColor('#1a3a5c')
ROW_ALT_COLOR       = colors.HexColor('#f4f8fc')
ACCENT_COLOR        = colors.HexColor('#2980b9')

STATUS_COLORS = {
    'COMPLIANT': COMPLIANT_COLOR,
    'NON-COMPLIANT': NON_COMPLIANT_COLOR,
    'NEEDS REVIEW': NEEDS_REVIEW_COLOR,
}


def _status_color(status: str):
    return STATUS_COLORS.get(status, colors.black)


def generate_report(
    job_id: str,
    image_path: str,
    compliance_results: list,
    score: float,
    site_name: str = "Unknown Site",
) -> str:
    """
    Generate a PDF compliance report.

    Args:
        job_id:             Unique job identifier (used in filename).
        image_path:         Path to the uploaded/analysed image.
        compliance_results: Output from compliance.classify_compliance().
        score:              Overall score (0–100) from score.compute_score().
        site_name:          Optional site label for the report header.

    Returns:
        Path to the generated PDF file.
    """
    output_path = str(Path(tempfile.gettempdir()) / f"report_{job_id}.pdf")
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()
    elements = []

    # ── Cover header ─────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=22,
        textColor=HEADER_COLOR,
        spaceAfter=4,
    )
    sub_style = ParagraphStyle(
        'Sub',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.grey,
        spaceAfter=2,
    )
    elements.append(Paragraph('AccessScan — Accessibility Compliance Report', title_style))
    elements.append(Paragraph('Disability Standards for Accessible Public Transport (DSAPT)', sub_style))
    elements.append(Paragraph(
        f'Site: {site_name}  |  Job ID: {job_id}  |  Generated: {datetime.now().strftime("%d %b %Y %H:%M")}',
        sub_style
    ))
    elements.append(HRFlowable(width="100%", thickness=2, color=ACCENT_COLOR, spaceAfter=10))

    # ── Score badge ───────────────────────────────────────────────────────────
    from score import score_label
    label = score_label(score)
    score_color = COMPLIANT_COLOR if score >= 80 else (NEEDS_REVIEW_COLOR if score >= 50 else NON_COMPLIANT_COLOR)
    score_style = ParagraphStyle(
        'Score',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=score_color,
        spaceAfter=10,
    )
    elements.append(Paragraph(f'Overall Compliance Score: {score}%  [{label}]', score_style))

    # ── Analysed image ────────────────────────────────────────────────────────
    if os.path.exists(image_path):
        elements.append(Paragraph('Analysed Image', styles['Heading3']))
        elements.append(RLImage(image_path, width=160 * mm, height=100 * mm))
        elements.append(Spacer(1, 10))

    # ── Compliance table ──────────────────────────────────────────────────────
    elements.append(Paragraph('Feature Compliance Results', styles['Heading3']))
    elements.append(Spacer(1, 4))

    table_data = [['Feature', 'DSAPT Clause', 'Conf.', 'Status', 'Note']]
    for r in compliance_results:
        table_data.append([
            r['class'],
            r['clause'],
            f"{r.get('confidence', 0):.0%}",
            r['status'],
            r['note'],
        ])

    col_widths = [28 * mm, 32 * mm, 14 * mm, 30 * mm, 60 * mm]
    table = Table(table_data, colWidths=col_widths, repeatRows=1)

    # Base style
    table_style = [
        ('BACKGROUND', (0, 0), (-1, 0), HEADER_COLOR),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, ROW_ALT_COLOR]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d0d0d0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]

    # Colour-code status cells
    for row_idx, r in enumerate(compliance_results, start=1):
        c = _status_color(r['status'])
        table_style.append(('TEXTCOLOR', (3, row_idx), (3, row_idx), c))
        table_style.append(('FONTNAME', (3, row_idx), (3, row_idx), 'Helvetica-Bold'))

    table.setStyle(TableStyle(table_style))
    elements.append(table)
    elements.append(Spacer(1, 12))

    # ── Summary counts ────────────────────────────────────────────────────────
    compliant_n    = sum(1 for r in compliance_results if r['status'] == 'COMPLIANT')
    non_compliant_n = sum(1 for r in compliance_results if r['status'] == 'NON-COMPLIANT')
    review_n       = sum(1 for r in compliance_results if r['status'] == 'NEEDS REVIEW')

    summary_style = ParagraphStyle('Summary', parent=styles['Normal'], fontSize=9, spaceAfter=2)
    elements.append(Paragraph('Summary', styles['Heading3']))
    elements.append(Paragraph(f'✔ Compliant: {compliant_n}', summary_style))
    elements.append(Paragraph(f'✘ Non-Compliant: {non_compliant_n}', summary_style))
    elements.append(Paragraph(f'⚠ Needs Review: {review_n}', summary_style))

    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    elements.append(Spacer(1, 4))
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=7, textColor=colors.grey)
    elements.append(Paragraph(
        'This report is a proof-of-concept automated assessment only. '
        'Results marked NEEDS REVIEW require manual on-site verification by a qualified accessibility auditor. '
        'AccessScan — La Trobe University × DITRDCA | TRL 3 Proof of Concept',
        footer_style
    ))

    doc.build(elements)
    print(f"[report] PDF generated: {output_path}")
    return output_path
