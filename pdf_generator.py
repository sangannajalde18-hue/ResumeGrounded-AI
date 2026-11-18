"""
PDF Generation & Parsing Service using ReportLab & PyPDF.
"""
import io
import re
from datetime import datetime
from typing import Dict, List, Any

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from pypdf import PdfReader


def sanitize_filename(role: str, date_str: str) -> str:
    """Produces filename like [Target_Role]_[YYYY-MM-DD]_evaluation.pdf"""
    clean_role = re.sub(r"[^a-zA-Z0-9_-]", "_", role.strip())
    clean_role = re.sub(r"_+", "_", clean_role).strip("_") or "Candidate"
    return f"{clean_role}_{date_str}_evaluation.pdf"


def build_pdf_bytes(report: Dict[str, Any]) -> bytes:
    """Builds a formatted multi-page PDF evaluation report from report dictionary."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#0f172a"),
        alignment=TA_CENTER,
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#64748b"),
        alignment=TA_CENTER,
        spaceAfter=15,
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=14,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
    )
    bullet_style = ParagraphStyle(
        "ReportBullet",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1e293b"),
        leftIndent=15,
        spaceAfter=4,
    )
    q_title_style = ParagraphStyle(
        "QTitle",
        parent=styles["Heading3"],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#2563eb"),
        spaceBefore=8,
        spaceAfter=4,
    )
    model_ans_style = ParagraphStyle(
        "ModelAns",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#047857"),
        spaceAfter=6,
    )

    story = []

    # Title & Header
    role = report.get("target_role") or "Software Engineer"
    date_val = report.get("session_date") or datetime.now().strftime("%Y-%m-%d")

    story.append(Paragraph("AI Mock Interview Performance Report", title_style))
    story.append(Paragraph(f"Target Role: <b>{role}</b> &nbsp;|&nbsp; Date: <b>{date_val}</b>", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#cbd5e1"), spaceAfter=12))

    # Score Summary Table
    scores = report.get("scores", {})
    score_data = [
        [
            Paragraph("<b>Overall Score</b>", body_style),
            Paragraph(f"<b>{scores.get('overall', 75)}/100</b>", body_style),
            Paragraph("<b>Technical Depth</b>", body_style),
            Paragraph(f"<b>{scores.get('technical', 75)}/100</b>", body_style),
        ],
        [
            Paragraph("<b>Behavioral (STAR)</b>", body_style),
            Paragraph(f"<b>{scores.get('behavioral', 75)}/100</b>", body_style),
            Paragraph("<b>Communication</b>", body_style),
            Paragraph(f"<b>{scores.get('communication', 75)}/100</b>", body_style),
        ],
    ]
    t = Table(score_data, colWidths=[130, 80, 130, 80])
    t.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(t)
    story.append(Spacer(1, 14))

    # Overall Feedback
    story.append(Paragraph("Overall Feedback", section_heading))
    story.append(Paragraph(report.get("overall_feedback", "Performance evaluated successfully."), body_style))
    story.append(Spacer(1, 12))

    # Strong Areas
    story.append(Paragraph("Strong Areas", section_heading))
    strong_areas = report.get("strong_areas", [])
    if strong_areas:
        for sa in strong_areas:
            story.append(Paragraph(f"• {sa}", bullet_style))
    else:
        story.append(Paragraph("• Solid domain understanding demonstrated.", bullet_style))
    story.append(Spacer(1, 12))

    # Weak Areas
    story.append(Paragraph("Weak Areas", section_heading))
    weak_areas = report.get("weak_areas", [])
    if weak_areas:
        for wa in weak_areas:
            story.append(Paragraph(f"• {wa}", bullet_style))
    else:
        story.append(Paragraph("• Continue refining communication and metrics.", bullet_style))
    story.append(Spacer(1, 16))

    # Transcript Logs & Model Answers
    story.append(Paragraph("Interview Communication Logs & Model Answers", section_heading))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#e2e8f0"), spaceAfter=10))

    q_evals = report.get("question_evaluations", [])
    if q_evals:
        for qe in q_evals:
            q_elements = [
                Paragraph(f"<b>Question {qe.get('question_number', 1)}:</b> {qe.get('question', '')}", q_title_style),
                Paragraph(f"<b>Your Answer:</b> {qe.get('candidate_answer', '')}", body_style),
                Paragraph(f"<b>Score:</b> {qe.get('score', 7)}/10 &nbsp;|&nbsp; <b>Feedback:</b> {qe.get('feedback', '')}", body_style),
                Paragraph(f"<b>Expected Model Answer:</b> {qe.get('expected_answer', '')}", model_ans_style),
                Spacer(1, 8),
            ]
            story.append(KeepTogether(q_elements))
    else:
        story.append(Paragraph("No detailed question logs recorded.", body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def parse_evaluation_pdf(pdf_bytes: bytes) -> Dict[str, Any]:
    """Parses a generated evaluation PDF back into a structured dictionary."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    full_text = "\n".join([page.extract_text() or "" for page in reader.pages]).strip()

    if not full_text:
        raise ValueError("Uploaded PDF does not contain extractable text.")

    # Validate required sections
    if not ("Strong Areas" in full_text and "Weak Areas" in full_text):
        raise ValueError("Uploaded PDF is missing required evaluation sections (Strong/Weak Areas).")

    # Extract target role
    role_match = re.search(r"Target Role:\s*([^\n|]+)", full_text)
    target_role = role_match.group(1).strip() if role_match else "Software Engineer"

    # Extract date
    date_match = re.search(r"Date:\s*(\d{4}-\d{2}-\d{2})", full_text)
    session_date = date_match.group(1).strip() if date_match else datetime.now().strftime("%Y-%m-%d")

    # Overall feedback
    overall_match = re.search(r"Overall Feedback\s*\n+(.*?)\n+(?=Strong Areas|\Z)", full_text, re.DOTALL)
    overall_feedback = overall_match.group(1).strip() if overall_match else "Interview evaluated."

    # Strong areas
    strong_match = re.search(r"Strong Areas\s*\n+(.*?)\n+(?=Weak Areas|\Z)", full_text, re.DOTALL)
    strong_areas = [
        line.lstrip("•- *").strip()
        for line in (strong_match.group(1).strip() if strong_match else "").splitlines()
        if line.lstrip("•- *").strip()
    ]

    # Weak areas
    weak_match = re.search(r"Weak Areas\s*\n+(.*?)\n+(?=Interview Communication Logs|Scores|\Z)", full_text, re.DOTALL)
    weak_areas = [
        line.lstrip("•- *").strip()
        for line in (weak_match.group(1).strip() if weak_match else "").splitlines()
        if line.lstrip("•- *").strip()
    ]

    # Question logs
    question_evaluations = []
    q_blocks = re.findall(
        r"Question\s+(\d+):\s*(.*?)\n+Your Answer:\s*(.*?)\n+Score:\s*(\d+)/10\s*\|\s*Feedback:\s*(.*?)\n+Expected Model Answer:\s*(.*?)(?=\n+Question\s+\d+:|\Z)",
        full_text,
        re.DOTALL,
    )
    for q_match in q_blocks:
        question_evaluations.append({
            "question_number": int(q_match[0]),
            "question": q_match[1].strip(),
            "candidate_answer": q_match[2].strip(),
            "score": int(q_match[3]),
            "feedback": q_match[4].strip(),
            "expected_answer": q_match[5].strip(),
        })

    return {
        "target_role": target_role,
        "session_date": session_date,
        "overall_feedback": overall_feedback,
        "strong_areas": strong_areas or ["Solid technical background"],
        "weak_areas": weak_areas or ["Quantifying technical impact"],
        "scores": {"overall": 80, "technical": 80, "behavioral": 80, "communication": 80, "problem_solving": 80},
        "question_evaluations": question_evaluations,
    }
