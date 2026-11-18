"""
AI Interviewer Platform — Professional Streamlit Application
Powered by LangChain and OpenAI (gpt-4o-mini)
"""
import os
import time
from datetime import datetime
import streamlit as st

from resume_parser import extract_resume_text
from pdf_generator import build_pdf_bytes, sanitize_filename, parse_evaluation_pdf
from interview_chain import (
    generate_first_question,
    generate_next_question,
    evaluate_interview,
    revision_coaching_turn,
    get_api_key,
)

# -----------------------------------------------------------------------------
# Page Configuration & Metadata
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="InterviewerAI — Executive Mock Interview Platform",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Enterprise Dark Theme & Custom Glassmorphic Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Base typography & dark luxury theme */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #f1f5f9;
        }

        .stApp {
            background: radial-gradient(circle at top right, #1e1b4b 0%, #0f172a 45%, #090d16 100%);
        }

        /* Glassmorphic Container Cards */
        .glass-card {
            background: rgba(15, 23, 42, 0.75);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        }

        .glass-card-subtle {
            background: rgba(30, 41, 59, 0.45);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 14px;
            padding: 18px;
            margin-bottom: 14px;
        }

        /* Header Hero Gradient */
        .gradient-title {
            background: linear-gradient(135deg, #a5b4fc 0%, #c084fc 50%, #f472b6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.3rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            line-height: 1.15;
            margin-bottom: 6px;
        }

        .gradient-badge {
            display: inline-flex;
            align-items: center;
            padding: 4px 12px;
            border-radius: 9999px;
            background: rgba(99, 102, 241, 0.12);
            border: 1px solid rgba(99, 102, 241, 0.3);
            color: #818cf8;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .metric-badge {
            background: #090d16;
            border: 1px solid #1e293b;
            border-radius: 12px;
            padding: 12px 16px;
            text-align: center;
        }

        .metric-val {
            font-size: 1.6rem;
            font-weight: 800;
            color: #60a5fa;
            line-height: 1.1;
        }

        .metric-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #94a3b8;
            margin-top: 4px;
        }

        /* Streamlit Input Styling overrides */
        .stTextInput input, .stTextArea textarea {
            background-color: #090d16 !important;
            border: 1px solid #334155 !important;
            border-radius: 12px !important;
            color: #f8fafc !important;
            font-family: inherit !important;
        }
        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: #6366f1 !important;
            box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25) !important;
        }

        /* Streamlit Buttons */
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 10px 24px !important;
            font-weight: 600 !important;
            letter-spacing: 0.01em !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3) !important;
        }
        div.stButton > button:first-child:hover {
            opacity: 0.94 !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px rgba(79, 70, 229, 0.45) !important;
        }

        /* Download button styling */
        div.stDownloadButton > button:first-child {
            background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 10px 24px !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3) !important;
        }

        /* Highlight Boxes */
        .highlight-expected {
            background: rgba(16, 185, 129, 0.08);
            border: 1px solid rgba(16, 185, 129, 0.25);
            border-radius: 12px;
            padding: 14px 16px;
            color: #a7f3d0;
            font-size: 0.92rem;
            line-height: 1.5;
            margin-top: 8px;
        }

        .highlight-critique {
            background: rgba(245, 158, 11, 0.08);
            border: 1px solid rgba(245, 158, 11, 0.25);
            border-radius: 12px;
            padding: 14px 16px;
            color: #fde68a;
            font-size: 0.92rem;
            line-height: 1.5;
            margin-top: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "stage" not in st.session_state:
    st.session_state.stage = "setup"  # "setup", "interview", "evaluation", "revision"

if "target_role" not in st.session_state:
    st.session_state.target_role = ""

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

if "turns" not in st.session_state:
    st.session_state.turns = []  # [{"role": "interviewer" | "candidate", "content": ..., "category": ...}]

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

if "total_questions" not in st.session_state:
    st.session_state.total_questions = 12

if "current_question" not in st.session_state:
    st.session_state.current_question = ""

if "current_category" not in st.session_state:
    st.session_state.current_category = "Resume Overview & Deep Dive"

if "interviewer_reaction" not in st.session_state:
    st.session_state.interviewer_reaction = None

if "evaluation_report" not in st.session_state:
    st.session_state.evaluation_report = None

if "revision_history" not in st.session_state:
    st.session_state.revision_history = []  # [{"role": "coach"|"user", "content": ..., "critique": ..., "improved": ...}]

if "revision_weak_area" not in st.session_state:
    st.session_state.revision_weak_area = ""


# -----------------------------------------------------------------------------
# Sidebar: System Status & API Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="gradient-badge">Executive AI Engine</div>', unsafe_allow_html=True)
    st.markdown("### 🎙️ InterviewerAI")
    st.caption("Adaptive 10–12 question mock interview platform with post-interview evaluation & revision coaching.")
    st.divider()

    # API Key Configuration
    st.markdown("#### 🔑 Model Configuration")
    env_key = get_api_key()
    has_env = bool(env_key)

    api_key_input = st.text_input(
        "OpenAI API Key",
        value=env_key if has_env else "",
        type="password",
        placeholder="sk-proj-...",
        help="Reads automatically from .env if configured.",
    )
    active_api_key = api_key_input.strip() or env_key

    if active_api_key:
        st.success("Connected to OpenAI (gpt-4o-mini)", icon="✅")
    else:
        st.info("Using simulated dynamic heuristic mode. Add key for live cloud LLM.", icon="ℹ️")

    st.divider()

    # Session Status indicator
    st.markdown("#### 📌 Session Overview")
    if st.session_state.target_role:
        st.markdown(f"**Target Role:** `{st.session_state.target_role}`")
        if st.session_state.stage == "interview":
            progress = min(1.0, st.session_state.question_count / st.session_state.total_questions)
            st.progress(progress)
            st.caption(f"Question {st.session_state.question_count} of {st.session_state.total_questions} ({int(progress * 100)}%)")
        elif st.session_state.stage == "evaluation":
            st.markdown("Status: **Evaluation Complete** 🏁")
        elif st.session_state.stage == "revision":
            st.markdown(f"Focus Area: **{st.session_state.revision_weak_area or 'Coaching Active'}** 🎯")
    else:
        st.caption("No interview in progress. Complete setup to begin.")

    st.divider()
    if st.button("🔄 Reset / Start New Interview", use_container_width=True):
        st.session_state.stage = "setup"
        st.session_state.target_role = ""
        st.session_state.resume_text = ""
        st.session_state.turns = []
        st.session_state.question_count = 0
        st.session_state.current_question = ""
        st.session_state.evaluation_report = None
        st.session_state.revision_history = []
        st.session_state.revision_weak_area = ""
        st.rerun()


# -----------------------------------------------------------------------------
# Top Navigation Bar / Tabs
# -----------------------------------------------------------------------------
tab_interview, tab_eval, tab_revision = st.tabs([
    "🎙️ 1. Mock Interview Arena",
    "📊 2. Performance Evaluation & PDF",
    "🎯 3. Revision & Coaching Lab",
])


# =============================================================================
# TAB 1: MOCK INTERVIEW ARENA
# =============================================================================
with tab_interview:
    # -------------------------------------------------------------------------
    # Sub-View A: Intake & Setup
    # -------------------------------------------------------------------------
    if st.session_state.stage == "setup":
        st.markdown('<div class="gradient-badge">Adaptive Real-Life Simulation</div>', unsafe_allow_html=True)
        st.markdown('<div class="gradient-title">Master Your Next Technical & Behavioral Interview</div>', unsafe_allow_html=True)
        st.markdown(
            "Upload your resume and choose your target position. Our AI interviewer dynamically constructs a personalized, "
            "**10 to 12 question live interview**—probing your actual project architecture, questioning edge cases, and testing STAR behavioral competencies."
        )
        st.write("")

        col_left, col_right = st.columns([1.1, 0.9], gap="large")

        with col_left:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 1️⃣ Candidate Resume Upload")
            st.caption("Supported formats: PDF, DOCX, TXT (up to 10 MB)")

            uploaded_file = st.file_uploader(
                "Upload Resume Document",
                type=["pdf", "docx", "txt"],
                label_visibility="collapsed",
            )

            st.write("")
            st.markdown("#### 2️⃣ Target Position")
            st.caption("Specify the exact title or level you are interviewing for")

            role_input = st.text_input(
                "Target Role",
                placeholder="e.g. Senior Backend Engineer (Python / FastAPI / Distributed Systems)",
                label_visibility="collapsed",
            )

            # Quick role chips
            st.caption("Quick Select Suggestions:")
            quick_cols = st.columns(3)
            with quick_cols[0]:
                if st.button("Senior Full Stack", use_container_width=True):
                    role_input = "Senior Full Stack Engineer"
            with quick_cols[1]:
                if st.button("Data Scientist / AI", use_container_width=True):
                    role_input = "Data Scientist & ML Engineer"
            with quick_cols[2]:
                if st.button("Cloud / DevOps", use_container_width=True):
                    role_input = "Cloud & DevOps Architect"

            st.write("")
            start_btn = st.button("🚀 Begin Realistic Mock Interview", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if start_btn:
                if not uploaded_file:
                    st.error("Please upload your resume to begin.")
                elif not role_input.strip():
                    st.error("Please specify your target role.")
                else:
                    with st.spinner("Analyzing resume highlights and formulating Question 1..."):
                        try:
                            file_bytes = uploaded_file.read()
                            resume_txt = extract_resume_text(file_bytes, uploaded_file.name)
                            st.session_state.resume_text = resume_txt
                            st.session_state.target_role = role_input.strip()

                            q1, cat1 = generate_first_question(resume_txt, role_input.strip(), active_api_key)
                            st.session_state.current_question = q1
                            st.session_state.current_category = cat1
                            st.session_state.question_count = 1
                            st.session_state.turns = [
                                {"role": "interviewer", "content": q1, "category": cat1}
                            ]
                            st.session_state.stage = "interview"
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to start interview: {e}")

        with col_right:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### ⚡ Platform Features")
            st.markdown(
                """
                - **Dynamic 10–12 Question Flow**: Questions are never static. The AI listens to your responses and probes trade-offs, scaling limits, and architectural choices.
                - **STAR Behavioral Scenarios**: Tests engineering ownership, cross-team conflict resolution, and handling high-pressure production outages.
                - **Tailored 10/10 Model Answers**: At completion, receive exact model answers correcting and elevating each of your responses.
                - **Downloadable PDF Report**: Official multi-page evaluation report formatted via ReportLab.
                - **Targeted Revision Lab**: Upload your PDF report to practice your weakest areas with iterative AI coaching.
                """
            )
            st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Sub-View B: Live Interview Arena
    # -------------------------------------------------------------------------
    elif st.session_state.stage == "interview":
        # Progress header card
        q_idx = st.session_state.question_count
        total_q = st.session_state.total_questions
        progress_val = min(1.0, q_idx / total_q)

        st.markdown(
            f"""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div>
                        <span class="gradient-badge">Live Simulation</span>
                        <h3 style="margin: 0; color: #ffffff; font-weight: 700;">{st.session_state.target_role}</h3>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 0.85rem; color: #94a3b8; font-weight: 600;">STAGE CATEGORY:</span>
                        <div style="color: #c084fc; font-weight: 700; font-size: 0.95rem;">{st.session_state.current_category}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(progress_val)
        st.caption(f"Progress: Question {q_idx} of {total_q} completed ({int(progress_val * 100)}%)")

        # Interviewer Question Card
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 14px;">
                <div style="width: 44px; height: 44px; border-radius: 12px; background: linear-gradient(135deg, #4f46e5, #9333ea); display: flex; align-items: center; justify-content: center; font-size: 1.3rem;">
                    🎙️
                </div>
                <div>
                    <div style="font-size: 0.88rem; font-weight: 700; color: #f8fafc;">Principal Interviewer</div>
                    <div style="font-size: 0.75rem; color: #94a3b8;">Hiring Board • Technical & Behavioral Review</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.interviewer_reaction:
            st.markdown(
                f"""
                <div style="padding: 10px 14px; background: rgba(99, 102, 241, 0.08); border-left: 3px solid #6366f1; border-radius: 8px; margin-bottom: 14px; font-size: 0.85rem; color: #c7d2fe; font-style: italic;">
                    <strong>Interviewer:</strong> "{st.session_state.interviewer_reaction}"
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div style="background: rgba(9, 13, 22, 0.7); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 14px; padding: 20px; font-size: 1.15rem; font-weight: 500; color: #f8fafc; line-height: 1.6;">
                {st.session_state.current_question}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Candidate Answer Form
        with st.form(key=f"answer_form_{q_idx}"):
            st.markdown("#### ✍️ Your Response")
            if "STAR" in st.session_state.current_category:
                st.caption("💡 Tip: Structure your answer using the STAR format (Situation, Task, Action, Result).")
            else:
                st.caption("💡 Tip: Highlight specific technical tools, architectural decisions, and quantifiable outcomes.")

            answer_input = st.text_area(
                "Candidate Response",
                height=160,
                placeholder="Type your response thoroughly. Detail your thought process, architectural choices, and the measurable results...",
                label_visibility="collapsed",
            )

            col_sub, col_end = st.columns([1, 1])
            with col_sub:
                submit_answer_btn = st.form_submit_button("Submit Response ➔", use_container_width=True)
            with col_end:
                end_early_btn = st.form_submit_button("🏁 Conclude & Evaluate Now", use_container_width=True)

        if submit_answer_btn:
            if not answer_input.strip():
                st.warning("Please type your answer before submitting.")
            else:
                # Append candidate turn
                st.session_state.turns.append({"role": "candidate", "content": answer_input.strip()})

                with st.spinner("Analyzing your response & generating next interview question..."):
                    next_q, cat, reaction = generate_next_question(
                        turns=st.session_state.turns,
                        target_role=st.session_state.target_role,
                        resume_text=st.session_state.resume_text,
                        question_count=st.session_state.question_count,
                        total_questions=st.session_state.total_questions,
                        api_key=active_api_key,
                    )

                    if next_q is None:
                        # Interview complete! Generate evaluation
                        with st.spinner("Generating your comprehensive performance evaluation & model answers..."):
                            report = evaluate_interview(
                                target_role=st.session_state.target_role,
                                resume_text=st.session_state.resume_text,
                                turns=st.session_state.turns,
                                api_key=active_api_key,
                            )
                            st.session_state.evaluation_report = report
                            st.session_state.stage = "evaluation"
                            st.rerun()
                    else:
                        st.session_state.current_question = next_q
                        st.session_state.current_category = cat
                        st.session_state.interviewer_reaction = reaction
                        st.session_state.question_count += 1
                        st.session_state.turns.append({"role": "interviewer", "content": next_q, "category": cat})
                        st.rerun()

        if end_early_btn:
            with st.spinner("Concluding interview and generating performance evaluation..."):
                if answer_input.strip():
                    st.session_state.turns.append({"role": "candidate", "content": answer_input.strip()})
                report = evaluate_interview(
                    target_role=st.session_state.target_role,
                    resume_text=st.session_state.resume_text,
                    turns=st.session_state.turns,
                    api_key=active_api_key,
                )
                st.session_state.evaluation_report = report
                st.session_state.stage = "evaluation"
                st.rerun()

        # Collapsible Transcript History
        if len(st.session_state.turns) > 1:
            with st.expander(f"📜 View Full Dialogue History ({len(st.session_state.turns)} turns)"):
                for turn in st.session_state.turns:
                    if turn.get("role") == "interviewer":
                        st.markdown(f"**🎙️ Interviewer ({turn.get('category', 'Question')}):**")
                        st.info(turn.get("content", ""))
                    else:
                        st.markdown("**👤 You (Candidate):**")
                        st.write(turn.get("content", ""))
                    st.divider()

    # If stage is already evaluated, prompt to switch tabs
    elif st.session_state.stage in ("evaluation", "revision"):
        st.info("Interview session complete! Review your results in the **Performance Evaluation & PDF** tab or practice in the **Revision & Coaching Lab**.", icon="🎉")


# =============================================================================
# TAB 2: PERFORMANCE EVALUATION & PDF REPORT
# =============================================================================
with tab_eval:
    report = st.session_state.evaluation_report

    if not report:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📊 No Evaluation Report Generated Yet")
        st.caption("Complete a mock interview in the first tab to view your scores, transcript logs, and downloadable PDF report.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="gradient-badge">Executive Hiring Evaluation</div>', unsafe_allow_html=True)
        st.markdown('<div class="gradient-title">Performance Scorecard & Detailed Review</div>', unsafe_allow_html=True)
        st.caption(f"Role: **{report.get('target_role')}** • Evaluation Date: **{report.get('session_date')}**")
        st.write("")

        # Competency Score Metrics Bar
        scores = report.get("scores", {})
        m_cols = st.columns(5)
        with m_cols[0]:
            st.markdown(
                f'<div class="metric-badge"><div class="metric-val" style="color: #818cf8;">{scores.get("overall", 80)}</div><div class="metric-label">Overall Score</div></div>',
                unsafe_allow_html=True,
            )
        with m_cols[1]:
            st.markdown(
                f'<div class="metric-badge"><div class="metric-val">{scores.get("technical", 80)}</div><div class="metric-label">Technical Depth</div></div>',
                unsafe_allow_html=True,
            )
        with m_cols[2]:
            st.markdown(
                f'<div class="metric-badge"><div class="metric-val" style="color: #c084fc;">{scores.get("behavioral", 80)}</div><div class="metric-label">STAR Behavioral</div></div>',
                unsafe_allow_html=True,
            )
        with m_cols[3]:
            st.markdown(
                f'<div class="metric-badge"><div class="metric-val" style="color: #f472b6;">{scores.get("communication", 80)}</div><div class="metric-label">Communication</div></div>',
                unsafe_allow_html=True,
            )
        with m_cols[4]:
            st.markdown(
                f'<div class="metric-badge"><div class="metric-val" style="color: #34d399;">{scores.get("problem_solving", 80)}</div><div class="metric-label">Problem Solving</div></div>',
                unsafe_allow_html=True,
            )

        st.write("")

        # PDF Download & Next Action Row
        pdf_bytes = build_pdf_bytes(report)
        pdf_filename = sanitize_filename(report.get("target_role", "Candidate"), report.get("session_date", "session"))

        action_col1, action_col2 = st.columns([1, 1])
        with action_col1:
            st.download_button(
                label="📥 Download Official PDF Evaluation Report",
                data=pdf_bytes,
                file_name=pdf_filename,
                mime="application/pdf",
                use_container_width=True,
            )
        with action_col2:
            if st.button("🎯 Open Revision Lab with this Report", use_container_width=True):
                st.session_state.stage = "revision"
                weak = report.get("weak_areas", ["Technical Trade-offs"])[0] if report.get("weak_areas") else "Technical Depth"
                st.session_state.revision_weak_area = weak
                st.rerun()

        st.write("")

        # Overall Feedback Card
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🧠 Executive Assessment Summary")
        st.write(report.get("overall_feedback", ""))
        st.markdown('</div>', unsafe_allow_html=True)

        # Strong vs Weak Areas Columns
        c_strong, c_weak = st.columns(2, gap="medium")
        with c_strong:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🌟 Key Demonstrated Strengths")
            for sa in report.get("strong_areas", []):
                st.markdown(f"• **{sa}**")
            st.markdown('</div>', unsafe_allow_html=True)

        with c_weak:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🔍 Targeted Areas for Improvement")
            for wa in report.get("weak_areas", []):
                st.markdown(f"• <span style='color: #fde68a;'>**{wa}**</span>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Question by Question Detailed Log & Corrected Model Answers
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📖 Full Communication Transcript & Tailored Model Answers")
        st.caption("Inspect each question asked, your response, evaluator feedback, and the upgraded 10/10 model answer.")

        q_evals = report.get("question_evaluations", [])
        if q_evals:
            for qe in q_evals:
                st.markdown(f"##### Q{qe.get('question_number', 1)}: {qe.get('question', '')}")
                st.markdown(f"**Score:** `{qe.get('score', 7)}/10`")
                st.markdown(f"**Your Answer:** {qe.get('candidate_answer', '')}")
                st.markdown(f"**Feedback:** {qe.get('feedback', '')}")
                st.markdown(
                    f"""
                    <div class="highlight-expected">
                        <strong>✨ Expected Model Answer (10/10):</strong><br>
                        {qe.get('expected_answer', '')}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.divider()
        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 3: REVISION & COACHING LAB
# =============================================================================
with tab_revision:
    st.markdown('<div class="gradient-badge">Post-Interview Coaching</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-title">Targeted Revision & Answer Refinement Lab</div>', unsafe_allow_html=True)
    st.markdown(
        "Upload your downloaded interview PDF report (or load from your current session) to inspect model answers, "
        "identify core technical gaps, and practice drilling your weak areas with instant AI answer refinement."
    )
    st.write("")

    # If no report loaded in revision yet: Upload or load option
    has_report = st.session_state.evaluation_report is not None

    if not has_report:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 📂 Load Evaluation Report to Begin Revision")
        st.caption("Upload the PDF evaluation report downloaded from your interview.")

        uploaded_pdf = st.file_uploader(
            "Upload Downloaded PDF Report",
            type=["pdf"],
            key="revision_pdf_uploader",
        )

        if uploaded_pdf:
            try:
                pdf_bytes = uploaded_pdf.read()
                parsed = parse_evaluation_pdf(pdf_bytes)
                st.session_state.evaluation_report = parsed
                st.session_state.revision_weak_area = parsed.get("weak_areas", ["Technical Depth"])[0] if parsed.get("weak_areas") else "Technical Depth"
                st.success("PDF Evaluation Report parsed successfully!", icon="✅")
                st.rerun()
            except Exception as e:
                st.error(f"Could not parse uploaded PDF: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        rev_rep = st.session_state.evaluation_report

        # Top summary card
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        col_r1, col_r2 = st.columns([1.2, 0.8])
        with col_r1:
            st.markdown(f"### 🎯 Coaching Target: `{rev_rep.get('target_role')}`")
            st.markdown(f"**Primary Focus Area:** <span style='color: #fde68a;'>**{st.session_state.revision_weak_area}**</span>", unsafe_allow_html=True)
            st.caption("The coach will ask questions targeting this weak area and rewrite your practice answers to a 10/10 standard.")
        with col_r2:
            st.markdown("#### 🔍 Areas to Improve")
            for w in rev_rep.get("weak_areas", []):
                st.markdown(f"• **{w}**")
        st.markdown('</div>', unsafe_allow_html=True)

        # Collapsible Model Answers Accordion
        with st.expander("📖 View All Expected Model Answers from Your Interview"):
            for qe in rev_rep.get("question_evaluations", []):
                st.markdown(f"**Q{qe.get('question_number')}: {qe.get('question')}**")
                st.markdown(
                    f"""
                    <div class="highlight-expected">
                        <strong>Expected Model Answer:</strong><br>
                        {qe.get('expected_answer')}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.write("")

        # Interactive Coaching Chat Container
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 💬 Interactive Practice Drill")

        # Initial prompt if revision history is empty
        if not st.session_state.revision_history:
            initial_prompt = (
                f"Welcome to your revision session for **{rev_rep.get('target_role')}**! "
                f"Based on your evaluation report, we are targeting your improvement area: **{st.session_state.revision_weak_area}**.\n\n"
                f"**Practice Question 1:** \"Describe a recent technical challenge or architectural trade-off in your projects. "
                f"Specifically detail the performance metrics before/after, alternative solutions considered, and your exact technical role.\""
            )
            st.session_state.revision_history.append({"role": "coach", "content": initial_prompt})

        # Render conversation history
        for msg in st.session_state.revision_history:
            if msg.get("role") == "coach":
                st.markdown(
                    f"""
                    <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(147, 51, 234, 0.25); border-radius: 14px; padding: 18px; margin-bottom: 14px;">
                        <div style="font-weight: 700; color: #c084fc; font-size: 0.85rem; margin-bottom: 6px;">🎯 AI Interview Coach</div>
                        <div style="color: #f1f5f9; font-size: 0.95rem; line-height: 1.5;">{msg.get('content')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if msg.get("critique"):
                    st.markdown(
                        f"""
                        <div class="highlight-critique">
                            <strong>💡 Feedback & Critique:</strong><br>
                            {msg.get('critique')}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                if msg.get("improved"):
                    st.markdown(
                        f"""
                        <div class="highlight-expected">
                            <strong>✨ Polished 10/10 Refined Answer:</strong><br>
                            {msg.get('improved')}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    f"""
                    <div style="background: rgba(79, 70, 229, 0.15); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 14px; padding: 18px; margin-bottom: 14px; margin-left: 10%;">
                        <div style="font-weight: 700; color: #818cf8; font-size: 0.85rem; margin-bottom: 6px;">👤 You (Practice Response)</div>
                        <div style="color: #f8fafc; font-size: 0.95rem; line-height: 1.5;">{msg.get('content')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Practice input form
        with st.form("practice_response_form"):
            practice_input = st.text_area(
                "Your Practice Answer",
                height=120,
                placeholder="Type your practice response. Focus on metrics, STAR structure, and concrete technical details...",
                label_visibility="collapsed",
            )
            submit_practice_btn = st.form_submit_button("Submit Practice Answer & Get 10/10 Refinement ➔", use_container_width=True)

        if submit_practice_btn:
            if not practice_input.strip():
                st.warning("Please type your practice answer.")
            else:
                st.session_state.revision_history.append({"role": "user", "content": practice_input.strip()})
                with st.spinner("AI Coach is evaluating your answer, rewriting it to a 10/10 standard, and formulating the next question..."):
                    coaching = revision_coaching_turn(
                        target_role=rev_rep.get("target_role", "Candidate"),
                        weak_area=st.session_state.revision_weak_area,
                        history=st.session_state.revision_history,
                        user_answer=practice_input.strip(),
                        api_key=active_api_key,
                    )

                    st.session_state.revision_history.append({
                        "role": "coach",
                        "content": f"**Next Practice Question:**\n\n{coaching.get('next_question')}",
                        "critique": coaching.get("critique"),
                        "improved": coaching.get("improved_answer"),
                    })
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
