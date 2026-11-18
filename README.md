# ResumeGrounded AI 🎙️🤖

An intelligent, adaptive AI mock interview platform built with **LangChain**, **OpenAI (`gpt-4o-mini`)**, and a modern executive **Streamlit** dark interface.

The platform conducts realistic, non-static interviews tailored to the candidate's uploaded resume and target role, produces comprehensive scorecards with downloadable PDF reports, and provides an interactive **Revision & Coaching Lab** where candidates practice weak areas with instant 10/10 answer refinement.

---

## 📁 Simple & Clean Project Structure

Instead of dozens of nested folders, the entire project is organized into 4 straightforward Python modules:

```
Ai interviewer/
├── app.py                # Executive modern Streamlit interface
├── interview_chain.py    # LangChain orchestration (Dynamic Qs, Evaluation, Coaching)
├── pdf_generator.py      # ReportLab PDF report generation & parsing
├── resume_parser.py      # Resume text extractor (PDF, DOCX, TXT)
├── .env                  # OpenAI API Key configuration
├── requirements.txt      # Python dependencies
└── start.bat / run.ps1   # 1-click launch script
```

---

## ✨ Core Features

1. **Non-Static Real-Life Interview Flow (10 to 12 Questions)**:
   - **Resume-Grounded Questions**: Drills into specific projects, tools, databases, and metrics from the candidate's actual resume.
   - **Dynamic Progression**: Adapts questions based on candidate answers across key interview stages:
     - *Resume Overview & Deep Dive* (Opening architectural walkthrough)
     - *Technical Depth & Architecture* (Scaling, latency, bottlenecks, consistency)
     - *Technical Probing & Edge Cases* (Challenging assumptions & deep follow-ups)
     - *Behavioral Competencies (STAR)* (Ownership, team disagreements, production outages)
     - *Role-Specific Scenarios & Leadership Retrospective*

2. **Performance Evaluation & Downloadable PDF Report**:
   - Executive scorecard with 5 core metrics (**Overall**, **Technical**, **Behavioral STAR**, **Communication**, **Problem Solving**).
   - Identified **Strong Areas** and **Weak Areas**.
   - **Full Communication Transcript Logs**: Every question asked, candidate's actual answer, evaluator score, feedback, and **Tailored 10/10 Expected Model Answer**.
   - One-click **Download Official PDF Report** formatted with ReportLab (`[Target_Role]_[YYYY-MM-DD]_evaluation.pdf`).

3. **Revision & Coaching Lab (Targeted Weak-Area Practice)**:
   - Upload any downloaded interview PDF report (or load directly from the current session).
   - **Executive Gap Analysis**: Clear checklist of where to improve.
   - **Expected Model Answer Bank**: Expandable library of ideal answers for all interview questions.
   - **Targeted Practice Drills**: Focuses on the candidate's weakest area.
   - **Iterative 10/10 Answer Refinement**: After the candidate submits a practice answer, the AI Coach evaluates it, highlights what was missing, and provides a **rewritten, polished 10/10 version** before posing the next drill question.

---

## 🚀 How to Run (1-Click)

### 1. Configure Your OpenAI API Key
Open [.env](file:///c:/Users/Shivraj/Desktop/Placement%20Material/Projects/Ai%20interviewer/.env) and ensure your key is set:
```env
OPENAI_API_KEY=sk-proj-your_key_here
OPENAI_MODEL=gpt-4o-mini
```

### 2. Launch the Application
Simply double-click **`start.bat`** (or run `.\run.ps1` in PowerShell):
```powershell
python -m streamlit run app.py
```

The app will open in your browser at **http://localhost:8501**.
