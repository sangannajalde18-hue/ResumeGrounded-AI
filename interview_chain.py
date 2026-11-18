"""
LangChain Orchestration Engine for AI Interviewer:
- Dynamic Question Generation (Resume grounded + STAR Behavioral)
- Comprehensive Evaluation with Tailored Model Answers
- Interactive Revision Coaching & Answer Upgrades
"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
logger = logging.getLogger(__name__)


def get_api_key(override_key: Optional[str] = None) -> str:
    """Retrieves OpenAI API key from override or environment."""
    return (override_key or "").strip() or os.getenv("OPENAI_API_KEY", "").strip()


def get_llm(api_key: Optional[str] = None, json_mode: bool = False, temperature: float = 0.4) -> Optional[ChatOpenAI]:
    """Initializes ChatOpenAI client via LangChain."""
    key = get_api_key(api_key)
    if not key:
        return None
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        return ChatOpenAI(
            model=model_name,
            api_key=key,
            temperature=temperature,
            timeout=70.0,
            model_kwargs=kwargs,
        )
    except Exception as e:
        logger.warning(f"Could not initialize ChatOpenAI: {e}")
        return None


def clean_json_text(text: str) -> str:
    """Cleans code fence markers from JSON strings."""
    t = text.strip()
    if t.startswith("```json"):
        t = t[7:]
    elif t.startswith("```"):
        t = t[3:]
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


def determine_category_for_index(index: int, total: int = 12) -> str:
    """Returns stage category based on current question progress."""
    if index == 1:
        return "Resume Overview & Deep Dive"
    elif 2 <= index <= 4:
        return "Technical Depth & Architecture"
    elif index in (5, 6):
        return "Technical Probing & Edge Cases"
    elif 7 <= index <= 9:
        return "Behavioral Competencies (STAR)"
    elif 10 <= index <= 11:
        return "Role-Specific Scenario & Problem Solving"
    else:
        return "Retrospective & Leadership"


def generate_first_question(resume_text: str, target_role: str, api_key: Optional[str] = None) -> Tuple[str, str]:
    """Generates the opening question derived from the candidate's resume."""
    category = "Resume Overview & Deep Dive"
    llm = get_llm(api_key, temperature=0.5)

    if llm:
        try:
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are an empathetic, sharp hiring manager interviewing a candidate for '{target_role}'.\n"
                    "Analyze their resume highlights. Warmly welcome them, mention the target role, and pick one specific "
                    "significant project or technical system mentioned on their resume to ask a deep opener on its architecture, "
                    "their key personal contribution, and the core challenge they solved."
                ),
                (
                    "human",
                    "Target Role: {target_role}\n\nResume Context:\n{resume_text}\n\nGenerate Question 1:"
                )
            ])
            chain = prompt | llm | StrOutputParser()
            q1 = chain.invoke({"target_role": target_role, "resume_text": resume_text[:3500]})
            if q1 and len(q1.strip()) > 30:
                return q1.strip(), category
        except Exception as e:
            logger.warning(f"LangChain Q1 generation error: {e}")

    # Fallback opener
    q1_fallback = (
        f"Welcome! I am looking forward to speaking with you today regarding the {target_role} position. "
        f"Reviewing your resume, I noticed your technical contributions and hands-on project work. "
        f"To kick off our conversation, could you walk me through the most technically demanding project you led or contributed to, "
        f"explaining your architectural choices, your direct responsibilities, and the measurable outcome?"
    )
    return q1_fallback, category


def generate_next_question(
    turns: List[Dict[str, str]],
    target_role: str,
    resume_text: str,
    question_count: int,
    total_questions: int = 12,
    api_key: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Generates the next dynamic question in the flow or signals completion.
    Returns: (next_question, category, reaction)
    """
    next_idx = question_count + 1
    if next_idx > total_questions:
        return None, None, "Thank you for sharing your experience. That completes all questions for our mock interview! Let's generate your evaluation report."

    category = determine_category_for_index(next_idx, total_questions)
    recent_context = "\n".join([f"{t.get('role', '').capitalize()}: {t.get('content', '')}" for t in turns[-6:]])
    last_answer = turns[-1].get("content", "") if turns else ""

    llm = get_llm(api_key, json_mode=True, temperature=0.4)
    if llm:
        try:
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are a Lead Technical Interviewer for '{target_role}'.\n"
                    "This is Question {next_idx} of {total_questions}. Category: '{category}'.\n"
                    "Instructions:\n"
                    "1. Provide a natural 1-2 sentence reaction to what the candidate just explained.\n"
                    "2. Formulate a sharp, non-generic next question following the category.\n"
                    "   - If Technical: probe bottlenecks, concurrency, schema decisions, scaling, or edge cases.\n"
                    "   - If Behavioral (STAR): ask for a past concrete scenario (conflict, delivery under pressure, production bug).\n"
                    "   - If Role Scenario: give a realistic workplace challenge for {target_role}.\n"
                    "3. Return JSON: {{\"reaction\": \"...\", \"question\": \"...\", \"category\": \"{category}\"}}"
                ),
                (
                    "human",
                    "Target Role: {target_role}\n"
                    "Resume: {resume_text}\n\n"
                    "Recent Turns:\n{recent_context}\n\n"
                    "Candidate's Latest Response:\n{last_answer}\n\n"
                    "Generate Question {next_idx} JSON:"
                )
            ])
            chain = prompt | llm | StrOutputParser()
            raw_out = chain.invoke({
                "target_role": target_role,
                "next_idx": next_idx,
                "total_questions": total_questions,
                "category": category,
                "resume_text": resume_text[:2000],
                "recent_context": recent_context,
                "last_answer": last_answer,
            })
            parsed = json.loads(clean_json_text(raw_out))
            q_text = parsed.get("question")
            reaction = parsed.get("reaction", "Thank you for explaining that.")
            cat = parsed.get("category", category)
            if q_text and len(q_text.strip()) > 15:
                return q_text.strip(), cat, reaction
        except Exception as e:
            logger.warning(f"LangChain next question error: {e}")

    # Tailored fallback questions by index
    fallback_map = {
        2: ("When designing systems for this role, how do you balance latency, consistency, and operational complexity? Can you detail an edge case you encountered?", "Thank you for that context."),
        3: (f"In production for a {target_role}, how do you ensure zero-downtime deployments and telemetry alerting when services degrade?", "Understood, that makes sense."),
        4: ("How do you handle backwards-incompatible API changes or database schema migrations without disrupting live customers?", "Good overview."),
        5: ("Suppose an upstream service begins intermittently returning 504 gateway timeouts under peak traffic. How would you systematically isolate the root cause?", "Thanks for clarifying."),
        6: (f"Security and access control are critical for {target_role}. How do you enforce authorization and protect sensitive data in transit and at rest?", "Great perspective."),
        7: ("Using the STAR method, tell me about a time you had a strong technical disagreement with a teammate or senior lead. How did you resolve it?", "That's helpful context."),
        8: ("Describe a project where requirements shifted drastically midway through delivery. How did you handle scope, stakeholders, and quality?", "Appreciate those details."),
        9: ("Can you share an experience where an unintended bug or outage was triggered by your change? What was your immediate response and long-term fix?", "Thank you for being candid."),
        10: (f"As a {target_role}, how do you persuade non-technical product managers to prioritize technical debt and architectural refactoring?", "Solid explanation."),
        11: ("Imagine a teammate's code reviews consistently miss error handling and edge cases. How do you coach them constructively without bottlenecking sprint delivery?", "Very good."),
        12: (f"Looking forward, what is one technical area you are actively learning right now, and how do you plan to create impact in your first 90 days as a {target_role}?", "Thank you for sharing your thoughts.")
    }
    q_default, r_default = fallback_map.get(next_idx, ("How do you approach continuous improvement in your engineering workflows?", "Thank you."))
    return q_default, category, r_default


def evaluate_interview(
    target_role: str,
    resume_text: str,
    turns: List[Dict[str, str]],
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluates completed interview dialogue:
    Produces scores, strong/weak areas, and tailored model answers correcting candidate responses.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")

    # Pair questions with candidate answers
    qa_pairs = []
    curr_q = None
    q_num = 1
    for t in turns:
        if t.get("role") == "interviewer":
            curr_q = t.get("content", "")
        elif t.get("role") == "candidate" and curr_q:
            qa_pairs.append({
                "question_number": q_num,
                "question": curr_q,
                "candidate_answer": t.get("content", "")
            })
            q_num += 1
            curr_q = None

    if not qa_pairs:
        return {
            "target_role": target_role,
            "session_date": date_str,
            "strong_areas": ["Enthusiasm and participation"],
            "weak_areas": ["Insufficient responses provided"],
            "overall_feedback": "The candidate commenced the mock interview session but completed limited questions. Please answer at least 10 questions for a comprehensive assessment.",
            "scores": {"overall": 50, "technical": 50, "behavioral": 50, "communication": 50, "problem_solving": 50},
            "question_evaluations": [],
        }

    llm = get_llm(api_key, json_mode=True, temperature=0.2)
    if llm:
        try:
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    f"You are an executive hiring board evaluator assessing a mock interview for '{target_role}'.\n"
                    "CRITICAL REQUIREMENTS:\n"
                    "1. For EVERY question in 'question_evaluations', generate a UNIQUE, QUESTION-SPECIFIC 'expected_answer'.\n"
                    "2. The 'expected_answer' must directly CORRECT, UPGRADE, and ELEVATE the candidate's actual answer into the ideal 10/10 response with concrete technical metrics, trade-offs, and best practices for that specific topic.\n"
                    "3. Keep each 'expected_answer' concise yet impactful (2-3 sentences max).\n"
                    "4. Output valid JSON: {{\n"
                    '  "strong_areas": ["strength 1", "strength 2"],\n'
                    '  "weak_areas": ["actionable gap 1", "actionable gap 2"],\n'
                    '  "scores": {{"overall": 82, "technical": 85, "behavioral": 80, "communication": 82, "problem_solving": 81}},\n'
                    '  "overall_feedback": "Detailed overall assessment paragraph (60+ words).",\n'
                    '  "question_evaluations": [\n'
                    '    {{\n'
                    '      "question_number": 1,\n'
                    '      "question": "question text",\n'
                    '      "candidate_answer": "candidate answer",\n'
                    '      "score": 8,\n'
                    '      "feedback": "Specific feedback highlighting what worked and what was missing.",\n'
                    '      "expected_answer": "Tailored 10/10 model answer correcting and elevating their response."\n'
                    '    }}\n'
                    '  ]\n'
                    "}}"
                ),
                (
                    "human",
                    "Target Role: {target_role}\n"
                    "Resume: {resume_text}\n\n"
                    "Interview Transcript:\n{qa_json}\n\n"
                    "Evaluate with distinct model answers in JSON:"
                )
            ])
            chain = prompt | llm | StrOutputParser()
            raw_out = chain.invoke({
                "target_role": target_role,
                "resume_text": resume_text[:2000],
                "qa_json": json.dumps(qa_pairs, indent=2),
            })
            parsed = json.loads(clean_json_text(raw_out))

            # Clean and validate evaluations
            evaluations = []
            for qe in parsed.get("question_evaluations", []):
                raw_score = qe.get("score", 7)
                try:
                    s_clean = str(raw_score).split("/")[0].replace("%", "").strip()
                    score_val = max(1, min(10, int(round(float(s_clean)))))
                except Exception:
                    score_val = 7

                evaluations.append({
                    "question_number": int(qe.get("question_number", len(evaluations) + 1)),
                    "question": qe.get("question", ""),
                    "candidate_answer": qe.get("candidate_answer", ""),
                    "score": score_val,
                    "feedback": qe.get("feedback", "Good communication; explore deeper technical parameters."),
                    "expected_answer": qe.get("expected_answer", ""),
                })

            raw_scores = parsed.get("scores", {})
            clean_scores = {}
            for k in ["overall", "technical", "behavioral", "communication", "problem_solving"]:
                val = raw_scores.get(k, 75)
                try:
                    clean_scores[k] = max(10, min(100, int(round(float(str(val).split("/")[0].replace("%", "").strip())))))
                except Exception:
                    clean_scores[k] = 75

            return {
                "target_role": target_role,
                "session_date": date_str,
                "strong_areas": parsed.get("strong_areas", ["Technical relevance", "Clear communication"]),
                "weak_areas": parsed.get("weak_areas", ["Could quantify business impact", "STAR behavioral structure"]),
                "overall_feedback": parsed.get("overall_feedback", "Solid interview performance with domain knowledge demonstrated."),
                "scores": clean_scores,
                "question_evaluations": evaluations,
            }
        except Exception as e:
            logger.warning(f"LangChain evaluation call error: {e}")

    # Tailored question-specific fallback
    evaluations = []
    total_score = 0
    for item in qa_pairs:
        q_text = item["question"]
        ans = item["candidate_answer"]
        score = 8 if len(ans.split()) > 35 else (6 if len(ans.split()) > 15 else 5)
        total_score += score
        q_lower = q_text.lower()

        if any(w in q_lower for w in ["tell me about", "project", "led", "contributed", "challenging"]):
            feedback = "Good overview of scope, but needs the specific architectural constraint, trade-offs evaluated, and measurable business result."
            expected = f"\"In my recent project for {target_role}, the core bottleneck was maintaining p99 latency <150ms under a 4x throughput spike. I chose an asynchronous event-driven pipeline over synchronous requests, reducing latency by 65% and maintaining 99.95% uptime.\""
        elif any(w in q_lower for w in ["bottleneck", "scale", "performance", "latency", "traffic", "sql"]):
            feedback = "Solid recognition of performance challenges. Elevate your answer with profiling tools, query execution plans, and caching invalidation strategies."
            expected = "\"I first analyzed APM traces to locate sequential database bottlenecks. I added composite indexes on join keys, introduced connection pooling, and deployed Redis caching with write-through invalidation, reducing query latency by 72%.\""
        elif any(w in q_lower for w in ["disagreement", "conflict", "behavioral", "star", "teammate", "team"]):
            feedback = "Tighten the STAR framework: emphasize objective data/benchmarks and empathy used to achieve team alignment."
            expected = "\"(Situation) A teammate favored an ORM abstraction while I proposed batch SQL. (Action) I built a load benchmark demonstrating a 5x throughput advantage and walked them through it collaboratively. (Result) We aligned on batch queries and shipped 3 days early.\""
        else:
            feedback = f"Good foundational understanding for {target_role}. Strengthen your response by stating the problem constraints, action taken, and quantifiable outcome."
            expected = f"\"Addressing this for {target_role}: Define the core SLOs and constraints, apply modular architectural best practices with automated test coverage (>85%), and validate the outcome against benchmark KPIs.\""

        evaluations.append({
            "question_number": item["question_number"],
            "question": item["question"],
            "candidate_answer": item["candidate_answer"],
            "score": score,
            "feedback": feedback,
            "expected_answer": expected,
        })

    avg_score = int((total_score / (len(qa_pairs) * 10)) * 100) if qa_pairs else 70
    return {
        "target_role": target_role,
        "session_date": date_str,
        "strong_areas": [
            f"Demonstrated domain knowledge for {target_role}",
            "Responsive engagement across technical and architectural questions",
            "Clear personal contribution on resume projects",
        ],
        "weak_areas": [
            "Quantifying impact (metrics, latency % improvements, or business ROI)",
            "Deep technical trade-off justification (why alternative approaches were discarded)",
            "Strict STAR method framing for behavioral and situational scenarios",
        ],
        "overall_feedback": (
            f"The candidate completed a comprehensive interview session for the {target_role} position. "
            f"They demonstrated solid foundational knowledge and answered questions with good clarity. "
            f"To reach the top candidate tier, focus on backing up architectural claims with measurable numbers, "
            f"explaining technical trade-offs explicitly, and adhering strictly to the STAR format during behavioral discussions."
        ),
        "scores": {
            "overall": avg_score,
            "technical": min(95, max(60, avg_score + 2)),
            "behavioral": min(95, max(60, avg_score - 3)),
            "communication": min(95, max(65, avg_score + 1)),
            "problem_solving": min(95, max(60, avg_score)),
        },
        "question_evaluations": evaluations,
    }


def revision_coaching_turn(
    target_role: str,
    weak_area: str,
    history: List[Dict[str, str]],
    user_answer: str,
    api_key: Optional[str] = None,
) -> Dict[str, str]:
    """
    Conducts iterative coaching turn:
    Critiques candidate practice response, provides an elevated 10/10 version, and asks next drill question.
    """
    llm = get_llm(api_key, json_mode=True, temperature=0.3)
    if llm:
        try:
            history_str = "\n".join([f"{h.get('role', '').capitalize()}: {h.get('content', '')}" for h in history[-4:]])
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    f"You are a Principal Interview Coach training a candidate for '{target_role}'.\n"
                    f"You are coaching them on their weak area: '{weak_area}'.\n"
                    "The candidate just gave a practice answer.\n"
                    "Instructions:\n"
                    "1. 'critique': 2-3 sentences analyzing what worked and what was missing (metrics, STAR structure, trade-offs).\n"
                    "2. 'improved_answer': A rewritten, polished 10/10 version of their answer demonstrating elite framing.\n"
                    "3. 'next_question': The next follow-up drill question to continue mastering the weak area.\n"
                    "Output valid JSON: {{\"critique\": \"...\", \"improved_answer\": \"...\", \"next_question\": \"...\"}}"
                ),
                (
                    "human",
                    "Target Role: {target_role}\n"
                    "Weak Area: {weak_area}\n"
                    "Recent Dialogue:\n{history_str}\n\n"
                    "Candidate Practice Answer:\n{user_answer}\n\n"
                    "Evaluate, rewrite to 10/10, and pose next question in JSON:"
                )
            ])
            chain = prompt | llm | StrOutputParser()
            raw_out = chain.invoke({
                "target_role": target_role,
                "weak_area": weak_area,
                "history_str": history_str,
                "user_answer": user_answer,
            })
            parsed = json.loads(clean_json_text(raw_out))
            return {
                "critique": parsed.get("critique", "Good effort communicating key points."),
                "improved_answer": parsed.get("improved_answer", "Here is how to elevate your response with metrics and clear technical structure."),
                "next_question": parsed.get("next_question", "How would you handle unexpected latency spikes in this proposed architecture?"),
            }
        except Exception as e:
            logger.warning(f"LangChain coaching turn error: {e}")

    # High quality coaching fallback
    return {
        "critique": (
            f"Your response demonstrates good domain intuition for {target_role}, but it can be strengthened by explicitly "
            f"quantifying the results (e.g. % performance increase, memory savings, or time to resolution) and tightening "
            f"the structure around the specific technical trade-offs."
        ),
        "improved_answer": (
            f"\"In my project, we faced a latency bottleneck of ~850ms under peak load. I analyzed the profile using tracing tools "
            f"and identified redundant database queries. I introduced an asynchronous cache invalidation strategy with Redis and "
            f"batch query pipeline, reducing p99 latency to 120ms (an 85% drop) and cutting DB CPU utilization by 40%.\""
        ),
        "next_question": (
            f"Let's practice the next scenario for **{weak_area}**: "
            f"If the team lead initially argued against your proposed strategy due to cache invalidation complexity, "
            f"how would you walk them through your proof of concept and address their concerns?"
        ),
    }
