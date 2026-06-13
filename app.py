import os
import re
import base64
import streamlit as st
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

# ── Load API key ───────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please add it to your .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ── Load CSS ───────────────────────────────────────────────────────────────
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ── Background image ───────────────────────────────────────────────────────
def set_background(image_filename):
    image_path = os.path.join(os.path.dirname(__file__), image_filename)
    if not os.path.exists(image_path):
        return
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    ext  = image_filename.split(".")[-1].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    st.markdown(
        f"""<style>
        .stApp {{
            background-image: url("data:{mime};base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>""",
        unsafe_allow_html=True,
    )

set_background("background.jpg")

# ── Page title ─────────────────────────────────────────────────────────────
st.title("🎤 AI Interview Coach")

# ── Session state defaults ─────────────────────────────────────────────────
defaults = {
    "scores":             [],
    "history":            [],
    "question_number":    1,
    "show_feedback":      False,
    "question":           None,
    "feedback":           None,
    "pending_next":       False,
    "saved_role":         "",
    "saved_qtype":        "",
    "answer_key":         0,
    "current_role":       "",
    "current_experience": "Fresher",
    "current_industry":   "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── GUARD ──────────────────────────────────────────────────────────────────
if st.session_state.pending_next:
    st.session_state.question_number += 1
    st.session_state.question         = None
    st.session_state.feedback         = None
    st.session_state.show_feedback    = False
    st.session_state.pending_next     = False
    st.session_state.answer_key      += 1

# ── Helper: call Groq AI ───────────────────────────────────────────────────
# This single function handles all AI calls.
# model="llama3-70b-8192" is free, fast, and very capable.
def call_ai(prompt: str) -> str:
    response = client.chat.completions.create(
    
model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )
    return response.choices[0].message.content.strip()

# ── Helper: extract score ──────────────────────────────────────────────────
def extract_score(feedback_text: str) -> int | None:
    match = re.search(r"Score:\s*(\d{1,2})/10", feedback_text)
    return int(match.group(1)) if match else None

# ── Helper: generate question ─────────────────────────────────────────────
def generate_question(role, experience, industry, question_type):
    # Build a list of previously asked questions to pass to the AI
    # so it knows what NOT to repeat
    asked = ""
    if st.session_state.history:
        asked_list = [item["question"] for item in st.session_state.history]
        asked = "\n".join(f"- {q}" for q in asked_list)
        asked = f"\n\nDo NOT repeat or ask anything similar to these already asked questions:\n{asked}"

    prompt = f"""
You are a professional interview coach.
Generate exactly ONE {question_type} interview question for the following candidate:
- Job Role: {role}
- Experience Level: {experience}
- Industry: {industry}

Guidelines:
- Technical question: focus on a real skill or tool used in the role.
- Behavioral question: focus on teamwork, communication, leadership, or conflict resolution.
- Situational question: describe a realistic workplace scenario and ask what they would do.
- Every question must be DIFFERENT and cover a NEW topic or skill.
- Do NOT repeat any concept, scenario, or topic you have already asked about.
{asked}

Return ONLY the question. No introduction, no numbering, no explanation.
""".strip()
    return call_ai(prompt)

# ── Helper: evaluate answer ───────────────────────────────────────────────
def evaluate_answer(question, answer, role, experience):
    prompt = f"""
You are a strict and honest interview coach evaluating a candidate's answer.

Job Role: {role}
Experience Level: {experience}
Question: {question}
Candidate's Answer: {answer}

Score the answer HONESTLY based on these strict criteria:

- 0: Completely meaningless answer, random characters, keyboard smashing, unrelated text, or no attempt to answer the question.
- 1-3: Very poor. Extremely weak answer, mostly irrelevant, or shows little understanding.
- 4-5: Below average. Vague, missing key points, weak structure.
- 6-7: Average. Some good points but lacks depth or examples.
- 8-9: Good. Clear, structured, relevant with good examples.
- 10: Excellent. Perfect structure, specific examples, complete answer.

IMPORTANT RULES:
- If the answer contains random text such as "asdfgh", "knhdbkankjvnkj", "123456", or meaningless words, give 0/10.
- If the answer does not address the question at all, give 0/10.
- Do NOT give 8 by default. Be honest.
- A one-line answer should NEVER score above 5.
- A vague answer without examples should NEVER score above 6.
- Only give 8+ if the answer is genuinely strong with specific details.

Respond in EXACTLY this format — do not change the structure:

📊 Score: X/10

✅ Strengths
- (one clear strength, or "No significant strengths" if answer is poor)

🔧 Improvements
- (one specific improvement)
- (another improvement)

💡 Better Answer
(Write a 3-4 line improved version of the answer in simple language)

🎯 Coaching Tip
(One practical sentence the candidate can apply immediately)

Use honest, constructive, beginner-friendly language.
""".strip()
    return call_ai(prompt)
# ── Inputs ─────────────────────────────────────────────────────────────────
role = st.text_input(
    "Enter Job Role",
    placeholder="e.g. Data Analyst,Cloud Engineer ",
    value=st.session_state.current_role,
)

experience = st.selectbox(
    "Select Experience Level",
    ["Fresher", "1-3 Years", "3-5 Years", "5+ Years"],
    index=["Fresher", "1-3 Years", "3-5 Years", "5+ Years"].index(
        st.session_state.current_experience
    ),
)

question_type = st.selectbox(
    "Select Question Type",
    ["Technical", "Behavioral", "Situational"],
)

industry = st.text_input(
    "Enter Industry",
    placeholder="e.g. Finance, Healthcare, IT",
    value=st.session_state.current_industry,
)

# ── Generate Question ──────────────────────────────────────────────────────
if st.button("🚀 Generate Question"):
    if not role.strip() or not industry.strip():
        st.warning("Please enter both Job Role and Industry.")
    else:
        st.session_state.current_role       = role
        st.session_state.current_experience = experience
        st.session_state.current_industry   = industry

        with st.spinner("Generating your interview question..."):
            try:
                raw_question = generate_question(role, experience, industry, question_type)
                st.session_state.question = (
                    f"Question {st.session_state.question_number}: {raw_question}"
                )
                st.session_state.show_feedback = False
                st.session_state.feedback      = None
            except Exception as e:
                st.error(f"Could not generate question: {e}")

# ── Question + Answer block ────────────────────────────────────────────────
if st.session_state.question:

    st.subheader("📝 Interview Question")
    st.write(st.session_state.question)

    answer = st.text_area(
        "Your Answer",
        height=200,
        placeholder="Type your answer here...",
        key=f"answer_{st.session_state.answer_key}",
    )

    if st.button("✅ Submit Answer"):
        if not answer.strip():
            st.warning("Please enter an answer before submitting.")
        else:
            with st.spinner("Evaluating your answer..."):
                try:
                    feedback_text = evaluate_answer(
                        st.session_state.question,
                        answer,
                        st.session_state.current_role,
                        st.session_state.current_experience,
                    )
                    st.session_state.feedback      = feedback_text
                    st.session_state.show_feedback = True

                    score = extract_score(feedback_text)
                    if score is not None:
                        st.session_state.scores.append(score)
                        st.session_state.history.append({
                            "question": st.session_state.question,
                            "score":    score,
                        })
                    else:
                        st.warning("Score could not be extracted from feedback.")

                except Exception as e:
                    st.error(f"Could not evaluate answer: {e}")

    # ── Feedback ───────────────────────────────────────────────────────────
    if st.session_state.show_feedback and st.session_state.feedback:

        st.subheader("📋 Feedback")
        st.markdown(
            f'<div class="feedback-box">{st.session_state.feedback}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        if st.button("➡️ Next Question"):
            st.session_state.saved_role   = st.session_state.current_role
            st.session_state.saved_qtype  = question_type
            st.session_state.pending_next = True
            st.rerun()

# ── Sidebar dashboard ──────────────────────────────────────────────────────
st.sidebar.title("📊 Dashboard")

if os.path.exists("dashboard.jpg"):
    st.sidebar.image("dashboard.jpg", use_column_width=True)

if st.session_state.scores:
    scores = st.session_state.scores
    st.sidebar.metric("Questions Answered", len(scores))
    st.sidebar.metric("Average Score",  round(sum(scores) / len(scores), 1))
    st.sidebar.metric("Highest Score",  max(scores))
    st.sidebar.metric("Lowest Score",   min(scores))
else:
    st.sidebar.info("Complete your first answer to see your stats here.")

# ── Interview history ──────────────────────────────────────────────────────
st.divider()
st.subheader("📜 Interview History")

if st.session_state.history:
    for i, item in enumerate(st.session_state.history, start=1):
        st.write(f"**Q{i}:** {item['question']}  →  Score **{item['score']}/10**")
else:
    st.write("No completed interviews yet. Answer your first question to get started!")

# ── Score trend chart ──────────────────────────────────────────────────────
if st.session_state.scores:
    st.divider()
    st.subheader("📈 Score Trend")
    chart_data = pd.DataFrame({
        "Interview": range(1, len(st.session_state.scores) + 1),
        "Score":     st.session_state.scores,
    })
    st.line_chart(chart_data.set_index("Interview"))

# ── Overall improvement ────────────────────────────────────────────────────
if len(st.session_state.scores) > 1:
    improvement = st.session_state.scores[-1] - st.session_state.scores[0]
    st.metric("Overall Improvement", f"{improvement:+}")
