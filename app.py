# ============================================================
# 🤖 AI-Powered Hiring Recommendation System (Streamlit + LangGraph)
# With Sidebar Controls
# ============================================================

import streamlit as st
import tempfile
import os, re
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Dict

# ============================================================
# 🧱 PAGE CONFIG
# ============================================================
st.set_page_config(page_title="AI Hiring Recommendation System", layout="wide")
st.title("🤖 AI-Powered Hiring Recommendation System")
st.write("Upload candidate resume and job description to generate a hiring recommendation.")

# ============================================================
# 🧭 SIDEBAR INPUTS
# ============================================================
st.sidebar.header("⚙️ Configuration")

# 🔑 API Key
openai_key = st.sidebar.text_input("🔑 Enter OpenAI API Key", type="password")

# 🧠 Model Selection
model_choice = st.sidebar.selectbox(
    "🧩 Choose Model",
    ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    index=0
)

# 📂 Resume Upload
resume_file = st.sidebar.file_uploader("📁 Upload Candidate Resume (PDF)", type=["pdf"])

# 📂 Job Description Upload
job_file = st.sidebar.file_uploader("📄 Upload Job Description (PDF)", type=["pdf"])

# Check basic inputs
if not openai_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
    st.stop()

if not resume_file or not job_file:
    st.info("Please upload both PDF files in the sidebar to begin.")
    st.stop()

# ============================================================
# 🗂️ TEMPORARY FILE STORAGE
# ============================================================
def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name

resume_path = save_uploaded_file(resume_file)
job_path = save_uploaded_file(job_file)

# ============================================================
# 🧩 STATE TYPE
# ============================================================
class HiringState(TypedDict):
    candidate_name: str
    resume_text: str
    job_text: str
    resume_summary: str
    job_summary: str
    scores: Dict
    overall_score: float
    recommendation: str
    final_report: str

# ============================================================
# 📄 LOAD RESUME
# ============================================================
def load_resume(state: HiringState) -> HiringState:
    st.write("📄 Loading resume PDF...")
    resume_loader = PyPDFLoader(resume_path)
    resume_text = " ".join([page.page_content for page in resume_loader.load()])

    llm = ChatOpenAI(model=model_choice, temperature=0, openai_api_key=openai_key)
    name_prompt = PromptTemplate.from_template("""
    Extract only the full name of the candidate from the following resume text.
    Resume:
    {resume_text}
    Respond with only the name, nothing else.
    """)
    name_chain = name_prompt | llm | StrOutputParser()
    candidate_name = name_chain.invoke({"resume_text": resume_text}).strip()

    st.success(f"✅ Resume loaded successfully for: {candidate_name}")
    return {"resume_text": resume_text, "candidate_name": candidate_name}

# ============================================================
# 📄 LOAD JOB DESCRIPTION
# ============================================================
def load_job_description(state: HiringState) -> HiringState:
    st.write("📄 Loading job description PDF...")
    job_loader = PyPDFLoader(job_path)
    job_text = " ".join([page.page_content for page in job_loader.load()])
    st.success("✅ Job description loaded successfully.")
    return {"job_text": job_text}

# ============================================================
# 🧾 RESUME SUMMARY
# ============================================================
def extract_resume_summary(state: HiringState) -> HiringState:
    llm = ChatOpenAI(model=model_choice, temperature=0, openai_api_key=openai_key)
    prompt = PromptTemplate.from_template("""
    Summarize the following resume into:
    - Skills
    - Experience
    - Education
    - Achievements

    Resume:
    {resume_text}

    Provide a concise, structured paragraph.
    """)
    chain = prompt | llm
    summary = chain.invoke({"resume_text": state["resume_text"]}).content
    return {"resume_summary": summary}

# ============================================================
# 🧾 JOB SUMMARY
# ============================================================
def extract_job_summary(state: HiringState) -> HiringState:
    llm = ChatOpenAI(model=model_choice, temperature=0, openai_api_key=openai_key)
    prompt = PromptTemplate.from_template("""
    Summarize the job description into:
    - Required skills
    - Experience level
    - Education requirements
    - Key responsibilities

    Job Description:
    {job_text}

    Provide a concise structured paragraph.
    """)
    chain = prompt | llm
    summary = chain.invoke({"job_text": state["job_text"]}).content
    return {"job_summary": summary}

# ============================================================
# ⚖️ COMPARISON LOGIC
# ============================================================
def compare_resume_with_job(state: HiringState) -> HiringState:
    resume_text = state["resume_text"].lower()
    job_text = state["job_text"].lower()

    skill_keywords = [
        "python", "java", "javascript", "typescript", "react", "node", "flask", "django",
        "aws", "azure", "docker", "kubernetes", "sql", "mongodb", "postgresql",
        "tensorflow", "pytorch", "machine learning", "data science", "api", "git",
        "html", "css", "vue", "angular"
    ]

    resume_skills = [s for s in skill_keywords if s in resume_text]
    job_skills = [s for s in skill_keywords if s in job_text]

    skill_overlap = len(set(resume_skills) & set(job_skills))
    skill_total = len(set(job_skills)) if job_skills else 1
    skills_score = int((skill_overlap / skill_total) * 100)

    def extract_years(text):
        matches = re.findall(r"(\d+)\+?\s*(?:years?|yrs?)", text)
        years = [int(m) for m in matches]
        return max(years) if years else 0

    resume_exp = extract_years(resume_text)
    job_exp = extract_years(job_text)
    exp_ratio = min(resume_exp / job_exp, 1) if job_exp > 0 else 1
    experience_score = int(exp_ratio * 100)

    education_levels = {"phd": 4, "master": 3, "bachelor": 2, "associate": 1}

    def extract_education_level(text):
        for key, val in education_levels.items():
            if key in text:
                return val
        return 0

    resume_edu = extract_education_level(resume_text)
    job_edu = extract_education_level(job_text)
    edu_ratio = min(resume_edu / job_edu, 1) if job_edu > 0 else 1
    education_score = int(edu_ratio * 100)

    overall = (
        skills_score * 0.5 +
        experience_score * 0.3 +
        education_score * 0.2
    )

    st.write(f"📊 **Skills Score:** {skills_score}")
    st.write(f"📊 **Experience Score:** {experience_score}")
    st.write(f"📊 **Education Score:** {education_score}")
    st.write(f"📈 **Overall Score:** {overall:.2f}")

    return {
        "scores": {
            "skills_score": skills_score,
            "experience_score": experience_score,
            "education_score": education_score
        },
        "overall_score": overall
    }

# ============================================================
# 🧭 DECISION LOGIC
# ============================================================
def decide_interview_stage(state: HiringState) -> str:
    score = state["overall_score"]
    if score >= 85:
        return "execute_one_interview_process"
    elif score >= 60:
        return "execute_two_interview_process"
    else:
        return "execute_rejection_process"

def execute_one_interview_process(state: HiringState) -> HiringState:
    st.success("🟢 Candidate shortlisted for one direct human interview.")
    return {"recommendation": "One Interview – Direct human interview"}

def execute_two_interview_process(state: HiringState) -> HiringState:
    st.warning("🟡 Candidate shortlisted for two rounds (screening + coding).")
    return {"recommendation": "Two Interviews – Initial screening + coding round"}

def execute_rejection_process(state: HiringState) -> HiringState:
    st.error("🔴 Candidate rejected due to insufficient match.")
    return {"recommendation": "Rejected – Insufficient match"}

# ============================================================
# 🧾 FINAL REPORT
# ============================================================
def generate_final_report(state: HiringState) -> HiringState:
    s = state["scores"]
    o = state["overall_score"]
    rec = state["recommendation"]
    name = state["candidate_name"]

    report = f"""
📋 **FINAL HIRING RECOMMENDATION REPORT**
-------------------------------------

**Candidate:** {name}

**SCORES**
- Skills Score: {s['skills_score']}/100  
- Experience Score: {s['experience_score']}/100  
- Education Score: {s['education_score']}/100  
- Overall Score: {o:.1f}

**Recommendation:** {rec}

---

📚 **Candidate Summary:**
{state['resume_summary']}

📄 **Job Fit Summary:**
{state['job_summary']}
"""
    st.subheader("📋 Final Report")
    st.markdown(report)
    return {"final_report": report}

# ============================================================
# 🔁 GRAPH PIPELINE
# ============================================================
def run_hiring_system():
    graph = StateGraph(HiringState)

    graph.add_node("load_resume", load_resume)
    graph.add_node("load_job_description", load_job_description)
    graph.add_node("extract_resume_summary", extract_resume_summary)
    graph.add_node("extract_job_summary", extract_job_summary)
    graph.add_node("compare_resume_with_job", compare_resume_with_job)
    graph.add_node("execute_one_interview_process", execute_one_interview_process)
    graph.add_node("execute_two_interview_process", execute_two_interview_process)
    graph.add_node("execute_rejection_process", execute_rejection_process)
    graph.add_node("generate_final_report", generate_final_report)

    graph.add_edge(START, "load_resume")
    graph.add_edge("load_resume", "load_job_description")
    graph.add_edge("load_job_description", "extract_resume_summary")
    graph.add_edge("load_job_description", "extract_job_summary")
    graph.add_edge("extract_resume_summary", "compare_resume_with_job")
    graph.add_edge("extract_job_summary", "compare_resume_with_job")

    graph.add_conditional_edges(
        "compare_resume_with_job",
        decide_interview_stage,
        {
            "execute_one_interview_process": "execute_one_interview_process",
            "execute_two_interview_process": "execute_two_interview_process",
            "execute_rejection_process": "execute_rejection_process",
        }
    )

    graph.add_edge("execute_one_interview_process", "generate_final_report")
    graph.add_edge("execute_two_interview_process", "generate_final_report")
    graph.add_edge("execute_rejection_process", "generate_final_report")
    graph.add_edge("generate_final_report", END)

    compiled_graph = graph.compile()
    compiled_graph.invoke({})
    st.success("✅ Hiring analysis completed successfully!")

# ============================================================
# 🚀 RUN BUTTON
# ============================================================
if st.sidebar.button("🚀 Run Hiring Analysis"):
    with st.spinner("Analyzing... Please wait..."):
        run_hiring_system()
