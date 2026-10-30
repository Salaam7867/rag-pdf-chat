import os
import hashlib
import tempfile

import streamlit as st
import numpy as np

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="InsightRAG", layout="wide")
st.title("🚀 InsightRAG: Context-Aware PDF QA System")
st.caption("AI-powered document assistant using Retrieval-Augmented Generation (RAG)")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 8

SUGGESTED_QUESTIONS = [
    "Summarize this document in 5 bullet points.",
    "What are the key skills mentioned in this resume?",
    "Would you shortlist this candidate for an AI role? Give strengths and weaknesses.",
    "What is the main purpose of this document?",
]

# -----------------------------
# API KEY
# -----------------------------
def get_groq_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY")


# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_llm(api_key: str):
    if not api_key:
        return None

    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        groq_api_key=api_key,
    )


embeddings = load_embeddings()
api_key = get_groq_api_key()
llm = load_llm(api_key)

# -----------------------------
# Similarity helpers
# -----------------------------
def cosine_similarity_matrix(query_vec, matrix):
    query_vec = np.asarray(query_vec, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)

    query_norm = np.linalg.norm(query_vec)
    matrix_norms = np.linalg.norm(matrix, axis=1)

    denom = matrix_norms * query_norm
    denom[denom == 0] = 1e-8

    return (matrix @ query_vec) / denom


def simple_retrieval(chunks, chunk_embeddings, query, k=3):
    if not chunks or chunk_embeddings is None or len(chunks) == 0:
        return []

    query_vec = embeddings.embed_query(query)
    scores = cosine_similarity_matrix(query_vec, chunk_embeddings)

    top_indices = np.argsort(scores)[::-1][:k]
    results = []

    for idx in top_indices:
        results.append(
            {
                "chunk": chunks[idx],
                "score": float(scores[idx]),
            }
        )

    return results


# -----------------------------
# PDF processing
# -----------------------------
@st.cache_data(show_spinner=False)
def process_pdf_bytes(pdf_bytes: bytes):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            temp_path = tmp.name

        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(docs)

        if not chunks:
            return [], 0, 0, np.empty((0, 0), dtype=np.float32)

        chunk_texts = [chunk.page_content for chunk in chunks]
        chunk_embeddings = embeddings.embed_documents(chunk_texts)
        chunk_embeddings = np.asarray(chunk_embeddings, dtype=np.float32)

        return chunks, len(docs), len(chunks), chunk_embeddings

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)



# -----------------------------
# Permanent System Instructions
# -----------------------------
SYSTEM_PROMPT = """
You are an expert technical recruiter and resume evaluator.

Your task is to evaluate a candidate's resume against a job description.

STRICT RULES:
1. Use ONLY facts explicitly supported by the supplied resume context.
2. Never invent skills, experience, projects, metrics, qualifications, or achievements.
3. Classify every relevant skill as:
   A. Demonstrated hands-on experience.
   B. Coursework or theoretical knowledge.
   C. Not demonstrated in the available resume.
4. Not demonstrated does NOT mean the candidate lacks that skill.
5. A skill listed in Technical Skills is not sufficient proof
   of hands-on experience. Look for project evidence.
6. Do not assume experience with one framework means experience
   with another framework.
7. Support major claims with resume excerpts and page numbers.
8. If evidence is unavailable, state:
   "Not stated in the resume."
9. Never recommend adding fabricated skills or achievements.
10. Evaluate fairly for the experience level of the job.

11. Never infer technical knowledge from a related concept.
    For example, embeddings do not automatically prove
    tokenization knowledge.

12. Never infer regular Git/version-control practices merely
    because projects are hosted on GitHub.

13. Never generate hypothetical resume bullets containing
    unverified skills, tools, metrics, or achievements.

14. Divide resume improvement suggestions into:
    - Verified improvements: supported by existing evidence.
    - Candidate confirmation required: ask whether the
      candidate has actually performed the activity.
    Never present unverified suggestions as ready-to-use bullets.

15. Do not classify soft skills such as analytical thinking
    as missing technical skills. Mark them as "Evidence unclear."

16. Page numbers must use human-readable numbering starting
    from 1. Convert zero-based PDF page metadata accordingly.

17. If the resume has one page, never cite page 2 or higher.

18. Always complete every requested output section,
    including all five interview questions and the final
    recommendation. Never silently omit a section.


SCORING:
- Technical skills: 35 points
- Relevant projects and experience: 30 points
- Required qualifications: 20 points
- Preferred skills: 10 points
- Education: 5 points

Award points only when supported by resume evidence.
Show earned points, possible points, and reasons for deductions.
Do not inflate scores based on skills listed without evidence.

OUTPUT FORMAT:

# 1. Overall Match
- Match score out of 100.
- Short explanation.
- Final recommendation.

# 2. Score Breakdown
Show all five categories, points earned, possible points,
and evidence-based deductions.

# 3. Required Skills Assessment
For each relevant skill, show:
- Skill
- Classification (A/B/C)
- Resume evidence
- Page number
- Explanation

# 4. Missing or Unclear Evidence
Separate missing information from skills that are
simply not demonstrated.

# 5. Three Strongest Matching Projects
Explain actual implementation, technologies, and relevance.
Include resume evidence and page numbers.

# 6. Resume Improvements
Give specific, truthful improvements for this job.
Do not fabricate metrics or experience.

# 7. Five Likely Interview Questions
Base questions on job requirements and evidence gaps.

# 8. Final Recommendation
Choose:
- Apply
- Apply after tailoring
- Low priority

Explain the decision briefly.
"""

# -----------------------------
# Answer generation
# -----------------------------
def generate_answer(context, question):
    if llm is None:
        return "GROQ_API_KEY is missing."

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "RESUME CONTEXT:\n"
                f"{context}\n\n"
                "JOB DESCRIPTION:\n"
                f"{question}\n\n"
                "Evaluate the resume against this job description."
            )
        )
    ]

    response = llm.invoke(messages)
    return response.content.strip()



# -----------------------------
# App
# -----------------------------
if not api_key:
    st.error("Add GROQ_API_KEY in secrets.")
    st.stop()

with st.sidebar:
    st.header("Demo prompts")
    for prompt_text in SUGGESTED_QUESTIONS:
        if st.button(prompt_text, use_container_width=True):
            st.session_state["question_text"] = prompt_text

file = st.file_uploader("Upload PDF", type="pdf")

if file:
    pdf_bytes = file.getvalue()
    file_hash = hashlib.md5(pdf_bytes).hexdigest()

    if (
        "file_hash" not in st.session_state
        or st.session_state.get("file_hash") != file_hash
    ):
        with st.spinner("Processing..."):
            chunks, pages, chunk_count, chunk_embeddings = process_pdf_bytes(pdf_bytes)

        st.session_state["file_hash"] = file_hash
        st.session_state["chunks"] = chunks
        st.session_state["chunk_embeddings"] = chunk_embeddings
        st.session_state["pages"] = pages
        st.session_state["chunk_count"] = chunk_count
        st.session_state["file_name"] = file.name

    chunks = st.session_state["chunks"]
    chunk_embeddings = st.session_state["chunk_embeddings"]
    pages = st.session_state["pages"]
    chunk_count = st.session_state["chunk_count"]
    file_name = st.session_state["file_name"]

    st.success(f"{file_name} | {pages} pages → {chunk_count} chunks")

    if "question_text" not in st.session_state:
        st.session_state["question_text"] = ""

    question = st.text_area(
    "Paste Job Description",
    key="question_text",
    height=250,
    placeholder="Paste the complete job description here..."
)

    if st.session_state.get("question_text", "").strip():
        question = st.session_state["question_text"]

        retrieved = simple_retrieval(chunks, chunk_embeddings, question, TOP_K)

        if not retrieved:
            st.warning("No relevant content found.")
        else:
            context = "\n\n".join([item["chunk"].page_content for item in retrieved])

            with st.spinner("Generating answer..."):
                answer = generate_answer(context, question)

            st.subheader("Answer")
            st.write(answer)

            st.caption(f"Top {len(retrieved)} retrieved chunks used.")

            with st.expander("Retrieved context"):
                st.write(context)

            st.subheader("Sources")
            for i, item in enumerate(retrieved, 1):
                chunk = item["chunk"]
                score = item["score"]
                page_num = chunk.metadata.get("page", "N/A")

                st.write(f"Source {i} | Page: {page_num} | Similarity: {score:.4f}")
                st.write(chunk.page_content[:400] + "...")
                st.divider()
else:
    st.info("Upload a PDF")
