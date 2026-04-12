import os
import hashlib
import tempfile

import streamlit as st
import numpy as np

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="InsightRAG", layout="wide")
st.title("🚀 InsightRAG: Context-Aware PDF QA System")
st.caption("AI-powered document assistant using Retrieval-Augmented Generation (RAG)")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

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
        model="llama-3.1-8b-instant",
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
# Answer generation
# -----------------------------
def generate_answer(context, question):
    if llm is None:
        return "GROQ_API_KEY is missing."

    prompt = (
        "You are a strict document assistant.\n"
        "Answer using ONLY the context below.\n"
        "If the answer is not explicitly present, reply exactly: Not found in document.\n"
        "Do not guess.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )

    response = llm.invoke([HumanMessage(content=prompt)])
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

    question = st.text_input(
        "Ask a question",
        value=st.session_state.get("question_text", ""),
        key="question_input",
    )

    if question.strip():
        retrieved = simple_retrieval(chunks, chunk_embeddings, question, TOP_K)

        if not retrieved:
            st.warning("No relevant content found.")
        else:
            context = "\n\n".join([item["chunk"].page_content for item in retrieved])
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
