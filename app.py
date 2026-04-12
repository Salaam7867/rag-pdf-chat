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
st.set_page_config(page_title="RAG PDF Chat", layout="wide")
st.title("📄 RAG – Chat with PDF")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

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
def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return -1.0
    return float(np.dot(a, b) / denom)

def simple_retrieval(chunks, query, k=3):
    if not chunks:
        return []

    query_vec = embeddings.embed_query(query)
    scored_chunks = []

    for chunk in chunks:
        chunk_vec = embeddings.embed_query(chunk.page_content)
        score = cosine_similarity(query_vec, chunk_vec)
        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored_chunks[:k]]

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

        return chunks, len(docs), len(chunks)
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
        "Answer using ONLY the context below.\n"
        "If not found, say exactly: Not found in document.\n\n"
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

file = st.file_uploader("Upload PDF", type="pdf")

if file:
    pdf_bytes = file.getvalue()
    file_hash = hashlib.md5(pdf_bytes).hexdigest()

    if (
        "file_hash" not in st.session_state
        or st.session_state.get("file_hash") != file_hash
    ):
        with st.spinner("Processing..."):
            chunks, pages, chunk_count = process_pdf_bytes(pdf_bytes)

        st.session_state["file_hash"] = file_hash
        st.session_state["chunks"] = chunks
        st.session_state["pages"] = pages
        st.session_state["chunk_count"] = chunk_count

    chunks = st.session_state["chunks"]
    pages = st.session_state["pages"]
    chunk_count = st.session_state["chunk_count"]

    st.success(f"{pages} pages → {chunk_count} chunks")

    question = st.text_input("Ask a question")

    if question.strip():
        docs = simple_retrieval(chunks, question, TOP_K)

        if not docs:
            st.warning("No relevant content found.")
        else:
            context = "\n\n".join([d.page_content for d in docs])
            answer = generate_answer(context, question)

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Sources")
            for i, d in enumerate(docs, 1):
                st.write(f"Source {i}")
                st.write(d.page_content[:300] + "...")
else:
    st.info("Upload a PDF")
