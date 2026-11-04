"""Streamlit interface for the InsightRAG PDF question-answering app."""

import hashlib

import streamlit as st

from src.config.settings import APP_CAPTION, APP_TITLE, SUGGESTED_QUESTIONS, TOP_K
from src.generation.answer import generate_answer, load_llm
from src.ingestion.pdf_processor import process_pdf_bytes
from src.retrieval.semantic import retrieve_relevant_chunks
from src.utils.secrets import get_groq_api_key


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(f"{APP_TITLE}: Context-Aware PDF QA System")
st.caption(APP_CAPTION)

api_key = get_groq_api_key()
if not api_key:
    st.error("GROQ_API_KEY is missing. Add it to .streamlit/secrets.toml or your environment.")
    st.stop()

llm = load_llm(api_key)

with st.sidebar:
    st.header("Demo prompts")
    for prompt_text in SUGGESTED_QUESTIONS:
        if st.button(prompt_text, use_container_width=True):
            st.session_state["question_text"] = prompt_text

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is None:
    st.info("Upload a PDF to get started.")
    st.stop()

pdf_bytes = uploaded_file.getvalue()
file_hash = hashlib.sha256(pdf_bytes).hexdigest()

if st.session_state.get("file_hash") != file_hash:
    try:
        with st.spinner("Processing PDF and creating embeddings..."):
            chunks, page_count, chunk_count, chunk_embeddings = process_pdf_bytes(pdf_bytes)
    except Exception as exc:
        st.error(f"Could not process this PDF: {exc}")
        st.stop()

    st.session_state.update(
        {
            "file_hash": file_hash,
            "chunks": chunks,
            "chunk_embeddings": chunk_embeddings,
            "pages": page_count,
            "chunk_count": chunk_count,
            "file_name": uploaded_file.name,
        }
    )

chunks = st.session_state["chunks"]
chunk_embeddings = st.session_state["chunk_embeddings"]
st.success(
    f"{st.session_state['file_name']} | "
    f"{st.session_state['pages']} pages → {st.session_state['chunk_count']} chunks"
)

question = st.text_input("Ask a question", key="question_text")
if question.strip():
    try:
        retrieved = retrieve_relevant_chunks(chunks, chunk_embeddings, question, TOP_K)
        if not retrieved:
            st.warning("No readable or relevant content was found in this PDF.")
            st.stop()

        context = "\n\n".join(item["chunk"].page_content for item in retrieved)
        with st.spinner("Generating answer..."):
            answer = generate_answer(context, question, llm)

        st.subheader("Answer")
        st.write(answer)
        st.caption(f"Top {len(retrieved)} retrieved chunks used.")

        with st.expander("Retrieved context"):
            st.write(context)

        st.subheader("Sources")
        for rank, item in enumerate(retrieved, start=1):
            chunk = item["chunk"]
            score = item["score"]
            # PyPDFLoader page metadata is zero-indexed; display human page numbers.
            raw_page = chunk.metadata.get("page")
            page_number = raw_page + 1 if isinstance(raw_page, int) else "N/A"

            st.write(f"Source {rank} | Page: {page_number} | Similarity: {score:.4f}")
            st.write(chunk.page_content[:400] + ("..." if len(chunk.page_content) > 400 else ""))
            st.divider()
    except Exception as exc:
        st.error(f"Something went wrong while answering your question: {exc}")
