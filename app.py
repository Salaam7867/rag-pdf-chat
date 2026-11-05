"""Streamlit interface for the InsightRAG PDF question-answering app."""

import hashlib
import streamlit as st

from src.config.settings import APP_CAPTION, APP_TITLE, SUGGESTED_QUESTIONS, CANDIDATE_POOL_K
from src.generation.answer import generate_answer, load_llm
from src.ingestion.pdf_processor import process_pdf_bytes
from src.retrieval.qdrant_manager import get_qdrant_client, index_chunks, hybrid_search
from src.utils.secrets import get_groq_api_key

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(f"{APP_TITLE}: Context-Aware PDF QA System")
st.caption(APP_CAPTION)

# 1. Initialize API Keys and Models
api_key = get_groq_api_key()
if not api_key:
    st.error("GROQ_API_KEY is missing. Add it to .streamlit/secrets.toml or your environment.")
    st.stop()

llm = load_llm(api_key)
qdrant_client = get_qdrant_client()

# Sidebar Prompts
with st.sidebar:
    st.header("Demo prompts")
    for prompt_text in SUGGESTED_QUESTIONS:
        if st.button(prompt_text, use_container_width=True):
            st.session_state["question_text"] = prompt_text

# 2. File Upload & Ingestion
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is None:
    st.info("Upload a PDF to get started.")
    st.stop()

pdf_bytes = uploaded_file.getvalue()
file_hash = hashlib.sha256(pdf_bytes).hexdigest()

# Process only if it's a new file
if st.session_state.get("file_hash") != file_hash:
    try:
        with st.spinner("Extracting text and indexing into Qdrant (FastEmbed)..."):
            # Extract chunks
            chunks, page_count, chunk_count = process_pdf_bytes(pdf_bytes)
            
            # Send raw chunks to Qdrant for automatic dense/sparse indexing
            index_chunks(
                client=qdrant_client,
                chunks=chunks,
                doc_name=uploaded_file.name,
                doc_hash=file_hash
            )
            
    except Exception as exc:
        st.error(f"Could not process this PDF: {exc}")
        st.stop()

    st.session_state.update(
        {
            "file_hash": file_hash,
            "pages": page_count,
            "chunk_count": chunk_count,
            "file_name": uploaded_file.name,
        }
    )

st.success(
    f"{st.session_state['file_name']} | "
    f"{st.session_state['pages']} pages → {st.session_state['chunk_count']} chunks indexed in Qdrant."
)

# 3. Hybrid Search & Generation
question = st.text_input("Ask a question", key="question_text")
if question.strip():
    try:
        # Run Qdrant native hybrid search (Dense + BM25 via RRF)
        retrieved_hits = hybrid_search(
            client=qdrant_client,
            query_text=question,
            doc_hash=st.session_state["file_hash"],
            limit=CANDIDATE_POOL_K
        )
        
        if not retrieved_hits:
            st.warning("No readable or relevant content was found in this PDF.")
            st.stop()

        # Format context for the LLM
        context = "\n\n".join(
            f"[Page {hit['page_number']}]\n{hit['text']}" 
            for hit in retrieved_hits
        )
        
        with st.spinner("Generating answer..."):
            answer = generate_answer(context, question, llm)

        # UI Display
        st.subheader("Answer")
        st.write(answer)
        st.caption(f"Top {len(retrieved_hits)} hybrid retrieved chunks used.")

        with st.expander("Retrieved context"):
            st.write(context)

        st.subheader("Sources")
        for rank, hit in enumerate(retrieved_hits, start=1):
            st.write(f"Source {rank} | Page: {hit['page_number']} | Fusion Score: {hit['score']:.4f}")
            st.write(hit['text'][:400] + ("..." if len(hit['text']) > 400 else ""))
            st.divider()
            
    except Exception as exc:
        st.error(f"Something went wrong while answering your question: {exc}")