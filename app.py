"""Streamlit interface for the InsightRAG PDF question-answering app."""

import hashlib
import streamlit as st

from src.config.settings import APP_CAPTION, APP_TITLE, SUGGESTED_QUESTIONS, CANDIDATE_POOL_K
from src.generation.answer import generate_answer, load_llm
from src.ingestion.pdf_processor import process_pdf_bytes
from src.retrieval.qdrant_manager import get_qdrant_client, index_chunks, hybrid_search
from src.retrieval.reranker import rerank_hits  # 1. Import the re-ranker
from src.utils.secrets import get_groq_api_key

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(f"{APP_TITLE}: Context-Aware PDF QA System")
st.caption(APP_CAPTION)

api_key = get_groq_api_key()
if not api_key:
    st.error("GROQ_API_KEY is missing. Add it to .streamlit/secrets.toml or your environment.")
    st.stop()

llm = load_llm(api_key)
qdrant_client = get_qdrant_client()

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
        with st.spinner("Extracting text and indexing into Qdrant (FastEmbed)..."):
            chunks, page_count, chunk_count = process_pdf_bytes(pdf_bytes)
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

question = st.text_input("Ask a question", key="question_text")
if question.strip():
    try:
        # 2. Get initial 15 candidates via Qdrant Hybrid Search
        initial_hits = hybrid_search(
            client=qdrant_client,
            query_text=question,
            doc_hash=st.session_state["file_hash"],
            limit=CANDIDATE_POOL_K
        )
        
        if not initial_hits:
            st.warning("No readable or relevant content was found in this PDF.")
            st.stop()

        # 3. Apply the Cross-Encoder to filter down to the Top 4 best chunks
        with st.spinner("Re-ranking retrieved evidence..."):
            final_hits = rerank_hits(question, initial_hits)

        # 4. Use the refined final_hits for LLM context
        context = "\n\n".join(
            f"[Page {hit['page_number']}]\n{hit['text']}" 
            for hit in final_hits
        )
        
        with st.spinner("Generating answer..."):
            answer = generate_answer(context, question, llm)

        st.subheader("Answer")
        st.write(answer)
        st.caption(f"Top {len(final_hits)} re-ranked chunks used (filtered from {len(initial_hits)} initial candidates).")

        with st.expander("Retrieved context"):
            st.write(context)

        st.subheader("Sources")
        for rank, hit in enumerate(final_hits, start=1):
            st.write(f"Source {rank} | Page: {hit['page_number']} | Re-rank Score: {hit['rerank_score']:.4f}")
            st.write(hit['text'][:400] + ("..." if len(hit['text']) > 400 else ""))
            st.divider()
            
    except Exception as exc:
        st.error(f"Something went wrong while answering your question: {exc}")