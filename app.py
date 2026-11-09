"""Streamlit interface for the InsightRAG PDF question-answering app."""

import hashlib
import streamlit as st

from src.config.settings import APP_CAPTION, APP_TITLE, SUGGESTED_QUESTIONS, CANDIDATE_POOL_K
from src.generation.answer import generate_answer, load_llm
from src.ingestion.pdf_processor import process_pdf_bytes
from src.retrieval.qdrant_manager import (
    get_qdrant_client,
    index_chunks,
    hybrid_search,
    document_exists,
)
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

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)



if not uploaded_files:
    st.info("Upload one or more PDFs to get started.")
    st.stop()

# Initialise a session-level cache of documents already processed.
if "indexed_docs" not in st.session_state:
    st.session_state["indexed_docs"] = {}

# Calculate hashes and remove duplicate uploads.
unique_files = {}

for uploaded_file in uploaded_files:
    file_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()

    if file_hash not in unique_files:
        unique_files[file_hash] = uploaded_file

current_hashes = list(unique_files.keys())


# Avoid reprocessing the same selection on every Streamlit rerun.
active_docs = []

for file_hash, uploaded_file in unique_files.items():

    # Already processed during this Streamlit session?
    if file_hash in st.session_state["indexed_docs"]:
        active_docs.append(
            st.session_state["indexed_docs"][file_hash]
        )
        continue

    try:
        # Check Qdrant only if this session hasn't seen the PDF.
        if document_exists(qdrant_client, file_hash):

            doc_info = {
                "file_hash": file_hash,
                "file_name": uploaded_file.name,
                "already_indexed": True,
                "pages": None,
                "chunk_count": None,
            }

            st.session_state["indexed_docs"][file_hash] = doc_info
            active_docs.append(doc_info)
            continue

        # Process only genuinely new PDFs.
        with st.spinner(f"Processing {uploaded_file.name}..."):

            pdf_bytes = uploaded_file.getvalue()

            chunks, page_count, chunk_count = process_pdf_bytes(
                pdf_bytes
            )

            index_chunks(
                client=qdrant_client,
                chunks=chunks,
                doc_name=uploaded_file.name,
                doc_hash=file_hash,
            )

        doc_info = {
            "file_hash": file_hash,
            "file_name": uploaded_file.name,
            "already_indexed": False,
            "pages": page_count,
            "chunk_count": chunk_count,
        }

        st.session_state["indexed_docs"][file_hash] = doc_info
        active_docs.append(doc_info)

    except Exception as exc:
        st.error(f"Could not process {uploaded_file.name}: {exc}")

st.session_state["active_hashes"] = current_hashes
st.session_state["active_docs"] = active_docs

# Display indexing status.
active_docs = st.session_state.get("active_docs", [])

if not active_docs:
    st.error("No documents could be processed successfully.")
    st.stop()

st.subheader("Uploaded documents")

for doc in active_docs:

    if doc["already_indexed"]:
        st.success(
            f"{doc['file_name']} is already indexed. "
            "Skipping extraction and indexing."
        )
    else:
        st.success(
            f"{doc['file_name']} | "
            f"{doc['pages']} pages → "
            f"{doc['chunk_count']} chunks indexed."
        )

st.caption(f"{len(active_docs)} document(s) available.")

question = st.text_input("Ask a question", key="question_text")

if question.strip():
    try:
        all_hits = []

        # Retrieve candidates independently from each PDF.
        final_hits = []
        total_candidates = 0

        # Retrieve and Re-Rank independently for each PDF
        for doc in active_docs:
            document_hits = hybrid_search(
                client=qdrant_client,
                query_text=question,
                doc_hash=doc["file_hash"],
                limit=CANDIDATE_POOL_K,
            )
            
            total_candidates += len(document_hits)
            
            if document_hits:
                with st.spinner(f"Re-ranking evidence for {doc['file_name']}..."):
                    # Take the top chunks PER DOCUMENT (e.g., top 2 or 3)
                    top_doc_hits = rerank_hits(question, document_hits, top_k=3) 
                    final_hits.extend(top_doc_hits)

        if not final_hits:
            st.warning("No relevant content found in the uploaded PDFs.")
            st.stop()

        # Optionally sort the combined final_hits by re-rank score
        final_hits = sorted(final_hits, key=lambda x: x["rerank_score"], reverse=True)

        # Preserve document-level source attribution.
        context = "\n\n".join(
            f"[Document: {hit['doc_name']} | Page: {hit['page_number']}]\n"
            f"{hit['text']}"
            for hit in final_hits
        )

        with st.spinner("Generating answer..."):
            answer = generate_answer(context, question, llm)

        st.subheader("Answer")
        st.write(answer)

        st.caption(
            f"Used {len(final_hits)} re-ranked chunks "
            f"from {len(all_hits)} retrieved candidates."
        )

        with st.expander("Retrieved context"):
            st.write(context)

        st.subheader("Sources")

        for rank, hit in enumerate(final_hits, start=1):

            st.write(
                f"Source {rank} | "
                f"Document: {hit['doc_name']} | "
                f"Page: {hit['page_number']} | "
                f"Re-rank Score: {hit['rerank_score']:.4f}"
            )

            st.write(hit["text"][:400])
            st.divider()

    except Exception as exc:
        st.error(f"Something went wrong while answering: {exc}")
