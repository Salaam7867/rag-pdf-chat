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

# 1. Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 2. Render previous messages from history on every rerun
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "caption" in msg:
            st.caption(msg["caption"])

            with st.expander("Retrieved context"):
                st.caption(f"Context sent to LLM: {len(msg['context']):,} characters")
                st.text(msg["context"])

            with st.expander("Sources"):
                for rank, hit in enumerate(msg["sources"], start=1):
                    st.write(
                        f"**Source {rank}** | "
                        f"Document: {hit['doc_name']} | "
                        f"Page: {hit['page_number']} | "
                        f"Re-rank Score: {hit['rerank_score']:.4f}"
                    )
                    st.write(hit["text"][:400])
                    st.divider()

# 3. Accept input from either the bottom chat bar OR your sidebar Demo Prompt buttons
question = st.chat_input("Ask a question...") or st.session_state.pop("question_text", None)

if question and question.strip():
    # Display and save the user message immediately
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state["messages"].append({"role": "user", "content": question})

    try:
        all_hits = []
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
            all_hits.extend(document_hits)

            if document_hits:
                with st.spinner(f"Re-ranking evidence for {doc['file_name']}..."):
                    top_doc_hits = rerank_hits(question, document_hits, top_k=3)
                    final_hits.extend(top_doc_hits)

        if not final_hits:
            with st.chat_message("assistant"):
                st.warning("No relevant content found in the uploaded PDFs.")
            st.stop()

        # Sort the combined final_hits by re-rank score
        final_hits = sorted(final_hits, key=lambda x: x["rerank_score"], reverse=True)

        # Preserve document-level source attribution
        context = "\n\n".join(
            f"[Document: {hit['doc_name']} | Page: {hit['page_number']}]\n"
            f"{hit['text']}"
            for hit in final_hits
        )

        # Generate and display the assistant response inside a chat bubble
        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                answer = generate_answer(context, question, llm)

            st.markdown(answer)

            stats_caption = (
                f"Retrieved {len(all_hits)} candidates across "
                f"{len(active_docs)} documents. "
                f"Selected {len(final_hits)} chunks after per-document reranking."
            )
            st.caption(stats_caption)

            with st.expander("Retrieved context"):
                st.caption(f"Context sent to LLM: {len(context):,} characters")
                st.text(context)

            with st.expander("Sources"):
                for rank, hit in enumerate(final_hits, start=1):
                    st.write(
                        f"**Source {rank}** | "
                        f"Document: {hit['doc_name']} | "
                        f"Page: {hit['page_number']} | "
                        f"Re-rank Score: {hit['rerank_score']:.4f}"
                    )
                    st.write(hit["text"][:400])
                    st.divider()

        # Save the assistant message + metadata so it stays visible on future questions
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": answer,
                "caption": stats_caption,
                "context": context,
                "sources": final_hits,
            }
        )

    except Exception as exc:
        st.error(f"Something went wrong while answering: {exc}")