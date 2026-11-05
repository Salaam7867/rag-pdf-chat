"""PDF extraction and chunking (Embeddings are now handled by Qdrant FastEmbed)."""

import os
import tempfile
from typing import Any

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import CHUNK_OVERLAP, CHUNK_SIZE


@st.cache_data(show_spinner=False)
def process_pdf_bytes(pdf_bytes: bytes) -> tuple[list[Any], int, int]:
    """Extract a PDF and split it into chunks.
    
    Returns:
        (chunks, page_count, chunk_count)
    """
    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            temp_path = tmp.name

        documents = PyPDFLoader(temp_path).load()
        if not documents:
            raise ValueError("No readable pages were found in this PDF.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(documents)

        return chunks, len(documents), len(chunks)

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)