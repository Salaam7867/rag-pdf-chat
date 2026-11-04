"""PDF extraction, chunking and document embedding."""

import os
import tempfile
from typing import Any

import numpy as np
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import CHUNK_OVERLAP, CHUNK_SIZE
from src.embeddings.provider import load_embeddings


@st.cache_data(show_spinner=False)
def process_pdf_bytes(pdf_bytes: bytes) -> tuple[list[Any], int, int, np.ndarray]:
    """Extract a PDF, split it into chunks and embed those chunks.

    Returns:
        (chunks, page_count, chunk_count, embedding_matrix)
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

        if not chunks:
            return [], len(documents), 0, np.empty((0, 0), dtype=np.float32)

        texts = [chunk.page_content for chunk in chunks]
        vectors = load_embeddings().embed_documents(texts)
        embedding_matrix = np.asarray(vectors, dtype=np.float32)

        return chunks, len(documents), len(chunks), embedding_matrix

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
