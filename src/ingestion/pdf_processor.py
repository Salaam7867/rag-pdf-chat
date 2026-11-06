"""PDF extraction and chunking with thread-safe / robust extraction."""

import io
import os
import tempfile
from typing import Any, List

import streamlit as st
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import CHUNK_OVERLAP, CHUNK_SIZE


@st.cache_data(show_spinner=False)
def process_pdf_bytes(pdf_bytes: bytes) -> tuple[List[Document], int, int]:
    """Extract a PDF and split it into chunks safely.
    
    Returns:
        (chunks, page_count, chunk_count)
    """
    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    documents: List[Document] = []
    
    # Use direct in-memory stream with pypdf and strict=False
    stream = io.BytesIO(pdf_bytes)
    reader = PdfReader(stream, strict=False)
    page_count = len(reader.pages)

    if page_count == 0:
        raise ValueError("No readable pages were found in this PDF.")

    for i, page in enumerate(reader.pages):
        try:
            # Extract plain text per page defensively
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"page": i}
                    )
                )
        except Exception:
            # Skip corrupted or non-extractable single pages without halting the document
            continue

    if not documents:
        raise ValueError("No extractable text was found in this PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    return chunks, page_count, len(chunks)