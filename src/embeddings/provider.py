"""Embedding model initialization."""

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.config.settings import EMBEDDING_MODEL_NAME


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings() -> HuggingFaceEmbeddings:
    """Load and cache the existing sentence-transformer embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
