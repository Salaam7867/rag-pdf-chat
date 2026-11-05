"""Cross-encoder re-ranking module."""

from typing import List, Dict, Any
import streamlit as st
from sentence_transformers import CrossEncoder

from src.config.settings import RERANKER_MODEL_NAME, TOP_K_FINAL


@st.cache_resource(show_spinner="Loading cross-encoder re-ranker...")
def load_reranker() -> CrossEncoder:
    """Load and cache the cross-encoder model."""
    return CrossEncoder(RERANKER_MODEL_NAME)


def rerank_hits(query: str, hits: List[Dict[str, Any]], top_k: int = TOP_K_FINAL) -> List[Dict[str, Any]]:
    """Re-rank hybrid search hits using a cross-encoder model."""
    if not hits:
        return []

    reranker = load_reranker()
    
    # Prepare query-document pairs for the cross-encoder
    pairs = [[query, hit["text"]] for hit in hits]
    
    # Predict relevance scores
    scores = reranker.predict(pairs)
    
    # Attach new scores to hits
    for hit, score in zip(hits, scores):
        hit["rerank_score"] = float(score)
        
    # Sort descending by cross-encoder score and take top_k
    ranked_hits = sorted(hits, key=lambda x: x["rerank_score"], reverse=True)
    return ranked_hits[:top_k]