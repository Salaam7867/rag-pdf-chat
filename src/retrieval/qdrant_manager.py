"""Qdrant local vector store management using FastEmbed."""

import uuid
from typing import Any, List, Optional
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config.settings import QDRANT_COLLECTION_NAME


@st.cache_resource
def get_qdrant_client() -> QdrantClient:
    """Initialize a persistent local Qdrant client and set FastEmbed models."""
    # Persistent local storage (fixes data disappearing on restart)
    client = QdrantClient(path="./qdrant_storage")
    
    # Configure FastEmbed models
    client.set_model("sentence-transformers/all-MiniLM-L6-v2")
    client.set_sparse_model("Qdrant/bm25")
    
    init_collection(client)
    return client


def init_collection(client: QdrantClient) -> None:
    """Create the collection with explicit dense and sparse vector configurations."""
    collections = [col.name for col in client.get_collections().collections]
    if QDRANT_COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config={
                "dense": models.VectorParams(
                    size=client.get_embedding_size("sentence-transformers/all-MiniLM-L6-v2"),
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams()
            },
        )


def index_chunks(
    client: QdrantClient,
    chunks: List[Any],
    doc_name: str,
    doc_hash: str,
    doc_role: str = "general"
) -> None:
    """Index raw text chunks into Qdrant, clearing stale chunks first."""
    
    if not chunks:
        return

    # 1. DELETE OLD STALE CHUNKS: 
    # If this document was uploaded before with a different chunk count,
    # clear out all old points matching this doc_hash to prevent ghost chunks.
    try:
        client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_hash",
                        match=models.MatchValue(value=doc_hash),
                    )
                ]
            ),
        )
    except Exception:
        pass  # Collection might be empty initially

    documents = [chunk.page_content for chunk in chunks]
    metadata = []
    ids = []
    
    for i, chunk in enumerate(chunks):
        page_num = chunk.metadata.get("page", 0) + 1
        
        # Deterministic UUID based on doc_hash and index
        chunk_unique_str = f"{doc_hash}_chunk_{i}"
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_unique_str)))
        
        metadata.append({
            "doc_hash": doc_hash,
            "doc_name": doc_name,
            "doc_role": doc_role,
            "chunk_index": i,
            "page_number": page_num,
            "text": chunk.page_content,
        })

    # Add fresh chunks via FastEmbed
    client.add(
        collection_name=QDRANT_COLLECTION_NAME,
        documents=documents,
        metadata=metadata,
        ids=ids
    )


def hybrid_search(
    client: QdrantClient,
    query_text: str,
    doc_hash: Optional[str] = None,
    limit: int = 15,
) -> List[dict]:
    """Execute native hybrid search using Qdrant's FastEmbed document generation."""
    query_filter = None
    if doc_hash:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_hash",
                    match=models.MatchValue(value=doc_hash),
                )
            ]
        )

    hits = client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=models.Document(
                    text=query_text,
                    model="sentence-transformers/all-MiniLM-L6-v2",
                ),
                using="dense",
                filter=query_filter,
                limit=limit,
            ),
            models.Prefetch(
                query=models.Document(
                    text=query_text,
                    model="Qdrant/bm25",
                ),
                using="sparse",
                filter=query_filter,
                limit=limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
    )

    return [
        {
            "id": hit.id,
            "score": hit.score,
            "text": hit.payload["text"],
            "page_number": hit.payload["page_number"],
            "doc_name": hit.payload["doc_name"],
            "doc_hash": hit.payload["doc_hash"],
        }
        for hit in hits.points
    ]