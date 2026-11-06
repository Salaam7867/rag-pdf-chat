"""Qdrant local vector store management using FastEmbed."""

import uuid
from typing import Any, List, Optional
import streamlit as st
from qdrant_client import QdrantClient, models
from src.utils.secrets import get_qdrant_credentials
from src.config.settings import QDRANT_COLLECTION_NAME

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"
DENSE_VECTOR_SIZE = 384  # Output dimension of all-MiniLM-L6-v2


@st.cache_resource
def get_qdrant_client() -> QdrantClient:
    """Connect to the Qdrant Cloud cluster."""

    qdrant_url, qdrant_api_key = get_qdrant_credentials()

    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )

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
                    size=DENSE_VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            },
        )

    client.create_payload_index(
        collection_name=QDRANT_COLLECTION_NAME,
        field_name="doc_hash",
        field_schema=models.PayloadSchemaType.KEYWORD,
        wait=True,
    )

def document_exists(client: QdrantClient, doc_hash: str) -> bool:
    """Check whether a document is already indexed in Qdrant."""

    result = client.count(
        collection_name=QDRANT_COLLECTION_NAME,
        count_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_hash",
                    match=models.MatchValue(value=doc_hash),
                )
            ]
        ),
        exact=True,
    )

    return result.count > 0


def index_chunks(
    client: QdrantClient,
    chunks: List[Any],
    doc_name: str,
    doc_hash: str,
    doc_role: str = "general"
) -> None:
    """Index raw text chunks into Qdrant using models.Document for FastEmbed."""
    if not chunks:
        return

    # 1. Delete old stale chunks for this document hash if re-uploading
    try:
        client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_hash",
                            match=models.MatchValue(value=doc_hash),
                        )
                    ]
                )
            ),
        )
    except Exception:
        pass

    # 2. Build PointStructs using models.Document so FastEmbed encodes them automatically
    points = []
    for i, chunk in enumerate(chunks):
        page_num = chunk.metadata.get("page", 0) + 1
        chunk_unique_str = f"{doc_hash}_chunk_{i}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_unique_str))

        points.append(
            models.PointStruct(
                id=point_id,
                vector={
                    "dense": models.Document(text=chunk.page_content, model=DENSE_MODEL),
                    "sparse": models.Document(text=chunk.page_content, model=SPARSE_MODEL),
                },
                payload={
                    "doc_hash": doc_hash,
                    "doc_name": doc_name,
                    "doc_role": doc_role,
                    "chunk_index": i,
                    "page_number": page_num,
                    "text": chunk.page_content,
                },
            )
        )

    # 3. Upsert points into Qdrant
    client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=points,
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
                    model=DENSE_MODEL,
                ),
                using="dense",
                filter=query_filter,
                limit=limit,
            ),
            models.Prefetch(
                query=models.Document(
                    text=query_text,
                    model=SPARSE_MODEL,
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