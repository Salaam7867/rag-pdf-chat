"""Central configuration for the InsightRAG pipeline."""

APP_TITLE = "InsightRAG"
APP_CAPTION = "Context-Aware Multi-Document & Hybrid RAG Assistant"

# Chunking settings (enlarged to preserve clause and paragraph context)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# Retrieval & Re-ranking counts
CANDIDATE_POOL_K = 15  # Candidates retrieved from Hybrid Search (Dense + Sparse)
TOP_K_FINAL = 4        # Top chunks retained after Cross-Encoder re-ranking

# Models
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL_NAME = "openai/gpt-oss-120b"
LLM_TEMPERATURE = 0

# Qdrant configuration
QDRANT_COLLECTION_NAME = "insight_documents"
DENSE_VECTOR_SIZE = 384  # Dimension for all-MiniLM-L6-v2

SUGGESTED_QUESTIONS = [
    "Summarize this document in 5 bullet points.",
    "What are the key obligations and SLAs specified?",
    "Identify any penalties, liabilities, or termination clauses.",
    "What are the primary technical specifications or requirements?",
]