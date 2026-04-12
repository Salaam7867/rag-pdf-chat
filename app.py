import os
import hashlib
import tempfile

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="RAG PDF Chat", layout="wide")
st.title("📄 RAG – Chat with PDF")

# -----------------------------
# Settings
# -----------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

# -----------------------------
# API key helper
# -----------------------------
def get_groq_api_key():
    try:
        if "GROQ_API_KEY" in st.secrets and st.secrets["GROQ_API_KEY"]:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY")


# -----------------------------
# Cached resources
# -----------------------------
@st.cache_resource
def load_embeddings_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_llm():
    api_key = get_groq_api_key()
    if not api_key:
        return None

    return ChatGroq(
        model=DEFAULT_MODEL_NAME,
        temperature=0,
        groq_api_key=api_key,
    )


embeddings_model = load_embeddings_model()
llm = load_llm()

# -----------------------------
# Vector store manager (in-memory, safe for Streamlit Cloud)
# -----------------------------
class VectorStoreManager:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.client = chromadb.Client()  # In-memory, not persistent
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG vector store for PDF embeddings"},
        )

    def add_documents(self, documents, embeddings):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents does not match number of embeddings")

        ids = []
        docs = []
        metadatas = []
        embs = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            ids.append(f"doc_{i}")
            docs.append(doc.page_content)
            metadatas.append(
                {
                    **dict(doc.metadata),
                    "chunk_index": i,
                    "content_length": len(doc.page_content),
                }
            )
            embs.append(embedding.tolist())

        self.collection.add(
            ids=ids,
            documents=docs,
            metadatas=metadatas,
            embeddings=embs,
        )

    def query(self, query_embedding, top_k=3):
        return self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )

    def count(self):
        return self.collection.count()


# -----------------------------
# PDF processing
# -----------------------------
@st.cache_data(show_spinner=False)
def extract_pdf_documents(pdf_bytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        temp_path = tmp.name

    try:
        loader = PyPDFLoader(temp_path)
        return loader.load()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def build_vectorstore(uploaded_file):
    pdf_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(pdf_bytes).hexdigest()  # nosec - used only for cache/keying
    collection_name = f"pdf_{file_hash[:10]}"

    raw_documents = extract_pdf_documents(pdf_bytes)
    chunks = split_documents(raw_documents)

    texts = [doc.page_content for doc in chunks]
    chunk_embeddings = embeddings_model.encode(texts, show_progress_bar=False)

    vector_store = VectorStoreManager(collection_name=collection_name)
    vector_store.add_documents(chunks, chunk_embeddings)

    return vector_store, len(raw_documents), len(chunks), collection_name


# -----------------------------
# LLM answer generation
# -----------------------------
def generate_answer(context, question):
    prompt = (
        "You are a helpful assistant. Answer using ONLY the context below. "
        "If the answer is not present, reply exactly: Not found in document.

"
        f"Context:
{context}

"
        f"Question:
{question}

"
        "Answer:"
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Settings")
st.sidebar.write(f"Embedding model: `{EMBEDDING_MODEL_NAME}`")
st.sidebar.write(f"LLM model: `{DEFAULT_MODEL_NAME}`")

if st.sidebar.button("Reset current session"):
    for key in ["vector_store", "file_hash", "file_name", "page_count", "chunk_count", "collection_name"]:
        st.session_state.pop(key, None)
    st.rerun()

# -----------------------------
# App body
# -----------------------------
api_key = get_groq_api_key()
if not api_key:
    st.error("Groq API key not found. Add GROQ_API_KEY in Streamlit secrets or environment variables.")
    st.stop()

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    current_hash = hashlib.md5(pdf_bytes).hexdigest()  # nosec - used only for cache/keying

    needs_rebuild = (
        "vector_store" not in st.session_state
        or st.session_state.get("file_hash") != current_hash
    )

    if needs_rebuild:
        with st.spinner("Indexing document..."):
            vector_store, page_count, chunk_count, collection_name = build_vectorstore(uploaded_file)

            st.session_state["vector_store"] = vector_store
            st.session_state["file_hash"] = current_hash
            st.session_state["file_name"] = uploaded_file.name
            st.session_state["page_count"] = page_count
            st.session_state["chunk_count"] = chunk_count
            st.session_state["collection_name"] = collection_name

    vector_store = st.session_state["vector_store"]

    st.success(f"Indexed {st.session_state['page_count']} pages into {st.session_state['chunk_count']} chunks.")
    st.caption(f"Collection: {st.session_state['collection_name']} | Stored chunks: {vector_store.count()}")

    question = st.text_input("Ask a question from the document")

    if question.strip():
        query_embedding = embeddings_model.encode([question])[0]
        results = vector_store.query(query_embedding, top_k=TOP_K)

        docs = []
        if results.get("documents") and results["documents"][0]:
            for i, doc_text in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                distance = results["distances"][0][i] if results.get("distances") else None
                docs.append({
                    "text": doc_text,
                    "metadata": metadata,
                    "distance": distance,
                    "rank": i + 1,
                })

        if not docs:
            st.warning("No relevant context found in the document.")
        else:
            context = "

".join([item["text"] for item in docs])
            answer = generate_answer(context, question)

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Sources")
            for item in docs:
                page_num = item["metadata"].get("page", "N/A")
                st.write(f"Source {item['rank']} | Page: {page_num} | Distance: {item['distance']}")
                st.write(item["text"][:500] + "...")
                st.json(item["metadata"])
                st.divider()
else:
    st.info("Upload a PDF to build the index and ask questions.")
