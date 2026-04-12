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
# Vector store (in-memory)
# -----------------------------
class VectorStoreManager:
    def __init__(self, collection_name: str):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, documents, embeddings):
        ids, docs, metas, embs = [], [], [], []

        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            ids.append(f"doc_{i}")
            docs.append(doc.page_content)
            metas.append(doc.metadata)
            embs.append(emb.tolist())

        self.collection.add(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embs,
        )

    def query(self, embedding, k=3):
        return self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=k,
        )

# -----------------------------
# PDF processing
# -----------------------------
@st.cache_data
def extract_pdf(pdf_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        path = tmp.name

    loader = PyPDFLoader(path)
    docs = loader.load()

    os.remove(path)
    return docs

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)

def build_vectorstore(file):
    pdf_bytes = file.getvalue()
    docs = extract_pdf(pdf_bytes)
    chunks = split_docs(docs)

    texts = [d.page_content for d in chunks]
    embeddings = embeddings_model.encode(texts)

    vs = VectorStoreManager("pdf_store")
    vs.add_documents(chunks, embeddings)

    return vs, len(docs), len(chunks)

# -----------------------------
# Answer generation
# -----------------------------
def generate_answer(context, question):
    prompt = (
        "You are a helpful assistant. Answer using ONLY the context below.\n"
        "If the answer is not present, reply exactly: Not found in document.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

# -----------------------------
# App
# -----------------------------
if not llm:
    st.error("❌ GROQ_API_KEY not found")
    st.stop()

file = st.file_uploader("Upload PDF", type="pdf")

if file:
    with st.spinner("Processing..."):
        vs, pages, chunks = build_vectorstore(file)

    st.success(f"{pages} pages → {chunks} chunks")

    question = st.text_input("Ask something")

    if question:
        q_emb = embeddings_model.encode([question])[0]
        results = vs.query(q_emb, TOP_K)

        docs = results["documents"][0]
        context = "\n\n".join(docs)

        answer = generate_answer(context, question)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for i, d in enumerate(docs):
            st.write(f"Source {i+1}")
            st.write(d[:300] + "...")
else:
    st.info("Upload a PDF")
