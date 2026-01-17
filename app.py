import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import tempfile
import os

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Local PDF RAG", layout="wide")
st.title("📄 Local RAG – Chat with PDF (No API)")

# -----------------------------
# Load LOCAL embeddings
# -----------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# -----------------------------
# Load LOCAL LLM (Mistral-7B-Instruct)
# -----------------------------
@st.cache_resource
def load_llm():
    return pipeline(
        "text2text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.1",
        device=-1,  # Set device to CPU; use device=0 for GPU if available
        max_new_tokens=200
    )

llm = load_llm()

# -----------------------------
# PDF upload
# -----------------------------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# -----------------------------
# Build vector store
# -----------------------------
def build_vectorstore(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.remove(pdf_path)
    return vectorstore

# -----------------------------
# Answer generation (Mistral-7B-Instruct)
# -----------------------------
def generate_answer(context, question):
    prompt = (
        "Answer the question using only the context below.\n"
        "If the answer is not in the context, say: Not found in document.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )
    result = llm(prompt)[0]["generated_text"]
    return result.strip()

# -----------------------------
# Main logic
# -----------------------------
if uploaded_file:
    with st.spinner("Indexing document..."):
        vectorstore = build_vectorstore(uploaded_file)

    question = st.text_input("Ask a question from the document")

    if question:
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        answer = generate_answer(context, question)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for i, doc in enumerate(docs, 1):
            st.write(f"Source {i}:")
            st.write(doc.page_content[:300] + "...")
