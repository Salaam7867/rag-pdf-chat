import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# Page config
st.set_page_config(page_title="PDF Chat (Local RAG)", page_icon="📄")
st.title("📄 Chat with your PDF (Local RAG)")

# Load local LLM (CPU-friendly)
@st.cache_resource
def load_llm():
    return pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )

llm = load_llm()

# Embeddings (LOCAL)
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

def build_vectorstore(pdf_bytes):
    with open("temp.pdf", "wb") as f:
        f.write(pdf_bytes)

    docs = PyPDFLoader("temp.pdf").load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    return FAISS.from_documents(chunks, embeddings)

if uploaded_file:
    with st.spinner("Processing PDF..."):
        vectorstore = build_vectorstore(uploaded_file.getvalue())

    question = st.text_input("Ask a question")

    if question:
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n".join(d.page_content for d in docs)

        prompt = f"""
Answer ONLY using the context below.
If not found, say "Not found in document".

Context:
{context}

Question:
{question}

You: output after generating the answer from that context
"""

        response = llm(prompt)[0]["generated_text"]
        st.subheader("Answer")
        st.write(response)
