import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import tempfile, os

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Local RAG (Phi-2)", layout="wide")
st.title("📄 Local RAG – Phi-2 (No API, No Echo)")

# -----------------------------
# Embeddings
# -----------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# -----------------------------
# LLM (Phi-2)
# -----------------------------
@st.cache_resource
def load_llm():
    return pipeline(
        "text-generation",
        model="microsoft/phi-2",
        torch_dtype="auto",
        device=-1,
        max_new_tokens=200,
        do_sample=False,
        temperature=0.0
    )

llm = load_llm()

# -----------------------------
# Vector store
# -----------------------------
def build_vectorstore(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    docs = PyPDFLoader(path).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    os.remove(path)
    return FAISS.from_documents(chunks, embeddings)

# -----------------------------
# Answer generation (IMPORTANT)
# -----------------------------
def answer_question(context, question):
    prompt = (
        "You are a factual assistant.\n"
        "Answer ONLY using the context.\n"
        "If the answer is missing, say: Not found in document.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    output = llm(prompt)[0]["generated_text"]
    return output.split("Answer:")[-1].strip()

# -----------------------------
# App logic
# -----------------------------
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with st.spinner("Indexing document..."):
        vectorstore = build_vectorstore(uploaded_file)

    question = st.text_input("Ask a question")

    if question:
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join(d.page_content for d in docs)

        answer = answer_question(context, question)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for i, d in enumerate(docs, 1):
            st.write(f"Source {i}:")
            st.write(d.page_content[:300] + "...")
