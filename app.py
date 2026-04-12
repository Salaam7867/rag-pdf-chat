import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="RAG PDF Chat", layout="wide")
st.title("📄 RAG – Chat with PDF")

# -----------------------------
# Settings
# -----------------------------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

# -----------------------------
# API KEY
# -----------------------------
def get_groq_api_key():
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except:
        pass
    return os.getenv("GROQ_API_KEY")

# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_llm():
    key = get_groq_api_key()
    if not key:
        return None

    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=key
    )

embeddings = load_embeddings()
llm = load_llm()

# -----------------------------
# PDF → chunks → FAISS
# -----------------------------
def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        path = tmp.name

    loader = PyPDFLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.remove(path)
    return vectorstore, len(docs), len(chunks)

# -----------------------------
# Answer
# -----------------------------
def generate_answer(context, question):
    prompt = (
        "Answer using ONLY the context below.\n"
        "If not found, say: Not found in document.\n\n"
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
    st.error("Add GROQ_API_KEY in secrets")
    st.stop()

file = st.file_uploader("Upload PDF", type="pdf")

if file:
    with st.spinner("Processing..."):
        vectorstore, pages, chunks = process_pdf(file)

    st.success(f"{pages} pages → {chunks} chunks")

    question = st.text_input("Ask a question")

    if question:
        docs = vectorstore.similarity_search(question, k=TOP_K)
        context = "\n\n".join([d.page_content for d in docs])

        answer = generate_answer(context, question)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for i, d in enumerate(docs):
            st.write(f"Source {i+1}")
            st.write(d.page_content[:300] + "...")
else:
    st.info("Upload a PDF")
