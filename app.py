import streamlit as st

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="RAG PDF Chat", page_icon="📄")
st.title("📄 Chat with your PDF (RAG)")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=st.secrets["GOOGLE_API_KEY"],
    temperature=0.2
)

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

def build_vectorstore(pdf_bytes: bytes):
    with open("temp.pdf", "wb") as f:
        f.write(pdf_bytes)

    docs = PyPDFLoader("temp.pdf").load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )

    return FAISS.from_documents(chunks, embeddings)

if uploaded_file:
    with st.spinner("Processing document..."):
        vectorstore = build_vectorstore(uploaded_file.getvalue())

    question = st.text_input("Ask a question from the document")

    if question:
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n".join(d.page_content for d in docs)

        response = llm.invoke(
            f"Answer ONLY from this context:\n{context}\n\nQuestion: {question}"
        )

        st.subheader("Answer")
        st.write(response.content)
