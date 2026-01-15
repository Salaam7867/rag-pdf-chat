import os
import streamlit as st
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# Page config
st.set_page_config(page_title="RAG PDF Chat", page_icon="📄")
st.title("📄 Chat with your PDF (RAG)")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2
)

# File upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

@st.cache_resource(show_spinner=False)
def build_vectorstore(pdf_bytes):
    with open("temp.pdf", "wb") as f:
        f.write(pdf_bytes)

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    return FAISS.from_documents(chunks, embeddings)

if uploaded_file:
    with st.spinner("Processing document..."):
        vectorstore = build_vectorstore(uploaded_file.getvalue())

    question = st.text_input("Ask a question from the document")

    if question:
        relevant_docs = vectorstore.similarity_search(question, k=3)
        context = "\n".join(doc.page_content for doc in relevant_docs)

        prompt = f"""
Answer the question ONLY using the context below.
If the answer is not in the context, say "Not found in document".

Context:
{context}

Question:
{question}
"""

        answer = llm.invoke(prompt)
        st.subheader("Answer")
        st.write(answer.content)
