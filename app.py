import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="RAG PDF Chat", page_icon="📄")
st.title("📄 Chat with your PDF (RAG)")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=st.secrets["GOOGLE_API_KEY"],
    temperature=0.2
)

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

@st.cache_resource
def build_vectorstore(pdf_bytes: bytes):
    with open("temp.pdf", "wb") as f:
        f.write(pdf_bytes)

    docs = PyPDFLoader("temp.pdf").load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(chunks, embeddings)

if uploaded_file:
    with st.spinner("Processing document..."):
        vectorstore = build_vectorstore(uploaded_file.getvalue())

    question = st.text_input("Ask a question from the document")

    if question:
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n".join(d.page_content for d in docs)

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
