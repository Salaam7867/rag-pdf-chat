# 🚀 InsightRAG – Context-Aware PDF QA System 
Link: https://rag-pdf-chat-7867.streamlit.app/

An AI-powered application that allows users to interact with PDF documents using natural language. Built using Retrieval-Augmented Generation (RAG) to ensure accurate and context-based answers.

---

## 🔍 Features

- 📄 Upload and analyze PDF documents
- 🧠 Intelligent context retrieval using embeddings
- 💬 Ask natural language questions
- ⚡ Fast responses using Groq (LLaMA 3)
- 🔎 Transparent retrieval (view source chunks)
- 🚫 Hallucination control (strict context-based answering)

---

## 🏗️ Architecture
PDF → Text Extraction → Chunking → Embeddings → Retrieval → LLM → Answer


---

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- HuggingFace Embeddings
- Groq API (LLaMA 3)
- NumPy

---

## ⚙️ How it Works

1. Upload a PDF
2. Document is split into smaller chunks
3. Each chunk is converted into embeddings
4. Relevant chunks are retrieved based on the query
5. LLM generates an answer using only retrieved context

---

## 🚀 Installation

```bash
git clone https://github.com/your-username/InsightRAG
cd InsightRAG
pip install -r requirements.txt
streamlit run app.py
