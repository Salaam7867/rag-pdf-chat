# 🚀 InsightRAG – Context-Aware PDF Question Answering

**InsightRAG** is an AI-powered application that enables users to interact with PDF documents using natural language. It leverages Retrieval-Augmented Generation (RAG) to deliver precise, context-grounded answers while minimizing hallucinations.

---

## 🔥 Key Highlights

* 📄 Upload and analyze PDF documents instantly
* 🧠 Context-aware retrieval using semantic embeddings
* 💬 Ask questions in natural language
* ⚡ Low-latency responses powered by Groq (LLaMA 3)
* 🔎 Source transparency with retrievable document chunks
* 🚫 Strict context-based answering to reduce hallucinations

---

## 🧠 How It Works

```
PDF → Text Extraction → Chunking → Embeddings → Retrieval → LLM → Answer
```

1. Upload a PDF document
2. Extract and split text into manageable chunks
3. Convert chunks into vector embeddings
4. Retrieve the most relevant chunks for a query
5. Generate answers using only retrieved context

---

## 🏗️ Tech Stack

* **Backend:** Python
* **Frontend:** Streamlit
* **Framework:** LangChain
* **Embeddings:** HuggingFace
* **LLM:** Groq API (LLaMA 3)
* **Utilities:** NumPy

---

## 🚀 Live Demo

Try it here:
👉 https://rag-pdf-chat-7867.streamlit.app/

---

## ⚙️ Installation & Setup

```bash
git clone https://github.com/Salaam7867/InsightRAG
cd InsightRAG
pip install -r requirements.txt
streamlit run app.py
```

---

## 🎯 Use Cases

* Research paper analysis
* Legal/document review
* Study material Q&A
* Business report insights

---

## 📌 Future Improvements

* Multi-document querying
* Chat history memory
* Advanced ranking (rerankers)
* Support for more file formats

---

## 🤝 Contributing

Contributions are welcome. Fork the repo, create a feature branch, and submit a pull request.

---

## 📄 License

This project is open-source and available under the MIT License.
