# InsightRAG — PDF Question Answering

InsightRAG lets users upload a PDF, ask natural-language questions, and receive answers grounded in retrieved document passages.

## Current pipeline

```text
PDF upload → PDF extraction → chunking → Hugging Face embeddings
→ NumPy cosine-similarity retrieval → Groq answer generation → source passages
```

## Project structure

```text
app.py
src/
  config/settings.py
  embeddings/provider.py
  ingestion/pdf_processor.py
  retrieval/semantic.py
  generation/answer.py
  utils/secrets.py
tests/
```

## Setup

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

Configure `GROQ_API_KEY` using either:

- Environment variable `GROQ_API_KEY`, or
- `.streamlit/secrets.toml` containing `GROQ_API_KEY = "your-key"`.

Run the application:

```bash
streamlit run app.py
```

Run the unit tests:

```bash
pytest -q
```

## Current scope

This milestone is a structural refactor. It intentionally preserves the existing embedding model, NumPy-based dense retrieval, Groq model, and Streamlit user experience. Persistent vector storage, hybrid retrieval, reranking, agentic workflows, and full RAG evaluation are planned for later milestones.
