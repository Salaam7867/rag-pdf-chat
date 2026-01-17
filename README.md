# 📄 Local RAG System (FLAN-T5)

## Overview
This project implements a **fully local Retrieval-Augmented Generation (RAG) system** that allows users to query PDF documents without using paid LLM APIs.

The primary goal is to understand **RAG fundamentals from first principles**, including document ingestion, chunking strategies, semantic retrieval, and grounded answer generation using a lightweight local instruction model (FLAN-T5).

This project is intentionally scoped as a **baseline / experimental system**, not a production deployment.

---

## Key Features
- PDF document ingestion
- Text chunking with overlap for improved retrieval recall
- Semantic search using vector embeddings
- Local vector database using FAISS
- Context-grounded answers using a local LLM
- Source chunk visibility for transparency
- Fully offline execution (no API keys required)

---
