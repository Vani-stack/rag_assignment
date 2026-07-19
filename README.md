# rag_assignment

Internship Assignment 2 — Cambridgeshire Career Academy
A simple Retrieval-Augmented Generation (RAG) app built with Streamlit, Sentence-Transformers, and Cohere.

## What it does
1. **Load & chunk** an uploaded PDF/TXT document (`load_and_chunk_document`)
2. **Embed** the chunks with `sentence-transformers` (`create_embeddings`)
3. **Search** for the most relevant chunks to a user's question via cosine similarity (`search_chunks`)
4. **Generate** a final answer from the retrieved chunks using the Cohere API (`generate_answer`)

## Files
- `functions.py` – all custom functions (Tasks 1, 2, 3, 5)
- `app.py` – Streamlit app (Task 4 + 5)
- `notebook.ipynb` – development/testing notebook showing each function working
- `requirements.txt` – dependencies
- `screenshots/` – screenshots of the app running

## How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
You'll need a free [Cohere API key](https://dashboard.cohere.com/api-keys) to generate answers.

## How to run in Google Colab
See `notebook.ipynb` — it installs dependencies and tests each function step by step.
