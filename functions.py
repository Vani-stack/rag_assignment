"""
functions.py
All custom functions for the RAG Assignment (Cambridgeshire Career Academy - Internship Assignment 2)
"""

import os
import cohere
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# TASK 1: Custom Document Loader
# ---------------------------------------------------------------------------
def load_and_chunk_document(file_path, chunk_size=300, overlap=50):
    """
    Loads text from a file (TXT or PDF) and splits it into overlapping chunks.

    Args:
        file_path (str): path to a .txt or .pdf file
        chunk_size (int): max characters per chunk
        overlap (int): overlap between chunks

    Returns:
        list[str]: list of text chunks
    """
    # 1. Read the file
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .txt or .pdf")

    # 2. Split into overlapping chunks (plain Python, no external splitter library)
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    # 3. Return chunks
    return chunks


# ---------------------------------------------------------------------------
# TASK 2: Custom Embedding Function
# ---------------------------------------------------------------------------
def create_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    """
    Creates embeddings for a list of text chunks.

    Args:
        chunks (list[str]): text chunks
        model_name (str): sentence-transformers model name

    Returns:
        list[list[float]]: list of embeddings, one per chunk
    """
    # 1. Load the model
    model = SentenceTransformer(model_name)

    # 2. Create embeddings
    embeddings = model.encode(chunks, show_progress_bar=False)

    # 3. Return embeddings as list of lists
    return embeddings.tolist()


# ---------------------------------------------------------------------------
# TASK 3: Search Function
# ---------------------------------------------------------------------------
def search_chunks(query, chunks, embeddings, k=3, model_name="all-MiniLM-L6-v2"):
    """
    Finds the top-k chunks most similar to the query.

    Args:
        query (str): user's question
        chunks (list[str]): original text chunks
        embeddings (list[list[float]]): embeddings for each chunk
        k (int): number of top results to return
        model_name (str): must match the model used in create_embeddings

    Returns:
        list[str]: top-k most similar chunks
    """
    # 1. Embed the query (same model as the chunks)
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query])

    # 2. Calculate cosine similarity
    similarities = cosine_similarity(query_embedding, np.array(embeddings))[0]

    # 3. Get top-k indices
    top_k_indices = np.argsort(similarities)[::-1][:k]

    # 4. Return top-k chunks
    top_chunks = [chunks[i] for i in top_k_indices]
    return top_chunks


# ---------------------------------------------------------------------------
# TASK 5: Cohere Answer Generation
# ---------------------------------------------------------------------------
def generate_answer(query, context, api_key):
    """
    Generates an answer to `query` using `context` chunks via the Cohere API.

    Args:
        query (str): user's question
        context (list[str] or str): retrieved chunks to use as context
        api_key (str): Cohere API key

    Returns:
        str: generated answer
    """
    # 1. Initialize Cohere client
    co = cohere.Client(api_key)

    # 2. Create a prompt with context + query
    if isinstance(context, list):
        context_text = "\n\n".join(context)
    else:
        context_text = context

    prompt = f"""Use the following context to answer the question.
If the answer is not contained in the context, say you don't know.

Context:
{context_text}

Question: {query}

Answer:"""

    # 3. Generate and return answer (command-r was retired; command-a-03-2025 is current)
    response = co.chat(
        message=prompt,
        model="command-a-03-2025",
        temperature=0.3,
    )
    return response.text
