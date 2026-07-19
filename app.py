"""
app.py
Simple Streamlit RAG interface (Task 4 + Task 5)
Run with: streamlit run app.py
"""

import streamlit as st
import tempfile
import os

from functions import (
    load_and_chunk_document,
    create_embeddings,
    search_chunks,
    generate_answer,
)

st.set_page_config(page_title="Mini RAG App", page_icon="📄")
st.title("📄 Mini RAG App")
st.write("Upload a document, ask a question, and get an AI-generated answer using retrieved context.")

# --- Sidebar: Cohere API key ---
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Cohere API Key", type="password")
k = st.sidebar.slider("Number of chunks to retrieve (k)", 1, 10, 3)

# --- File uploader ---
uploaded_file = st.file_uploader("Upload a document (PDF or TXT)", type=["pdf", "txt"])

# --- Text input for question ---
query = st.text_input("Ask a question about the document")

# --- Search button ---
if st.button("Search"):
    if not uploaded_file:
        st.warning("Please upload a file first.")
    elif not query:
        st.warning("Please enter a question.")
    else:
        # Save uploaded file to a temp path
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("1/4 Loading and chunking document..."):
            chunks = load_and_chunk_document(tmp_path)
        st.success(f"Document split into {len(chunks)} chunks.")

        with st.spinner("2/4 Creating embeddings..."):
            embeddings = create_embeddings(chunks)
        st.success("Embeddings created.")

        with st.spinner("3/4 Searching for relevant chunks..."):
            top_chunks = search_chunks(query, chunks, embeddings, k=k)

        st.subheader("🔎 Top matching chunks")
        for i, chunk in enumerate(top_chunks, 1):
            with st.expander(f"Chunk {i}"):
                st.write(chunk)

        # Task 5: Cohere answer generation
        if api_key:
            with st.spinner("4/4 Generating answer with Cohere..."):
                try:
                    answer = generate_answer(query, top_chunks, api_key)
                    st.subheader("💬 Generated Answer")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Cohere generation failed: {e}")
        else:
            st.info("Enter a Cohere API key in the sidebar to generate a full answer.")

        os.remove(tmp_path)
