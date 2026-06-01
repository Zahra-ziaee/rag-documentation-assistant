import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import load_document, save_uploaded_file
from src.rag_chain import answer_question
from src.text_splitter import split_documents
from src.vector_store import create_vector_store


def main():
    st.set_page_config(
        page_title="RAG Documentation Assistant",
        layout="wide",
    )

    st.title("📚 RAG Technical Documentation Assistant")

    st.write(
        "Upload a technical document, build a local vector database, and ask "
        "questions based on the uploaded content. This version uses local "
        "SentenceTransformer embeddings and does not require OpenAI API billing."
    )

    st.divider()

    st.subheader("1. Upload Document")

    uploaded_file = st.file_uploader(
        "Upload PDF, TXT, or Markdown file",
        type=["pdf", "txt", "md"],
    )

    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Saving and processing document..."):
                file_path = save_uploaded_file(uploaded_file)

                documents = load_document(file_path)
                chunks = split_documents(documents)
                create_vector_store(chunks)

            st.success(
                f"Document processed successfully. Created {len(chunks)} chunks."
            )

            st.session_state["document_processed"] = True
            st.session_state["document_name"] = uploaded_file.name

    st.divider()

    st.subheader("2. Ask Questions")

    if "document_processed" not in st.session_state:
        st.info("Please upload and process a document first.")
        return

    st.write(f"Current document: `{st.session_state['document_name']}`")

    question = st.text_input(
        "Ask a question about the uploaded document",
        placeholder="Example: What does this project do?",
    )

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Retrieving relevant context..."):
            result = answer_question(question)

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Retrieved Sources")

        for index, source in enumerate(result["sources"], start=1):
            with st.expander(
                f"Source {index}: {source['source']} | Chunk {source['chunk_id']}"
            ):
                st.write(source["preview"])


if __name__ == "__main__":
    main()