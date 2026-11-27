"""Document-scoped local semantic search and extractive question answering."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import load_document, save_uploaded_file
from src.document_scope import new_document_id
from src.config import MIN_RELEVANCE_SCORE
from src.rag_chain import answer_question
from src.text_splitter import split_documents
from src.vector_store import create_vector_store


def main():
    st.set_page_config(
        page_title="Technical Documentation Search Assistant",
        layout="wide",
    )
    st.title("Technical Documentation Search Assistant")
    st.write(
        "Upload a PDF, TXT or Markdown file and retrieve relevant excerpts "
        "from that document. Uses local embeddings and extractive responses; "
        "it does not call a generative LLM."
    )
    st.caption(
        "Each processed upload has a separate Chroma collection. "
        "Only the current document is searched in this browser session."
    )

    st.divider()
    st.subheader("1. Upload Document")
    uploaded_file = st.file_uploader(
        "Upload PDF, TXT, or Markdown file (up to 20 MB)",
        type=["pdf", "txt", "md"],
    )
    if uploaded_file is not None and st.button("Process Document"):
        try:
            with st.spinner("Saving and indexing the document..."):
                file_path = save_uploaded_file(uploaded_file)
                documents = load_document(
                    file_path, source_name=uploaded_file.name
                )
                chunks = split_documents(documents)
                document_id = new_document_id()
                create_vector_store(chunks, document_id=document_id)
        except (ValueError, OSError, RuntimeError) as exc:
            st.error(f"Document processing failed: {exc}")
        else:
            st.session_state["active_document_id"] = document_id
            st.session_state["document_name"] = uploaded_file.name
            st.success(f"Document processed: {len(chunks)} indexed chunks.")

    st.divider()
    st.subheader("2. Ask Questions")
    st.caption(
        f"Only excerpts with cosine similarity >= {MIN_RELEVANCE_SCORE:.2f} "
        "are shown. This heuristic threshold is not a confidence probability."
    )
    active_id = st.session_state.get("active_document_id")
    if active_id is None:
        st.info("Please upload and process a document first.")
        return

    st.write(f"Current document: `{st.session_state['document_name']}`")
    question = st.text_input(
        "Ask a question about the current document",
        placeholder="Example: What does this project do?",
    )
    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
            return
        try:
            with st.spinner("Retrieving relevant excerpts..."):
                result = answer_question(question, document_id=active_id)
        except (ValueError, RuntimeError) as exc:
            st.error(f"Search failed: {exc}")
            return
        st.subheader("Relevant Excerpts")
        st.write(result["answer"])
        st.subheader("Retrieved Sources")
        if not result["sources"]:
            st.info("No excerpts passed the relevance threshold.")
        for index, source in enumerate(result["sources"], start=1):
            page_label = (
                f" | PDF page {source['page'] + 1}"
                if isinstance(source.get("page"), int)
                else ""
            )
            relevance = source.get("relevance_score")
            score_label = (
                f" | Cosine similarity {relevance:.2f}"
                if isinstance(relevance, (int, float)) else ""
            )
            with st.expander(
                f"Source {index}: {source['source']} "
                f"| Chunk {source['chunk_id']}{page_label}{score_label}"
            ):
                st.write(source["preview"])


if __name__ == "__main__":
    main()
