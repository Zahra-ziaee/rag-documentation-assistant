"""Source-grounded, extractive answers with relevance-based abstention."""

from typing import List

from langchain_core.documents import Document

from src.vector_store import retrieve_relevant_chunks


def format_context(documents: List[Document]) -> str:
    formatted_chunks = []
    for document in documents:
        source = document.metadata.get("source", "unknown")
        chunk_id = document.metadata.get("chunk_id", "unknown")
        formatted_chunks.append(
            f"[Source: {source} | Chunk: {chunk_id}]\n{document.page_content}"
        )
    return "\n\n---\n\n".join(formatted_chunks)


def create_extractive_answer(question: str, documents: List[Document]) -> str:
    """Show excerpts only when relevant, not an invented LLM answer."""
    if not documents:
        return (
            "I could not find sufficiently relevant information in the "
            "current document. Please rephrase your question or upload "
            "a document covering this topic."
        )

    answer_parts = [
        "Based on the uploaded document, the most relevant information is:"
    ]
    for index, document in enumerate(documents[:3], start=1):
        source = document.metadata.get("source", "unknown")
        chunk_id = document.metadata.get("chunk_id", "unknown")
        content = document.page_content.strip()
        answer_parts.append(
            f"\n{index}. From `{source}`, chunk {chunk_id}:\n{content[:700]}"
        )

    answer_parts.append(
        "\nNote: These are retrieved excerpts using local embeddings, "
        "not an LLM-generated answer. Relevance filtering is heuristic."
    )
    return "\n".join(answer_parts)


def answer_question(question: str, *, document_id: str) -> dict:
    relevant_chunks = retrieve_relevant_chunks(question, document_id=document_id)
    answer = create_extractive_answer(question, relevant_chunks)

    sources = []
    for document in relevant_chunks:
        sources.append(
            {
                "source": document.metadata.get("source", "unknown"),
                "chunk_id": document.metadata.get("chunk_id", "unknown"),
                "page": document.metadata.get("page"),
                "preview": document.page_content[:300],
                "relevance_score": document.metadata.get("relevance_score"),
            }
        )

    return {"answer": answer, "sources": sources}
