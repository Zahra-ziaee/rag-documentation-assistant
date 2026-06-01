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
    """
    No-cost answer generation.
    Instead of calling an LLM, this returns the most relevant retrieved context
    in a clean answer format.
    """
    if not documents:
        return "I could not find the answer in the uploaded document."

    answer_parts = []

    answer_parts.append(
        "Based on the uploaded document, the most relevant information is:"
    )

    for index, document in enumerate(documents[:3], start=1):
        source = document.metadata.get("source", "unknown")
        chunk_id = document.metadata.get("chunk_id", "unknown")
        content = document.page_content.strip()

        answer_parts.append(
            f"\n{index}. From `{source}`, chunk {chunk_id}:\n{content[:700]}"
        )

    answer_parts.append(
        "\nNote: This answer is generated from retrieved document chunks using local embeddings."
    )

    return "\n".join(answer_parts)


def answer_question(question: str) -> dict:
    relevant_chunks = retrieve_relevant_chunks(question)

    answer = create_extractive_answer(
        question=question,
        documents=relevant_chunks,
    )

    sources = []

    for document in relevant_chunks:
        sources.append(
            {
                "source": document.metadata.get("source", "unknown"),
                "chunk_id": document.metadata.get("chunk_id", "unknown"),
                "preview": document.page_content[:300],
            }
        )

    return {
        "answer": answer,
        "sources": sources,
    }