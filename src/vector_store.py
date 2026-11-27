"""Document-scoped Chroma retrieval with explicit semantic relevance filtering.

The score is calculated directly from the embedding vectors. This deliberately
avoids assuming whether an existing Chroma collection stores cosine distance,
inner-product distance, or squared L2 distance. No re-indexing is required.
"""

from functools import lru_cache
from math import sqrt
from typing import List, Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    EMBEDDING_MODEL,
    MIN_RELEVANCE_SCORE,
    RETRIEVAL_K,
    VECTORSTORE_DIR,
)
from src.document_scope import collection_name_for


@lru_cache(maxsize=1)
def get_embeddings():
    """Load one embedding model per Python process."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vector_store(chunks: List[Document], *, document_id: str) -> Chroma:
    """Store one document in its own collection."""
    collection_name = collection_name_for(document_id)
    if not chunks:
        raise ValueError("Cannot index a document without text chunks.")

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        chunk.metadata["document_id"] = document_id

    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=collection_name,
        persist_directory=str(VECTORSTORE_DIR),
    )


def load_vector_store(*, document_id: str) -> Chroma:
    """Open only the collection of the selected document."""
    collection_name = collection_name_for(document_id)
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(VECTORSTORE_DIR),
    )


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Calculate cosine similarity independent of vector database distance."""
    if len(a) != len(b) or not a:
        raise ValueError("Embedding vectors must have equal nonzero dimensions.")

    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    norm_a = sqrt(sum(float(x) ** 2 for x in a))
    norm_b = sqrt(sum(float(y) ** 2 for y in b))
    if not norm_a or not norm_b:
        return 0.0

    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


def retrieve_relevant_chunks(question: str, *, document_id: str) -> List[Document]:
    """Only return current-document chunks above the similarity threshold.

    Chroma retrieves candidates; then cosine similarity is recomputed using
    the local embeddings model. This also works with existing L2 collections.
    The returned Document objects carry a display-only relevance_score field.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must not be empty.")

    # Fail closed if an invalid document ID was supplied.
    store = load_vector_store(document_id=document_id)
    candidates = store.as_retriever(search_kwargs={"k": RETRIEVAL_K}).invoke(
        question.strip()
    )
    if not candidates:
        return []

    # An additional check prevents accidentally returning a chunk from a
    # different document even if a store is misconfigured.
    candidates = [
        document
        for document in candidates
        if document.metadata.get("document_id") == document_id
    ]
    if not candidates:
        return []

    embeddings = get_embeddings()
    query_embedding = embeddings.embed_query(question.strip())
    document_embeddings = embeddings.embed_documents(
        [document.page_content for document in candidates]
    )
    if len(document_embeddings) != len(candidates):
        raise RuntimeError("Embedding model returned an unexpected number of vectors.")

    relevant = []
    for document, vector in zip(candidates, document_embeddings):
        score = cosine_similarity(query_embedding, vector)
        if score >= MIN_RELEVANCE_SCORE:
            document.metadata["relevance_score"] = round(score, 4)
            relevant.append((document, score))

    relevant.sort(key=lambda pair: pair[1], reverse=True)
    return [document for document, _ in relevant]
