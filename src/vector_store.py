from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    RETRIEVAL_K,
    VECTORSTORE_DIR,
)


def get_embeddings():
    """
    Local embeddings using SentenceTransformers.
    This does not require OpenAI API key or billing.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vector_store(chunks: List[Document]) -> Chroma:
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR),
    )

    return vector_store


def load_vector_store() -> Chroma:
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(VECTORSTORE_DIR),
    )


def retrieve_relevant_chunks(question: str) -> List[Document]:
    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(
        search_kwargs={"k": RETRIEVAL_K}
    )

    return retriever.invoke(question)