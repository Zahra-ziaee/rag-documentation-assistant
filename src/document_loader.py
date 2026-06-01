from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from src.config import DOCUMENTS_DIR


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def save_uploaded_file(uploaded_file) -> Path:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    file_path = DOCUMENTS_DIR / uploaded_file.name

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    return file_path


def load_document(file_path: Path) -> List[Document]:
    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {suffix}. Supported types: PDF, TXT, MD."
        )

    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()

    else:
        loader = TextLoader(str(file_path), encoding="utf-8")
        documents = loader.load()

    for document in documents:
        document.metadata["source"] = file_path.name

    return documents